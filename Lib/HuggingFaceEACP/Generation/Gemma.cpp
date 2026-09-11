#include "Gemma.h"

#include "ResourcesDirectory.h"

#include <HuggingFaceEACP/Model/ModelError.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>

namespace HF
{
using eacp::GPU::Buffer;
using eacp::GPU::BufferRange;
using eacp::GPU::BufferUsage;
using eacp::GPU::CommandBuffer;
using eacp::GPU::ComputePass;
using eacp::GPU::Device;
using eacp::GPU::DispatchOrder;

namespace
{
constexpr auto slotBytes = (int) sizeof(std::uint32_t);

constexpr int floatBytes(int elementCount)
{
    return elementCount * (int) sizeof(float);
}

bool isFile(const std::filesystem::path& path)
{
    auto error = std::error_code {};
    return std::filesystem::is_regular_file(path, error);
}

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// eacp's Buffer takes its size as an int, which plan.md's second gap is about:
// a logits buffer is the prompt capacity times the vocabulary, so at Gemma's
// 256,000 columns a capacity of 2048 rows is past what an int holds. Counted in
// 64 bits and refused here, because the alternative is a truncated allocation
// and a kernel writing outside it.
int requireBufferBytes(std::int64_t elements, std::string_view what)
{
    const auto bytes = elements * (std::int64_t) sizeof(float);

    if (bytes > (std::int64_t) std::numeric_limits<int>::max())
        throw ModelError {"the " + std::string {what} + " would be "
                          + std::to_string(bytes)
                          + " bytes, and a GPU buffer's size is an int — see "
                            "plan.md's second gap"};

    return (int) bytes;
}
} // namespace

void Gemma::load(const std::filesystem::path& modelDirectory)
{
    load(ModelFiles::fromDirectory(modelDirectory));
}

// config.json first, since it is what says the tokenizer's vocabulary is the
// one this model gathers from, and generation_config.json rides along inside
// GemmaConfig::fromModelFiles — the repo's authority on the three token ids.
void Gemma::load(const ModelFiles& files)
{
    modelConfig = GemmaConfig::fromModelFiles(files);

    buildTokenizer(files);

    weightFile.emplace(ShardedTensors::fromModelFiles(files));

    maximumTokenCount = 0;
}

std::filesystem::path Gemma::bundledModelDirectory()
{
    return resourcesDirectory() / bundledModelDirectoryName;
}

bool Gemma::hasBundledModel()
{
    const auto directory = bundledModelDirectory();

    const auto hasWeights = isFile(directory / ModelFileNames::weights)
                            || isFile(directory / ModelFileNames::shardIndex);

    return isFile(directory / ModelFileNames::config) && hasWeights
           && isFile(directory / ModelFileNames::tokenizerJson);
}

// No directory at all is a build that never asked for the copy, and the
// message says how to ask. One that is there but short a file is a copy that
// did not finish, and load() names the file the way it would for any
// directory.
void Gemma::loadBundled()
{
    const auto directory = bundledModelDirectory();
    auto error = std::error_code {};

    if (!std::filesystem::is_directory(directory, error))
        throw ModelError {"this binary ships no model: there is no "
                          + directory.string()
                          + "; a model is copied there by calling "
                            "hf_bundle_model(<target>) in the target's "
                            "CMakeLists, in a build configured with "
                            "-DHF_EACP_FETCH_MODEL=ON"};

    load(directory);
}

// ModelFiles records the tokenizer without requiring it, because a caller
// loading weights to test a kernel has no use for one. Generation does, so the
// requirement is stated here and names the file.
void Gemma::buildTokenizer(const ModelFiles& files)
{
    if (files.tokenizerJson.empty())
        throw ModelError {"the model directory " + files.directory.string()
                          + " has no " + ModelFileNames::tokenizerJson
                          + ", and a string in and a string out needs one"};

    vocabulary.emplace(Tokenizer::fromFile(files.tokenizerJson));

    // A token past the embedding table is a gather outside the weights, which
    // is silent on both backends. The other way round is ordinary — a config
    // may pad its vocabulary past the pieces the tokenizer names.
    if (vocabulary->vocabularySize() > modelConfig.vocabularySize)
        throw ModelError {"this tokenizer names "
                          + std::to_string(vocabulary->vocabularySize())
                          + " tokens and config.json's embedding holds "
                          + std::to_string(modelConfig.vocabularySize)};
}

const Tokenizer& Gemma::tokenizer() const
{
    requireLoaded();
    return *vocabulary;
}

const DecoderShape& Gemma::shape() const
{
    requirePrepared();
    return decoderShape;
}

void Gemma::requireLoaded() const
{
    if (!isLoaded())
        throw ModelError {"this Gemma has not loaded a model yet"};
}

void Gemma::requirePrepared() const
{
    if (!isPrepared())
        throw ModelError {"this Gemma has not been prepared on a device yet"};
}

void Gemma::setMaximumTokens(int count)
{
    if (count < 0)
        throw ModelError {"a token limit is a count, or zero for the window's "
                          "own bound, and this one is "
                          + std::to_string(count)};

    maximumTokenCount = count;
}

void Gemma::setPromptCapacity(int rows)
{
    if (rows <= 0)
        throw ModelError {"a prefill block has to carry at least one row"};

    promptRowCapacity = rows;
}

// Through a Sampler rather than stored raw, so the options are validated where
// SamplingError says they are and a temperature that describes no distribution
// is refused here rather than at the first draw.
void Gemma::setSampling(SamplingOptions options)
{
    sampler = Sampler {options};
    samplingOptions = sampler.options();
}

// The window is the caches', and the prompt capacity is every per-step
// intermediate's — the separation DecoderShape::maxStepRows exists for, and the
// difference between 302 MB of KV cache and 2.2 GB of buffers a decode step
// uses one row of.
void Gemma::prepare(Device& device)
{
    requireLoaded();

    gpu = &device;

    decoderShape = DecoderShape::fromConfig(modelConfig);
    decoderShape.maxStepRows =
        std::min(promptRowCapacity, decoderShape.maxPositions);
    decoderShape.validate();

    // The weights first, because the decoder compiles the product programs
    // their storage needs and no others — see Decoder::prepare.
    weights.emplace(*weightFile, decoderShape);

    decoder.emplace(decoderShape);
    decoder->prepare(device, *weights);

    selection.prepare(device, 1, decoderShape.logitElementCount());

    buildSuppressionMask();

    suppressionBuffer.emplace(device.makeBuffer(
        suppression.data(), floatBytes(suppression.size()), BufferUsage::Storage));

    const auto logitElements = (std::int64_t) decoderShape.stepRowCapacity()
                               * decoderShape.logitElementCount();

    logitBuffer.emplace(device.makeBuffer(
        requireBufferBytes(logitElements, "logits buffer"), BufferUsage::Storage));

    sequenceTokens.emplace(device.makeBuffer(
        slotBytes * (decoderShape.maxPositions + 1), BufferUsage::Storage));

    logitsRow.resize(decoderShape.logitElementCount());
}

void Gemma::prepare()
{
    prepare(Device::shared());
}

// Zeros, because gemma-2b's generation_config suppresses nothing: there is no
// suppress_tokens list and no begin_suppress_tokens, where Whisper's config has
// ninety of the first and two of the second. Kept and bound anyway so that a
// repo which does carry one needs a fill here and nothing else — and because
// neither backend defines what a shader reading an unbound buffer gets.
void Gemma::buildSuppressionMask()
{
    suppression.clear();
    suppression.resize(modelConfig.vocabularySize);

    for (auto index = 0; index < suppression.size(); ++index)
        suppression[index] = 0.f;
}

void Gemma::uploadPrompt(Span<const TokenId> promptTokens)
{
    promptIds.resize(promptTokens.size());

    for (auto index = 0; index < promptIds.size(); ++index)
        promptIds[index] = (std::uint32_t) promptTokens[index];

    sequenceTokens->update(promptIds.data(), slotBytes * promptIds.size());
}

BufferRange Gemma::sequenceSlots(int first, int count) const
{
    return {&*sequenceTokens, first * slotBytes, count * slotBytes};
}

TokenId Gemma::tokenInSlot(CommandBuffer& commands, int slot) const
{
    auto id = std::uint32_t {};
    commands.read(*sequenceTokens, &id, slotBytes, slotBytes * slot);

    return (TokenId) id;
}

TokenId Gemma::tokenInSlot(int slot) const
{
    auto id = std::uint32_t {};
    sequenceTokens->read(&id, slotBytes, slotBytes * slot);

    return (TokenId) id;
}

void Gemma::writeTokenToSlot(TokenId token, int slot)
{
    const auto id = (std::uint32_t) token;
    sequenceTokens->update(&id, slotBytes, slotBytes * slot);
}

// Greedy sampling over the last row of a step's logits, which is the
// distribution over the token that follows everything decoded so far. The row
// is bound as a range rather than the whole buffer — the same ranged bind the
// KV cache needed — so a prefill block's earlier rows are neither scanned nor
// copied, and the answer lands in the sequence slot the next step embeds.
void Gemma::encodeSampling(ComputePass& pass, int rowCount, int slot)
{
    const auto rowLength = decoderShape.logitElementCount();
    const auto rowBytes = floatBytes(rowLength);

    selection.encode(
        pass,
        BufferRange {&*logitBuffer, (rowCount - 1) * rowBytes, rowBytes},
        *suppressionBuffer,
        sequenceSlots(slot, 1),
        1,
        rowLength);
}

void Gemma::encodeGeneratedStep(ComputePass& pass,
                                int promptLength,
                                int step,
                                bool samples)
{
    decoder->step(
        pass, sequenceSlots(promptLength + step - 1, 1), 1, *weights, *logitBuffer);

    if (!samples)
        return;

    pass.barrier();
    encodeSampling(pass, 1, promptLength + step);
}

int Gemma::finalPrefillRows(int promptLength) const
{
    const auto capacity = decoderShape.stepRowCapacity();
    const auto remainder = promptLength % capacity;

    return remainder == 0 ? capacity : remainder;
}

// A block per command buffer, each committed before the next is recorded. The
// blocks exist because the per-step intermediates hold promptCapacity() rows
// and a prompt may be longer; they change nothing about the rows, since the
// decoder appends to the same caches at the same positions either way.
//
// Only the last block's last row is ever sampled: the rows before it are the
// distributions over tokens the prompt already names.
double Gemma::prefill(int promptLength, bool samples)
{
    const auto capacity = decoderShape.stepRowCapacity();
    const auto start = Clock::now();

    for (auto first = 0; first < promptLength; first += capacity)
    {
        const auto rows = std::min(capacity, promptLength - first);
        const auto isLast = first + rows >= promptLength;

        auto commands = gpu->makeCommandBuffer();

        {
            auto pass = commands.beginCompute({}, DispatchOrder::Concurrent);

            decoder->step(
                pass, sequenceSlots(first, rows), rows, *weights, *logitBuffer);

            if (isLast && samples)
            {
                pass.barrier();
                encodeSampling(pass, rows, promptLength);
            }
        }

        commands.commit();
        ++stepCount;
    }

    return secondsSince(start);
}

void Gemma::readLastLogitsRow(int rowCount)
{
    const auto rowLength = decoderShape.logitElementCount();

    logitBuffer->read(logitsRow.data(),
                      floatBytes(rowLength),
                      floatBytes((rowCount - 1) * rowLength));
}

// The sequence is the prompt and everything after it, and the window bounds the
// whole of it — so a prompt of maxPositions - 1 tokens has room for one. This is
// HF's max_length rule, and it is one token stricter than WhisperEACP's, which
// allows a last token that is sampled and never fed back.
int Gemma::tokenLimitFor(int promptLength) const
{
    const auto window = decoderShape.maxPositions - promptLength;

    return maximumTokenCount > 0 ? std::min(maximumTokenCount, window) : window;
}

Vector<TokenId> Gemma::generate(Span<const TokenId> promptTokens)
{
    requirePrepared();

    const auto promptLength = promptTokens.size();

    if (promptLength <= 0)
        throw ModelError {"a run needs at least one prompt token: gemma-2b is "
                          "trained to see its beginning-of-sequence token in "
                          "front of every sequence, which encodeWithBos puts "
                          "there"};

    if (promptLength > decoderShape.maxPositions)
        throw ModelError {"this prompt holds " + std::to_string(promptLength)
                          + " tokens and the window is "
                          + std::to_string(decoderShape.maxPositions)};

    for (auto index = 0; index < promptLength; ++index)
        if (promptTokens[index] < 0
            || promptTokens[index] >= decoderShape.vocabularySize)
            throw ModelError {"prompt token " + std::to_string(promptTokens[index])
                              + " is outside a vocabulary of "
                              + std::to_string(decoderShape.vocabularySize)};

    decoder->beginSequence();
    uploadPrompt(promptTokens);

    prefillSeconds = 0.0;
    decodeSeconds = 0.0;
    stepCount = 0;

    const auto tokenLimit = tokenLimitFor(promptLength);

    // Temperature zero is the on-device path, which is both the default and the
    // fast one: nothing about a token reaches the host until after it was
    // chosen. Anything above it is a draw, and a draw is on the CPU behind a
    // readback.
    const auto greedy = samplingOptions.temperature == 0.f;

    prefillSeconds = prefill(promptLength, greedy && tokenLimit > 0);

    if (tokenLimit <= 0)
        return {};

    return greedy ? generateGreedily(promptLength, tokenLimit)
                  : generateBySampling(promptLength, tokenLimit);
}

// One command buffer to a step and stepsInFlight of them in the air: step k + 1
// is recorded and submitted before the host asks the GPU for step k, so the two
// overlap instead of taking turns. Each step's token is read out of that step's
// own command buffer, which waits for it alone — Buffer::read waits for the
// newest submission, and would therefore wait for the step still running.
//
// The tail starts at step one because token zero is the prefill's: the Argmax
// over the last prompt row already wrote it into slot promptLength, and it is
// there to be read the moment the prefill's last commit returns.
Vector<TokenId> Gemma::generateGreedily(int promptLength, int tokenLimit)
{
    const auto endOfSequence = modelConfig.endOfSequenceToken;

    // A CommandBuffer is neither copyable nor movable, so the ring holds them
    // in place. Slot j % stepsInFlight belongs to step j, and is replaced only
    // once that step's token has been read.
    auto inFlight = std::array<std::optional<CommandBuffer>, stepsInFlight> {};

    const auto startStep = [&](int step)
    {
        auto& commands = inFlight[step % stepsInFlight].emplace(*gpu);

        // Concurrent because Decoder.cpp spells its ordering out: a barrier
        // sits at every boundary where a stage reads what the stage before it
        // wrote, and the one place with none — a layer's three projections,
        // which read the same normalised rows and write three buffers nothing
        // else has touched — is exactly what a concurrent pass lets overlap.
        {
            auto pass = commands.beginCompute({}, DispatchOrder::Concurrent);
            encodeGeneratedStep(pass, promptLength, step, true);
        }

        commands.submit();
        ++stepCount;
    };

    auto generated = Vector<TokenId> {};
    auto nextStep = 1;

    const auto decodeStart = Clock::now();

    while (nextStep < stepsInFlight && nextStep < tokenLimit)
    {
        startStep(nextStep);
        ++nextStep;
    }

    for (auto step = 0; step < nextStep; ++step)
    {
        const auto token = step == 0 ? tokenInSlot(promptLength)
                                     : tokenInSlot(*inFlight[step % stepsInFlight],
                                                   promptLength + step);

        if (token == endOfSequence)
            break;

        generated.add(token);
        onToken(token);

        // The command buffer just read is the ring slot the next step records
        // into, and exactly one step is started per step read, so the ring
        // refills without ever overwriting one that is still running.
        if (nextStep < tokenLimit)
        {
            startStep(nextStep);
            ++nextStep;
        }
    }

    decodeSeconds = secondsSince(decodeStart);

    // The steps still in the air were computed past the end of the run and
    // nothing reads their tokens — but they are still writing the sequence
    // buffer, and the host writes the same buffer the moment the next run
    // uploads its prompt. A GPU write and a host memcpy into shared storage are
    // ordered by nothing at all, where the next run's *dispatches* are ordered
    // behind these by the queue. So the clock stops first, the way Whisper's
    // does, and then this waits.
    for (auto& commands: inFlight)
        if (commands.has_value())
            commands->wait();

    return generated;
}

// plan.md's "start as a readback", and that is all it is: the last logits row
// comes back to the host — 1 MB at Gemma's vocabulary — the Sampler draws from
// it, the id is written into the sequence slot the next step embeds, and only
// then is that step recorded. One step in the air and a round trip per token,
// against the greedy path's two steps and none.
//
// It is the same sequence buffer and the same slots, so the two paths differ in
// who fills a slot and nowhere else. Moving the temperature and the two cuts
// onto the device is a later round; until then this is what a temperature
// costs.
Vector<TokenId> Gemma::generateBySampling(int promptLength, int tokenLimit)
{
    const auto endOfSequence = modelConfig.endOfSequenceToken;

    sampler = Sampler {samplingOptions};

    auto generated = Vector<TokenId> {};

    // The prefill left its last block in the logits buffer, so the first draw
    // reads that row rather than one from a step of its own.
    auto rowCount = finalPrefillRows(promptLength);

    const auto decodeStart = Clock::now();

    for (auto step = 0; step < tokenLimit; ++step)
    {
        readLastLogitsRow(rowCount);

        const auto token = sampler.sample(logitsRow, suppression);

        if (token == endOfSequence)
            break;

        generated.add(token);
        onToken(token);

        if (step + 1 >= tokenLimit)
            break;

        writeTokenToSlot(token, promptLength + step);

        auto commands = gpu->makeCommandBuffer();

        {
            auto pass = commands.beginCompute({}, DispatchOrder::Concurrent);
            encodeGeneratedStep(pass, promptLength, step + 1, false);
        }

        commands.commit();
        ++stepCount;

        rowCount = 1;
    }

    decodeSeconds = secondsSince(decodeStart);

    return generated;
}

// The tokens are named rather than passed straight through: a Span cannot bind
// to a temporary Vector, which is EA::Span refusing to dangle.
Vector<TokenId> Gemma::generateFromText(std::string_view prompt)
{
    requirePrepared();

    const auto promptTokens = tokenizer().encodeWithBos(prompt);

    return generate(promptTokens);
}

std::string Gemma::generateText(std::string_view prompt)
{
    const auto tokens = generateFromText(prompt);
    return textForTokens(tokens);
}

std::string Gemma::textForTokens(Span<const TokenId> tokens) const
{
    requireLoaded();
    return vocabulary->decode(tokens);
}
} // namespace HF

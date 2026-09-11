#pragma once

// The module's class and, through what it includes, its umbrella: the decoder,
// the tokenizer and the sampler all come with it, the way Decoder/Decoder.h
// brings DecoderShape and DecoderWeights. Model/ has a Model.h separate from
// its classes because it has no one class; this has one.

#include <HuggingFaceEACP/Decoder/Decoder.h>
#include <HuggingFaceEACP/Kernels/Argmax.h>
#include <HuggingFaceEACP/Sampling/Sampling.h>
#include <HuggingFaceEACP/Tokenizer/Tokenizer.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace HF
{
// The whole runtime for a decoder-only model, a string in and a string out:
// the checkpoint, the tokenizer, the KV-cached decoder and the search that
// turns the last logit row into the next token. WhisperEACP's Whisper class
// without the audio half, which is most of it.
//
// This is the layer the decoder deliberately left empty. Which tokens are
// suppressed at which step is generation config, so Argmax is bound here rather
// than inside Decoder, and the mask below is the whole of that config —
// gemma-2b's generation_config carries no suppression list at all, so it is
// zeros, and it exists so that a repo which does have one needs no new plumbing.
//
// A run is two phases:
//
//   1. the prefill — the prompt through the decoder, in blocks of at most
//      promptCapacity() rows, each block its own command buffer;
//   2. one command buffer per generated token, with stepsInFlight of them in
//      the air at a time.
//
// **The greedy path never brings a token back to sample it.** Argmax writes the
// chosen id into a slot of the sequence buffer and the next step's Embed reads
// that same slot on the device, so a step is recorded and submitted without the
// host knowing what the one before it produced. A slot is read back for the
// stop condition and the onToken callback alone, and CommandBuffer::read waits
// for that one step rather than for the newest submission, so step k's token is
// read while step k + 1 is already running.
//
// **The sampled path is a readback**, which is what plan.md's sixth step says
// it starts as: at a temperature above zero the last logits row comes back to
// the host each step — 1 MB at Gemma's vocabulary — Sampler decides, the id is
// written into the sequence slot, and only then is the next step recorded. One
// step in the air, and a round trip per token. Moving temperature, top-k and
// top-p onto the device is a later round; the tokens are the same either way,
// which is what the tests over the two paths say.
//
// **Memory.** Gemma's weights are BF16 and stay BF16: every product and the
// embedding gather read them through eacp's readBFloat16, so the device holds
// about 5 GB of them rather than the 10 GB a widened copy cost, with the
// 1.05 GB embedding bound twice — once as the gather's table and once as the
// tied logits projection. On top of that the logits buffer is promptCapacity()
// rows of the vocabulary, 524 MB at the default 512 rows, and it is now the
// largest single allocation here after the embedding, as well as the one buffer
// that scales with that setting.
class Gemma
{
public:
    Gemma() = default;

    Gemma(const Gemma&) = delete;
    Gemma& operator=(const Gemma&) = delete;

    // A HuggingFace Gemma directory: config.json, the weights as one
    // safetensors or as shards behind an index, tokenizer.json, and
    // generation_config.json where the repo ships one. A missing required file
    // is a ModelError naming it, since "load failed" about a directory is a
    // message that costs the caller the `ls`.
    //
    // No GPU is touched here — the weights are mapped, not uploaded. That is
    // prepare()'s half.
    void load(const std::filesystem::path& modelDirectory);

    // The same files, already located. ModelFiles is a view of a directory:
    // the paths must still name the files when prepare() reads them.
    void load(const ModelFiles& files);

    // The model the build copied beside this binary, if it copied one: the
    // files under bundledModelDirectoryName, in the bundle's Resources on
    // macOS and next to the executable otherwise. A ModelError names the
    // directory it looked for when there is none, since a build that meant to
    // ship a model and never asked for the copy should not read as a corrupt
    // one; a directory that is there but short a file fails the way
    // load(path) does, naming the file.
    //
    // A binary gets one by calling hf_bundle_model(<target>) in its
    // CMakeLists, in a build configured with -DHF_EACP_FETCH_MODEL=ON — the
    // default. Nothing here copies a model on its own: 5 GB beside every
    // executable is a decision the build makes, not the runtime.
    void loadBundled();

    static bool hasBundledModel();

    // resourcesDirectory() / bundledModelDirectoryName, whether or not
    // anything is there.
    static std::filesystem::path bundledModelDirectory();

    // The directory hf_bundle_model copies into. Spelled here and in
    // Model/CMakeLists.txt, and nowhere else.
    static constexpr auto bundledModelDirectoryName = "GemmaModel";

    bool isLoaded() const { return weightFile.has_value(); }

    // Compiles every kernel, sizes every intermediate and every cache, uploads
    // the weights and the suppression mask, and allocates the sequence buffer.
    //
    // The device argument reaches the pipelines only. The weight buffers go up
    // through ShardedTensors::makeBuffer, which uses Device::shared(), so a
    // prepare() against a second Device would compile there and upload here.
    // Recorded rather than hidden: it is a seam in the loader, not in this.
    void prepare(eacp::GPU::Device& device);
    void prepare();

    bool isPrepared() const { return decoder.has_value(); }

    const GemmaConfig& config() const { return modelConfig; }
    const Tokenizer& tokenizer() const;
    const DecoderShape& shape() const;

    // The most tokens generate() may produce, the end-of-sequence token
    // excluded.
    //
    // Zero, the default, is HF's max_length rule at the window's own size: a
    // run of n prompt tokens may produce maxPositions - n, so the sequence
    // never passes the window. A prompt's length is not known until generate()
    // is called, which is why the default reads back as zero rather than as a
    // number; setting zero puts it back.
    int maximumTokens() const { return maximumTokenCount; }
    void setMaximumTokens(int count);

    // How many rows one prefill block may carry, which becomes the decoder's
    // maxStepRows. A longer prompt is prefilled in that many rows at a time,
    // and only the last block's logits are sampled.
    //
    // It is a memory setting rather than a correctness one — the blocks produce
    // the rows one call would have — and it is the number the per-step
    // intermediates are sized by: 512 rows of Gemma's vocabulary is a 524 MB
    // logits buffer, and the window's 8192 would be 8.4 GB for buffers a decode
    // step uses one row of. Set it before prepare(), which is where the sizes
    // are taken.
    int promptCapacity() const { return promptRowCapacity; }
    void setPromptCapacity(int rows);

    // Greedy by default, which is the fast path: temperature zero is the
    // on-device Argmax and never reads a logits row back.
    //
    // Every run reseeds from these options rather than carrying the generator
    // on, so the same prompt at the same seed produces the same continuation
    // however many runs came before it — which is the only form of
    // reproducibility a caller can check.
    const SamplingOptions& sampling() const { return samplingOptions; }
    void setSampling(SamplingOptions options);

    // One call per generated token, in order, on the thread that runs
    // generate() and once the token is known. A no-op by default so the loop
    // calls it unconditionally.
    std::function<void(TokenId)> onToken = [](TokenId) {};

    // The prompt through the decoder and then a token at a time until the
    // end-of-sequence token, maximumTokens() or the window. Returns the
    // generated tokens only, with the end-of-sequence token excluded — it was
    // produced and then dropped, which is what lastStepCount() counts and this
    // does not.
    //
    // The sequence is opened here: beginSequence() puts the decoder's position
    // back to zero, so a second call is a fresh run over the same buffers.
    Vector<TokenId> generate(Span<const TokenId> promptTokens);

    // The same, from text: encoded with the beginning-of-sequence token in
    // front, which is what gemma-2b is trained to see.
    Vector<TokenId> generateFromText(std::string_view prompt);

    // And decoded back, the special tokens dropped — the continuation as text,
    // without the prompt in front of it.
    std::string generateText(std::string_view prompt);

    std::string textForTokens(Span<const TokenId> tokens) const;

    // Wall clock around the last run's two halves, which is the only clock
    // available: eacp's FrameTimer is driven by Frame and an off-screen compute
    // pass is timed from the host.
    //
    // The prefill is its own command buffers end to end, their commits — the
    // first token included, since the Argmax over the last prompt row rides in
    // the last of them. The decode runs from the first thing the tail does — a
    // submit on the greedy path, the prefill row's readback on the sampled one
    // — to the read of the last token, and stops at that read rather than
    // waiting for the steps still in the air behind it. So the two hold the run
    // between them and neither counts the other.
    double lastPrefillSeconds() const { return prefillSeconds; }
    double lastDecodeSeconds() const { return decodeSeconds; }

    // How many Decoder::step calls the last run consumed: the prefill's blocks
    // plus one per token after the first. A step is counted when it is
    // recorded, so the steps still in the air when a run ends are counted and
    // the tokens they produced are not — they were computed past the end and
    // never looked at.
    int lastStepCount() const { return stepCount; }

    // How many generated steps are in the air at a time. Each is a command
    // buffer of its own, recorded and submitted before the host waits on the
    // step before it, so the GPU takes the next one up the moment it finishes
    // instead of idling across a commit; the price is that the last
    // stepsInFlight - 1 steps of a run are computed past its end, which a
    // continuation of any length pays once. Greedy only — the sampled path has
    // to see a row before it can record the step that follows it.
    static constexpr int stepsInFlight = 2;

    // The prefill block a run takes unless it is told otherwise. Large enough
    // that an ordinary prompt is one block and takes the many-row tiled
    // product, small enough that the logits buffer behind it is half a gigabyte
    // rather than eight.
    static constexpr int defaultPromptCapacity = 512;

private:
    void requireLoaded() const;
    void requirePrepared() const;

    void buildTokenizer(const ModelFiles& files);
    void buildSuppressionMask();

    void uploadPrompt(Span<const TokenId> promptTokens);

    // Every block of the prompt, each its own command buffer, and — when the
    // run has room for a token — the Argmax that turns the last block's last
    // row into the first one. Returns what the whole of it cost.
    double prefill(int promptLength, bool samples);

    // How many rows the prompt's last block carried, which is where the
    // sampled path's first draw finds its row in the logits buffer.
    int finalPrefillRows(int promptLength) const;

    void encodeSampling(eacp::GPU::ComputePass& pass, int rowCount, int slot);

    // Step j of the generated tail embeds the token in slot promptLength + j -
    // 1 and, on the greedy path, samples into slot promptLength + j. Token
    // zero comes out of the prefill, so the tail starts at one.
    void encodeGeneratedStep(eacp::GPU::ComputePass& pass,
                             int promptLength,
                             int step,
                             bool samples);

    eacp::GPU::BufferRange sequenceSlots(int first, int count) const;

    // The token in a slot, out of the command buffer that wrote it: that one
    // waited for, and nothing submitted behind it.
    TokenId tokenInSlot(eacp::GPU::CommandBuffer& commands, int slot) const;
    TokenId tokenInSlot(int slot) const;

    void writeTokenToSlot(TokenId token, int slot);

    // Into logitsRow: the last row of the logits buffer after a step of
    // rowCount rows, which is the distribution over the token that follows
    // everything decoded so far.
    void readLastLogitsRow(int rowCount);

    Vector<TokenId> generateGreedily(int promptLength, int tokenLimit);
    Vector<TokenId> generateBySampling(int promptLength, int tokenLimit);

    // How many tokens this run may produce: the window's own bound, narrowed by
    // maximumTokens() when one was set.
    int tokenLimitFor(int promptLength) const;

    // The device every command buffer and every buffer this class owns comes
    // from — the one prepare() was handed, which the weight buffers are the
    // documented exception to.
    eacp::GPU::Device* gpu = nullptr;

    GemmaConfig modelConfig;
    std::optional<Tokenizer> vocabulary;
    std::optional<ShardedTensors> weightFile;

    DecoderShape decoderShape;
    std::optional<Decoder> decoder;
    std::optional<DecoderWeights> weights;
    Argmax selection;
    Sampler sampler {SamplingOptions::greedy()};

    SamplingOptions samplingOptions = SamplingOptions::greedy();
    int maximumTokenCount = 0;
    int promptRowCapacity = defaultPromptCapacity;

    // One float per vocabulary entry, nonzero at a token that may not be
    // chosen, in the layout Argmax's mask buffer has and Sampler::sample reads.
    Vector<float> suppression;

    Vector<std::uint32_t> promptIds;
    Vector<float> logitsRow;

    std::optional<eacp::GPU::Buffer> logitBuffer;
    std::optional<eacp::GPU::Buffer> suppressionBuffer;

    // The whole sequence as unsigned ids, one slot per position and one more
    // for the token produced at the last: the prompt written into the first
    // slots, and every slot after them filled by the Argmax of one step and
    // read by the Embed of the next, on the device. The host reads a slot back
    // only to learn what the token was and whether the run is over.
    std::optional<eacp::GPU::Buffer> sequenceTokens;

    double prefillSeconds = 0.0;
    double decodeSeconds = 0.0;
    int stepCount = 0;
};
} // namespace HF

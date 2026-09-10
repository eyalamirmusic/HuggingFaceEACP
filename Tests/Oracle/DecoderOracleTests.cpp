#include "Common.h"

// Tests/Kernels/Common.h by relative path rather than a second copy of what it
// holds — the upload and readback plumbing and the isClose bound live there,
// and Tests/Decoder/Common.h reaches them the same way. It is a test header
// rather than a library one, so nothing but this include depends on it.
#include "../Kernels/Common.h"

#include <HuggingFaceEACP/Decoder/Decoder.h>
#include <HuggingFaceEACP/Sampling/Sampler.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// plan.md's third step: our GPU decoder's logits over the real google/gemma-2b
// weights against llama.cpp's over the F32 GGUF of the same checkpoint.
//
// This is the deepest claim the suite can make. Between a token and a logit,
// everything is ours on one side and the reference's on the other — eighteen
// layers of independently written arithmetic — and, unlike WhisperEACP's
// version of this comparison, both sides hold the same weights at the same
// width: the GGUF is the F32 conversion of the same safetensors our loader
// widens from BF16, so a disagreement is arithmetic rather than a format.
//
// **Which assertion is the real one.** The argmax of every row and the set of
// its five largest tokens. That is what a generation step consumes, and it is
// robust to the one thing the two sides genuinely differ in: the order they
// accumulate in. Our products tile and split their inner sums across a
// threadgroup, ggml's walk them along a row with its own vector width, and a
// 2048-wide dot product summed two ways in float32 does not give the same last
// bits — nor do exp, rsqrt and tanh, which are the backend's rather than
// libm's.
//
// **The elementwise bound is provisional.** isClose(a, e, 2e-3) allows
// |a - e| <= 2e-3 * (1 + |e|), which at a logit of 20 is about 0.04. Nothing
// has run this yet, so that number is an expectation and not a measurement:
// what a run should be read for is the max |a - e| each test prints, and the
// bound replaced with a stated multiple of it the way Tests/Decoder's 5e-6 is
// a multiple of its own measured 2.1e-7. If the elementwise check fails while
// the argmax and the top five agree, the tolerance is what is wrong; if the
// argmax disagrees, we are.
//
// Everything skips without a device, without a checkpoint, and without the
// GGUF beside the safetensors — which the fetched mirror does not carry, so it
// takes GEMMA_MODEL_DIR pointing at Google's own download.

using namespace nano;
using namespace HF;
using namespace HF::Testing;
using namespace eacp::GPU;

namespace
{
// The window our decoder is built for. Every intermediate is sized at
// maxPositions rows — at the config's 8192 the gated feed-forward's pair alone
// is 1.6 GB — so a comparison that feeds a handful of tokens builds for a
// handful. Matched to the oracle's own context so neither side is the one that
// runs out first, and so the two are the same claim about the same window.
constexpr auto oracleWindow = LlamaOracle::defaultContextSize;

// See the note above: an expectation, not a measurement.
constexpr auto logitTolerance = 2e-3;

// How many of a row's largest tokens have to be the same set. Five is past
// where a sampler's top-k usually cuts and far enough down the row that
// agreeing on it is a claim about the distribution rather than about its peak.
constexpr auto topTokens = 5;

constexpr auto capitalPrompt = std::string_view {"The capital of France is"};
constexpr auto greedySteps = 8;

// Three prompts rather than one: a bare noun phrase, a sentence with
// punctuation and digits, and the one the greedy test continues. Tokenized by
// our tokenizer on both sides, since TokenizerOracleTests already checks that
// it agrees with llama.cpp's piece for piece — so a difference here cannot be
// a different sequence of tokens.
const auto promptCases = std::vector<std::string> {
    std::string {capitalPrompt},
    "In 1969 the first crewed landing on the Moon took place.",
    "def add(a, b):",
};

bool canRun()
{
    return Device::shared().isValid() && hasGemmaModel() && hasOracleModel()
           && hasOurTokenizer();
}

// The numbers comparison needs neither the device nor ten gigabytes of
// weights: it reads config.json and the GGUF's header and nothing else.
bool canCompareNumbers()
{
    return hasGemmaModel() && hasOracleModel();
}

// ---------------------------------------------------------------------------
// Our side of the comparison
// ---------------------------------------------------------------------------

// The whole of our model above the tokens: the checkpoint's config, its
// weights on the device, and a decoder prepared for them.
//
// Uploading a token vector and reading a step's logits back is the only host
// plumbing a comparison needs, and it is here rather than in Common.h because
// Common.h is what the tokenizer and file comparisons include and neither of
// those touches a device.
class OurDecoding
{
public:
    OurDecoding()
        : files(gemmaModelFiles())
        , modelConfig(GemmaConfig::fromModelFiles(files))
        , decoderShape(windowedShape(modelConfig))
        , weights(ShardedTensors::fromModelFiles(files), decoderShape)
        , decoder(decoderShape)
    {
        decoder.prepare(Device::shared());
    }

    const GemmaConfig& config() const { return modelConfig; }
    const DecoderShape& shape() const { return decoderShape; }
    int vocabularySize() const { return decoderShape.logitElementCount(); }
    int position() const { return decoder.position(); }

    void beginSequence() { decoder.beginSequence(); }

    // Appends the tokens and hands back the rows they produced:
    // [tokens.size(), vocabularySize] row-major, which is the layout
    // LlamaOracle::logits answers in.
    Vector<float> step(Span<const TokenId> tokens)
    {
        const auto rowCount = tokens.size();
        const auto elements = rowCount * vocabularySize();

        // Both of these outlive the commit rather than the recording call: a
        // bind names a buffer and does not copy it, so a temporary would be
        // freed while the command buffer still referred to it.
        const auto ids = uploaded(tokens);
        const auto logits = outputFor(elements);

        auto commands = Device::shared().makeCommandBuffer();

        {
            auto pass = commands.beginCompute();
            decoder.step(pass, ids, rowCount, weights, logits);
        }

        commands.commit();

        return readBack(logits, elements);
    }

private:
    static DecoderShape windowedShape(const GemmaConfig& config)
    {
        auto shape = DecoderShape::fromConfig(config);
        shape.maxPositions = oracleWindow;

        return shape;
    }

    // Our ids are signed and the embed kernel gathers by an unsigned one, so
    // the conversion is here and a test never spells it.
    static Buffer uploaded(Span<const TokenId> tokens)
    {
        auto ids = unsignedSized(tokens.size());

        for (auto index = 0; index < ids.size(); ++index)
            ids[index] = (std::uint32_t) tokens[index];

        return storageOf(ids);
    }

    ModelFiles files;
    GemmaConfig modelConfig;
    DecoderShape decoderShape;
    DecoderWeights weights;
    Decoder decoder;
};

// One of these for the executable. The weights are ten gigabytes of BF16
// widened to F32 on the way to the device and preparing compiles every kernel,
// so a second would double a run that already holds llama.cpp's ten gigabytes
// beside it — the same reason sharedOracle() in Common.h is one oracle.
//
// Every test below opens with beginSequence(), so sharing one carries no state
// from the test before it.
OurDecoding& ourDecoding()
{
    static const auto decoding = std::make_unique<OurDecoding>();
    return *decoding;
}

// ---------------------------------------------------------------------------
// Comparing two rows of a 256,000-wide vocabulary
// ---------------------------------------------------------------------------

Span<const float> rowOf(const Vector<float>& rows, int index, int width)
{
    return {rows.data() + index * width, width};
}

Span<const float> lastRowOf(const Vector<float>& rows, int width)
{
    return {rows.data() + rows.size() - width, width};
}

// What two implementations' logits differ by, accumulated rather than checked
// per element: a row is a quarter of a million floats and a check apiece would
// be the whole run's output. The magnitude is carried alongside because an
// absolute difference means nothing without the range it sits in.
struct RowDifference
{
    double absolute = 0.0;
    double relative = 0.0;
    double magnitude = 0.0;
    bool everyElementIsClose = true;

    void take(const RowDifference& other)
    {
        absolute = std::max(absolute, other.absolute);
        relative = std::max(relative, other.relative);
        magnitude = std::max(magnitude, other.magnitude);
        everyElementIsClose = everyElementIsClose && other.everyElementIsClose;
    }
};

RowDifference compareRow(Span<const float> ours, Span<const float> theirs)
{
    auto difference = RowDifference {};

    for (auto index = 0; index < ours.size(); ++index)
    {
        const auto expected = (double) theirs[index];
        const auto gap = std::abs((double) ours[index] - expected);

        difference.absolute = std::max(difference.absolute, gap);
        difference.relative =
            std::max(difference.relative, gap / (1.0 + std::abs(expected)));
        difference.magnitude = std::max(difference.magnitude, std::abs(expected));

        difference.everyElementIsClose =
            difference.everyElementIsClose
            && isClose(ours[index], expected, logitTolerance);
    }

    return difference;
}

// The indices of the count largest entries, largest first and the lower token
// on an exact tie — which is Sampler::ranksBefore's order, so the head of what
// this returns is the token a greedy draw would have taken.
Vector<TokenId> largestTokens(Span<const float> row, int count)
{
    auto ranked = Vector<TokenId> {};
    ranked.resize(row.size());
    std::iota(ranked.begin(), ranked.end(), 0);

    const auto ranksBefore = [&](TokenId left, TokenId right)
    {
        if (row[left] != row[right])
            return row[left] > row[right];

        return left < right;
    };

    std::partial_sort(
        ranked.begin(), ranked.begin() + count, ranked.end(), ranksBefore);
    ranked.resize(count);

    return ranked;
}

// By value, because the comparison is of sets and sorting is how that is
// spelled: two rows may rank the same five tokens in a different order without
// disagreeing about which five they are.
bool sameTokenSet(Vector<TokenId> ours, Vector<TokenId> theirs)
{
    std::sort(ours.begin(), ours.end());
    std::sort(theirs.begin(), theirs.end());

    return std::equal(ours.begin(), ours.end(), theirs.begin(), theirs.end());
}

bool sameTokens(Span<const TokenId> ours, Span<const TokenId> theirs)
{
    return std::equal(ours.begin(), ours.end(), theirs.begin(), theirs.end());
}

// Every row of one step against the oracle's rows for the same tokens, with
// the two assertions that matter reported separately from the one that is a
// guess. Returns what the run measured, for the caller to print.
struct RowAgreement
{
    RowDifference difference;
    bool everyArgmaxAgrees = true;
    bool everyTopSetAgrees = true;
};

RowAgreement compareRows(const Vector<float>& ours,
                         const Vector<float>& theirs,
                         int firstRow,
                         int rowCount,
                         int width)
{
    auto agreement = RowAgreement {};

    for (auto row = 0; row < rowCount; ++row)
    {
        const auto mine = rowOf(ours, row, width);
        const auto reference = rowOf(theirs, firstRow + row, width);

        agreement.difference.take(compareRow(mine, reference));

        agreement.everyArgmaxAgrees =
            agreement.everyArgmaxAgrees
            && Sampler::argmax(mine) == Sampler::argmax(reference);

        agreement.everyTopSetAgrees =
            agreement.everyTopSetAgrees
            && sameTokenSet(largestTokens(mine, topTokens),
                            largestTokens(reference, topTokens));
    }

    return agreement;
}

void reportAndCheck(std::string_view what, const RowAgreement& agreement)
{
    std::cout << "  " << what << ": max |a - e| " << agreement.difference.absolute
              << ", max relative " << agreement.difference.relative
              << ", over logits reaching " << agreement.difference.magnitude << "\n";

    // The two the comparison exists for.
    check(agreement.everyArgmaxAgrees, "every row's argmax agrees");
    check(agreement.everyTopSetAgrees, "every row's five largest are the same five");

    // And the provisional one, which is the number a run is read for.
    check(agreement.difference.everyElementIsClose,
          "every logit is inside the elementwise bound");
}

// ---------------------------------------------------------------------------
// The GGUF's own header against config.json
// ---------------------------------------------------------------------------

std::optional<double> numberIn(const std::string& text)
{
    auto* end = (char*) nullptr;
    const auto value = std::strtod(text.c_str(), &end);

    if (end == text.c_str())
        return {};

    return value;
}

// One number out of the GGUF's header against the config.json field the
// conversion read it from.
//
// A key the file does not carry is reported rather than asserted on: a
// converter is free to leave out a value that took its default, and a missing
// key is a gap in this comparison rather than a disagreement.
//
// The tolerance is relative and loose because llama.cpp stores every metadata
// value as the string std::to_string gives it — six decimal places for a
// float32 — so rope_theta arrives as "10000.000000" and rms_norm_eps as
// "0.000001", which is the whole of the precision the header can carry for it.
void checkAgainstMetadata(const LlamaOracle& oracle,
                          const char* key,
                          double expected,
                          const char* field)
{
    const auto text = oracle.metadata(key);

    if (text.empty())
    {
        std::cout << "  the GGUF carries no " << key << ", so " << field
                  << " goes unchecked\n";

        return;
    }

    const auto value = numberIn(text);
    check(value.has_value(), "the metadata value is a number");

    if (!value.has_value())
        return;

    if (*value == 0.0 && expected != 0.0)
    {
        std::cout << "  " << key << " is \"" << text
                  << "\", which is six decimals rounding " << expected
                  << " away, so " << field << " goes unchecked\n";

        return;
    }

    check(std::abs(*value - expected) <= 1e-3 * std::abs(expected), field);
}
} // namespace

// The prompt's own rows, which is prefill: one step of several tokens, causally
// masked, against the oracle decoding the same tokens in one batch.
//
// Both sides are handed the same ids — ours, out of our tokenizer with BOS in
// front — so nothing here can be a tokenization difference. That is checked
// separately in TokenizerOracleTests and is not re-litigated per prompt.
auto tPromptLogitsAgree = test("Oracle/Decoder/promptLogitsAgree") = []
{
    if (!canRun())
        return;

    auto& oracle = sharedOracle();
    check(oracle.isValid(), "the GGUF loaded");

    if (!oracle.isValid())
        return;

    auto& ours = ourDecoding();
    const auto width = ours.vocabularySize();

    check(width == oracle.vocabularySize(), "the two vocabularies are one size");

    if (width != oracle.vocabularySize())
        return;

    for (const auto& prompt: promptCases)
    {
        const auto tokens = ourTokenizer().encodeWithBos(prompt);

        check(tokens.size() > 1, "the prompt tokenizes");
        check(tokens.size() <= ours.shape().maxPositions,
              "the prompt fits the window");

        const auto reference = oracle.logits(tokens);
        check(reference.size() == tokens.size() * width,
              "the oracle answered one row per token");

        ours.beginSequence();
        const auto rows = ours.step(tokens);

        check(rows.size() == reference.size(), "and so did we");
        check(ours.position() == tokens.size());

        if (rows.size() != reference.size())
            continue;

        reportAndCheck(prompt,
                       compareRows(rows, reference, 0, tokens.size(), width));
    }
};

// The same prompt one token per step, which is what every step of a real
// generation is: the split product and the decode attention rather than the
// tiled product and the prefill one, and a KV cache that has to hold what the
// steps before it wrote.
//
// The oracle computes the whole sequence in one batch from position zero, so
// this says our cache reproduces a full re-run — the property plan.md's fourth
// step is built on, checked here at the real width against an implementation
// that shares none of our code.
auto tTokenByTokenAgrees = test("Oracle/Decoder/tokenByTokenAgrees") = []
{
    if (!canRun())
        return;

    auto& oracle = sharedOracle();
    check(oracle.isValid(), "the GGUF loaded");

    if (!oracle.isValid())
        return;

    auto& ours = ourDecoding();
    const auto width = ours.vocabularySize();
    const auto tokens = ourTokenizer().encodeWithBos(capitalPrompt);

    const auto reference = oracle.logits(tokens);
    check(reference.size() == tokens.size() * width, "the oracle answered");

    if (reference.size() != tokens.size() * width)
        return;

    ours.beginSequence();

    auto agreement = RowAgreement {};

    for (auto index = 0; index < tokens.size(); ++index)
    {
        const auto single = Span<const TokenId> {tokens.data() + index, 1};
        const auto row = ours.step(single);

        check(row.size() == width, "one row per step");
        check(ours.position() == index + 1);

        if (row.size() != width)
            return;

        const auto step = compareRows(row, reference, index, 1, width);

        agreement.difference.take(step.difference);
        agreement.everyArgmaxAgrees =
            agreement.everyArgmaxAgrees && step.everyArgmaxAgrees;
        agreement.everyTopSetAgrees =
            agreement.everyTopSetAgrees && step.everyTopSetAgrees;
    }

    reportAndCheck("one token at a time", agreement);
};

// The loop itself: our decoder driven greedily off its own choices, against
// llama.cpp's greedy continuation of the same prompt.
//
// Nothing feeds either side the other's token, which is the point — a
// disagreement at step k sends the two down different sequences and every step
// after it disagrees too. That is a harsher test than following the reference,
// and it is the one that matches what a generation loop will do.
//
// Sampler::argmax over a row read back is the CPU half of a step, and it is
// Kernels/Argmax.h's rule: the largest logit, the lower index on a tie.
auto tGreedyContinuationAgrees = test("Oracle/Decoder/greedyContinuationAgrees") = []
{
    if (!canRun())
        return;

    auto& oracle = sharedOracle();
    check(oracle.isValid(), "the GGUF loaded");

    if (!oracle.isValid())
        return;

    auto& ours = ourDecoding();
    const auto width = ours.vocabularySize();
    const auto prompt = ourTokenizer().encodeWithBos(capitalPrompt);

    const auto expected = oracle.greedy(prompt, greedySteps);
    check(!expected.empty(), "the oracle continued the prompt");

    ours.beginSequence();

    auto rows = ours.step(prompt);
    auto generated = Vector<TokenId> {};

    for (auto step = 0; step < greedySteps; ++step)
    {
        const auto next = Sampler::argmax(lastRowOf(rows, width));

        // The oracle stops on any end-of-generation token and this stops on
        // the config's end of sequence, so a checkpoint whose EOG is something
        // else would show up as one sequence being shorter than the other
        // rather than as a wrong token.
        if (next == ours.config().endOfSequenceToken)
            break;

        generated.add(next);

        if (step + 1 == greedySteps)
            break;

        rows = ours.step(Span<const TokenId> {&next, 1});
    }

    const auto text = ourTokenizer().decode(generated);

    std::cout << "  \"" << capitalPrompt << "\" -> \"" << text << "\", ours "
              << spelled(generated) << ", llama.cpp " << spelled(expected) << "\n";

    check(sameTokens(generated, expected), "the greedy continuations agree");
    check(text.find("Paris") != std::string::npos,
          "Paris arrives within eight tokens");
};

// plan.md's "confirm every number against config.json", from the other side:
// the GGUF's header was written by a conversion that read the same
// config.json, so the two agreeing is what retires that caveat — and a
// disagreement says the two files beside each other in the model directory are
// not the same checkpoint, which is the first thing to rule out when a logit
// comparison fails.
//
// Cheap on purpose: no device, no decoder, no ten gigabytes. Five of the
// numbers are llama.cpp accessors at this tag — llama_model_n_embd, n_layer,
// n_head, n_head_kv and n_ctx_train — and the rest reach it only as GGUF
// metadata, since llama.cpp has no getter for a head width, a feed-forward
// width, a rope base or an RMS epsilon.
auto tConfigMatchesGguf = test("Oracle/Decoder/configMatchesGguf") = []
{
    if (!canCompareNumbers())
        return;

    auto& oracle = sharedOracle();
    check(oracle.isValid(), "the GGUF loaded");

    if (!oracle.isValid())
        return;

    const auto config = GemmaConfig::fromModelFiles(gemmaModelFiles());

    std::cout << "  llama.cpp calls it \"" << oracle.description() << "\"\n";

    check(oracle.vocabularySize() == config.vocabularySize, "vocab_size");
    check(oracle.bos() == config.beginningOfSequenceToken, "bos_token_id");
    check(oracle.eos() == config.endOfSequenceToken, "eos_token_id");

    check(oracle.embeddingWidth() == config.hiddenSize, "hidden_size");
    check(oracle.layerCount() == config.layerCount, "num_hidden_layers");
    check(oracle.attentionHeads() == config.attentionHeads, "num_attention_heads");
    check(oracle.keyValueHeads() == config.keyValueHeads, "num_key_value_heads");
    check(oracle.trainingContext() == config.maxPositions,
          "max_position_embeddings");

    // Gemma 2B's heads are 256 wide over a 2048-wide model, so head_dim is not
    // hidden_size / num_attention_heads by definition — it only happens to be
    // here, and a config that said otherwise would be the one this project
    // follows. Checked both ways for that reason.
    check(oracle.embeddingWidth() == config.attentionHeads * config.headWidth,
          "the heads fill the width");

    checkAgainstMetadata(
        oracle, "gemma.attention.key_length", config.headWidth, "head_dim");

    checkAgainstMetadata(oracle,
                         "gemma.feed_forward_length",
                         config.intermediateSize,
                         "intermediate_size");

    checkAgainstMetadata(
        oracle, "gemma.rope.freq_base", config.ropeTheta, "rope_theta");

    checkAgainstMetadata(oracle,
                         "gemma.attention.layer_norm_rms_epsilon",
                         config.rmsNormEpsilon,
                         "rms_norm_eps");
};

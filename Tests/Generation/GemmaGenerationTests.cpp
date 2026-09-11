#include "Common.h"

#include "../Support/GemmaModel.h"

#include <iostream>
#include <string>

// The real checkpoint: the gemma-2b the build fetched, or the one
// GEMMA_MODEL_DIR names instead. Every test below returns early when neither
// is there — a build configured with -DHF_EACP_FETCH_MODEL=OFF — the way they
// all do without a device.
//
// This is the first test in the tree that asserts something about what the
// model *says* rather than about a number: the exact continuation greedy
// gemma-2b gives "The capital of France is", which is the one assertion that
// fails if the loader, the tokenizer, the eighteen layers, the KV cache and
// the greedy search are each nearly right. The tiers below it are what say
// which of them it was.
//
// The continuation is not " Paris", which is what this asserted until three
// independent implementations were asked. At the last prompt position the base
// model ranks "▁a" at -16.529 over "▁Paris" at -16.855 — a third of a logit,
// so the sentence it continues into is "a city of contrasts" rather than the
// answer to a question nobody asked it. Both llama.cpp over the F32 GGUF, in
// Tests/Oracle, and transformers 5.17.0 in float32 and in bfloat16 produce
// exactly these eight ids, so the expectation below is what the model says and
// not what our arithmetic does.

using namespace nano;
using namespace HF;
using namespace HF::Testing;
using namespace eacp::GPU;

namespace
{
bool hasRealCheckpoint()
{
    return Device::shared().isValid() && hasGemmaModel();
}

// Small enough that a first run is seconds rather than a minute, and long
// enough that a continuation is a clause rather than a word.
constexpr auto smokeTokens = 8;

// What greedy gemma-2b answers "The capital of France is" with, over those
// eight tokens: ids 476, 3413, 576, 82777, 235265, 1165, 603, 476. Asserted as
// the whole string rather than as a word in it, because the claim is that our
// stack reproduces the reference token for token and a substring would pass on
// a sequence that had drifted after the first few.
constexpr auto capitalContinuation = " a city of contrasts. It is a";

void reportTimings(const Gemma& gemma, int tokenCount)
{
    const auto decode = gemma.lastDecodeSeconds();
    const auto perToken = tokenCount > 0 ? decode / tokenCount : 0.0;

    std::cout << "  prefill               " << gemma.lastPrefillSeconds() << " s\n";
    std::cout << "  decode, " << tokenCount << " tokens    " << decode << " s   "
              << (perToken > 0.0 ? 1.0 / perToken : 0.0) << " tokens/s over "
              << gemma.lastStepCount() << " steps\n";
}
} // namespace

// The whole runtime end to end, which is what plan.md's fourth step is: a
// string in and a string out.
auto tGemmaGeneratesText = test("Generation/Gemma/completesAPrompt") = []
{
    if (!hasRealCheckpoint())
        return;

    auto gemma = Gemma {};
    gemma.load(gemmaModelDirectory());

    check(gemma.isLoaded());
    check(gemma.config().vocabularySize == 256000);
    check(gemma.tokenizer().bos() == gemma.config().beginningOfSequenceToken);

    gemma.prepare();

    check(gemma.isPrepared());
    check(gemma.shape().maxPositions == gemma.config().maxPositions);
    check(gemma.shape().stepRowCapacity() == Gemma::defaultPromptCapacity);

    gemma.setMaximumTokens(smokeTokens);

    // Streamed as it arrives, which is what the callback is for and what the
    // Generate app prints: a run that stalls says where.
    auto streamed = Vector<TokenId> {};
    gemma.onToken = [&streamed](TokenId token) { streamed.add(token); };

    const auto text = gemma.generateText("The capital of France is");

    check(streamed.size() > 0);
    check(streamed.size() <= smokeTokens);

    std::cout << "  \"The capital of France is\" ->\"" << text << "\"\n";
    reportTimings(gemma, streamed.size());

    check(text == capitalContinuation);
};

// The two sampling paths over the real model, which is the one comparison that
// cannot be made at the synthetic shape: a temperature just above zero with
// top-k one leaves exactly the largest logit in the running, so the readback
// draw has to land on the token the on-device Argmax took.
auto tGemmaSampledPathAgrees = test("Generation/Gemma/sampledPathAgrees") = []
{
    if (!hasRealCheckpoint())
        return;

    auto gemma = Gemma {};
    gemma.load(gemmaModelDirectory());
    gemma.prepare();

    gemma.setMaximumTokens(4);

    const auto greedy = gemma.generateFromText("The capital of France is");

    gemma.setSampling(SamplingOptions {.temperature = 1.0e-3f, .topK = 1});

    const auto sampled = gemma.generateFromText("The capital of France is");

    check(sameTokens(greedy, sampled));

    std::cout << "  greedy and readback agree over " << greedy.size() << " tokens\n";
};

// A prompt longer than one prefill block, at the real widths: the blocks append
// to the same caches at the same positions, so the continuation is the one a
// single block would have produced. Run at a small capacity on purpose, since
// the default swallows any prompt a test would write.
auto tGemmaBlockedPrefillAgrees = test("Generation/Gemma/blockedPrefillAgrees") = []
{
    if (!hasRealCheckpoint())
        return;

    constexpr auto prompt =
        "Paris is the capital of France, and Rome is the capital of";

    auto whole = Gemma {};
    whole.load(gemmaModelDirectory());
    whole.prepare();
    whole.setMaximumTokens(4);

    const auto unblocked = whole.generateFromText(prompt);
    check(whole.lastStepCount() == 1 + unblocked.size() - 1);

    auto blocked = Gemma {};
    blocked.load(gemmaModelDirectory());
    blocked.setPromptCapacity(4);
    blocked.prepare();
    blocked.setMaximumTokens(4);

    check(blocked.shape().stepRowCapacity() == 4);

    const auto inBlocks = blocked.generateFromText(prompt);

    check(sameTokens(unblocked, inBlocks));
    check(blocked.lastStepCount() > whole.lastStepCount());

    std::cout << "  blocked prefill took " << blocked.lastStepCount()
              << " steps against " << whole.lastStepCount() << "\n";
};

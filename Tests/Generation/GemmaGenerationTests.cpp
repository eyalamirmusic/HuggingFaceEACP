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
// model *says* rather than about a number: a base completion model asked for
// the capital of France answers " Paris", and it is the one assertion that
// fails if the loader, the tokenizer, the eighteen layers, the KV cache and
// the greedy search are each nearly right. The tiers below it are what say
// which of them it was.
//
// The prompt is a question because "The capital of France is" is not one this
// model answers: greedy gemma-2b continues that with " a city of contrasts",
// and it is right to — ' a' beats ' Paris' there by 0.326 logits, which
// llama.cpp over the F32 GGUF and Hugging Face transformers in fp32 both agree
// with token for token. An assertion on a 0.326-logit margin measures which
// side of a near-tie a rounding landed on. The question form puts ' Paris'
// 4.89 logits clear of the runner-up, which is a hundred times the largest
// disagreement Tests/Oracle has ever measured between our logits and
// llama.cpp's, so what it measures is our arithmetic.

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

// The probe: the answer is the very first token greedy takes, and the note at
// the top is why it is this prompt rather than a bare completion.
constexpr auto parisPrompt = "Q: What is the capital of France?\nA:";

// Small enough that a first run is seconds rather than a minute, and long
// enough that a continuation has somewhere to put the word.
constexpr auto smokeTokens = 8;

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

    const auto text = gemma.generateText(parisPrompt);

    check(streamed.size() > 0);
    check(streamed.size() <= smokeTokens);

    std::cout << "  \"" << parisPrompt << "\" ->\"" << text << "\"\n";
    reportTimings(gemma, streamed.size());

    check(text.find("Paris") != std::string::npos);
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

    const auto greedy = gemma.generateFromText(parisPrompt);

    gemma.setSampling(SamplingOptions {.temperature = 1.0e-3f, .topK = 1});

    const auto sampled = gemma.generateFromText(parisPrompt);

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

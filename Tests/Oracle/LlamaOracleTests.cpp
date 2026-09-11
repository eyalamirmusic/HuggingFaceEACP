#include "Common.h"

#include <cmath>
#include <iostream>
#include <string>
#include <string_view>

// The harness's own sanity, before anything of ours is compared against it.
//
// A reference that loaded the wrong file, or decoded into the wrong rows,
// would make every comparison built on it meaningless in a way that reads as
// our bug — so the shape of what LlamaOracle hands back is asserted here, and
// one end-to-end claim about the model's behaviour with it: greedy decoding
// from "The capital of France is" gives the continuation gemma-2b gives.
//
// Everything skips without gemma-2b.gguf, which comes with Google's own gated
// download rather than with the mirror the build fetches — GEMMA_MODEL_DIR is
// how a machine that has done that download names it.

using namespace nano;
using namespace HF;
using namespace HF::Testing;

namespace
{
constexpr auto capitalPrompt = std::string_view {"The capital of France is"};
constexpr auto greedySteps = 8;

// The base model's own answer, which is a sentence rather than the capital:
// "▁a" beats "▁Paris" by a third of a logit at the last prompt position.
// transformers 5.17.0 in float32 gives these eight ids too, so the string is
// gemma-2b's and not this reference's.
constexpr auto capitalContinuation =
    std::string_view {" a city of contrasts. It is a"};

bool canRun()
{
    return hasOracleModel();
}
} // namespace

// plan.md's numbers for gemma-2b, read off the reference rather than off the
// safetensors. A disagreement here says the GGUF is a different model, which
// is the first thing to rule out when a logit comparison fails.
auto tOracleVocabulary = test("Oracle/Llama/vocabulary") = []
{
    if (!canRun())
        return;

    auto& oracle = sharedOracle();
    check(oracle.isValid(), "the GGUF loaded");

    if (!oracle.isValid())
        return;

    check(oracle.vocabularySize() == 256000, "256k pieces");
    check(oracle.bos() == 2, "bos is 2");
    check(oracle.eos() == 1, "eos is 1");
};

// One batch, every row asked for. The shape is what a decoder comparison will
// index into, and finiteness is what says the CPU path ran rather than
// returning a buffer nobody wrote.
auto tOracleLogitsShape = test("Oracle/Llama/logitsShape") = []
{
    if (!canRun())
        return;

    auto& oracle = sharedOracle();
    check(oracle.isValid(), "the GGUF loaded");

    if (!oracle.isValid())
        return;

    const auto tokens = oracle.tokenize(capitalPrompt, true);

    check(tokens.size() > 1, "the prompt tokenizes");
    check(tokens[0] == oracle.bos(), "behind bos");

    const auto rows = oracle.logits(tokens);

    check(rows.size() == tokens.size() * oracle.vocabularySize(),
          "one row of the vocabulary per token");

    // Accumulated rather than checked per value: this is a million and a half
    // floats, and a check apiece would be the whole run's output.
    auto allFinite = true;

    for (const auto value: rows)
        allFinite = allFinite && std::isfinite(value);

    check(allFinite, "every logit is finite");
};

// The one claim here about what the model does rather than what the harness
// returns, and the reason it is worth making is that it exercises the KV cache
// path: greedy() decodes the prompt once and then one token at a time, so a
// cache that was not carried between steps would produce a continuation that
// reads as nonsense rather than as an error.
auto tOracleGreedyContinues = test("Oracle/Llama/greedyContinuesThePrompt") = []
{
    if (!canRun())
        return;

    auto& oracle = sharedOracle();
    check(oracle.isValid(), "the GGUF loaded");

    if (!oracle.isValid())
        return;

    const auto prompt = oracle.tokenize(capitalPrompt, true);
    const auto continuation = oracle.greedy(prompt, greedySteps);

    check(!continuation.empty(), "the oracle continued the prompt");

    const auto text = oracle.detokenize(continuation);

    std::cout << "  \"" << capitalPrompt << "\" -> \"" << text << "\"\n";

    check(text == capitalContinuation, "the continuation is the model's own");
};

#include "ReferenceDecoder.h"

#include <cmath>
#include <iostream>
#include <vector>

// The real checkpoint, which nothing here has: google/gemma-2b is gated, this
// machine has no copy, and a build cannot fetch one unauthenticated. So every
// test below returns early until GEMMA_MODEL_DIR names a directory — plan.md's
// fourth gap and its answer.
//
// It is a smoke test on purpose. What it says is that the loader, the shapes
// and the whole recorded step survive contact with 18 layers of 2048 over a
// 256,000-token vocabulary: every logit is a number, the rows are the right
// count, and a second step off the cache produces another row. The comparison
// against llama.cpp over the F32 GGUF is plan.md's third step and needs an
// oracle this file does not link.

using namespace nano;
using namespace HF;
using namespace HF::Testing;
using namespace eacp::GPU;

namespace
{
// The whole 8192-position window, with a step capacity of eight rows. The two
// used to be one number, and the smoke test had to shrink the window to keep
// the gated feed-forward's pair off 1.6 GB; now the caches are the config's —
// 302 MB across eighteen layers, which is what a real run holds — and only the
// per-step intermediates are the prompt's size.
constexpr auto smokeStepRows = 8;

// BOS, then a few ids that exist in any Gemma vocabulary. No tokenizer here on
// purpose: what this exercises is the decoder, and a tokenizer between it and
// the assertion would be a second thing that could fail.
std::vector<int> smokeTokens()
{
    return {2, 651, 3403, 603};
}

bool everyValueIsFinite(const Vector<float>& values)
{
    for (auto index = 0; index < values.size(); ++index)
        if (!std::isfinite(values[index]))
            return false;

    return true;
}
} // namespace

auto tGemmaPromptRuns = test("Decoder/Gemma/promptRunsAndIsFinite") = []
{
    if (!Device::shared().isValid() || !ModelFiles::hasDirectoryInEnvironment())
        return;

    const auto files = ModelFiles::fromEnvironment();
    const auto config = GemmaConfig::fromModelFiles(files);
    const auto weightFile = ShardedTensors::fromModelFiles(files);

    // The catalogue first: a checkpoint that is not the architecture this was
    // written for should say so by name rather than by a wrong-looking logit.
    GemmaTensors::checkAgainst(weightFile, config);

    auto shape = DecoderShape::fromConfig(config);
    check(shape.layers == config.layerCount);
    check(shape.queryWidth() == config.attentionHeads * config.headWidth);

    check(shape.maxPositions == config.maxPositions);

    shape.maxStepRows = smokeStepRows;

    const auto weights = DecoderWeights {weightFile, shape};
    auto decoder = Decoder {shape};
    decoder.prepare();

    const auto tokens = smokeTokens();
    const auto rowCount = (int) tokens.size();

    auto ids = unsignedSized(rowCount);

    for (auto index = 0; index < rowCount; ++index)
        ids[index] = (std::uint32_t) tokens[(std::size_t) index];

    const auto tokenBuffer = storageOf(ids);
    const auto logits = outputFor(rowCount * shape.logitElementCount());

    auto commands = Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        decoder.step(pass, tokenBuffer, rowCount, weights, logits);
    }

    commands.commit();

    check(decoder.position() == rowCount);

    const auto rows = readBack(logits, rowCount * shape.logitElementCount());
    check(rows.size() == rowCount * shape.vocabularySize);
    check(everyValueIsFinite(rows));

    const auto hidden = readBack(decoder.hiddenStates(), rowCount * shape.width);
    check(hidden.size() == rowCount * shape.width);
    check(everyValueIsFinite(hidden));

    // A single token off the cache, which is the shape every step of a real
    // generation takes: the split product and the decode attention rather than
    // the tiled product and the prefill one.
    const auto nextBuffer = storageOf(unsignedSized(1));
    const auto nextLogits = outputFor(shape.logitElementCount());

    auto second = Device::shared().makeCommandBuffer();

    {
        auto pass = second.beginCompute();
        decoder.step(pass, nextBuffer, 1, weights, nextLogits);
    }

    second.commit();

    check(decoder.position() == rowCount + 1);
    check(everyValueIsFinite(readBack(nextLogits, shape.logitElementCount())));

    std::cout << "  gemma: " << shape.layers << " layers of " << shape.width
              << " over " << shape.vocabularySize << " tokens\n";
};

// The row capacity a decoder was built for is its own, so a block wider than
// the intermediates is refused the same way it is at the small shape — which is
// what keeps a run that sized its step buffers for a prompt from dispatching
// past their end at 256,000 columns.
auto tGemmaStepCapacityIsEnforced = test("Decoder/Gemma/stepCapacityIsEnforced") = []
{
    if (!Device::shared().isValid() || !ModelFiles::hasDirectoryInEnvironment())
        return;

    const auto files = ModelFiles::fromEnvironment();
    const auto config = GemmaConfig::fromModelFiles(files);

    auto shape = DecoderShape::fromConfig(config);
    shape.maxStepRows = smokeStepRows;

    const auto weights =
        DecoderWeights {ShardedTensors::fromModelFiles(files), shape};

    auto decoder = Decoder {shape};
    decoder.prepare();

    const auto rows = smokeStepRows + 1;
    const auto tokens = storageOf(unsignedSized(rows));
    const auto logits = outputFor(rows * shape.logitElementCount());

    auto commands = Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();

        check(throwsModelError(
            [&] { decoder.step(pass, tokens, rows, weights, logits); }));
    }

    check(decoder.position() == 0);
};

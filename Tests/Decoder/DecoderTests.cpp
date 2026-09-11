#include "ReferenceDecoder.h"

#include <iostream>
#include <vector>

// The GPU decoder against the scalar reference, mutation-checked rather than
// trusted green. Each of these was applied to the module and the suite rerun,
// against the five comparisons below and the one weights test that reads the
// fused buffer back:
//
//   rotating the queries and not the keys              5 of the 6 fail
//   binding the cache one row past where it belongs    5 fail
//   rotating at position zero on every step            2 fail
//   stacking up before gate in the fused weight        all 6 fail
//   the post-attention norm reading *attended          5 fail
//
// The third is the one worth reading twice: it is invisible to every tier that
// only ever decodes from position zero, and only the two that continue a
// sequence catch it. The causal bound is the mirror of that — a step of one
// query against n cached keys masks nothing, since the query stands at the last
// of them, so the per-row key count is only load bearing when a step carries
// more than one token. Which is why the prompt is decoded in one call in half
// the tests here and one token at a time in the other half.

using namespace nano;
using namespace HF;
using namespace HF::Testing;
using namespace eacp::GPU;

namespace
{
// How far the GPU may sit from the same forward pass in double precision, as
// |actual - expected| over 1 + |expected| — so at outputs this size it is an
// absolute bound. The measured worst is 2.0e-7 for a prompt in one step and
// 2.1e-7 for the same tokens one at a time, on Metal: the products accumulate
// in float32 where the reference accumulates in double, and exp, rsqrt, tanh
// and the rotary table are each a float32 value whose last digits are the
// backend's rather than libm's. 5e-6 is more than an order of magnitude over
// that, which is the margin a second backend gets. A stride is wrong by whole
// digits rather than by the seventh.
constexpr auto tolerance = 5e-6;

// The prompt the small-shape tests decode: seven ids, none of them zero, none
// repeated, and all inside a 64-token vocabulary — so a gather reading the
// wrong row lands on a row that exists and produces numbers rather than
// failing.
//
// Seven rows is past Decoder::splitRowLimit, so a prompt in one step takes the
// tiled product and a single token takes the split one. The two-block test
// below crosses between them inside one sequence.
std::vector<int> promptTokens()
{
    return {5, 41, 3, 17, 28, 9, 60};
}

// Both halves of a step against the reference rows it should have reproduced,
// reported as a number rather than only as a pass: a tolerance nobody has
// measured against is a tolerance that was guessed.
double checkStepAgainstReference(const StepResult& result,
                                 const ReferenceDecoding& expected,
                                 const DecoderShape& shape,
                                 int firstRow,
                                 int rowCount)
{
    const auto hidden = worstError(
        result.hidden, rowsOf(expected.hidden, firstRow, rowCount, shape.width));

    const auto logits = worstError(
        result.logits,
        rowsOf(expected.logits, firstRow, rowCount, shape.logitElementCount()));

    check(hidden <= tolerance);
    check(logits <= tolerance);

    return std::max(hidden, logits);
}

ReferenceDecoding decodeOnTheCpu(const SyntheticCheckpoint& checkpoint,
                                 const std::vector<int>& tokens)
{
    const auto shape = checkpoint.shape();

    return referenceDecode(
        readReferenceModel(checkpoint.weights(), shape), shape, tokens);
}
} // namespace

// The tier plan.md's third step calls the bulk of the suite: a whole prompt
// through the GPU decoder in one step, against a scalar reference written from
// HuggingFace's definition, over weights that went through a real safetensors
// header and a real config.json.
//
// hiddenStates() is checked alongside the logits, which is what makes a
// disagreement bisect: rows that match with logits that do not is the tied
// projection, and rows that do not is everything before it.
auto tPromptMatchesReference = test("Decoder/promptMatchesReference") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint = SyntheticCheckpoint {"decoder-prompt"};
    const auto shape = checkpoint.shape();
    const auto tokens = promptTokens();

    auto run = DecoderRun {shape, checkpoint.weights()};
    const auto result = run.step(tokens);

    check(run.position() == (int) tokens.size());

    const auto worst = checkStepAgainstReference(
        result, decodeOnTheCpu(checkpoint, tokens), shape, 0, (int) tokens.size());

    std::cout << "  prompt in one step: worst error " << worst << "\n";
};

// The KV cache, stated as the property it exists for: the same tokens fed one
// at a time have to produce the rows the prompt did. The reference is what both
// are compared to rather than to each other, so a cache that agreed with a
// prompt path that was itself wrong would still fail.
//
// Every step here is one row, which is the split product and the decode
// attention — the other half of both switches in Decoder.cpp.
auto tCachedStepsMatchTheReference =
    test("Decoder/cachedStepsMatchTheReference") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint = SyntheticCheckpoint {"decoder-cached"};
    const auto shape = checkpoint.shape();
    const auto tokens = promptTokens();
    const auto expected = decodeOnTheCpu(checkpoint, tokens);

    auto run = DecoderRun {shape, checkpoint.weights()};
    auto worst = 0.0;

    for (auto index = 0; index < (int) tokens.size(); ++index)
    {
        const auto result = run.step({tokens[(std::size_t) index]});
        check(run.position() == index + 1);

        worst = std::max(
            worst, checkStepAgainstReference(result, expected, shape, index, 1));
    }

    std::cout << "  one token at a time: worst error " << worst << "\n";
};

// The same prompt in two blocks, which is what a run that appends to a system
// prefix does — and which crosses both product paths and both attention
// kernels inside one sequence: four rows take the tiled product and the prefill
// attention at position zero, three take them at position four.
auto tPromptInTwoBlocks = test("Decoder/promptInTwoBlocksMatchesReference") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint = SyntheticCheckpoint {"decoder-blocks"};
    const auto shape = checkpoint.shape();
    const auto tokens = promptTokens();
    const auto expected = decodeOnTheCpu(checkpoint, tokens);

    const auto first = std::vector<int> {tokens.begin(), tokens.begin() + 4};
    const auto second = std::vector<int> {tokens.begin() + 4, tokens.end()};

    auto run = DecoderRun {shape, checkpoint.weights()};

    auto worst = checkStepAgainstReference(
        run.step(first), expected, shape, 0, (int) first.size());

    check(run.position() == 4);

    worst = std::max(worst,
                     checkStepAgainstReference(
                         run.step(second), expected, shape, 4, (int) second.size()));

    check(run.position() == (int) tokens.size());

    std::cout << "  four then three: worst error " << worst << "\n";
};

// A decoder whose intermediates hold four rows, fed the same seven tokens in
// blocks of at most four: the sequence is the window's and only the step is
// short, so the rows have to be the reference's exactly as they are at the full
// capacity. This is the property the separation of the two capacities rests on,
// since a generation loop is precisely a prompt in blocks and then single rows.
auto tShortStepCapacityMatchesReference =
    test("Decoder/shortStepCapacityMatchesReference") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint = SyntheticCheckpoint {"decoder-step-capacity"};
    const auto tokens = promptTokens();
    const auto expected = decodeOnTheCpu(checkpoint, tokens);

    auto shape = checkpoint.shape();
    shape.maxStepRows = 4;

    check(shape.stepRowCapacity() == 4);
    check(shape.cacheElementCount() == checkpoint.shape().cacheElementCount());

    auto run = DecoderRun {shape, checkpoint.weights()};

    const auto first = std::vector<int> {tokens.begin(), tokens.begin() + 4};
    const auto second = std::vector<int> {tokens.begin() + 4, tokens.end()};

    auto worst = checkStepAgainstReference(
        run.step(first), expected, shape, 0, (int) first.size());

    worst = std::max(worst,
                     checkStepAgainstReference(
                         run.step(second), expected, shape, 4, (int) second.size()));

    // And one more row off the cache, which is the shape every step of a real
    // generation takes against intermediates sized for the prompt.
    run.step({12});
    check(run.position() == (int) tokens.size() + 1);

    std::cout << "  four-row capacity: worst error " << worst << "\n";
};

// A block wider than the intermediates, which the decoder has to refuse for the
// reason it refuses a step past the window: a dispatch taller than the buffer
// it writes runs off its end silently on both backends.
auto tStepPastTheRowCapacityIsAnError =
    test("Decoder/stepPastTheRowCapacityIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint = SyntheticCheckpoint {"decoder-step-capacity-error"};

    auto shape = checkpoint.shape();
    shape.maxStepRows = 4;

    auto run = DecoderRun {shape, checkpoint.weights()};

    // Five rows against a capacity of four, with the window's sixteen positions
    // wide open — so it is the step capacity being enforced and not the window.
    check(run.stepThrowsLeavingLogitsUntouched({5, 41, 3, 17, 28}));
    check(run.position() == 0);

    // Four is the capacity and runs, which is what makes the refusal a bound
    // rather than an off-by-one.
    run.step({5, 41, 3, 17});
    check(run.position() == 4);
};

// A step past the end of the window, which is the one bound the decoder has to
// enforce itself: the cache row it would bind is outside its buffer, and a
// BufferRange past a buffer's end binds nothing at all rather than failing. So
// the check is on the host, before anything is recorded — and "before" is what
// the untouched logits say.
auto tStepPastTheWindowIsAnError = test("Decoder/stepPastTheWindowIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint = SyntheticCheckpoint {"decoder-window"};
    const auto shape = checkpoint.shape();

    auto run = DecoderRun {shape, checkpoint.weights()};
    auto whole = std::vector<int> {};

    for (auto index = 0; index < shape.maxPositions; ++index)
        whole.push_back(index % shape.vocabularySize);

    run.step(whole);
    check(run.position() == shape.maxPositions);

    check(run.stepThrowsLeavingLogitsUntouched({1}));
    check(run.position() == shape.maxPositions);

    // And a block that would only overrun by one, which is the off-by-one a
    // bound written with the wrong comparison gets wrong.
    run.beginSequence();
    run.step({whole.begin(), whole.begin() + shape.maxPositions - 1});
    check(run.stepThrowsLeavingLogitsUntouched({1, 2}));
};

// beginSequence resets the position and nothing else, so the caches are
// overwritten from row zero rather than appended to: a second run of the same
// tokens has to reproduce the first bit for bit, since it is the same
// arithmetic in the same order over the same weights.
auto tBeginSequenceResets = test("Decoder/beginSequenceResetsTheSequence") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint = SyntheticCheckpoint {"decoder-reset"};
    const auto shape = checkpoint.shape();
    const auto tokens = promptTokens();
    const auto expected = decodeOnTheCpu(checkpoint, tokens);

    auto run = DecoderRun {shape, checkpoint.weights()};

    // A different sequence first, so the caches hold rows a reset has to make
    // irrelevant rather than rows that happen to be right.
    run.step({11, 22, 33, 44, 55});
    check(run.position() == 5);

    run.beginSequence();
    check(run.position() == 0);

    const auto first = run.step(tokens);
    checkStepAgainstReference(first, expected, shape, 0, (int) tokens.size());

    run.beginSequence();
    const auto second = run.step(tokens);

    for (auto index = 0; index < first.logits.size(); ++index)
        check(first.logits[index] == second.logits[index]);

    for (auto index = 0; index < first.hidden.size(); ++index)
        check(first.hidden[index] == second.hidden[index]);
};

// The same decoding out of a checkpoint split across two shards, which is the
// shape gemma-2b itself ships in. The weights are identical, so this is about
// the index resolving every name rather than about the arithmetic.
auto tShardedCheckpointDecodesTheSame =
    test("Decoder/shardedCheckpointDecodesTheSame") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint =
        SyntheticCheckpoint {"decoder-sharded-run", Sharding::TwoShards};

    const auto shape = checkpoint.shape();
    const auto tokens = promptTokens();

    auto run = DecoderRun {shape, checkpoint.weights()};
    const auto worst = checkStepAgainstReference(run.step(tokens),
                                                 decodeOnTheCpu(checkpoint, tokens),
                                                 shape,
                                                 0,
                                                 (int) tokens.size());

    std::cout << "  two shards: worst error " << worst << "\n";
};

// The storage gemma-2b actually ships, which every weight here now keeps all
// the way to the device: the same forward pass over a checkpoint written in
// BF16, through the tiled product a prompt takes and the split product a token
// takes, at the tolerance the F32 checkpoint is held to.
//
// The tolerance is unchanged because the storage is not a precision: the file
// holds rounded values, readFloats hands the reference those same rounded
// values, and the widening a kernel does on the way in is exact. What would
// fail here is a packed read that landed on the wrong half of a word or a
// fused weight whose two halves were stacked in the wrong storage — both of
// which are wrong by whole digits rather than by the seventh.
auto tBFloat16CheckpointMatchesReference =
    test("Decoder/bfloat16CheckpointMatchesReference") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint =
        SyntheticCheckpoint {"decoder-bfloat16", TensorType::BF16};

    const auto shape = checkpoint.shape();
    const auto tokens = promptTokens();
    const auto expected = decodeOnTheCpu(checkpoint, tokens);

    auto prompt = DecoderRun {shape, checkpoint.weights()};

    auto worst = checkStepAgainstReference(
        prompt.step(tokens), expected, shape, 0, (int) tokens.size());

    auto cached = DecoderRun {shape, checkpoint.weights()};

    for (auto index = 0; index < (int) tokens.size(); ++index)
    {
        const auto result = cached.step({tokens[(std::size_t) index]});

        worst = std::max(
            worst, checkStepAgainstReference(result, expected, shape, index, 1));
    }

    std::cout << "  bf16 checkpoint: worst error " << worst << "\n";
};

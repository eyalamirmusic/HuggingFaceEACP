#pragma once

#include "Gelu.h"
#include "WeightStorage.h"

#include <algorithm>

namespace HF
{
// y = x W^T + b, which is every projection in a transformer, with W stored the
// way PyTorch's nn.Linear stores it: [outputWidth, innerCount] row-major, one
// output's whole input row contiguous.
//
//   output[row, o] = bias[o]
//                  + sum over k of input[row * innerCount + k]
//                                 * weight[o * innerCount + k]
//
// That is the shape model.layers.0.self_attn.q_proj.weight already has in the
// safetensors file, so the loader hands the bytes over untransposed and
// nothing in the model transposes a weight. MatMul next door is the same
// product with the operand stored the other way round, which is the shape a
// matrix multiplication is written in rather than the shape a weight is
// shipped in; both exist because both occur.
//
// One thread per output element over a 2D grid, accumulating serially:
// dispatch(kernel, outputWidth, rowCount). The row count is the dispatch
// height and never reaches the kernel body, which is why it is not a uniform;
// the other two are, because they are the strides the input, the weight and
// the output are walked at.
//
// The bias buffer is always bound, for MatMul's reason: Gemma has no bias
// anywhere, so every projection binds a zero buffer of outputWidth floats,
// since neither backend defines what a shader reading a buffer nothing was
// bound to gets.
template <WeightStorage weightStorage>
struct LinearProgram final : ComputeProgram
{
    LinearProgram() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto column = position.x;
        auto row = position.y;

        auto inputBase = row * innerCount;
        auto weightBase = column * innerCount;

        auto total = var(0.f);
        auto step = var(0u);

        loop(step < innerCount,
             [&]
             {
                 total += input[inputBase + step] * weight(weightBase + step);
                 step += 1u;
             });

        write(output, row * outputWidth + column, total.get() + bias[column]);
    }

    Float weight(const UInt& index)
    {
        return storedWeight<weightStorage>(weights, index, scaleBase());
    }

    // Where the weight's per-block scales begin, in halves: past its
    // outputWidth * innerCount elements, both of which are already uniforms.
    // Read by the quantized storage alone — see storedWeight.
    UInt scaleBase() { return outputWidth * innerCount / 2u; }

    Uniform<InputBuffer> input;
    Uniform<InputBuffer> weights;
    Uniform<InputBuffer> bias;
    Uniform<OutputBuffer> output;
    Uniform<UInt> innerCount;
    Uniform<UInt> outputWidth;

    EACP_SHADER(input, weights, bias, output, innerCount, outputWidth)
};

using Linear = LinearProgram<WeightStorage::Float>;
using HalfWeightLinear = LinearProgram<WeightStorage::PackedHalf>;
using BFloat16WeightLinear = LinearProgram<WeightStorage::PackedBFloat16>;
using Int8WeightLinear = LinearProgram<WeightStorage::Int8Blocks>;

// The same product for a handful of rows — a decode step's one token, or the
// prompt's two — where a thread per output is a thread per 2048 or 16384
// serial multiply-adds and the GPU is nearly idle. The inner sum is split
// across splitCount lanes instead: the grid is [splitCount, rowCount *
// outputWidth], each lane of a group row takes every splitCount'th run of four
// inputs of one output, and the group folds the partial sums before the one
// store. Adjacent lanes read adjacent words of the same weight row, which is
// the access the memory system serves whole.
//
// **The split count is the shape's, and it is what makes this kernel fast.**
// It sets both how much of the inner sum a lane walks and how many groups the
// dispatch has, and the second is what a decode step is short of. The caller
// names the count its shapes want, the way ReducingProgram takes its lane
// count; the two Gemma will want are a projection's and the logits', both of
// which have to be measured rather than guessed.
//
// A group is splitCount lanes by however many outputs 64 threads then hold,
// and one output at or past 64. At one output the fold is groupSum(), which is
// a SIMD reduction rather than the shared array and the serial walk of it a
// group holding several outputs still needs.
//
// dispatch(pass, outputWidth, rowCount), which rounds the height up to whole
// groups since the fold is a barrier; a thread past the last output computes
// against the last row and stores nothing. A lane's run is sixteen weights,
// eight, four or one, the widest of them the row's extent and the lane count
// leave whole — see the choice at the loop itself — so the kernel is correct at
// every shape and fast at the model's.
//
// Two flags fold the stages either side of a projection into its store, each a
// dispatch of its own otherwise and a few microseconds of GPU whatever its
// size: gelu applies the model's activation to the result, and residual adds
// the result to what the output already holds, which is what follows every
// o_proj and down_proj — in place on the residual stream, the element read and
// stored by the one lane that stores it. That one lane matters: the residual
// is a read-modify-write, so every path here stores from a single lane of the
// group rather than from all of them holding the same folded value.
//
// The activation the flag applies is the tanh GELU, Gemma's own. Gemma's own
// feed-forward is gated and so does not use it — GeGLU.h is what follows the
// concatenated gate-and-up product — but an ungated projection through this
// kernel should activate the way the rest of the model does.
template <WeightStorage weightStorage>
struct SplitLinearProgram final : ComputeProgram
{
    explicit SplitLinearProgram(int splitCount = groupSize2D)
        : ComputeProgram({splitCount, std::max(1, groupWidth / splitCount)})
        , splits((unsigned) groupShape().x)
        , outputsPerGroup(groupShape().y)
    {
        compile();
    }

    void dispatch(ComputePass& pass, int outputWidth, int rowCount)
    {
        const auto outputs = outputWidth * rowCount;
        const auto height =
            (outputs + outputsPerGroup - 1) / outputsPerGroup * outputsPerGroup;

        pass.dispatch(*this, (int) splits, height);
    }

    void define() override
    {
        auto position = threadPosition();
        auto split = position.x;
        auto element = position.y;
        auto row = element / outputWidth;
        auto column = element % outputWidth;

        auto inputBase = min(row, rowCount - 1u) * innerCount;
        auto weightBase = column * innerCount;

        auto total = var(0.f);

        // Lane `split` takes every splitCount'th record of `width` weights,
        // whatever the width is, so adjacent lanes always read adjacent
        // records of the same weight row — the access the memory system serves
        // whole — and every record of the row is taken by exactly one lane.
        auto walk = [&](unsigned width, auto&& accumulate)
        {
            auto step = var(split * width);

            loop(step.get() < innerCount,
                 [&]
                 {
                     accumulate(step.get());
                     step += splits * width;
                 });
        };

        auto sixteenWide = [&](const UInt& at)
        {
            auto row4 = (inputBase + at) / 4u;
            auto quad = weight16(weightBase + at);

            total += dot(input.read4(row4), quad.a)
                     + dot(input.read4(row4 + 1u), quad.b)
                     + dot(input.read4(row4 + 2u), quad.c)
                     + dot(input.read4(row4 + 3u), quad.d);
        };

        auto eightWide = [&](const UInt& at)
        {
            auto row4 = (inputBase + at) / 4u;
            auto pair = weight8(weightBase + at);

            total += dot(input.read4(row4), pair.low)
                     + dot(input.read4(row4 + 1u), pair.high);
        };

        auto fourWide = [&](const UInt& at)
        {
            total +=
                dot(input.read4((inputBase + at) / 4u), weight4(weightBase + at));
        };

        auto oneAtATime = [&](const UInt& at)
        { total += input[inputBase + at] * weight(weightBase + at); };

        // **The widest record the row divides by, and never mind whether it
        // leaves lanes idle.** A record of sixteen is one load of a quantized
        // weight where four records of four are four, so at a row of 2048 it is
        // 128 records — and a group of 256 lanes then has half of them with
        // nothing to do. That costs less than the loads it saves, and it was
        // measured rather than assumed: eight-wide records with every lane busy
        // give 115.7 decode tokens/s over 64 against sixteen-wide records with
        // half the lanes idle at 122.8, on the same device and the same step.
        // What the idle half really says is that the lane count was chosen for
        // a narrower record — see Decoder::stepSplitCount, which is now 128 so
        // that a 2048-wide row is exactly one sixteen-record to a lane.
        //
        // Every branch is correct at every shape: nothing here requires
        // innerCount to be a multiple of anything, since a width the row does
        // not divide by falls to the next one down and the last of them walks
        // single elements.
        ifThen(
            innerCount % 16u == 0u,
            [&] { walk(16u, sixteenWide); },
            [&]
            {
                ifThen(
                    innerCount % 8u == 0u,
                    [&] { walk(8u, eightWide); },
                    [&]
                    {
                        ifThen(
                            innerCount % 4u == 0u,
                            [&] { walk(4u, fourWide); },
                            [&] { walk(1u, oneAtATime); });
                    });
            });

        auto local = localPosition();

        // **One output to a SIMD group is the fold this kernel wants.** At
        // splits == simdWidth the group is [32, 2], threads are flattened x
        // first, and each SIMD group is therefore exactly one output's lanes —
        // so simdSum hands every lane its own output's inner sum with neither
        // threadgroup scratch nor a barrier, where the general case below
        // publishes 64 partial sums and walks 32 of them serially. That count
        // is the logits projection's own, which is the widest product a decode
        // step runs.
        //
        // Narrower and a SIMD group spans several outputs, which would fold
        // them together; wider and one output spans several, which would leave
        // each holding a part. Both keep what they had.
        if (splits == (unsigned) simdWidth)
        {
            auto sum = var(simdSum(total.get()));

            ifThen(local.x == 0u, [&] { store(sum.get(), element, column, row); });
            return;
        }

        if (outputsPerGroup == 1)
        {
            auto sum = var(groupSum(total.get()));

            ifThen(local.x == 0u, [&] { store(sum.get(), element, column, row); });
            return;
        }

        auto tile = shared<Float>((int) splits * outputsPerGroup);

        write(tile, local.y * splits + local.x, total.get());
        barrier();

        auto sum = var(0.f);

        for (auto part = 0u; part < splits; ++part)
            sum += tile[local.y * splits + part];

        ifThen(local.x == 0u, [&] { store(sum.get(), element, column, row); });
    }

    // The one store, behind the row test a rounded-up dispatch height needs:
    // the bias, then the two stages folded into it.
    void store(const Float& sum,
               const UInt& element,
               const UInt& column,
               const UInt& row)
    {
        ifThen(row < rowCount,
               [&]
               {
                   auto value = sum + bias[column];
                   auto activated = select(gelu != 0u, tanhGelu(value), value);
                   auto carried = select(residual != 0u, output[element], 0.f);

                   write(output, element, activated + carried);
               });
    }

    Float weight(const UInt& index)
    {
        return storedWeight<weightStorage>(weights, index, scaleBase());
    }

    Float4 weight4(const UInt& index)
    {
        return storedWeight4<weightStorage>(weights, index, scaleBase());
    }

    Float4Pair weight8(const UInt& index)
    {
        return storedWeight8<weightStorage>(weights, index, scaleBase());
    }

    Float4Quad weight16(const UInt& index)
    {
        return storedWeight16<weightStorage>(weights, index, scaleBase());
    }

    // LinearProgram's, at this kernel's own uniforms. **There is nothing to
    // hoist between iterations of the loop above**, which is worth saying
    // because the obvious optimisation is to read one scale per block of
    // thirty-two rather than one per record: a lane's records are
    // splitCount * recordWidth elements apart, so at every split count this
    // kernel is dispatched at, consecutive iterations are in different blocks
    // and each of them reads its own scale exactly once. What shares a scale is
    // neighbouring lanes, and those read the same half in the same cycle. What
    // there is to hoist is inside one iteration, which is what the wide records
    // do: sixteen weights from one block cost one scale read between them.
    UInt scaleBase() { return outputWidth * innerCount / 2u; }

    Uniform<InputBuffer> input;
    Uniform<InputBuffer> weights;
    Uniform<InputBuffer> bias;
    Uniform<OutputBuffer> output;
    Uniform<UInt> innerCount;
    Uniform<UInt> outputWidth;
    Uniform<UInt> rowCount;
    Uniform<UInt> gelu;
    Uniform<UInt> residual;

    // How many lanes share one output's inner sum, and how many outputs one
    // group therefore holds: the group is [splits, groupWidth / splits] below
    // the group width, so it stays 64 threads, and [splits, 1] at or above it.
    const unsigned splits;
    const int outputsPerGroup;

    EACP_SHADER(input,
                weights,
                bias,
                output,
                innerCount,
                outputWidth,
                rowCount,
                gelu,
                residual)
};

using SplitLinear = SplitLinearProgram<WeightStorage::Float>;
using HalfWeightSplitLinear = SplitLinearProgram<WeightStorage::PackedHalf>;
using BFloat16WeightSplitLinear = SplitLinearProgram<WeightStorage::PackedBFloat16>;
using Int8WeightSplitLinear = SplitLinearProgram<WeightStorage::Int8Blocks>;
} // namespace HF

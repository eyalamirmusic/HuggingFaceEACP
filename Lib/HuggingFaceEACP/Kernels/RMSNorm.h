#pragma once

#include "Reduce.h"

namespace HF
{
// y = x * rsqrt(mean(x^2) + epsilon) * (1 + weight), over rows of rowLength
// elements, with weight one row long and shared by every row. This is the only
// normalisation Gemma has — no mean subtraction, no bias, and the scale
// offset by one.
//
// **The `1 +` is inside the kernel and not folded into the weight by the
// loader.** Hugging Face's GemmaRMSNorm stores the tensor the checkpoint
// shipped and writes `x * (1 + w)` at every call, so the numbers on disk are
// centred near zero; a loader that added one on the way in would be correct
// and would also mean the bytes in the buffer are no longer the bytes in the
// file, which is the thing a tensor-level test compares.
//
// One group per row: dispatchRows(pass, rowCount). The sum of squares is a
// strided walk and one group reduction, accumulated in float32 whatever the
// activations were stored as — which is the whole point of separating this
// from a fused product, since a bf16 or fp16 accumulation of 2048 squares
// loses the small terms that decide the scale.
//
// The row length is a uniform rather than a constant because this is what the
// decoder gets written out of: eighteen layers of two norms each, plus the
// final one, all through one pipeline.
//
// The lane count is the caller's, because the two shapes this is dispatched at
// are opposite. A prompt step normalises many rows at once and fills the
// machine on its own, where a wider group only adds barriers; a decode step is
// one row of 2048 with nothing else running, where the widest group that still
// has work for every lane is what hides the latency. LayerNorm in WhisperEACP
// measured to opposite answers at the same two shapes, which is why the count
// is a parameter rather than a constant; what the decoder's own shapes measure
// to is in Decoder.h beside normLanes.
struct RMSNorm final : ReducingProgram
{
    // config.json's rms_norm_eps. Inside the square root, as Hugging Face
    // applies it: rsqrt(mean + eps), not rsqrt(mean) + eps and not
    // rsqrt(mean * (1 + eps)).
    static constexpr auto gemmaEpsilon = 1e-6f;

    explicit RMSNorm(int laneCount = groupWidth)
        : ReducingProgram(laneCount)
    {
        epsilon = gemmaEpsilon;
        compile();
    }

    void define() override
    {
        auto lane = localId();
        auto base = groupId() * rowLength;
        auto width = toFloat(rowLength);

        auto squares = var(0.f);
        auto summing = var(lane);

        loop(summing.get() < rowLength,
             [&]
             {
                 auto value = input[base + summing.get()];
                 squares += value * value;
                 summing += lanes;
             });

        auto scale = var(rsqrt(groupSum(squares.get()) / width + epsilon));
        auto writing = var(lane);

        loop(writing.get() < rowLength,
             [&]
             {
                 auto at = base + writing.get();
                 auto column = writing.get();

                 write(output, at, input[at] * scale.get() * (1.f + weight[column]));

                 writing += lanes;
             });
    }

    Uniform<InputBuffer> input;
    Uniform<InputBuffer> weight;
    Uniform<OutputBuffer> output;
    Uniform<UInt> rowLength;
    Uniform<Float> epsilon;

    EACP_SHADER(input, weight, output, rowLength, epsilon)
};
} // namespace HF

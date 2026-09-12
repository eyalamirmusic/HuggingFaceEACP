#pragma once

#include "WeightStorage.h"

namespace HF
{
// C = A * B + bias, all row-major: A is rowCount x innerCount, B is
// innerCount x columnCount, C is rowCount x columnCount, and bias is one value
// per output column, broadcast down every row — the shape a linear layer's
// bias has.
//
// One thread per output element over a 2D grid, accumulating serially:
// dispatch(kernel, columnCount, rowCount). The row count is the dispatch
// height and never reaches the kernel body, which is why it is not a uniform;
// the other two are, because they are the strides A, B and C are walked at.
//
// The bias buffer is always bound. Gemma has no bias anywhere, so every
// dispatch here binds a zero buffer of columnCount floats — which is the
// cheapest way to say it: neither backend defines what a shader reading a
// buffer nothing was bound to gets, so there is no unbound slot to branch
// around and a flag would only guard a read that must not happen at all.
//
// B is the weight, and the only operand that can be packed: a model's weights
// are what a repo ships narrow, and A, bias and C are the activations, which
// this stack computes in float32 throughout. The packed forms index the same B
// by element and widen each on read, so all three take identical uniforms and
// differ only in the buffer bound to b.
template <WeightStorage weightStorage>
struct MatMulProgram final : ComputeProgram
{
    MatMulProgram() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto column = position.x;
        auto row = position.y;

        auto total = var(0.f);
        auto step = var(0u);

        loop(step < innerCount,
             [&]
             {
                 total += a[row * innerCount + step]
                          * weight(step * columnCount + column);
                 step += 1u;
             });

        write(output, row * columnCount + column, total.get() + bias[column]);
    }

    Float weight(const UInt& index)
    {
        return storedWeight<weightStorage>(b, index, scaleBase());
    }

    // Where B's per-block scales begin, in halves: past its innerCount *
    // columnCount elements, both of which are already uniforms. Read by the
    // quantized storage alone — see storedWeight.
    UInt scaleBase() { return innerCount * columnCount / 2u; }

    Uniform<InputBuffer> a;
    Uniform<InputBuffer> b;
    Uniform<InputBuffer> bias;
    Uniform<OutputBuffer> output;
    Uniform<UInt> innerCount;
    Uniform<UInt> columnCount;

    EACP_SHADER(a, b, bias, output, innerCount, columnCount)
};

using MatMul = MatMulProgram<WeightStorage::Float>;
using HalfWeightMatMul = MatMulProgram<WeightStorage::PackedHalf>;
using BFloat16WeightMatMul = MatMulProgram<WeightStorage::PackedBFloat16>;
using Int8WeightMatMul = MatMulProgram<WeightStorage::Int8Blocks>;
} // namespace HF

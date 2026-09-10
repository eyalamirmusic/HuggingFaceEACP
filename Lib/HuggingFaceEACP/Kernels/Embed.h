#pragma once

#include "KernelTypes.h"

namespace HF
{
// The decoder's input row, which is a gather and a multiply:
//
//   output[t, c] = scale * tokenTable[token[t], c]
//
// Whisper's Embed added a learned positional row here. Gemma has no positional
// table at all — position enters through RoPE on the queries and keys — so
// what is left is the gather, and the scale the model applies to it.
//
// The layouts, over a model width W:
//
//   tokens      [tokenCount]        one token id per step, unsigned
//   tokenTable  [vocabulary, W]     row-major, the tied embedding
//   output      [tokenCount, W]     row-major
//
// One thread per (c, t) over a 2D grid: dispatch(kernel, W, tokenCount). The
// token count is the dispatch height and never reaches the body; W is a
// uniform, since it is the stride both the table and the output are walked at.
//
// scale is a uniform rather than sqrt(W) computed here, and that is not a
// convenience. Gemma multiplies the embedding by the hidden size's square root
// rounded through bfloat16 — the constant is built as a bf16 value in the
// reference implementation, so at W = 2048 the factor is 45.25 exactly and not
// sqrt(2048) = 45.254833. Baking the square root in would put a kernel one
// bf16 rounding away from the reference on every token; taking the number from
// the caller keeps that decision in the model, where config.json is.
//
// The ids arrive through an integer buffer, which is what lets the id Argmax
// wrote at the end of one step be the id this reads at the start of the next
// without a trip through the host: a vocabulary is indexed, not measured, and
// the buffer's element type says so on both backends.
struct Embed final : ComputeProgram
{
    Embed() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto channel = position.x;
        auto step = position.y;

        auto token = tokens[step];

        write(output,
              step * width + channel,
              scale * tokenTable[token * width + channel]);
    }

    Uniform<UIntInputBuffer> tokens;
    Uniform<InputBuffer> tokenTable;
    Uniform<OutputBuffer> output;
    Uniform<UInt> width;
    Uniform<Float> scale;

    EACP_SHADER(tokens, tokenTable, output, width, scale)
};
} // namespace HF

#pragma once

#include "WeightStorage.h"

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
// convenience: which square root it is depends on what the model is running
// in, and that is the config's decision rather than this kernel's. Hugging
// Face builds the constant in the model's own dtype, so a bf16 run rounds it
// to 45.25 at W = 2048 where sqrt(2048) is 45.254833; llama.cpp over an F32
// GGUF — the oracle plan.md's third step compares against — uses sqrtf(n_embd)
// unrounded, and so does DecoderShape::embeddingScale. Baking either in would
// put a kernel one rounding away from whichever reference it is being read
// against; taking the number from the caller keeps that where config.json is.
//
// The ids arrive through an integer buffer, which is what lets the id Argmax
// wrote at the end of one step be the id this reads at the start of the next
// without a trip through the host: a vocabulary is indexed, not measured, and
// the buffer's element type says so on both backends.
//
// The table takes a WeightStorage for the reason the products do, and it is the
// tied embedding that asks: the same buffer is the gather's table and the
// logits projection's weight, so a gather with only a float form would have
// forced the largest tensor in the model to be widened however the product read
// it. storedWeight is the same widening either kernel does.
template <WeightStorage tableStorage>
struct EmbedProgram final : ComputeProgram
{
    EmbedProgram() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto channel = position.x;
        auto step = position.y;

        auto token = tokens[step];
        auto element = token * width + channel;
        auto gathered = storedWeight<tableStorage>(tokenTable, element);

        write(output, step * width + channel, scale * gathered);
    }

    Uniform<UIntInputBuffer> tokens;
    Uniform<InputBuffer> tokenTable;
    Uniform<OutputBuffer> output;
    Uniform<UInt> width;
    Uniform<Float> scale;

    EACP_SHADER(tokens, tokenTable, output, width, scale)
};

using Embed = EmbedProgram<WeightStorage::Float>;
using HalfWeightEmbed = EmbedProgram<WeightStorage::PackedHalf>;
using BFloat16WeightEmbed = EmbedProgram<WeightStorage::PackedBFloat16>;
} // namespace HF

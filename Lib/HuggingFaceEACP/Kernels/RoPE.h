#pragma once

#include "KernelTypes.h"

namespace HF
{
// Rotary position embedding, rotate-half form, applied to the queries and the
// keys after their projections. Over a head of headDim channels, with
// half = headDim / 2 and the angle a = position * theta^(-2i/headDim):
//
//   out[i]        = x[i]        * cos(a) - x[i + half] * sin(a)
//   out[i + half] = x[i + half] * cos(a) + x[i]        * sin(a)
//
// which is the pair (x[i], x[i + half]) turned through a. **Rotate-half, not
// pairwise**: Hugging Face's implementation rotates channel i against channel
// i + half rather than against its neighbour, and the two orderings give
// different vectors from the same weights. The choice is a permutation of the
// projection's output columns, so a model whose weights were converted for one
// and run under the other produces plausible-looking nonsense rather than an
// error.
//
// The layouts, over a buffer of `rows` rows of `heads` heads of headDim:
//
//   input, output  [rows, rowStride]      row-major, head h at h * headDim
//   cosines, sines [maxPositions, half]   row-major, uploaded once
//
// rowStride is a uniform rather than heads * headDim so that a head group
// inside a wider activation is rotated where it lies: a fused q-k-v product
// writes one [rows, (heads + 2) * headDim] buffer, and the q heads and the k
// head are then two dispatches of this kernel into it at different offsets,
// with no copy between.
//
// Row r carries position firstPosition + r, which is a KV cache's own shape:
// a step appending one token to a cache of n binds firstPosition = n and
// dispatches one row, while the prompt that opens a sequence binds zero and
// dispatches its whole length.
//
// The same program does Gemma's queries and its keys. Multi-query attention
// gives the eight query heads one shared KV head, so q is dispatched with
// headCount 8 and k with headCount 1 over the identical table — nothing about
// the rotation knows how many heads it is being run over, which is the point.
//
// One thread per rotation pair over a 2D grid: dispatch(pass, rows, heads,
// headDim) gives [half, rows * heads], so each thread owns both halves of one
// pair and reads and writes both. That is what makes it safe in place: the two
// elements a thread stores are the two it read, and no other thread touches
// either — but both are read into vars before either is stored, since a buffer
// read in eacp names the element rather than the value and re-materialises
// after a store to its slot.
struct RoPE final : ComputeProgram
{
    RoPE() { compile(); }

    void dispatch(ComputePass& pass, int rows, int heads, int channelsPerHead)
    {
        pass.dispatch(*this, channelsPerHead / 2, rows * heads);
    }

    void define() override
    {
        auto position = threadPosition();
        auto pair = position.x;
        auto slot = position.y;

        auto row = slot / headCount;
        auto head = slot % headCount;
        auto half = headDim / 2u;

        auto low = row * rowStride + head * headDim + pair;
        auto high = low + half;

        auto angle = (firstPosition + row) * half + pair;
        auto cosine = cosines[angle];
        auto sine = sines[angle];

        auto x = var(input[low]);
        auto y = var(input[high]);

        write(output, low, x.get() * cosine - y.get() * sine);
        write(output, high, y.get() * cosine + x.get() * sine);
    }

    Uniform<InputBuffer> input;
    Uniform<InputBuffer> cosines;
    Uniform<InputBuffer> sines;
    Uniform<OutputBuffer> output;
    Uniform<UInt> rowStride;
    Uniform<UInt> headDim;
    Uniform<UInt> headCount;
    Uniform<UInt> firstPosition;

    EACP_SHADER(
        input, cosines, sines, output, rowStride, headDim, headCount, firstPosition)
};

// The two tables the kernel reads, each [maxPositions, headDim / 2] row-major.
struct RotaryTable
{
    Vector<float> cosines;
    Vector<float> sines;
};

// theta from config.json's rope_theta. 10000 is the transformer's original and
// Gemma 2B's own; a long-context variant raises it, which is why it is an
// argument.
inline constexpr auto gemmaRopeTheta = 10000.0;

// Built once on the CPU in double precision and narrowed on the store, rather
// than computed per thread in the shader. Two reasons, and the second is the
// one that matters: theta^(-2i/headDim) at i = 127 of 256 is 1e-4, and a
// float32 pow of that against a position of 8191 leaves an angle whose last
// digits are the pow's error rather than the position's — while std::cos and
// std::sin of a double argument are correctly rounded to well under a float
// ulp. The first reason is only that 8192 x 128 x 2 floats is 8 MB read once
// against a transcendental pair per channel per token.
RotaryTable
    makeRotaryTable(int maxPositions, int headDim, double theta = gemmaRopeTheta);
} // namespace HF

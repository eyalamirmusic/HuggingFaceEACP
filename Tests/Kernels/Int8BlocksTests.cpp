#include "Common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

// The block-quantized weight format itself, before any kernel reads one: what
// the quantizer writes, what the dequantizer gets back, and which shapes it
// refuses. The products over it are beside their own kernels — LinearTests,
// MatMulTests, the two tiled suites and EmbedTests each have an Int8Blocks case
// next to their bf16 one.
//
// **What is asserted is not "close".** The dequantizer is the shader's own
// arithmetic written out on the host — a byte sign-extended the way every
// backend spells it, times the block's fp16 scale widened — so every value it
// hands back is exactly a scale times a small integer, and a product over it
// disagrees with the GPU only about the order it summed in. That is what lets
// the kernel tests below use the same tolerances the exact bf16 path answers
// to.

using namespace nano;
using namespace HF;

namespace
{
// The two numbers a block is defined by, recomputed here from the definition
// rather than taken from the quantizer: the largest magnitude over 127,
// narrowed to fp16.
float expectedScale(Span<const float> block)
{
    auto largest = 0.f;

    for (auto index = 0; index < block.size(); ++index)
        largest = std::max(largest, std::abs(block[index]));

    return eacp::GPU::halfToFloat(eacp::GPU::halfFromFloat(largest / 127.f));
}

Vector<std::uint8_t> quantized(const Vector<float>& values)
{
    auto bytes = Vector<std::uint8_t> {};
    bytes.resize((int) int8BlocksByteCount(values.size()));

    quantizeInt8Blocks(Span<const float> {values.data(), values.size()},
                       Span<std::uint8_t> {bytes.data(), bytes.size()});

    return bytes;
}

Vector<float> dequantized(const Vector<std::uint8_t>& bytes, int elementCount)
{
    auto values = sized(elementCount);

    dequantizeInt8Blocks(Span<const std::uint8_t> {bytes.data(), bytes.size()},
                         Span<float> {values.data(), elementCount});

    return values;
}

// The largest relative error over a run of values, reported rather than only
// checked: a tolerance nobody has measured against is a tolerance that was
// guessed, and this is the one place the format's own accuracy is visible.
double worstRelativeError(const Vector<float>& original,
                          const Vector<float>& recovered,
                          int first,
                          int count)
{
    auto worst = 0.0;

    for (auto index = first; index < first + count; ++index)
    {
        const auto expected = (double) original[index];
        const auto gap = std::abs((double) recovered[index] - expected);

        worst = std::max(worst, gap / (1.0 + std::abs(expected)));
    }

    return worst;
}
} // namespace

// The format's own claim, element by element: every value comes back as its
// block's scale times a whole number in [-127, 127], the scale is the one the
// definition gives, and nothing is off by a rounding the shader would not make.
//
// No device: this is host arithmetic, and it is the arithmetic the shader
// mirrors rather than an approximation of it.
auto tInt8BlocksRoundTrip = test("Kernels/int8BlocksRoundTrip") = []
{
    constexpr auto elements = 32 * 40;

    const auto values = spreadValues(elements, 8181u, 3.f);
    const auto bytes = quantized(values);
    const auto recovered = dequantized(bytes, elements);

    check(bytes.size() == elements + 2 * (elements / int8BlockSize));
    check(recovered.size() == elements);

    for (auto block = 0; block < elements / int8BlockSize; ++block)
    {
        const auto first = block * int8BlockSize;
        const auto scale =
            expectedScale(Span<const float> {values.data() + first, int8BlockSize});

        check(scale > 0.f);

        for (auto index = first; index < first + int8BlockSize; ++index)
        {
            const auto steps = std::nearbyint(recovered[index] / scale);

            check(std::abs(steps) <= 127.0);
            check(recovered[index] == scale * (float) steps);
            check(std::abs(recovered[index] - values[index]) <= 0.5f * scale);
        }
    }

    std::cout << "  int8 blocks: worst relative error "
              << worstRelativeError(values, recovered, 0, elements) << "\n";
};

// **Blocks of very different magnitudes**, which is what a per-block scale
// exists for: a run at 1e-2 beside a run at 1e4 and a run of zeroes. A single
// scale over the whole tensor would flush the first to nothing; a scale per
// block keeps every one of them at seven bits of its own magnitude.
auto tInt8BlocksFollowTheBlockMagnitude =
    test("Kernels/int8BlocksFollowTheBlockMagnitude") = []
{
    constexpr auto blocks = 6;
    constexpr auto elements = blocks * int8BlockSize;

    // Six orders of magnitude apart, with a block of zeroes and a block too
    // small for any fp16 scale at all. The four that are held span 1e-2 to 1e4,
    // which is where a real checkpoint's blocks are and where the scale is a
    // normal fp16 rather than a subnormal one — see Int8Blocks.h on why that
    // boundary is the format's own.
    const auto magnitudes =
        Vector<float> {1.0e-2f, 1.0e3f, 0.f, 1.f, 6.0e-8f, 1.0e4f};

    auto values = spreadValues(elements, 1717u, 1.f);

    for (auto block = 0; block < blocks; ++block)
        for (auto index = 0; index < int8BlockSize; ++index)
            values[block * int8BlockSize + index] *= magnitudes[block];

    const auto recovered = dequantized(quantized(values), elements);

    // Every block but the two the format cannot hold keeps its own magnitude to
    // within half a step of its own scale, whatever its neighbours are: the
    // 1e-2 block is held to 4e-5 while the 1e4 block beside it is held to 40.
    for (auto block = 0; block < blocks; ++block)
    {
        const auto first = block * int8BlockSize;
        const auto worst =
            worstRelativeError(values, recovered, first, int8BlockSize);

        std::cout << "  block at " << magnitudes[block] << ": worst relative "
                  << worst << "\n";

        if (magnitudes[block] == 0.f)
        {
            for (auto index = first; index < first + int8BlockSize; ++index)
                check(recovered[index] == 0.f);

            continue;
        }

        // 6e-8 is under fp16's smallest subnormal times 127, so its scale
        // narrows to nothing and the block is stored as zeroes — deliberately,
        // since a scale of zero has no quantization to offer. It is also 1e-8
        // of a weight, which is why that is the right answer rather than a
        // reason to widen the scale.
        if (magnitudes[block] < 1.0e-7f)
        {
            for (auto index = first; index < first + int8BlockSize; ++index)
                check(recovered[index] == 0.f);

            continue;
        }

        // Half a step of this block's own scale, which is what a per-block
        // scale buys and one scale over the whole tensor would not.
        for (auto index = first; index < first + int8BlockSize; ++index)
            check(std::abs(recovered[index] - values[index])
                  <= 0.5f * magnitudes[block] / 127.f);
    }
};

// Which shapes the format takes, which is what the loader refuses a tensor by.
// The blocks run along the contiguous dimension, so that dimension is a whole
// number of them; and both regions are read a word at a time, so the element
// count is a whole number of scale words — sixty-four elements, not thirty-two.
//
// Gemma's own are all in: a projection weight's contiguous dimension is 2048 or
// 16384, and the smallest tensor a product reads is [256, 2048].
auto tInt8BlocksShapes = test("Kernels/int8BlocksShapes") = []
{
    check(quantizesAsInt8Blocks(2048, 2048 * 2048));
    check(quantizesAsInt8Blocks(16384, 2048 * 16384));
    check(quantizesAsInt8Blocks(2048, 256000 * 2048));
    check(quantizesAsInt8Blocks(32, 64));

    // The contiguous dimension is not a whole number of blocks. 48 is the
    // synthetic test model's own feed-forward width, which is why the
    // quantized decoder tests run at 64 instead.
    check(!quantizesAsInt8Blocks(48, 32 * 48));

    // A whole number of blocks, an odd number of them, so the scales region
    // would end half a word short.
    check(!quantizesAsInt8Blocks(32, 32 * 3));

    check(!quantizesAsInt8Blocks(0, 0));

    check(int8BlocksByteCount(64) == 68);
    check(int8BlocksByteCount(2048 * 2048) == 2048 * 2048 + 2048 * 2048 / 16);
};

// A run quantized in pieces is the run quantized whole, which is what lets the
// loader split five gigabytes across every core it has: the blocks are
// independent, so a thread that took elements [begin, end) writes exactly the
// bytes a single pass would have written there.
auto tInt8BlocksQuantizeInPieces = test("Kernels/int8BlocksQuantizeInPieces") = []
{
    constexpr auto elements = 32 * 20;

    const auto values = spreadValues(elements, 5150u, 2.f);
    const auto whole = quantized(values);

    auto pieces = Vector<std::uint8_t> {};
    pieces.resize(whole.size());

    const auto destination = Span<std::uint8_t> {pieces.data(), pieces.size()};

    for (auto first = 0; first < elements; first += 4 * int8BlockSize)
        quantizeInt8Blocks(
            Span<const float> {values.data() + first, 4 * int8BlockSize},
            first,
            elements,
            destination);

    for (auto index = 0; index < whole.size(); ++index)
        check(pieces[index] == whole[index]);
};

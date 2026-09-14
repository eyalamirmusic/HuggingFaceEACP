#include "Int8Blocks.h"

#include <eacp/GPU/GPU.h>

#include <algorithm>
#include <cmath>

namespace HF
{
namespace
{
// Seven bits and a sign, with -128 left out: a symmetric range is what makes
// the dequantized zero exactly zero, and the one value that would break the
// symmetry buys nothing.
constexpr auto largestQuantized = 127.0;

// The scale one block is stored with, as the sixteen bits the buffer holds: the
// largest magnitude in the block over 127, narrowed to fp16.
//
// Zero when the block is all zeroes, and zero too when the narrowing leaves
// nothing to divide by — a block whose values are under fp16's smallest
// subnormal times 127, or past its largest. A scale that is zero, infinite or a
// NaN has no quantization to offer, and zeroes are at least what such a block
// rounds to.
std::uint16_t blockScaleBits(Span<const float> block)
{
    auto largest = 0.f;

    for (auto index = 0; index < block.size(); ++index)
        largest = std::max(largest, std::abs(block[index]));

    const auto bits = eacp::GPU::halfFromFloat(largest / (float) largestQuantized);
    const auto scale = eacp::GPU::halfToFloat(bits);

    return scale > 0.f && std::isfinite(scale) ? bits : (std::uint16_t) 0u;
}

// Round to nearest, ties to even — the standard's default rounding direction,
// which is what std::nearbyint takes from the environment and what nothing here
// changes.
std::uint8_t quantizedByte(float value, float scale)
{
    const auto rounded = std::nearbyint((double) value / (double) scale);
    const auto clamped = std::clamp(rounded, -largestQuantized, largestQuantized);

    return (std::uint8_t) (std::int8_t) clamped;
}

void writeScale(Span<std::uint8_t> destination, std::int64_t at, std::uint16_t bits)
{
    destination[(int) at] = (std::uint8_t) (bits & 0xFFu);
    destination[(int) at + 1] = (std::uint8_t) (bits >> 8);
}

std::uint16_t readScale(Span<const std::uint8_t> stored, std::int64_t at)
{
    return (std::uint16_t) (stored[(int) at]
                            | ((std::uint32_t) stored[(int) at + 1] << 8));
}

// The shader's sign extension written the shader's way rather than as a cast,
// for the reason eacp's own host helpers give: the two sides have to be the
// same arithmetic if a disagreement is to mean anything.
float signedByte(std::uint8_t stored)
{
    return (float) ((int) (stored ^ 0x80u) - 128);
}
} // namespace

std::int64_t int8BlockCount(std::int64_t elementCount)
{
    return elementCount / int8BlockSize;
}

std::int64_t int8BlocksByteCount(std::int64_t elementCount)
{
    return elementCount + 2 * int8BlockCount(elementCount);
}

bool quantizesAsInt8Blocks(std::int64_t contiguousExtent, std::int64_t elementCount)
{
    return contiguousExtent > 0 && elementCount > 0
           && contiguousExtent % int8BlockSize == 0
           && elementCount % (2 * int8BlockSize) == 0;
}

void quantizeInt8Blocks(Span<const float> values, Span<std::uint8_t> destination)
{
    quantizeInt8Blocks(values, 0, values.size(), destination);
}

void quantizeInt8Blocks(Span<const float> values,
                        std::int64_t first,
                        std::int64_t elementCount,
                        Span<std::uint8_t> destination)
{
    for (auto at = std::int64_t {}; at < values.size(); at += int8BlockSize)
    {
        const auto block = Span<const float> {values.data() + at, int8BlockSize};
        const auto bits = blockScaleBits(block);
        const auto scale = eacp::GPU::halfToFloat(bits);
        const auto element = first + at;

        writeScale(destination, elementCount + 2 * (element / int8BlockSize), bits);

        for (auto index = 0; index < int8BlockSize; ++index)
            destination[(int) element + index] =
                bits == 0 ? (std::uint8_t) 0 : quantizedByte(block[index], scale);
    }
}

void dequantizeInt8Blocks(Span<const std::uint8_t> stored, Span<float> values)
{
    const auto elementCount = (std::int64_t) values.size();

    for (auto index = std::int64_t {}; index < elementCount; ++index)
    {
        const auto bits =
            readScale(stored, elementCount + 2 * (index / int8BlockSize));

        values[(int) index] =
            eacp::GPU::halfToFloat(bits) * signedByte(stored[(int) index]);
    }
}
} // namespace HF

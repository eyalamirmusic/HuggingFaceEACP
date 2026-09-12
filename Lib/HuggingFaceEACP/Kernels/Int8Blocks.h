#pragma once

#include <HuggingFaceEACP/Core/Core.h>

#include <cstdint>

namespace HF
{
// The host side of WeightStorage::Int8Blocks: the layout a block-quantized
// weight lies in on the device, and the two functions that write and read it.
//
// Blocks of thirty-two consecutive elements, one scale each, symmetric and with
// no zero point — llama.cpp's Q8_0 shape, taken because it is the
// well-characterised one and because a later comparison against a Q8_0 GGUF
// stays possible. Per element that is one byte plus a sixteenth of a scale:
// 1.0625 bytes against bf16's two, which on a decode step whose cost is how
// many bytes of weight it reads is the whole of the win.
//
// **One buffer, in two regions**, because a kernel that needed two would need a
// second InputBuffer member and every product's member list would change:
//
//   [0, elementCount)             the quantized elements, one signed byte each,
//                                 four to a word, in the order the tensor
//                                 already has them
//   [elementCount, + 2 * blocks)  one fp16 scale per block, two to a word, in
//                                 the same block order
//
// So element i is byte i, its block is i / int8BlockSize, and the scale of
// block b is the half at elementCount / 2 + b — which is the one number the
// shader has to be told, since nothing about a buffer says where the halves
// begin.
//
// **The scale is fp16 rather than fp32**, which is two bytes per thirty-two
// elements rather than four: an fp32 scale is 12.5% more bytes on a step whose
// cost is bytes. bf16 would be the wrong sixteen — an 8-bit mantissa under a
// 7-bit quantized magnitude puts visible error on the scale itself — so fp16 is
// the one of the three that is both small and precise enough to disappear.
//
// **The quantizer divides by the narrowed scale**, not by amax / 127 before it
// is narrowed. That is what makes a CPU reference bit-exact against the shader
// rather than merely close to it: both sides multiply the same fp16 value by
// the same small integer, and a float multiply gives the same answer on every
// backend. llama.cpp's Q8_0 divides by the unnarrowed one and stores the
// narrowed, so its dequantized values are not the ones its quantizer aimed at;
// this way round they are.
//
// The one place the narrowing shows is a block whose scale is an fp16
// subnormal, which is a block whose largest magnitude is under about 7.8e-3.
// There the rounding of the scale is a percent rather than a part in four
// thousand, and the largest element of the block clamps to 127 rather than
// landing on it. A real checkpoint's blocks are nowhere near — Gemma's weights
// are 1e-2 to 1e-1, so its scales are 1e-4 to 1e-3, normal fp16 throughout —
// and a block whose largest magnitude is under fp16's smallest subnormal times
// 127 is stored as zeroes outright, since a scale of zero has no quantization
// to offer.
inline constexpr auto int8BlockSize = 32;

// How many blocks a tensor of this many elements is, and how many bytes the two
// regions above come to.
std::int64_t int8BlockCount(std::int64_t elementCount);
std::int64_t int8BlocksByteCount(std::int64_t elementCount);

// Whether a tensor of this shape can be stored this way at all. The blocks run
// along the contiguous dimension, so that dimension is a whole number of them;
// and both regions are read a word at a time, so the element count is a
// multiple of twice the block — four elements to a word of quants, two scales
// to a word of scales. Every Gemma tensor passes: the contiguous dimension of a
// projection weight is 2048 or 16384, and of the embedding 2048.
bool quantizesAsInt8Blocks(std::int64_t contiguousExtent, std::int64_t elementCount);

// The whole tensor at once, for a caller holding all of it.
void quantizeInt8Blocks(Span<const float> values, Span<std::uint8_t> destination);

// One run of whole blocks of a larger tensor, so a loader can quantize a five
// gigabyte weight a piece at a time and on several threads. `destination` is
// the whole tensor's bytes and `elementCount` how many elements it holds;
// `first` is the element `values` begins at, and it and values.size() are both
// multiples of int8BlockSize.
void quantizeInt8Blocks(Span<const float> values,
                        std::int64_t first,
                        std::int64_t elementCount,
                        Span<std::uint8_t> destination);

// The values the shader reads back out of those bytes, which is what a
// reference asserting against a quantized product runs on. Exactly the shader's
// arithmetic: the byte sign-extended the way every backend spells it, times the
// block's fp16 scale widened.
void dequantizeInt8Blocks(Span<const std::uint8_t> stored, Span<float> values);
} // namespace HF

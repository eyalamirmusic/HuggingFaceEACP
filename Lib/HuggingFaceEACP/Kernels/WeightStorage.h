#pragma once

#include "Int8Blocks.h"
#include "KernelTypes.h"

#include <string_view>

namespace HF
{
// How a checkpoint's weights lie in the buffer a kernel was handed. Nothing
// about a GPU::Buffer says which of the four it is, so the kernel is told at
// compile time and the caller picks the program that matches the buffer it
// loaded.
//
// Every kernel that reads a weight takes this, which is the products and the
// embedding gather — the gather because Gemma ties the embedding to the logits
// projection, so the same buffer is read by both and a gather with only a float
// form would have forced the largest tensor in the model to be widened.
//
// Gemma ships bfloat16, and PackedBFloat16 is the case that keeps it as it
// lies: eacp's readBFloat16 widens it in the shader, which is a shift and a
// bitcast and therefore exact on every backend, where widening on the CPU
// doubled every weight on the device.
//
// Int8Blocks is the only one of the four the checkpoint does not ship. It is
// made at load: thirty-two elements to a block, one fp16 scale each, which is
// Int8Blocks.h's layout and 1.0625 bytes an element against bf16's two. Int4 is
// the fifth case and is the same shape at half the bytes — the same block, the
// same scale, readInt4x16 in place of readInt8x16.
enum class WeightStorage
{
    Float,
    PackedHalf,
    PackedBFloat16,
    Int8Blocks
};

// What a run reports having read its weights through, for an app that prints it
// and a test that asserts on it. The checkpoint's own word where it has one,
// and the format's where the format is ours.
constexpr std::string_view weightStorageName(WeightStorage storage)
{
    switch (storage)
    {
        case WeightStorage::PackedHalf:
            return "fp16";

        case WeightStorage::PackedBFloat16:
            return "bf16";

        case WeightStorage::Int8Blocks:
            return "int8 blocks of 32";

        case WeightStorage::Float:
            break;
    }

    return "fp32";
}

// One weight element, however it is stored: a subscript of a float buffer, the
// widening read a packed format has, or a quantized byte against its block's
// scale. The three widenings are exact, so a packed operand gives the same
// number a widened one does rather than the same number to a tolerance; the
// quantized one is a different number, and it is the same different number on
// the CPU and on the device.
//
// bf16 may not be reached through the fp16 path and the other way round. bf16
// has eight exponent bits against fp16's five, so 1e-6 and 1e30 are ordinary
// bf16 values that fp16 flushes to zero and takes to infinity — a checkpoint
// read through the wrong one is not less precise, it is wrong.
//
// **scaleBase is where this buffer's per-block scales begin**, counted in
// halves from its start, which is its element count over two. Only Int8Blocks
// reads it; the other three take it and ignore it, and the expression a kernel
// hands over is never emitted for them, since an unreferenced node reaches no
// statement. Every product derives it from the shape uniforms it already has —
// a weight is rows × K and both are uniforms — and the gather, which knows its
// width and not its vocabulary, carries it as one uniform of its own.
template <WeightStorage storage>
Float storedWeight(const InputBuffer& weights,
                   const UInt& index,
                   const UInt& scaleBase)
{
    if constexpr (storage == WeightStorage::PackedHalf)
        return weights.readHalf(index);
    else if constexpr (storage == WeightStorage::PackedBFloat16)
        return weights.readBFloat16(index);
    else if constexpr (storage == WeightStorage::Int8Blocks)
        return weights.readInt8(index)
               * weights.readHalf(scaleBase + index / (unsigned) int8BlockSize);
    else
        return weights[index];
}

// Four consecutive weights from a four-aligned index, and the same index
// arithmetic in all four cases: every read counts records of four, so the call
// site never spells how wide an element is. A packed record is the two words
// holding it, which eacp fetches as one eight-byte load on Metal — where two
// two-wide reads were two indexed loads and the unpacking of each separately.
//
// A quantized record is one word of four bytes and the one scale all four share
// — the block is thirty-two elements and the index is four-aligned, so a record
// never straddles two of them. That is two loads per four weights against the
// packed forms' one, and half the bytes.
template <WeightStorage storage>
Float4 storedWeight4(const InputBuffer& weights,
                     const UInt& index,
                     const UInt& scaleBase)
{
    if constexpr (storage == WeightStorage::PackedHalf)
        return weights.readHalf4(index / 4u);
    else if constexpr (storage == WeightStorage::PackedBFloat16)
        return weights.readBFloat16x4(index / 4u);
    else if constexpr (storage == WeightStorage::Int8Blocks)
        return weights.readInt8x4(index / 4u)
               * weights.readHalf(scaleBase + index / (unsigned) int8BlockSize);
    else
        return weights.read4(index / 4u);
}

// Eight and sixteen consecutive weights from an eight- or sixteen-aligned
// index, on storedWeight4's convention one and two widths up: every read still
// counts records, so the call site never spells how wide an element is. They
// hand back eacp's own Float4Pair and Float4Quad rather than an aggregate of
// ours, because that is what the reads underneath return and wrapping them
// would be a copy and a second name for one thing — .low and .high are bytes
// 0..3 and 4..7, and .a through .d are the four Float4s in address order, so
// .a.x is the record's first weight and .d.w its sixteenth.
//
// **This is what a quantized decode step is for.** readInt8x4 fetches one
// four-byte word where readBFloat16x4 fetches an eight-byte one for the same
// four weights, so the two walks issue the same number of loads and the
// quantized one carries half the data in each — which is what held int8 to
// 1.32x the bf16 decode rate where its bytes predict 1.88x. readInt8x8 is one
// eight-byte load for eight weights and readInt8x16 one sixteen-byte load for
// sixteen, so a quantized walk issues a quarter of the loads rather than the
// same number.
//
// The block's scale comes with them once. A run of sixteen weights from a
// sixteen-aligned index lies inside one block of thirty-two, and eight from an
// eight-aligned index likewise, so a record costs exactly one readHalf where
// storedWeight4 costs one per four.
//
// The other three storages have no wider load to reach for — eight bytes is
// already the whole of a packed record of four, sixteen of a float one — so
// they are two or four calls of what storedWeight4 makes one. That is the same
// loads they issued before at the same widths, so nothing regresses by walking
// wide records in a kernel whose weights are not quantized.
template <WeightStorage storage>
Float4Pair storedWeight8(const InputBuffer& weights,
                         const UInt& index,
                         const UInt& scaleBase)
{
    if constexpr (storage == WeightStorage::Int8Blocks)
    {
        auto quantized = weights.readInt8x8(index / 8u);
        auto scale = weights.readHalf(scaleBase + index / (unsigned) int8BlockSize);

        return {quantized.low * scale, quantized.high * scale};
    }
    else
    {
        return {storedWeight4<storage>(weights, index, scaleBase),
                storedWeight4<storage>(weights, index + 4u, scaleBase)};
    }
}

template <WeightStorage storage>
Float4Quad storedWeight16(const InputBuffer& weights,
                          const UInt& index,
                          const UInt& scaleBase)
{
    if constexpr (storage == WeightStorage::Int8Blocks)
    {
        auto quantized = weights.readInt8x16(index / 16u);
        auto scale = weights.readHalf(scaleBase + index / (unsigned) int8BlockSize);

        return {quantized.a * scale,
                quantized.b * scale,
                quantized.c * scale,
                quantized.d * scale};
    }
    else
    {
        return {storedWeight4<storage>(weights, index, scaleBase),
                storedWeight4<storage>(weights, index + 4u, scaleBase),
                storedWeight4<storage>(weights, index + 8u, scaleBase),
                storedWeight4<storage>(weights, index + 12u, scaleBase)};
    }
}
} // namespace HF

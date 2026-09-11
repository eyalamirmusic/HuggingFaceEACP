#pragma once

#include "KernelTypes.h"

namespace HF
{
// How a checkpoint's weights lie in the buffer a kernel was handed. Nothing
// about a GPU::Buffer says which of the three it is, so the kernel is told at
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
enum class WeightStorage
{
    Float,
    PackedHalf,
    PackedBFloat16
};

// One weight element, however the checkpoint stored it: a subscript of a float
// buffer, or the widening read the packed format has. Both widenings are
// exact, so a packed operand gives the same number a widened one does rather
// than the same number to a tolerance.
//
// bf16 may not be reached through the fp16 path and the other way round. bf16
// has eight exponent bits against fp16's five, so 1e-6 and 1e30 are ordinary
// bf16 values that fp16 flushes to zero and takes to infinity — a checkpoint
// read through the wrong one is not less precise, it is wrong.
template <WeightStorage storage>
Float storedWeight(const InputBuffer& weights, const UInt& index)
{
    if constexpr (storage == WeightStorage::PackedHalf)
        return weights.readHalf(index);
    else if constexpr (storage == WeightStorage::PackedBFloat16)
        return weights.readBFloat16(index);
    else
        return weights[index];
}

// Four consecutive weights from a four-aligned index, and the same index
// arithmetic in all three cases: every read counts records of four, so the call
// site never spells how wide an element is. A packed record is the two words
// holding it, which eacp fetches as one eight-byte load on Metal — where two
// two-wide reads were two indexed loads and the unpacking of each separately.
template <WeightStorage storage>
Float4 storedWeight4(const InputBuffer& weights, const UInt& index)
{
    if constexpr (storage == WeightStorage::PackedHalf)
        return weights.readHalf4(index / 4u);
    else if constexpr (storage == WeightStorage::PackedBFloat16)
        return weights.readBFloat16x4(index / 4u);
    else
        return weights.read4(index / 4u);
}
} // namespace HF

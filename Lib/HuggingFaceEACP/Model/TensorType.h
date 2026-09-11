#pragma once

#include <optional>
#include <string_view>

namespace HF
{
// The dtype tags safetensors names in its header. BF16 is what
// google/gemma-2b ships; F32 and F16 are what a converted or a smaller repo
// uses. The integer tags are here so an unknown one is an error rather than a
// silent misread of the blob.
enum class TensorType
{
    Bool,
    U8,
    I8,
    U16,
    I16,
    F16,
    BF16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64
};

int bytesPerElement(TensorType type);
bool isFloatingPoint(TensorType type);
std::string_view tensorTypeName(TensorType type);

// Whether the type is one of the two sixteen-bit floats a kernel reads two to a
// word — F16 through eacp's readHalf, BF16 through readBFloat16. Those are the
// tensors that go to the device still packed, and the word is what makes the
// packing a question at all: a read fetches the word an index lands in, so a
// packed buffer is a whole number of them and an odd element count is padded.
//
// F32 is deliberately not in here. It reaches the device untouched too, but as
// the floats a subscript reads, and every caller that asks this question has
// already answered the F32 one.
bool isPackedSixteenBit(TensorType type);

// Empty for a tag this build does not know, which is the whole reason it
// returns an optional rather than a default.
std::optional<TensorType> findTensorType(std::string_view name);
} // namespace HF

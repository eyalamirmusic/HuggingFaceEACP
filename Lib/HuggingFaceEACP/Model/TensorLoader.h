#pragma once

#include <HuggingFaceEACP/Model/ShardedTensors.h>

#include <initializer_list>
#include <string>
#include <string_view>

namespace HF
{
// The shape a caller was compiled for, outermost axis first, as the config
// gives it rather than as the file claims it.
using TensorShape = std::initializer_list<int>;

// The checks a weight struct runs on the way from a checkpoint to the
// TensorBuffer it hands a kernel: the checkpoint carries the tensor, the
// tensor has the shape the config implies, and its storage is one the program
// bound to it can read. Each failure is a ModelError naming the tensor — the
// alternative is a kernel walking a buffer at a stride it does not have, which
// is silent on both backends.
//
// `component` is the word those messages use for the caller — "decoder",
// "layer 3" — so a refusal says which part of the model would not take the
// checkpoint rather than leaving that to a stack trace.
//
// **Packed storage.** A projection weight may stay packed as fp16: the product
// kernels have a half-reading variant and the caller picks it by
// TensorBuffer::storage. Every other tensor is bound to a program with no half
// read, so a packed one there would be wrong by a factor of two in every index
// while staying silent, and loadFloatTensor refuses it. Gemma's own weights
// are BF16, which has no shader read at all yet and therefore arrives widened.
struct TensorLoader
{
    const ShardedTensors& file;
    std::string_view component;

    const TensorInfo& require(const std::string& name) const;

    void checkShape(const TensorInfo& tensor, Span<const int> expected) const;
    void checkShape(const TensorInfo& tensor, TensorShape expected) const;

    void rejectPackedHalf(const TensorBuffer& loaded,
                          const std::string& name,
                          std::string_view reader) const;

    // Bound to a program that subscripts a float buffer, so a packed one would
    // be read at half the stride it was written at — silently, on both
    // backends. `reader` is the kernel the refusal names.
    TensorBuffer loadFloatTensor(const std::string& name,
                                 TensorShape expected,
                                 std::string_view reader) const;

    // A projection weight, which is the one operand that may stay packed.
    TensorBuffer loadProjectionWeight(const std::string& name,
                                      TensorShape expected) const;
};
} // namespace HF

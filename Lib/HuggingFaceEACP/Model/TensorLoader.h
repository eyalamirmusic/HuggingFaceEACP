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
// **Packed storage.** A weight a product or the embedding gather reads may stay
// packed as it lies in the checkpoint — fp16 or Gemma's own bf16 — because
// those kernels have a variant per storage and the caller picks it by
// TensorBuffer::storage. The norm scales are the exception: RMSNorm subscripts
// floats and has no packed read, so a packed one there would be read at half
// the stride it was written at, silently, on both backends. loadFloatTensor
// widens those on the way up, which is kilobytes for a row of a weight rather
// than the copy of a matrix the packed path exists to avoid.
struct TensorLoader
{
    const ShardedTensors& file;
    std::string_view component;

    const TensorInfo& require(const std::string& name) const;

    void checkShape(const TensorInfo& tensor, Span<const int> expected) const;
    void checkShape(const TensorInfo& tensor, TensorShape expected) const;

    // Widened to F32 whatever the checkpoint holds, for the tensors bound to a
    // program that subscripts a float buffer.
    TensorBuffer loadFloatTensor(const std::string& name,
                                 TensorShape expected) const;

    // A weight, which reaches the device in the storage the checkpoint shipped
    // it in.
    TensorBuffer loadWeight(const std::string& name, TensorShape expected) const;
};
} // namespace HF

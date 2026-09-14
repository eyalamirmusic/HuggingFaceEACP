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

// What a product weight goes to the device as. AsShipped is the checkpoint's
// own — F32, fp16 or bf16, the bytes as they lie — and is what everything here
// did before there was a choice.
//
// Int8Blocks quantizes on the way up, into the layout Kernels/Int8Blocks.h
// describes: thirty-two elements to a block, one fp16 scale each, 1.0625 bytes
// an element against bf16's two. It costs a pass over every weight at load and
// a little accuracy per element, and it buys the only thing a decode step is
// short of, which is bytes.
//
// Only the tensors a product or the gather reads take it. The norm scales are
// widened to F32 whatever this says, for the reason DecoderWeights gives.
enum class WeightPrecision
{
    AsShipped,
    Int8Blocks
};

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
// **Packed storage.** A weight a product kernel reads may stay as the
// checkpoint ships it — F32, fp16 or bf16 — since every product has a variant
// per storage and the caller picks it by TensorBuffer::storage. Everything else
// is bound to a program that only subscripts floats, and loadFloatTensor is
// what widens those on the way up: a packed buffer there would be read at half
// the stride it was written at, silently, on both backends.
struct TensorLoader
{
    const ShardedTensors& file;
    std::string_view component;

    const TensorInfo& require(const std::string& name) const;

    void checkShape(const TensorInfo& tensor, Span<const int> expected) const;
    void checkShape(const TensorInfo& tensor, TensorShape expected) const;

    // Widened to F32 whatever the checkpoint stores it as, for the programs
    // that subscript a float buffer.
    TensorBuffer loadFloatTensor(const std::string& name,
                                 TensorShape expected) const;

    // A weight a product kernel reads, which is the one that may stay packed —
    // or be quantized, when the caller asks for Int8Blocks. A tensor whose
    // contiguous dimension is not a whole number of blocks, or whose element
    // count is not a whole number of scale words, is a ModelError naming it and
    // both numbers rather than a silently different layout; every Gemma tensor
    // passes.
    TensorBuffer loadProjectionWeight(
        const std::string& name,
        TensorShape expected,
        WeightPrecision precision = WeightPrecision::AsShipped) const;

    // Two weights of one shape as a single buffer, the first's rows then the
    // second's — the concatenation Gemma's fused gate-and-up weight is. Both
    // are checked at their shipped shapes before a byte is read, so a
    // checkpoint whose halves disagree is a ModelError naming the tensor rather
    // than a stack of two matrices of different widths.
    //
    // **The bytes are stacked, not the values.** A packed pair goes up still
    // packed, at the storage both halves share, so a bf16 or fp16 repo pays one
    // copy of its largest weight rather than a widened one — which is what the
    // "no raw-bytes concatenation" item in plan.md asked for. What stops it is
    // an odd element count in either half, since a packed read fetches whole
    // words; that falls back to widening both, as does a pair whose halves are
    // stored differently or in a type no kernel reads.
    //
    // The stack is the one buffer here that is twice a tensor's size, so it is
    // also where a GPU buffer's int-sized byte count is reachable — plan.md's
    // second gap. It is counted in 64 bits and refused by name rather than
    // truncated.
    //
    // **Quantized, the stack is one tensor rather than two.** The halves are
    // read, widened and quantized into the one buffer in order, so a block is
    // thirty-two elements of whichever half it fell in — which is the same
    // blocks quantizing them separately would have made, since each half's
    // element count is a whole number of them.
    TensorBuffer loadStackedProjectionWeights(
        const std::string& first,
        const std::string& second,
        TensorShape expected,
        WeightPrecision precision = WeightPrecision::AsShipped) const;
};
} // namespace HF

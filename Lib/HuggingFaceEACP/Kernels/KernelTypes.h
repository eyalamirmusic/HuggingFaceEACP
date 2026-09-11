#pragma once

#include <HuggingFaceEACP/Core/Core.h>

#include <eacp/GPU/GPU.h>

namespace HF
{
// The EDSL vocabulary every kernel in this directory is declared in, brought in
// by name the way Core/Common.h brings in ea_data_structures'. The intrinsics
// and operators need no import: their arguments are eacp::GPU types, so ADL
// finds them.
using eacp::GPU::AtomicBuffer;
using eacp::GPU::ComputePass;
using eacp::GPU::ComputeProgram;
using eacp::GPU::Float;
using eacp::GPU::Float4;
using eacp::GPU::InputBuffer;
using eacp::GPU::OutputBuffer;
using eacp::GPU::Shared;
using eacp::GPU::UInt;
using eacp::GPU::UIntInputBuffer;
using eacp::GPU::UIntOutputBuffer;
using eacp::GPU::Uniform;
using eacp::GPU::Var;

// What a weight operand's buffer holds. Nothing about a GPU::Buffer says which
// of the three it is, so the kernel is told at compile time and the caller
// picks the program that matches the buffer it loaded.
//
// The two packed cases are the two 16-bit floats, and they are not
// interchangeable: bfloat16 has eight exponent bits where fp16 has five, so a
// bf16 checkpoint read through readHalf is not less precise, it is wrong.
// Gemma ships bfloat16 and reaches the device in it — widening a bf16 is a
// shift and a bitcast, which is exact, so a packed product answers with the
// bits the float one does over the same weights at half the traffic.
enum class WeightStorage
{
    Float,
    PackedHalf,
    PackedBFloat16
};
} // namespace HF

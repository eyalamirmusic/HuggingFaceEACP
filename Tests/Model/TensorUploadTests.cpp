#include "Common.h"

#include <HuggingFaceEACP/Model/TensorLoader.h>

#include <eacp/GPU/GPU.h>

#include <cstring>
#include <string>

// The one part of this module that reaches the device, and the reason it has
// to: what a tensor is uploaded *as* is the claim under test. A BF16 weight now
// goes up as the bytes the blob holds, which is a buffer half the size a
// widened one was and one no float subscript can read — and neither half of
// that is visible from the CPU side alone. Every test here returns early
// without a device, the way a GPU suite's does.

using namespace nano;
using namespace HF;
using namespace HF::Testing;
using namespace eacp::GPU;

namespace
{
// Narrowed through eacp's own host converters, which are the encodings
// InputBuffer::readBFloat16 and readHalf widen: what the blob holds and what a
// shader would read are then the same numbers rather than nearly the same ones.
template <typename Narrow>
Vector<std::uint8_t> packedBlob(const Vector<float>& values, Narrow narrow)
{
    auto bytes = Vector<std::uint8_t> {};

    for (auto value: values)
    {
        const auto bits = narrow(value);

        bytes.add((std::uint8_t) (bits & 0xFFu));
        bytes.add((std::uint8_t) (bits >> 8));
    }

    return bytes;
}

Vector<std::uint8_t> floatBlob(const Vector<float>& values)
{
    auto bytes = Vector<std::uint8_t> {};
    bytes.resize((int) sizeof(float) * values.size());
    std::memcpy(
        bytes.data(), values.data(), sizeof(float) * (std::size_t) values.size());

    return bytes;
}

Vector<std::uint8_t> doubleBlob(const Vector<float>& values)
{
    auto bytes = Vector<std::uint8_t> {};
    bytes.resize((int) sizeof(double) * values.size());

    for (auto index = 0; index < values.size(); ++index)
    {
        const auto wide = (double) values[index];
        std::memcpy(
            bytes.data() + index * (int) sizeof(double), &wide, sizeof(wide));
    }

    return bytes;
}

Vector<std::uint8_t> blobOf(std::string_view dtype, const Vector<float>& values)
{
    if (dtype == "BF16")
        return packedBlob(values, bfloat16FromFloat);

    if (dtype == "F16")
        return packedBlob(values, halfFromFloat);

    if (dtype == "F64")
        return doubleBlob(values);

    return floatBlob(values);
}

// Values both sixteen-bit mantissas hold exactly, so the assertions below are
// equalities and a narrowing that lost a bit is a wrong number rather than a
// tolerance.
Vector<float> exactValues(int count)
{
    auto values = Vector<float> {};
    values.resize(count);

    for (auto index = 0; index < count; ++index)
        values[index] = 0.25f * (float) (index + 1 - count / 2);

    return values;
}

// A second set, disjoint from the first and still inside bf16's seven mantissa
// bits, so a stack that put the two halves the wrong way round is a wrong
// number rather than the same number twice.
Vector<float> otherExactValues(int count)
{
    auto values = Vector<float> {};
    values.resize(count);

    for (auto index = 0; index < count; ++index)
        values[index] = 8.f + 0.25f * (float) index;

    return values;
}

Vector<float> readBackFloats(const Buffer& buffer, int count)
{
    auto values = Vector<float> {};
    values.resize(count);
    buffer.read(values.data(), (int) sizeof(float) * count);

    return values;
}

// A packed buffer read back as the values a shader widens it to: its elements
// are words holding two, the low half first.
template <typename Widen>
Vector<float> readBackPacked(const Buffer& buffer, int count, Widen widen)
{
    const auto words = readBackFloats(buffer, (count + 1) / 2);

    auto values = Vector<float> {};
    values.resize(count);

    for (auto index = 0; index < count; ++index)
    {
        auto word = std::uint32_t {};
        std::memcpy(&word, &words[index / 2], sizeof(word));

        const auto half = index % 2 == 0 ? word & 0xFFFFu : word >> 16;
        values[index] = widen((std::uint16_t) half);
    }

    return values;
}

std::string tensorEntry(std::string_view name,
                        std::string_view dtype,
                        int count,
                        std::uint64_t begin,
                        std::uint64_t end)
{
    return "\"" + std::string {name} + "\":{\"dtype\":\"" + std::string {dtype}
           + "\",\"shape\":[" + std::to_string(count) + "],\"data_offsets\":["
           + std::to_string(begin) + "," + std::to_string(end) + "]}";
}

Vector<std::uint8_t> oneTensorFile(std::string_view dtype,
                                   const Vector<float>& values)
{
    const auto blob = blobOf(dtype, values);
    const auto header =
        "{" + tensorEntry("w", dtype, values.size(), 0, (std::uint64_t) blob.size())
        + "}";

    return assemble(header, blob);
}

// Two tensors of one shape under the names the stacking takes, "gate" first,
// which is also the order an ordered header reads them in. The two dtypes are
// separate because a pair whose halves disagree is one of the shapes the byte
// stacking has to refuse.
Vector<std::uint8_t> pairFile(std::string_view firstDtype,
                              const Vector<float>& first,
                              std::string_view secondDtype,
                              const Vector<float>& second)
{
    auto blob = blobOf(firstDtype, first);
    const auto half = (std::uint64_t) blob.size();

    for (auto byte: blobOf(secondDtype, second))
        blob.add(byte);

    const auto header =
        "{" + tensorEntry("gate", firstDtype, first.size(), 0, half) + ","
        + tensorEntry(
            "up", secondDtype, second.size(), half, (std::uint64_t) blob.size())
        + "}";

    return assemble(header, blob);
}

TensorBuffer stackPair(const ScratchDirectory& scratch,
                       std::string_view firstDtype,
                       const Vector<float>& first,
                       std::string_view secondDtype,
                       const Vector<float>& second)
{
    const auto path = scratch.write(
        ModelFileNames::weights, pairFile(firstDtype, first, secondDtype, second));

    const auto file = ShardedTensors::fromFile(path);

    return TensorLoader {file, "test"}.loadStackedProjectionWeights(
        "gate", "up", {first.size()});
}

TensorBuffer stackPair(const ScratchDirectory& scratch,
                       std::string_view dtype,
                       const Vector<float>& first,
                       const Vector<float>& second)
{
    return stackPair(scratch, dtype, first, dtype, second);
}
} // namespace

// The whole point of the packed path: a BF16 tensor reaches the device as the
// bytes the blob holds, at two bytes an element and tagged so the caller binds
// it to a program that widens on read. Gemma is BF16 throughout, so this is
// what halves the 10 GB the widened path uploaded.
auto tBFloat16UploadsPacked = test("Model/Upload/bfloat16UploadsPacked") = []
{
    if (!Device::shared().isValid())
        return;

    const auto values = exactValues(8);
    const auto file = SafeTensors::fromBytes(oneTensorFile("BF16", values));
    const auto uploaded = file.makeBuffer("w");

    check(uploaded.storage == TensorType::BF16);
    check(uploaded.isPackedBFloat16());
    check(!uploaded.isPackedHalf());
    check(uploaded.buffer.size() == 2 * values.size());

    const auto read =
        readBackPacked(uploaded.buffer, values.size(), bfloat16ToFloat);

    for (auto index = 0; index < values.size(); ++index)
        check(read[index] == values[index]);
};

// An odd element count, where the last element sits in the low half of a word
// whose high half nothing wrote: the buffer is padded to whole words, since
// readBFloat16 fetches the word an index lands in and a read one element past
// the allocation is what the alternative would be.
auto tOddBFloat16CountIsPadded = test("Model/Upload/oddBFloat16CountIsPadded") = []
{
    if (!Device::shared().isValid())
        return;

    const auto values = exactValues(7);
    const auto file = SafeTensors::fromBytes(oneTensorFile("BF16", values));
    const auto uploaded = file.makeBuffer("w");

    check(uploaded.isPackedBFloat16());
    check(uploaded.buffer.size() == 16);

    const auto read =
        readBackPacked(uploaded.buffer, values.size(), bfloat16ToFloat);

    for (auto index = 0; index < values.size(); ++index)
        check(read[index] == values[index]);
};

// The other upload, for the tensors bound to a program that only subscripts
// floats — Gemma's norm scales. Same tensor, four bytes an element, and the
// values the widening gives, which for bf16 is exact.
auto tFloatBufferWidensBFloat16 = test("Model/Upload/floatBufferWidensBFloat16") = []
{
    if (!Device::shared().isValid())
        return;

    const auto values = exactValues(8);
    const auto file = SafeTensors::fromBytes(oneTensorFile("BF16", values));
    const auto widened = file.makeFloatBuffer("w");

    check(widened.storage == TensorType::F32);
    check(!widened.isPackedBFloat16());
    check(widened.buffer.size() == 4 * values.size());

    const auto read = readBackFloats(widened.buffer, values.size());

    for (auto index = 0; index < values.size(); ++index)
        check(read[index] == values[index]);
};

// The concatenation Gemma's fused gate-and-up weight is, with the bytes left
// packed: one buffer of twice one half's bytes, still BF16, holding the gate
// rows and then the up rows. Widening through readFloats and stacking that is
// what this used to do, and it cost a doubled copy of the largest weight in
// every layer.
auto tStacksBFloat16WeightsPacked =
    test("Model/Upload/stacksBFloat16WeightsPacked") = []
{
    if (!Device::shared().isValid())
        return;

    const auto scratch = ScratchDirectory {"stack-bf16"};
    const auto gate = exactValues(6);
    const auto up = otherExactValues(6);

    const auto stacked = stackPair(scratch, "BF16", gate, up);

    check(stacked.isPackedBFloat16());
    check(stacked.buffer.size() == 2 * 2 * gate.size());

    const auto values =
        readBackPacked(stacked.buffer, 2 * gate.size(), bfloat16ToFloat);

    for (auto index = 0; index < gate.size(); ++index)
    {
        check(values[index] == gate[index]);
        check(values[gate.size() + index] == up[index]);
    }
};

// An fp16 repo stacks packed too, which is the case plan.md's "no raw-bytes
// concatenation" item was actually about: Gemma's own checkpoint was widened
// either way before the bf16 read existed, and an fp16 one paid for it with
// nothing to show.
auto tStacksHalfWeightsPacked = test("Model/Upload/stacksHalfWeightsPacked") = []
{
    if (!Device::shared().isValid())
        return;

    const auto scratch = ScratchDirectory {"stack-f16"};
    const auto gate = exactValues(6);
    const auto up = otherExactValues(6);

    const auto stacked = stackPair(scratch, "F16", gate, up);

    check(stacked.isPackedHalf());
    check(!stacked.isPackedBFloat16());
    check(stacked.buffer.size() == 2 * 2 * gate.size());

    const auto values = readBackPacked(stacked.buffer, 2 * gate.size(), halfToFloat);

    for (auto index = 0; index < gate.size(); ++index)
    {
        check(values[index] == gate[index]);
        check(values[gate.size() + index] == up[index]);
    }
};

// F32 stacks as bytes as well, which it always could — the same code path, and
// the one that says the packed cases are a storage question rather than a
// second implementation.
auto tStacksFloatWeights = test("Model/Upload/stacksFloatWeights") = []
{
    if (!Device::shared().isValid())
        return;

    const auto scratch = ScratchDirectory {"stack-f32"};
    const auto gate = exactValues(5);
    const auto up = otherExactValues(5);

    const auto stacked = stackPair(scratch, "F32", gate, up);

    check(stacked.storage == TensorType::F32);
    check(stacked.buffer.size() == 4 * 2 * gate.size());

    const auto values = readBackFloats(stacked.buffer, 2 * gate.size());

    for (auto index = 0; index < gate.size(); ++index)
    {
        check(values[index] == gate[index]);
        check(values[gate.size() + index] == up[index]);
    }
};

// An odd element count in the first half is the one shape a packed stack cannot
// take: the second half's elements would straddle the words a packed read
// fetches, so both halves are widened instead and the buffer says F32. Gemma's
// own [16384, 2048] halves are even by a factor of thirty-two million, so this
// is the guard rather than the case.
auto tOddPackedHalfFallsBackToFloats =
    test("Model/Upload/oddPackedHalfFallsBackToFloats") = []
{
    if (!Device::shared().isValid())
        return;

    const auto scratch = ScratchDirectory {"stack-bf16-odd"};
    const auto gate = exactValues(5);
    const auto up = otherExactValues(5);

    const auto stacked = stackPair(scratch, "BF16", gate, up);

    check(stacked.storage == TensorType::F32);
    check(stacked.buffer.size() == 4 * 2 * gate.size());

    const auto values = readBackFloats(stacked.buffer, 2 * gate.size());

    for (auto index = 0; index < gate.size(); ++index)
    {
        check(values[index] == gate[index]);
        check(values[gate.size() + index] == up[index]);
    }
};

// Halves stored differently, which the byte stacking cannot take for the plain
// reason that there is no one storage to report: the buffer would be bf16 for
// its first half and F32 for its second, and a kernel reads a buffer one way.
// Both are widened instead, and the result is the stride an F32 stack has.
auto tMixedTypesFallBackToFloats =
    test("Model/Upload/mixedTypesFallBackToFloats") = []
{
    if (!Device::shared().isValid())
        return;

    const auto scratch = ScratchDirectory {"stack-mixed"};
    const auto gate = exactValues(6);
    const auto up = otherExactValues(6);

    const auto stacked = stackPair(scratch, "F32", gate, "BF16", up);

    check(stacked.storage == TensorType::F32);
    check(!stacked.isPackedBFloat16());
    check(stacked.buffer.size() == 4 * 2 * gate.size());

    const auto values = readBackFloats(stacked.buffer, 2 * gate.size());

    for (auto index = 0; index < gate.size(); ++index)
    {
        check(values[index] == gate[index]);
        check(values[gate.size() + index] == up[index]);
    }
};

// A type no kernel reads at all. F64 is the one a converted checkpoint
// plausibly carries, and there is no shader subscript for it, so the pair is
// widened the way a single F64 tensor is — eight bytes an element on disk, four
// in the buffer.
auto tDoublesFallBackToFloats = test("Model/Upload/doublesFallBackToFloats") = []
{
    if (!Device::shared().isValid())
        return;

    const auto scratch = ScratchDirectory {"stack-f64"};
    const auto gate = exactValues(6);
    const auto up = otherExactValues(6);

    const auto stacked = stackPair(scratch, "F64", gate, up);

    check(stacked.storage == TensorType::F32);
    check(stacked.buffer.size() == 4 * 2 * gate.size());

    const auto values = readBackFloats(stacked.buffer, 2 * gate.size());

    for (auto index = 0; index < gate.size(); ++index)
    {
        check(values[index] == gate[index]);
        check(values[gate.size() + index] == up[index]);
    }
};

#include "Common.h"

#include <eacp/GPU/GPU.h>

#include <bit>
#include <cmath>
#include <cstdint>

using namespace nano;
using namespace HF;
using namespace HF::Testing;

namespace
{
const auto singleTensorHeader = std::string {
    R"({"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}})"};

Vector<std::uint8_t> singleTensorFile()
{
    return assemble(singleTensorHeader,
                    toBytes<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

SafeTensors parseHeaderOnly(const std::string& header)
{
    return SafeTensors::fromBytes(assemble(header));
}

// bfloat16 is the top sixteen bits of a float32 and nothing else, so this is
// the whole of the reference the widening is checked against — not an
// approximation of it, and not our own implementation restated.
float bfloatReference(std::uint16_t bits)
{
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

// Zero, negative zero, the smallest subnormal, the largest subnormal, one, a
// few ordinary magnitudes, the largest finite value, both infinities, a quiet
// NaN and a signalling one. Every class bfloat16 has.
const auto bfloatSweep = std::initializer_list<std::uint16_t> {
    0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x4049, 0x3EAA,
    0xC2F0, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x0002, 0x00FF, 0x4780};
} // namespace

auto tSingleTensor = test("Model/SafeTensors/singleTensor") = []
{
    const auto file = SafeTensors::fromBytes(singleTensorFile());

    check(file.tensors().size() == 1);
    check(file.contains("weight"));
    check(!file.contains("bias"));

    const auto& weight = file.info("weight");
    check(weight.name == "weight");
    check(weight.type == TensorType::F32);
    check(weight.rank() == 2);
    check(weight.dimension(0) == 2);
    check(weight.dimension(1) == 3);
    check(weight.elementCount() == 6);
    check(weight.byteCount == 24);
    check(file.rawBytes(weight).size() == 24);
};

// F32 is the one type that reaches the CPU as the very bytes the file holds,
// so the round trip is bit-exact rather than merely close.
auto tFloatRoundTrips = test("Model/SafeTensors/floatRoundTrips") = []
{
    const auto written = std::initializer_list<float> {
        0.0f, -0.0f, 1.0f, -2.5f, 3.14159265f, 1.0e-38f, 3.4028235e38f};

    const auto file = SafeTensors::fromBytes(
        assemble(R"({"w":{"dtype":"F32","shape":[7],"data_offsets":[0,28]}})",
                 toBytes<float>(written)));

    const auto values = file.readFloats("w");
    check(values.size() == 7);

    auto index = 0;

    for (auto expected: written)
        check(std::bit_cast<std::uint32_t>(values[index++])
              == std::bit_cast<std::uint32_t>(expected));
};

// The CPU widening, which is what the references a kernel is checked against
// read a checkpoint through — the device gets the packed bytes instead, see
// bfloat16UploadsPacked below. Compared by bits, not by value, so a NaN
// pattern is checked as exactly as a finite one.
auto tBfloatWidensExactly = test("Model/SafeTensors/bfloatWidensExactly") = []
{
    const auto bits = toBytes<std::uint16_t>(bfloatSweep);
    const auto count = static_cast<int>(bfloatSweep.size());

    const auto header = R"({"w":{"dtype":"BF16","shape":[)" + std::to_string(count)
                        + R"(],"data_offsets":[0,)" + std::to_string(count * 2)
                        + R"(]}})";

    const auto file = SafeTensors::fromBytes(assemble(header, bits));
    const auto values = file.readFloats("w");

    check(values.size() == count);

    auto index = 0;

    for (auto pattern: bfloatSweep)
        check(std::bit_cast<std::uint32_t>(values[index++])
              == std::bit_cast<std::uint32_t>(bfloatReference(pattern)));
};

// The upload side of the same tensor, and the whole of what reading bf16
// packed asks of the loader: the blob's bytes reach the device unchanged, and
// the buffer says which packing they are, since nothing about a GPU buffer
// does. A word holds two bfloat16s little-endian in the file and the same two
// in the buffer, which is the layout InputBuffer::readBFloat16 indexes — so
// what is asserted here is that no byte moved.
auto tBFloat16UploadsPacked = test("Model/SafeTensors/bfloat16UploadsPacked") = []
{
    if (!eacp::GPU::Device::shared().isValid())
        return;

    const auto bits = toBytes<std::uint16_t>(bfloatSweep);
    const auto count = static_cast<int>(bfloatSweep.size());

    const auto header = R"({"w":{"dtype":"BF16","shape":[)" + std::to_string(count)
                        + R"(],"data_offsets":[0,)" + std::to_string(count * 2)
                        + R"(]}})";

    const auto file = SafeTensors::fromBytes(assemble(header, bits));
    const auto loaded = file.makeBuffer("w");

    check(loaded.storage == TensorType::BF16);
    check(loaded.isPackedBFloat16());
    check(!loaded.isPackedHalf());
    check(loaded.buffer.size() == ((count + 1) / 2) * 4);

    auto uploaded = Vector<std::uint8_t> {};
    uploaded.resize(loaded.buffer.size());
    loaded.buffer.read(uploaded.data(), uploaded.size());

    for (auto index = 0; index < bits.size(); ++index)
        check(uploaded[index] == bits[index]);

    // The widened buffer is still there for the one reader that has no packed
    // read — the norm scales — and it is F32 at twice the bytes.
    const auto widened = file.makeWidenedBuffer("w");

    check(widened.storage == TensorType::F32);
    check(!widened.isPackedBFloat16());
    check(widened.buffer.size() == count * (int) sizeof(float));
};

// An odd count of bfloat16s, which is a half-empty last word: readBFloat16
// fetches word i / 2 whichever half it wants, so the buffer is padded to a
// whole word rather than read one element past its end.
auto tOddBFloat16CountIsPadded =
    test("Model/SafeTensors/oddBFloat16CountIsPadded") = []
{
    if (!eacp::GPU::Device::shared().isValid())
        return;

    const auto bits = toBytes<std::uint16_t>({0x3F80, 0xC000, 0x4049});
    const auto file = SafeTensors::fromBytes(assemble(
        R"({"w":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}})", bits));

    const auto loaded = file.makeBuffer("w");

    check(loaded.isPackedBFloat16());
    check(loaded.buffer.size() == 8);

    auto uploaded = Vector<std::uint8_t> {};
    uploaded.resize(loaded.buffer.size());
    loaded.buffer.read(uploaded.data(), uploaded.size());

    for (auto index = 0; index < bits.size(); ++index)
        check(uploaded[index] == bits[index]);
};

// The same sweep read through the value the standard defines, so a widening
// that agreed with the shift and disagreed with IEEE 754 would still fail.
auto tBfloatClasses = test("Model/SafeTensors/bfloatClasses") = []
{
    const auto bits = toBytes<std::uint16_t>(
        {0x0000, 0x8000, 0x3F80, 0xC000, 0x4049, 0x7F80, 0xFF80, 0x7FC0, 0x0001});

    const auto file = SafeTensors::fromBytes(assemble(
        R"({"w":{"dtype":"BF16","shape":[9],"data_offsets":[0,18]}})", bits));

    const auto values = file.readFloats("w");

    check(values[0] == 0.0f && !std::signbit(values[0]));
    check(values[1] == 0.0f && std::signbit(values[1]));
    check(values[2] == 1.0f);
    check(values[3] == -2.0f);
    check(nearlyEqual(values[4], 3.140625f, 1.0e-7f));
    check(std::isinf(values[5]) && values[5] > 0.0f);
    check(std::isinf(values[6]) && values[6] < 0.0f);
    check(std::isnan(values[7]));
    check(values[8] > 0.0f && values[8] < 1.0e-38f);
};

auto tHalfWidensToFloat = test("Model/SafeTensors/halfWidensToFloat") = []
{
    const auto bits = toBytes<std::uint16_t>(
        {0x0000, 0x8000, 0x3C00, 0xC000, 0x3555, 0x0001, 0x03FF, 0x7BFF});

    const auto file = SafeTensors::fromBytes(assemble(
        R"({"w":{"dtype":"F16","shape":[8],"data_offsets":[0,16]}})", bits));

    const auto values = file.readFloats("w");

    check(values[0] == 0.0f);
    check(values[1] == 0.0f && std::signbit(values[1]));
    check(values[2] == 1.0f);
    check(values[3] == -2.0f);
    check(nearlyEqual(values[4], 0.333251953125f, 1.0e-9f));
    check(nearlyEqual(values[5], 5.9604644775390625e-8f, 1.0e-12f));
    check(nearlyEqual(values[6], 6.0975551605224609e-5f, 1.0e-10f));
    check(values[7] == 65504.0f);
};

auto tReadIntoDestination = test("Model/SafeTensors/readIntoDestination") = []
{
    const auto file = SafeTensors::fromBytes(singleTensorFile());

    auto destination = Vector<float> {};
    destination.resize(6);
    file.readFloats("weight", destination);

    check(nearlyEqual(destination[5], 6.0f));
    check(throwsModelError(
        [&]
        {
            auto tooSmall = Vector<float> {};
            tooSmall.resize(5);
            file.readFloats("weight", tooSmall);
        }));
};

auto tMultipleTensorsAndMetadata =
    test("Model/SafeTensors/multipleTensorsAndMetadata") = []
{
    const auto header =
        std::string {R"({"__metadata__":{"format":"pt"},)"
                     R"("b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]},)"
                     R"("a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})"};

    const auto file = SafeTensors::fromBytes(
        assemble(header, toBytes<float>({1.0f, 2.0f, 3.0f, 4.0f})));

    check(file.tensors().size() == 2);
    check(file.metadata().size() == 1);
    check(file.metadata().at("format") == "pt");
    check(!file.contains("__metadata__"));

    const auto names = file.names();
    check(names.size() == 2);
    check(names[0] == "a");
    check(names[1] == "b");

    check(nearlyEqual(file.readFloats("a")[0], 1.0f));
    check(nearlyEqual(file.readFloats("b")[0], 3.0f));
};

auto tIntegerTensorsAreNotFloats =
    test("Model/SafeTensors/integerTensorsAreNotFloats") = []
{
    const auto file = SafeTensors::fromBytes(
        assemble(R"({"w":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}})",
                 toBytes<std::int32_t>({1, 2})));

    check(file.info("w").type == TensorType::I32);
    check(file.rawBytes("w").size() == 8);
    check(throwsModelError([&] { return file.readFloats("w"); }));
};

auto tUnknownTensorLookup = test("Model/SafeTensors/unknownTensorLookup") = []
{
    const auto file = SafeTensors::fromBytes(singleTensorFile());

    check(file.find("nope") == nullptr);
    check(throwsModelError([&] { return file.info("nope"); }));
    check(throwsModelError([&] { return file.rawBytes("nope"); }));
};

auto tMalformedHeaders = test("Model/SafeTensors/malformedHeaders") = []
{
    auto truncated = Vector<std::uint8_t> {};
    truncated.resize(7);
    check(throwsModelError([&] { return SafeTensors::fromBytes(truncated); }));

    const auto header = std::string {"{}"};

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(
                assembleWithLength(header.size() + 1, header, {}));
        }));

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(
                assembleWithLength(~std::uint64_t {}, header, {}));
        }));

    check(throwsModelError([] { return parseHeaderOnly("{not json"); }));
    check(throwsModelError([] { return parseHeaderOnly("[1,2,3]"); }));
    check(throwsModelError([] { return parseHeaderOnly(""); }));
    check(throwsModelError([] { return parseHeaderOnly(R"({"w":42})"); }));
};

auto tMissingFields = test("Model/SafeTensors/missingFields") = []
{
    check(throwsModelError(
        []
        { return parseHeaderOnly(R"({"w":{"shape":[1],"data_offsets":[0,4]}})"); }));

    check(throwsModelError(
        []
        {
            return parseHeaderOnly(R"({"w":{"dtype":"F32","data_offsets":[0,4]}})");
        }));

    check(throwsModelError(
        [] { return parseHeaderOnly(R"({"w":{"dtype":"F32","shape":[1]}})"); }));

    check(throwsModelError(
        []
        {
            return parseHeaderOnly(
                R"({"w":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}})");
        }));
};

// The check that matters most: a shape and a byte range that disagree mean a
// kernel would walk the blob at a stride it does not have.
auto tShapeDisagreesWithRange =
    test("Model/SafeTensors/shapeDisagreesWithRange") = []
{
    const auto blob = toBytes<float>({1.0f, 2.0f});

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(assemble(
                R"({"w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})", blob));
        }));

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(assemble(
                R"({"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,8]}})", blob));
        }));

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(
                assemble(R"({"w":{"dtype":"F32","shape":[4],)"
                         R"("data_offsets":[0,16]}})",
                         blob));
        }));

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(assemble(
                R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}})", blob));
        }));

    check(throwsModelError(
        [&]
        {
            return SafeTensors::fromBytes(assemble(
                R"({"w":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}})", blob));
        }));
};

auto tMetadataMustBeStrings = test("Model/SafeTensors/metadataMustBeStrings") = []
{
    check(throwsModelError(
        [] { return parseHeaderOnly(R"({"__metadata__":{"format":7}})"); }));

    check(throwsModelError([] { return parseHeaderOnly(R"({"__metadata__":7})"); }));
};

auto tEmptyTensorIsLegal = test("Model/SafeTensors/emptyTensorIsLegal") = []
{
    const auto file = parseHeaderOnly(
        R"({"empty":{"dtype":"F32","shape":[0,4],"data_offsets":[0,0]}})");

    check(file.info("empty").elementCount() == 0);
    check(file.readFloats("empty").empty());
};

auto tMissingFileIsAnError = test("Model/SafeTensors/missingFileIsAnError") = []
{
    check(throwsModelError(
        [] { return SafeTensors::fromFile("no/such/model.safetensors"); }));

    check(throwsModelError(
        []
        { return SafeTensors::fromFile(std::filesystem::temp_directory_path()); }));
};

// fromFile maps the file rather than reading it, which is an implementation
// detail of where the bytes live: the same bytes on disk have to parse to the
// same tensors as the buffer they were written from.
auto tMappedFileMatchesBytes = test("Model/SafeTensors/mappedFileMatchesBytes") = []
{
    const auto scratch = ScratchDirectory {"mapped"};
    const auto bytes = singleTensorFile();
    const auto path = scratch.write("model.safetensors", bytes);

    const auto mapped = SafeTensors::fromFile(path);
    const auto inMemory = SafeTensors::fromBytes(bytes);

    check(mapped.names() == inMemory.names());
    check(mapped.info("weight").byteCount == inMemory.info("weight").byteCount);

    const auto fromDisk = mapped.readFloats("weight");
    const auto fromBuffer = inMemory.readFloats("weight");

    check(fromDisk.size() == fromBuffer.size());

    for (auto index = 0; index < fromDisk.size(); ++index)
        check(fromDisk[index] == fromBuffer[index]);
};

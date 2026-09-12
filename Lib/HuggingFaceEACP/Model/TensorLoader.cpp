#include "TensorLoader.h"

#include <HuggingFaceEACP/Kernels/Int8Blocks.h>

#include <eacp/GPU/Device/Device.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace HF
{
namespace
{
std::string shapeText(const TensorInfo& tensor)
{
    auto text = std::string {"["};

    for (auto axis = 0; axis < tensor.rank(); ++axis)
        text += (axis == 0 ? "" : ", ") + std::to_string(tensor.dimension(axis));

    return text + "]";
}

std::string shapeText(Span<const int> shape)
{
    auto text = std::string {"["};

    for (auto axis = 0; axis < shape.size(); ++axis)
        text += (axis == 0 ? "" : ", ") + std::to_string(shape[axis]);

    return text + "]";
}

// eacp::GPU::Buffer counts its bytes in an int, which is plan.md's second gap,
// and a stacked weight is the one buffer this file makes that is twice a
// tensor's size — Gemma's fused gate and up is 134 MB packed and 268 MB
// widened. Counted in 64 bits and refused by name, because the alternative is a
// truncated allocation and a kernel writing outside it.
int requireStackedBytes(const TensorInfo& first,
                        const TensorInfo& second,
                        std::int64_t bytesPerStoredElement)
{
    const auto elements = first.elementCount() + second.elementCount();
    const auto bytes = elements * bytesPerStoredElement;

    if (bytes > (std::int64_t) std::numeric_limits<int>::max())
        throw ModelError {"tensors '" + first.name + "' and '" + second.name
                          + "' are " + std::to_string(bytes)
                          + " bytes stacked, which is too large for a GPU "
                            "buffer"};

    return (int) bytes;
}

eacp::GPU::Buffer uploadStackedBytes(Span<const std::uint8_t> first,
                                     Span<const std::uint8_t> second,
                                     int byteCount)
{
    auto stacked = Vector<std::uint8_t> {};
    stacked.resize(byteCount);

    std::memcpy(stacked.data(), first.data(), (std::size_t) first.size());
    std::memcpy(
        stacked.data() + first.size(), second.data(), (std::size_t) second.size());

    return eacp::GPU::Device::shared().makeBuffer(
        stacked.data(), byteCount, eacp::GPU::BufferUsage::Storage);
}

eacp::GPU::Buffer uploadStackedFloats(const ShardedTensors& file,
                                      const TensorInfo& first,
                                      const TensorInfo& second,
                                      int byteCount)
{
    const auto firstCount = (int) first.elementCount();

    auto stacked = Vector<float> {};
    stacked.resize(byteCount / (int) sizeof(float));

    file.readFloats(first.name, Span<float> {stacked.data(), firstCount});
    file.readFloats(
        second.name,
        Span<float> {stacked.data() + firstCount, (int) second.elementCount()});

    return eacp::GPU::Device::shared().makeBuffer(
        stacked.data(), byteCount, eacp::GPU::BufferUsage::Storage);
}

// Whether the two halves can be handed over as the bytes they already are.
// F32 always can. A packed pair can when both element counts are even: a packed
// read fetches the word an index lands in, so an odd first half would leave the
// second one straddling words, and an odd second half would leave the buffer
// itself a half-word short of the last element's word. Both are checked rather
// than only the first, since nothing here says the two halves are the same
// shape — that is the caller's `expected`, not this function's.
bool stacksAsBytes(const TensorInfo& first, const TensorInfo& second)
{
    if (first.type != second.type)
        return false;

    if (first.type == TensorType::F32)
        return true;

    return isPackedSixteenBit(first.type) && first.elementCount() % 2 == 0
           && second.elementCount() % 2 == 0;
}

// ---------------------------------------------------------------------------
// Quantizing on the way up
// ---------------------------------------------------------------------------

// How many elements one pass of the widen-and-quantize loop holds at a time.
// The blocks are independent, so this is a scratch size rather than a shape: a
// megabyte of elements is four megabytes of floats, which stays in cache-sized
// work per thread and never widens a five gigabyte tensor whole.
constexpr auto quantizeChunkElements = std::int64_t {1} << 20;

std::string blockShapeText(const TensorInfo& tensor)
{
    return "tensor '" + tensor.name + "' is " + shapeText(tensor) + " of "
           + std::to_string(tensor.elementCount()) + " elements";
}

// The two things Kernels/Int8Blocks.h needs of a shape, named by tensor when
// they do not hold. Every google/gemma-2b tensor a product reads passes: the
// contiguous dimension is 2048 or 16384 and the smallest element count is
// 256 * 2048.
void requireQuantizable(const TensorInfo& tensor)
{
    const auto contiguous = tensor.dimension(tensor.rank() - 1);

    if (quantizesAsInt8Blocks(contiguous, tensor.elementCount()))
        return;

    throw ModelError {blockShapeText(tensor) + ", which int8 blocks of "
                      + std::to_string(int8BlockSize)
                      + " cannot hold: the contiguous dimension must be a whole "
                        "number of blocks and the element count a whole number "
                        "of scale words"};
}

int requireQuantizedBytes(std::int64_t elementCount, const std::string& what)
{
    const auto bytes = int8BlocksByteCount(elementCount);

    if (bytes > (std::int64_t) std::numeric_limits<int>::max())
        throw ModelError {what + " is " + std::to_string(bytes)
                          + " bytes quantized, which is too large for a GPU "
                            "buffer"};

    return (int) bytes;
}

// One tensor's elements, widened a chunk at a time and quantized into the place
// they occupy in the destination. `firstElement` is where this tensor begins in
// that destination, which is zero for a lone weight and the first half's count
// for the second half of a stack.
void quantizeElements(const TensorInfo& tensor,
                      Span<const std::uint8_t> bytes,
                      std::int64_t firstElement,
                      std::int64_t elementCount,
                      Span<std::uint8_t> destination,
                      std::int64_t begin,
                      std::int64_t end)
{
    auto scratch = Vector<float> {};
    scratch.resize((int) quantizeChunkElements);

    for (auto at = begin; at < end; at += quantizeChunkElements)
    {
        const auto count = (int) std::min(quantizeChunkElements, end - at);

        widenTensorElements(tensor, bytes, at, Span<float> {scratch.data(), count});

        quantizeInt8Blocks(Span<const float> {scratch.data(), count},
                           firstElement + at,
                           elementCount,
                           destination);
    }
}

// **Quantizing 2.5 billion elements is seconds of a load that is otherwise
// under one**, and the blocks are independent, so the work splits by element
// range. Each thread writes a disjoint run of the destination and reads a
// disjoint run of the mapping, so there is nothing to synchronise but the join.
void quantizeInParallel(const TensorInfo& tensor,
                        Span<const std::uint8_t> bytes,
                        std::int64_t firstElement,
                        std::int64_t elementCount,
                        Span<std::uint8_t> destination)
{
    const auto total = tensor.elementCount();
    const auto blocks = int8BlockCount(total);
    const auto cores =
        (std::int64_t) std::max(1u, std::thread::hardware_concurrency());
    const auto workers = std::max(std::int64_t {1}, std::min(cores, blocks));
    const auto blocksEach = (blocks + workers - 1) / workers;

    auto threads = std::vector<std::thread> {};

    for (auto worker = std::int64_t {}; worker < workers; ++worker)
    {
        const auto begin = worker * blocksEach * int8BlockSize;
        const auto end = std::min(total, begin + blocksEach * int8BlockSize);

        if (begin >= end)
            break;

        threads.emplace_back(
            [&, begin, end]
            {
                quantizeElements(tensor,
                                 bytes,
                                 firstElement,
                                 elementCount,
                                 destination,
                                 begin,
                                 end);
            });
    }

    for (auto& thread: threads)
        thread.join();
}

eacp::GPU::Buffer uploadQuantized(Span<std::uint8_t> bytes)
{
    return eacp::GPU::Device::shared().makeBuffer(
        bytes.data(), bytes.size(), eacp::GPU::BufferUsage::Storage);
}
} // namespace

const TensorInfo& TensorLoader::require(const std::string& name) const
{
    if (const auto* found = file.find(name))
        return *found;

    throw ModelError {"the " + std::string {component} + " needs a tensor named '"
                      + name + "', and the checkpoint has none"};
}

void TensorLoader::checkShape(const TensorInfo& tensor,
                              Span<const int> expected) const
{
    auto matches = tensor.rank() == expected.size();

    for (auto axis = 0; matches && axis < expected.size(); ++axis)
        matches = tensor.dimension(axis) == expected[axis];

    if (!matches)
        throw ModelError {"tensor '" + tensor.name + "' is " + shapeText(tensor)
                          + ", and the " + std::string {component}
                          + " was built for " + shapeText(expected)};
}

void TensorLoader::checkShape(const TensorInfo& tensor, TensorShape expected) const
{
    checkShape(
        tensor,
        Span<const int> {expected.begin(), static_cast<int>(expected.size())});
}

TensorBuffer TensorLoader::loadFloatTensor(const std::string& name,
                                           TensorShape expected) const
{
    checkShape(require(name), expected);
    return file.makeFloatBuffer(name);
}

TensorBuffer TensorLoader::loadProjectionWeight(const std::string& name,
                                                TensorShape expected,
                                                WeightPrecision precision) const
{
    const auto& tensor = require(name);
    checkShape(tensor, expected);

    if (precision == WeightPrecision::AsShipped)
        return file.makeBuffer(name);

    requireQuantizable(tensor);

    const auto elements = tensor.elementCount();

    auto bytes = Vector<std::uint8_t> {};
    bytes.resize(requireQuantizedBytes(elements, "tensor '" + name + "'"));

    const auto destination = Span<std::uint8_t> {bytes.data(), bytes.size()};

    quantizeInParallel(tensor, file.rawBytes(name), 0, elements, destination);

    return {uploadQuantized(destination), TensorType::I8, BufferLayout::Int8Blocks};
}

TensorBuffer
    TensorLoader::loadStackedProjectionWeights(const std::string& first,
                                               const std::string& second,
                                               TensorShape expected,
                                               WeightPrecision precision) const
{
    const auto& top = require(first);
    const auto& bottom = require(second);

    checkShape(top, expected);
    checkShape(bottom, expected);

    if (precision == WeightPrecision::Int8Blocks)
    {
        requireQuantizable(top);
        requireQuantizable(bottom);

        const auto elements = top.elementCount() + bottom.elementCount();
        const auto what = "tensors '" + first + "' and '" + second + "' stacked";

        auto bytes = Vector<std::uint8_t> {};
        bytes.resize(requireQuantizedBytes(elements, what));

        const auto destination = Span<std::uint8_t> {bytes.data(), bytes.size()};

        quantizeInParallel(top, file.rawBytes(first), 0, elements, destination);
        quantizeInParallel(bottom,
                           file.rawBytes(second),
                           top.elementCount(),
                           elements,
                           destination);

        return {
            uploadQuantized(destination), TensorType::I8, BufferLayout::Int8Blocks};
    }

    if (stacksAsBytes(top, bottom))
    {
        const auto bytes =
            requireStackedBytes(top, bottom, bytesPerElement(top.type));

        return {
            uploadStackedBytes(file.rawBytes(first), file.rawBytes(second), bytes),
            top.type};
    }

    const auto bytes = requireStackedBytes(top, bottom, sizeof(float));

    return {uploadStackedFloats(file, top, bottom, bytes), TensorType::F32};
}
} // namespace HF

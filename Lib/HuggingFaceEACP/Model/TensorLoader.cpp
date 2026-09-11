#include "TensorLoader.h"

#include <eacp/GPU/Device/Device.h>

#include <cstdint>
#include <cstring>
#include <limits>

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
                                                TensorShape expected) const
{
    checkShape(require(name), expected);
    return file.makeBuffer(name);
}

TensorBuffer TensorLoader::loadStackedProjectionWeights(const std::string& first,
                                                        const std::string& second,
                                                        TensorShape expected) const
{
    const auto& top = require(first);
    const auto& bottom = require(second);

    checkShape(top, expected);
    checkShape(bottom, expected);

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

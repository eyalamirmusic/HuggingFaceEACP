#include "TensorLoader.h"

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

void TensorLoader::rejectPackedHalf(const TensorBuffer& loaded,
                                    const std::string& name,
                                    std::string_view reader) const
{
    if (loaded.isPackedHalf())
        throw ModelError {"tensor '" + name + "' is fp16, and this "
                          + std::string {component} + " binds it to "
                          + std::string {reader}
                          + ", which has no packed-half read: ship the tensor "
                            "as F32"};
}

TensorBuffer TensorLoader::loadFloatTensor(const std::string& name,
                                           TensorShape expected,
                                           std::string_view reader) const
{
    checkShape(require(name), expected);
    auto loaded = file.makeBuffer(name);
    rejectPackedHalf(loaded, name, reader);

    return loaded;
}

TensorBuffer TensorLoader::loadProjectionWeight(const std::string& name,
                                                TensorShape expected) const
{
    checkShape(require(name), expected);
    return file.makeBuffer(name);
}
} // namespace HF

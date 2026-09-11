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

TensorBuffer TensorLoader::loadFloatTensor(const std::string& name,
                                           TensorShape expected) const
{
    checkShape(require(name), expected);
    return file.makeWidenedBuffer(name);
}

TensorBuffer TensorLoader::loadWeight(const std::string& name,
                                      TensorShape expected) const
{
    checkShape(require(name), expected);
    return file.makeBuffer(name);
}
} // namespace HF

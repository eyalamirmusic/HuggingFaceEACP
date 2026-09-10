#include "DecoderWeights.h"

#include <HuggingFaceEACP/Model/GemmaTensors.h>
#include <HuggingFaceEACP/Model/TensorLoader.h>

#include <eacp/GPU/GPU.h>

#include <string>

namespace HF
{
namespace
{
// The gather has no class in Kernels/ that names it in a signature, so the
// tensors it reads name it by what it does rather than by a type this file
// would have to keep in step.
constexpr auto embeddingReader = "the embedding gather";

// The word every refusal uses for a top-level tensor, and the one a layer's
// tensors use, so a message says which part of the model would not take the
// checkpoint rather than leaving that to a stack trace.
TensorLoader decoderTensors(const ShardedTensors& file)
{
    return {file, "gemma decoder"};
}

std::string layerComponent(int index)
{
    return "layer " + std::to_string(index);
}

// Before the first tensor is looked up rather than after the last, since a
// shape whose head width the attention cannot hold would otherwise load five
// gigabytes and fail on the first dispatch. The first member's initialiser is
// the only place that runs before the others.
const DecoderShape& validated(const DecoderShape& shape)
{
    shape.validate();
    return shape;
}

TensorBuffer loadTokenEmbedding(const ShardedTensors& file,
                                const DecoderShape& shape)
{
    return decoderTensors(file).loadFloatTensor(GemmaTensors::embedding,
                                                {shape.vocabularySize, shape.width},
                                                embeddingReader);
}

// gate_proj and up_proj stacked into the one [2 * intermediate, width] weight
// the feed-forward's single product reads, gate rows first — which is the
// halving of a row GeGLU's primary form is written against.
//
// Both are checked at their shipped shapes before a byte is read, so a
// checkpoint whose gate and up disagree is a ModelError naming the tensor
// rather than a stack of two matrices of different widths.
TensorBuffer
    loadFusedGateUp(const ShardedTensors& file, const DecoderShape& shape, int index)
{
    const auto tensors = TensorLoader {file, layerComponent(index)};
    const auto gateName =
        GemmaTensors::layerTensorName(index, GemmaTensors::gateProjection);
    const auto upName =
        GemmaTensors::layerTensorName(index, GemmaTensors::upProjection);

    tensors.checkShape(tensors.require(gateName), {shape.intermediate, shape.width});
    tensors.checkShape(tensors.require(upName), {shape.intermediate, shape.width});

    const auto halfCount = shape.intermediate * shape.width;

    auto stacked = Vector<float> {};
    stacked.resize(2 * halfCount);

    file.readFloats(gateName, Span<float> {stacked.data(), halfCount});
    file.readFloats(upName, Span<float> {stacked.data() + halfCount, halfCount});

    auto buffer =
        eacp::GPU::Device::shared().makeBuffer(stacked.data(),
                                               (int) sizeof(float) * stacked.size(),
                                               eacp::GPU::BufferUsage::Storage);

    return {std::move(buffer), TensorType::F32};
}
} // namespace

DecoderLayerWeights::DecoderLayerWeights(const ShardedTensors& file,
                                         const DecoderShape& shape,
                                         int index)
    : inputNorm(TensorLoader {file, layerComponent(index)}.loadFloatTensor(
          GemmaTensors::layerTensorName(index, GemmaTensors::inputNorm),
          {shape.width},
          "RMSNorm"))
    , query(TensorLoader {file, layerComponent(index)}.loadProjectionWeight(
          GemmaTensors::layerTensorName(index, GemmaTensors::queryProjection),
          {shape.queryWidth(), shape.width}))
    , key(TensorLoader {file, layerComponent(index)}.loadProjectionWeight(
          GemmaTensors::layerTensorName(index, GemmaTensors::keyProjection),
          {shape.kvWidth(), shape.width}))
    , value(TensorLoader {file, layerComponent(index)}.loadProjectionWeight(
          GemmaTensors::layerTensorName(index, GemmaTensors::valueProjection),
          {shape.kvWidth(), shape.width}))
    , output(TensorLoader {file, layerComponent(index)}.loadProjectionWeight(
          GemmaTensors::layerTensorName(index, GemmaTensors::outputProjection),
          {shape.width, shape.queryWidth()}))
    , fusedGateUp(loadFusedGateUp(file, shape, index))
    , down(TensorLoader {file, layerComponent(index)}.loadProjectionWeight(
          GemmaTensors::layerTensorName(index, GemmaTensors::downProjection),
          {shape.width, shape.intermediate}))
    , postAttentionNorm(TensorLoader {file, layerComponent(index)}.loadFloatTensor(
          GemmaTensors::layerTensorName(index, GemmaTensors::postAttentionNorm),
          {shape.width},
          "RMSNorm"))
{
}

DecoderWeights::DecoderWeights(const ShardedTensors& file,
                               const DecoderShape& shapeToUse)
    : shape(validated(shapeToUse))
    , tokenEmbedding(loadTokenEmbedding(file, shapeToUse))
    , finalNorm(decoderTensors(file).loadFloatTensor(
          GemmaTensors::finalNorm, {shapeToUse.width}, "RMSNorm"))
{
    layers.reserve(shapeToUse.layers);

    for (auto index = 0; index < shapeToUse.layers; ++index)
        layers.emplace_back(file, shapeToUse, index);
}
} // namespace HF

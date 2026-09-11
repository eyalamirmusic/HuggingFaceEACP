#include "DecoderWeights.h"

#include <HuggingFaceEACP/Model/GemmaTensors.h>
#include <HuggingFaceEACP/Model/TensorLoader.h>

#include <eacp/GPU/GPU.h>

#include <string>

namespace HF
{
namespace
{
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
    return decoderTensors(file).loadProjectionWeight(
        GemmaTensors::embedding, {shape.vocabularySize, shape.width});
}

// gate_proj and up_proj stacked into the one [2 * intermediate, width] weight
// the feed-forward's single product reads, gate rows first — which is the
// halving of a row GeGLU's primary form is written against.
TensorBuffer
    loadFusedGateUp(const ShardedTensors& file, const DecoderShape& shape, int index)
{
    return TensorLoader {file, layerComponent(index)}.loadStackedProjectionWeights(
        GemmaTensors::layerTensorName(index, GemmaTensors::gateProjection),
        GemmaTensors::layerTensorName(index, GemmaTensors::upProjection),
        {shape.intermediate, shape.width});
}
} // namespace

DecoderLayerWeights::DecoderLayerWeights(const ShardedTensors& file,
                                         const DecoderShape& shape,
                                         int index)
    : inputNorm(TensorLoader {file, layerComponent(index)}.loadFloatTensor(
          GemmaTensors::layerTensorName(index, GemmaTensors::inputNorm),
          {shape.width}))
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
          {shape.width}))
{
}

WeightStorage weightStorageOf(const TensorBuffer& weight)
{
    if (weight.isPackedHalf())
        return WeightStorage::PackedHalf;

    if (weight.isPackedBFloat16())
        return WeightStorage::PackedBFloat16;

    return WeightStorage::Float;
}

DecoderWeights::DecoderWeights(const ShardedTensors& file,
                               const DecoderShape& shapeToUse)
    : shape(validated(shapeToUse))
    , tokenEmbedding(loadTokenEmbedding(file, shapeToUse))
    , finalNorm(decoderTensors(file).loadFloatTensor(GemmaTensors::finalNorm,
                                                     {shapeToUse.width}))
{
    layers.reserve(shapeToUse.layers);

    for (auto index = 0; index < shapeToUse.layers; ++index)
        layers.emplace_back(file, shapeToUse, index);
}

Vector<WeightStorage> DecoderWeights::storages() const
{
    auto found = Vector<WeightStorage> {};

    auto note = [&](const TensorBuffer& weight)
    { found.addIfNotThere(weightStorageOf(weight)); };

    note(tokenEmbedding);

    for (const auto& layer: layers)
    {
        note(layer.query);
        note(layer.key);
        note(layer.value);
        note(layer.output);
        note(layer.fusedGateUp);
        note(layer.down);
    }

    return found;
}
} // namespace HF

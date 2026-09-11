#include "DecoderWeights.h"

#include <HuggingFaceEACP/Model/GemmaTensors.h>
#include <HuggingFaceEACP/Model/TensorLoader.h>

#include <eacp/GPU/GPU.h>

#include <cstring>
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
    return decoderTensors(file).loadWeight(GemmaTensors::embedding,
                                           {shape.vocabularySize, shape.width});
}

// Whether a kernel reads this storage as pairs in a word rather than as floats
// it subscripts, which is what decides whether the two halves below can be
// joined as the bytes they already are.
bool isPackedPair(TensorType type)
{
    return type == TensorType::F16 || type == TensorType::BF16;
}

template <typename Element>
eacp::GPU::Buffer uploadStacked(Span<const Element> first,
                                Span<const Element> second)
{
    auto stacked = Vector<Element> {};
    stacked.resize(first.size() + second.size());

    std::memcpy(stacked.data(), first.data(), sizeof(Element) * first.size());
    std::memcpy(stacked.data() + first.size(),
                second.data(),
                sizeof(Element) * second.size());

    return eacp::GPU::Device::shared().makeBuffer(stacked.data(),
                                                  (int) sizeof(Element)
                                                      * stacked.size(),
                                                  eacp::GPU::BufferUsage::Storage);
}

// gate_proj and up_proj stacked into the one [2 * intermediate, width] weight
// the feed-forward's single product reads, gate rows first — which is the
// halving of a row GeGLU's primary form is written against.
//
// Both are checked at their shipped shapes before a byte is read, so a
// checkpoint whose gate and up disagree is a ModelError naming the tensor
// rather than a stack of two matrices of different widths.
//
// The stack is of the raw bytes when the two arrive in the same packed
// storage, so the layer's largest weight is joined without being widened to be
// joined — the whole point of reading bf16 packed, since this is the tensor a
// widening would cost the most. Two 16-bit values share a word, so the up
// rows can only start where the gate rows end if the gate half is a whole
// number of words; an odd half, which no real shape has, widens instead.
TensorBuffer
    loadFusedGateUp(const ShardedTensors& file, const DecoderShape& shape, int index)
{
    const auto tensors = TensorLoader {file, layerComponent(index)};
    const auto gateName =
        GemmaTensors::layerTensorName(index, GemmaTensors::gateProjection);
    const auto upName =
        GemmaTensors::layerTensorName(index, GemmaTensors::upProjection);

    const auto& gate = tensors.require(gateName);
    const auto& up = tensors.require(upName);

    tensors.checkShape(gate, {shape.intermediate, shape.width});
    tensors.checkShape(up, {shape.intermediate, shape.width});

    const auto halfCount = shape.intermediate * shape.width;

    if (gate.type == up.type && isPackedPair(gate.type) && halfCount % 2 == 0)
        return {uploadStacked(file.rawBytes(gateName), file.rawBytes(upName)),
                gate.type};

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
          {shape.width}))
    , query(TensorLoader {file, layerComponent(index)}.loadWeight(
          GemmaTensors::layerTensorName(index, GemmaTensors::queryProjection),
          {shape.queryWidth(), shape.width}))
    , key(TensorLoader {file, layerComponent(index)}.loadWeight(
          GemmaTensors::layerTensorName(index, GemmaTensors::keyProjection),
          {shape.kvWidth(), shape.width}))
    , value(TensorLoader {file, layerComponent(index)}.loadWeight(
          GemmaTensors::layerTensorName(index, GemmaTensors::valueProjection),
          {shape.kvWidth(), shape.width}))
    , output(TensorLoader {file, layerComponent(index)}.loadWeight(
          GemmaTensors::layerTensorName(index, GemmaTensors::outputProjection),
          {shape.width, shape.queryWidth()}))
    , fusedGateUp(loadFusedGateUp(file, shape, index))
    , down(TensorLoader {file, layerComponent(index)}.loadWeight(
          GemmaTensors::layerTensorName(index, GemmaTensors::downProjection),
          {shape.width, shape.intermediate}))
    , postAttentionNorm(TensorLoader {file, layerComponent(index)}.loadFloatTensor(
          GemmaTensors::layerTensorName(index, GemmaTensors::postAttentionNorm),
          {shape.width}))
{
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
} // namespace HF

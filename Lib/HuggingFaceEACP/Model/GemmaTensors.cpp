#include "GemmaTensors.h"

#include "TensorLoader.h"

namespace HF::GemmaTensors
{
namespace
{
TensorEntry entry(std::string name, Vector<int> shape)
{
    return {std::move(name), std::move(shape)};
}

TensorEntry layerEntry(int layer, std::string_view suffix, Vector<int> shape)
{
    return entry(layerTensorName(layer, suffix), std::move(shape));
}
} // namespace

std::string layerTensorName(int layer, std::string_view suffix)
{
    return "model.layers." + std::to_string(layer) + "." + std::string {suffix};
}

Vector<TensorEntry> forLayer(const GemmaConfig& config, int layer)
{
    const auto hidden = config.hiddenSize;
    const auto queries = config.queryWidth();
    const auto keyValues = config.keyValueWidth();
    const auto intermediate = config.intermediateSize;

    auto entries = Vector<TensorEntry> {};

    entries.add(layerEntry(layer, queryProjection, {queries, hidden}));
    entries.add(layerEntry(layer, keyProjection, {keyValues, hidden}));
    entries.add(layerEntry(layer, valueProjection, {keyValues, hidden}));
    entries.add(layerEntry(layer, outputProjection, {hidden, queries}));
    entries.add(layerEntry(layer, gateProjection, {intermediate, hidden}));
    entries.add(layerEntry(layer, upProjection, {intermediate, hidden}));
    entries.add(layerEntry(layer, downProjection, {hidden, intermediate}));
    entries.add(layerEntry(layer, inputNorm, {hidden}));
    entries.add(layerEntry(layer, postAttentionNorm, {hidden}));

    return entries;
}

Vector<TensorEntry> all(const GemmaConfig& config)
{
    auto entries = Vector<TensorEntry> {};
    entries.add(entry(embedding, {config.vocabularySize, config.hiddenSize}));

    for (auto layer = 0; layer < config.layerCount; ++layer)
        for (auto& tensor: forLayer(config, layer))
            entries.add(std::move(tensor));

    entries.add(entry(finalNorm, {config.hiddenSize}));

    return entries;
}

void checkAgainst(const ShardedTensors& weights, const GemmaConfig& config)
{
    const auto loader = TensorLoader {weights, "gemma decoder"};

    for (const auto& tensor: all(config))
        loader.checkShape(loader.require(tensor.name), tensor.shape);
}
} // namespace HF::GemmaTensors

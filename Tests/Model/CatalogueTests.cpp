#include "Common.h"

using namespace nano;
using namespace HF;
using namespace HF::Testing;

namespace
{
// Gemma's shape, two orders of magnitude down: the ratios that matter are the
// ones the catalogue is built out of — 2 query heads over 1 KV head, a head
// wider than hiddenSize / heads, and a gated MLP twice the width.
GemmaConfig tinyConfig()
{
    auto config = GemmaConfig {};
    config.vocabularySize = 6;
    config.hiddenSize = 4;
    config.intermediateSize = 8;
    config.layerCount = 2;
    config.attentionHeads = 2;
    config.keyValueHeads = 1;
    config.headWidth = 3;
    config.maxPositions = 16;

    return config;
}

std::string shapeJson(const Vector<int>& shape)
{
    auto text = std::string {"["};

    for (auto axis = 0; axis < shape.size(); ++axis)
        text += (axis == 0 ? "" : ",") + std::to_string(shape[axis]);

    return text + "]";
}

int elementCountOf(const Vector<int>& shape)
{
    auto count = 1;

    for (auto extent: shape)
        count *= extent;

    return count;
}

// A checkpoint holding exactly the tensors named, all F32 zeros. Written from
// the catalogue itself, so the assertion below is that the loader and the
// catalogue agree about a file neither of them invented alone.
Vector<std::uint8_t> buildCheckpoint(const Vector<TensorEntry>& entries)
{
    auto header = std::string {"{"};
    auto cursor = 0;

    for (auto index = 0; index < entries.size(); ++index)
    {
        const auto bytes =
            elementCountOf(entries[index].shape) * static_cast<int>(sizeof(float));

        header += (index == 0 ? "" : ",");
        header += "\"" + entries[index].name + R"(":{"dtype":"F32","shape":)"
                  + shapeJson(entries[index].shape) + R"(,"data_offsets":[)"
                  + std::to_string(cursor) + "," + std::to_string(cursor + bytes)
                  + "]}";

        cursor += bytes;
    }

    header += "}";

    auto blob = Vector<std::uint8_t> {};
    blob.resize(cursor);

    return assemble(header, blob);
}

ShardedTensors checkpointIn(const ScratchDirectory& scratch,
                            const Vector<TensorEntry>& entries)
{
    scratch.writeText(ModelFileNames::config, R"({"model_type":"gemma"})");
    scratch.write(ModelFileNames::weights, buildCheckpoint(entries));

    return ShardedTensors::fromDirectory(scratch.path());
}
} // namespace

auto tCatalogueNames = test("Model/Catalogue/names") = []
{
    const auto config = tinyConfig();
    const auto entries = GemmaTensors::all(config);

    // The embedding, nine tensors a layer, and the final norm.
    check(entries.size() == 1 + 9 * config.layerCount + 1);
    check(entries.front().name == "model.embed_tokens.weight");
    check(entries.back().name == "model.norm.weight");

    check(GemmaTensors::layerTensorName(3, GemmaTensors::queryProjection)
          == "model.layers.3.self_attn.q_proj.weight");
    check(GemmaTensors::layerTensorName(0, GemmaTensors::postAttentionNorm)
          == "model.layers.0.post_attention_layernorm.weight");
    check(GemmaTensors::layerTensorName(17, GemmaTensors::downProjection)
          == "model.layers.17.mlp.down_proj.weight");
};

// The shapes are the config read back out, which is what makes a checkpoint
// checkable before a byte of it reaches the GPU.
auto tCatalogueShapes = test("Model/Catalogue/shapes") = []
{
    const auto config = tinyConfig();
    const auto layer = GemmaTensors::forLayer(config, 0);

    const auto shapeOf = [&](std::string_view suffix)
    {
        const auto name = GemmaTensors::layerTensorName(0, suffix);

        for (const auto& entry: layer)
            if (entry.name == name)
                return entry.shape;

        return Vector<int> {};
    };

    check(shapeOf(GemmaTensors::queryProjection) == Vector<int> {6, 4});
    check(shapeOf(GemmaTensors::keyProjection) == Vector<int> {3, 4});
    check(shapeOf(GemmaTensors::valueProjection) == Vector<int> {3, 4});
    check(shapeOf(GemmaTensors::outputProjection) == Vector<int> {4, 6});
    check(shapeOf(GemmaTensors::gateProjection) == Vector<int> {8, 4});
    check(shapeOf(GemmaTensors::upProjection) == Vector<int> {8, 4});
    check(shapeOf(GemmaTensors::downProjection) == Vector<int> {4, 8});
    check(shapeOf(GemmaTensors::inputNorm) == Vector<int> {4});
    check(shapeOf(GemmaTensors::postAttentionNorm) == Vector<int> {4});

    const auto all = GemmaTensors::all(config);
    check(all.front().shape == Vector<int> {6, 4});
    check(all.back().shape == Vector<int> {4});
};

// Gemma 2B's own numbers, so the catalogue is checked against plan.md's table
// rather than only against itself. Unconfirmed against the real config.json —
// the repo is gated and was not fetched — and CheckpointTests is where a real
// download would say so.
auto tCatalogueAtGemmaSize = test("Model/Catalogue/atGemmaSize") = []
{
    const auto config = GemmaConfig {};
    const auto entries = GemmaTensors::all(config);

    check(entries.size() == 1 + 9 * 18 + 1);
    check(entries.front().shape == Vector<int> {256000, 2048});

    const auto layer = GemmaTensors::forLayer(config, 0);
    check(layer.front().shape == Vector<int> {2048, 2048});
    check(layer[1].shape == Vector<int> {256, 2048});
};

auto tCatalogueAcceptsAMatchingCheckpoint =
    test("Model/Catalogue/acceptsMatchingCheckpoint") = []
{
    const auto config = tinyConfig();
    const auto scratch = ScratchDirectory {"catalogue-match"};
    const auto weights = checkpointIn(scratch, GemmaTensors::all(config));

    GemmaTensors::checkAgainst(weights, config);

    check(weights.tensorCount() == GemmaTensors::all(config).size());
};

auto tCatalogueRefusesAWrongShape = test("Model/Catalogue/refusesWrongShape") = []
{
    const auto config = tinyConfig();

    auto entries = GemmaTensors::all(config);
    entries[1].shape = Vector<int> {7, 4};

    const auto scratch = ScratchDirectory {"catalogue-wrong-shape"};
    const auto weights = checkpointIn(scratch, entries);

    check(throwsModelError([&] { GemmaTensors::checkAgainst(weights, config); }));
};

auto tCatalogueRefusesAMissingTensor =
    test("Model/Catalogue/refusesMissingTensor") = []
{
    const auto config = tinyConfig();

    auto entries = GemmaTensors::all(config);
    entries.removeAt(entries.size() - 1);

    const auto scratch = ScratchDirectory {"catalogue-missing"};
    const auto weights = checkpointIn(scratch, entries);

    check(throwsModelError([&] { GemmaTensors::checkAgainst(weights, config); }));
};

// A checkpoint of the right shape for one config is the wrong shape for
// another, and the refusal names the tensor rather than the config.
auto tCatalogueRefusesAnotherConfig =
    test("Model/Catalogue/refusesAnotherConfig") = []
{
    const auto config = tinyConfig();
    const auto scratch = ScratchDirectory {"catalogue-other-config"};
    const auto weights = checkpointIn(scratch, GemmaTensors::all(config));

    auto wider = config;
    wider.hiddenSize = 8;

    check(throwsModelError([&] { GemmaTensors::checkAgainst(weights, wider); }));
};

auto tLoaderReportsTheShapeItWantedItsRank =
    test("Model/Catalogue/loaderMessagesNameTheTensor") = []
{
    const auto config = tinyConfig();
    const auto scratch = ScratchDirectory {"catalogue-messages"};
    const auto weights = checkpointIn(scratch, GemmaTensors::all(config));

    const auto loader = TensorLoader {weights, "decoder"};
    const auto& norm = loader.require("model.norm.weight");

    loader.checkShape(norm, {4});

    check(throwsModelError([&] { loader.checkShape(norm, {4, 4}); }));
    check(throwsModelError([&] { loader.checkShape(norm, {5}); }));
    check(throwsModelError([&] { return loader.require("model.nothing"); }));

    try
    {
        loader.checkShape(norm, {5});
    }
    catch (const ModelError& error)
    {
        const auto message = std::string {error.what()};
        check(message.find("model.norm.weight") != std::string::npos);
        check(message.find("decoder") != std::string::npos);
    }
};

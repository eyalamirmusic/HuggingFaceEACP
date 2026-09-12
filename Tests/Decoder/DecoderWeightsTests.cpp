#include "ReferenceDecoder.h"

using namespace nano;
using namespace HF;
using namespace HF::Testing;
using namespace eacp::GPU;

namespace
{
int bufferElements(const TensorBuffer& tensor)
{
    return tensor.buffer.size() / (int) sizeof(float);
}

// The catalogue with one tensor gone, which is what a truncated download or a
// repo of a different architecture looks like from here.
TensorEdit without(std::string_view name)
{
    return [wanted = std::string {name}](Vector<SyntheticTensor>& tensors)
    {
        for (auto at = 0; at < tensors.size(); ++at)
        {
            if (tensors[at].name == wanted)
            {
                tensors.erase(tensors.begin() + at);
                return;
            }
        }
    };
}

// One tensor's outer axis changed and its values shortened to match, so the
// file is internally consistent and only disagrees with the config — which is
// the failure a header check cannot catch and a shape check must.
TensorEdit reshaped(std::string_view name, int firstDimension)
{
    return [wanted = std::string {name},
            firstDimension](Vector<SyntheticTensor>& tensors)
    {
        for (auto& tensor: tensors)
        {
            if (tensor.name != wanted)
                continue;

            auto elements = firstDimension;

            for (auto axis = 1; axis < tensor.shape.size(); ++axis)
                elements *= tensor.shape[axis];

            tensor.shape[0] = firstDimension;
            tensor.values.resize(elements, 0.f);
            return;
        }
    };
}
} // namespace

// Every tensor uploaded at the size the config implies, over a checkpoint that
// went through a real safetensors header — so the loader's name and shape
// mapping is under test rather than a convention shared with it.
auto tWeightsLoad = test("Decoder/weightsLoadAtTheConfigsShapes") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint = SyntheticCheckpoint {"decoder-weights"};
    const auto shape = checkpoint.shape();
    const auto weights = DecoderWeights {checkpoint.weights(), shape};

    check(weights.shape == shape);
    check(weights.layers.size() == shape.layers);

    check(bufferElements(weights.tokenEmbedding)
          == shape.vocabularySize * shape.width);
    check(bufferElements(weights.finalNorm) == shape.width);

    for (const auto& layer: weights.layers)
    {
        check(bufferElements(layer.inputNorm) == shape.width);
        check(bufferElements(layer.postAttentionNorm) == shape.width);
        check(bufferElements(layer.query) == shape.queryWidth() * shape.width);
        check(bufferElements(layer.key) == shape.kvWidth() * shape.width);
        check(bufferElements(layer.value) == shape.kvWidth() * shape.width);
        check(bufferElements(layer.output) == shape.width * shape.queryWidth());
        check(bufferElements(layer.down) == shape.width * shape.intermediate);

        // The one buffer that is not a tensor in the file: gate and up stacked
        // into the single weight the feed-forward's one product reads.
        check(bufferElements(layer.fusedGateUp)
              == 2 * shape.intermediate * shape.width);
    }
};

// The stacking itself, read back off the device: the gate rows first and the up
// rows after them, each element for element what the checkpoint holds. A
// concatenation the other way round would still be the right size and would
// still run, and the model it produced would be a different model.
auto tFusedGateUpIsGateThenUp = test("Decoder/fusedGateUpIsGateThenUp") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint = SyntheticCheckpoint {"decoder-fused"};
    const auto shape = checkpoint.shape();
    const auto weights = DecoderWeights {checkpoint.weights(), shape};

    const auto halfCount = shape.intermediate * shape.width;
    const auto stacked =
        readBack(weights.layers[0].fusedGateUp.buffer, 2 * halfCount);

    const auto gate = checkpoint.weights().readFloats(
        GemmaTensors::layerTensorName(0, GemmaTensors::gateProjection));
    const auto up = checkpoint.weights().readFloats(
        GemmaTensors::layerTensorName(0, GemmaTensors::upProjection));

    for (auto index = 0; index < halfCount; ++index)
    {
        check(stacked[index] == gate[index]);
        check(stacked[halfCount + index] == up[index]);
    }
};

// A BF16 checkpoint, which is what google/gemma-2b ships: every weight a
// product or the gather reads stays packed, at half the bytes the widened path
// uploaded, and the two norm scales are widened because RMSNorm subscripts
// floats. The embedding is the one to watch — it is bound to both the gather
// and the tied logits projection, so a packed one is 1.05 GB in the real model
// instead of 2.10 GB.
auto tBFloat16WeightsStayPacked = test("Decoder/bfloat16WeightsStayPacked") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint =
        SyntheticCheckpoint {"decoder-bf16-weights", Sharding::Single, asBFloat16()};

    const auto shape = checkpoint.shape();
    const auto weights = DecoderWeights {checkpoint.weights(), shape};

    check(weights.tokenEmbedding.isPackedBFloat16());
    check(weights.tokenEmbedding.buffer.size()
          == 2 * shape.vocabularySize * shape.width);

    check(!weights.finalNorm.isPackedBFloat16());
    check(bufferElements(weights.finalNorm) == shape.width);

    for (const auto& layer: weights.layers)
    {
        check(layer.query.isPackedBFloat16());
        check(layer.key.isPackedBFloat16());
        check(layer.value.isPackedBFloat16());
        check(layer.output.isPackedBFloat16());
        check(layer.down.isPackedBFloat16());

        check(!layer.inputNorm.isPackedBFloat16());
        check(!layer.postAttentionNorm.isPackedBFloat16());
        check(bufferElements(layer.inputNorm) == shape.width);

        check(layer.query.buffer.size() == 2 * shape.queryWidth() * shape.width);
    }
};

// The stacking with the bytes left packed, which is what makes the fused
// gate-and-up weight half what a widened concatenation cost: the buffer is
// bf16, it is twice one half's bytes, and widening it back gives the gate rows
// and then the up rows exactly as the checkpoint rounded them.
auto tBFloat16FusedGateUpStaysPacked =
    test("Decoder/bfloat16FusedGateUpStaysPacked") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint =
        SyntheticCheckpoint {"decoder-bf16-fused", Sharding::Single, asBFloat16()};

    const auto shape = checkpoint.shape();
    const auto weights = DecoderWeights {checkpoint.weights(), shape};
    const auto& fused = weights.layers[0].fusedGateUp;

    const auto halfCount = shape.intermediate * shape.width;

    check(fused.isPackedBFloat16());
    check(fused.buffer.size() == 2 * 2 * halfCount);

    const auto stacked = readBackBFloat16(fused.buffer, 2 * halfCount);

    const auto gate = checkpoint.weights().readFloats(
        GemmaTensors::layerTensorName(0, GemmaTensors::gateProjection));
    const auto up = checkpoint.weights().readFloats(
        GemmaTensors::layerTensorName(0, GemmaTensors::upProjection));

    for (auto index = 0; index < halfCount; ++index)
    {
        check(stacked[index] == gate[index]);
        check(stacked[halfCount + index] == up[index]);
    }
};

// The same checkpoint split across two shards with an index naming them, which
// is the shape gemma-2b itself ships in: the loader has to resolve every name
// through the index, and neither shard holds a whole model.
auto tWeightsLoadThroughTheShardIndex =
    test("Decoder/weightsLoadThroughTheShardIndex") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint =
        SyntheticCheckpoint {"decoder-shards", Sharding::TwoShards};

    check(checkpoint.modelFiles().isSharded());
    check(checkpoint.weights().shardCount() == 2);

    const auto shape = checkpoint.shape();
    const auto weights = DecoderWeights {checkpoint.weights(), shape};

    check(weights.layers.size() == shape.layers);
    check(bufferElements(weights.finalNorm) == shape.width);
};

// A name the checkpoint does not carry is a ModelError naming it and the layer
// that wanted it, since the alternative is a kernel walking a buffer that was
// never bound.
auto tMissingTensorIsAnError = test("Decoder/missingTensorIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto missingLayer = SyntheticCheckpoint {
        "decoder-missing-layer",
        Sharding::Single,
        without(GemmaTensors::layerTensorName(1, GemmaTensors::upProjection))};

    check(throwsModelError(
        [&]
        { return DecoderWeights {missingLayer.weights(), missingLayer.shape()}; }));

    const auto missingNorm = SyntheticCheckpoint {
        "decoder-missing-norm", Sharding::Single, without(GemmaTensors::finalNorm)};

    check(throwsModelError(
        [&]
        { return DecoderWeights {missingNorm.weights(), missingNorm.shape()}; }));
};

// A shape the config does not imply is the failure that would otherwise be
// silent: the bytes are there, the header is consistent, and only the stride a
// kernel would walk them at is wrong.
auto tWrongShapeIsAnError = test("Decoder/wrongTensorShapeIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto shortEmbedding =
        SyntheticCheckpoint {"decoder-short-embedding",
                             Sharding::Single,
                             reshaped(GemmaTensors::embedding, 32)};

    check(throwsModelError(
        [&]
        {
            return DecoderWeights {shortEmbedding.weights(), shortEmbedding.shape()};
        }));

    // The gate and up pair is checked before either is read, so a mismatched
    // half is refused rather than stacked into a weight of the wrong height.
    const auto shortGate = SyntheticCheckpoint {
        "decoder-short-gate",
        Sharding::Single,
        reshaped(GemmaTensors::layerTensorName(0, GemmaTensors::gateProjection),
                 16)};

    check(throwsModelError(
        [&] { return DecoderWeights {shortGate.weights(), shortGate.shape()}; }));
};

// Weights are loaded against one shape and dispatched against another, and
// nothing about a GPU buffer says which — so step() compares the two outright
// before it records anything.
auto tShapeMismatchIsRefused = test("Decoder/weightsForAnotherShapeAreRefused") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint = SyntheticCheckpoint {"decoder-shape-mismatch"};
    const auto shape = checkpoint.shape();
    const auto weights = DecoderWeights {checkpoint.weights(), shape};

    auto other = shape;
    other.maxPositions -= 1;

    auto decoder = Decoder {other};
    decoder.prepare(weights);

    const auto tokens = storageOf(Vector<std::uint32_t> {1u});
    const auto logits = outputFor(shape.logitElementCount());

    auto commands = Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();

        check(throwsModelError([&]
                               { decoder.step(pass, tokens, 1, weights, logits); }));
    }

    check(decoder.position() == 0);
};

// prepare() is what compiles the kernels and sizes the caches, so a step
// before it has no buffers to bind and says so rather than dereferencing an
// empty optional.
auto tUnpreparedStepIsAnError = test("Decoder/unpreparedStepIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint = SyntheticCheckpoint {"decoder-unprepared"};
    const auto shape = checkpoint.shape();
    const auto weights = DecoderWeights {checkpoint.weights(), shape};

    auto decoder = Decoder {shape};

    const auto tokens = storageOf(Vector<std::uint32_t> {1u});
    const auto logits = outputFor(shape.logitElementCount());

    auto commands = Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();

        check(throwsModelError([&]
                               { decoder.step(pass, tokens, 1, weights, logits); }));
    }
};

// The quantized upload: every tensor a product or the gather reads becomes int8
// blocks, the two norm scales stay F32 floats, and the buffers are the size the
// layout says — one byte an element plus one fp16 scale per thirty-two, which
// at the real model's [256000, 2048] embedding is 557 MB rather than 1.05 GB.
//
// The bytes themselves are checked against the dequantizer rather than trusted:
// the fused gate-and-up weight read back through it has to be the gate rows and
// then the up rows, quantized, which is the same claim the bf16 stacking test
// makes about a stack of packed bytes.
auto tInt8WeightsQuantizeOnUpload = test("Decoder/int8WeightsQuantizeOnUpload") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint = SyntheticCheckpoint {"decoder-int8-weights",
                                                 Sharding::Single,
                                                 asBFloat16(),
                                                 smallQuantizableGemmaConfig()};

    const auto shape = checkpoint.shape();
    const auto weights =
        DecoderWeights {checkpoint.weights(), shape, WeightPrecision::Int8Blocks};

    const auto quantizedBytes = [](int elements)
    { return (int) int8BlocksByteCount(elements); };

    check(weights.tokenEmbedding.isInt8Blocks());
    check(weights.tokenEmbedding.buffer.size()
          == quantizedBytes(shape.vocabularySize * shape.width));

    check(!weights.finalNorm.isInt8Blocks());
    check(bufferElements(weights.finalNorm) == shape.width);

    for (const auto& layer: weights.layers)
    {
        check(layer.query.isInt8Blocks());
        check(layer.key.isInt8Blocks());
        check(layer.value.isInt8Blocks());
        check(layer.output.isInt8Blocks());
        check(layer.down.isInt8Blocks());
        check(layer.fusedGateUp.isInt8Blocks());

        check(!layer.inputNorm.isInt8Blocks());
        check(!layer.postAttentionNorm.isInt8Blocks());

        check(layer.query.buffer.size()
              == quantizedBytes(shape.queryWidth() * shape.width));
    }

    const auto storages = weights.storages();

    check(storages.size() == 1);
    check(storages[0] == WeightStorage::Int8Blocks);

    const auto halfCount = shape.intermediate * shape.width;
    const auto& fused = weights.layers[0].fusedGateUp;

    check(fused.buffer.size() == quantizedBytes(2 * halfCount));

    auto bytes = Vector<std::uint8_t> {};
    bytes.resize(fused.buffer.size());
    fused.buffer.read(bytes.data(), bytes.size());

    auto stacked = sized(2 * halfCount);
    dequantizeInt8Blocks(Span<const std::uint8_t> {bytes.data(), bytes.size()},
                         Span<float> {stacked.data(), stacked.size()});

    auto expectedHalf = [&](std::string_view suffix)
    {
        auto values = checkpoint.weights().readFloats(
            GemmaTensors::layerTensorName(0, suffix));

        auto widened = Vector<float> {};
        TiledProduct::quantizedInt8Blocks(values, widened);

        return widened;
    };

    const auto gate = expectedHalf(GemmaTensors::gateProjection);
    const auto up = expectedHalf(GemmaTensors::upProjection);

    for (auto index = 0; index < halfCount; ++index)
    {
        check(stacked[index] == gate[index]);
        check(stacked[halfCount + index] == up[index]);
    }
};

// The shape the block format cannot hold, refused by name rather than stored at
// a layout the shader would read differently. The suite's own 48-wide
// feed-forward is exactly that shape, which is why the quantized tiers run at
// 64 — so this is the one test that wants the ordinary small config.
auto tUnquantizableShapeIsAnError = test("Decoder/unquantizableShapeIsAnError") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint =
        SyntheticCheckpoint {"decoder-int8-refused", Sharding::Single, asBFloat16()};

    const auto shape = checkpoint.shape();

    check(!quantizesAsInt8Blocks(shape.intermediate,
                                 shape.width * shape.intermediate));

    check(throwsModelError(
        [&]
        {
            return DecoderWeights {
                checkpoint.weights(), shape, WeightPrecision::Int8Blocks};
        }));
};

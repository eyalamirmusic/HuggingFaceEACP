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

// A checkpoint in Gemma's own storage: every weight reaches the device packed,
// the norm scales are widened because RMSNorm subscripts floats, and the
// buffers are half the bytes their widened selves would be.
auto tBFloat16WeightsStayPacked = test("Decoder/bfloat16WeightsStayPacked") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint =
        SyntheticCheckpoint {"decoder-bfloat16-weights", TensorType::BF16};

    const auto shape = checkpoint.shape();
    const auto weights = DecoderWeights {checkpoint.weights(), shape};

    check(weights.tokenEmbedding.isPackedBFloat16());
    check(weights.tokenEmbedding.buffer.size()
          == shape.vocabularySize * shape.width * 2);

    check(!weights.finalNorm.isPackedBFloat16());
    check(bufferElements(weights.finalNorm) == shape.width);

    for (const auto& layer: weights.layers)
    {
        check(layer.query.isPackedBFloat16());
        check(layer.key.isPackedBFloat16());
        check(layer.value.isPackedBFloat16());
        check(layer.output.isPackedBFloat16());
        check(layer.down.isPackedBFloat16());
        check(layer.fusedGateUp.isPackedBFloat16());

        check(!layer.inputNorm.isPackedBFloat16());
        check(!layer.postAttentionNorm.isPackedBFloat16());

        check(layer.down.buffer.size() == shape.width * shape.intermediate * 2);
    }
};

// The stacking done on the packed bytes rather than through a widening — the
// whole point of gap 1 for this tensor, since gate and up are the largest
// weights in a layer. Read back as the words they are and widened here, so a
// stack that shifted the up rows by a half-word, or joined them in the wrong
// order, fails.
auto tPackedFusedGateUpIsGateThenUp =
    test("Decoder/packedFusedGateUpIsGateThenUp") = []
{
    if (!Device::shared().isValid())
        return;

    const auto checkpoint =
        SyntheticCheckpoint {"decoder-fused-bfloat16", TensorType::BF16};

    const auto shape = checkpoint.shape();
    const auto weights = DecoderWeights {checkpoint.weights(), shape};

    const auto halfCount = shape.intermediate * shape.width;
    const auto& fused = weights.layers[0].fusedGateUp;

    check(fused.isPackedBFloat16());
    check(fused.buffer.size() == 2 * halfCount * 2);

    auto packed = Vector<std::uint16_t> {};
    packed.resize(2 * halfCount);
    fused.buffer.read(packed.data(), fused.buffer.size());

    const auto gate = checkpoint.weights().readFloats(
        GemmaTensors::layerTensorName(0, GemmaTensors::gateProjection));
    const auto up = checkpoint.weights().readFloats(
        GemmaTensors::layerTensorName(0, GemmaTensors::upProjection));

    for (auto index = 0; index < halfCount; ++index)
    {
        check(bfloat16ToFloat(packed[index]) == gate[index]);
        check(bfloat16ToFloat(packed[halfCount + index]) == up[index]);
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
    decoder.prepare();

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

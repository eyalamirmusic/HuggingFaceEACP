#include "Decoder.h"

#include <HuggingFaceEACP/Model/ModelError.h>

#include <cstdint>
#include <string>

namespace HF
{
using eacp::GPU::Buffer;
using eacp::GPU::BufferRange;
using eacp::GPU::BufferUsage;
using eacp::GPU::ComputePass;
using eacp::GPU::Device;

namespace
{
constexpr int floatBytes(int elementCount)
{
    return elementCount * (int) sizeof(float);
}

Buffer allocate(Device& device, int elementCount)
{
    return device.makeBuffer(floatBytes(elementCount), BufferUsage::Storage);
}

Buffer upload(Device& device, const Vector<float>& values)
{
    return device.makeBuffer(
        values.data(), floatBytes(values.size()), BufferUsage::Storage);
}

Buffer allocateZeroed(Device& device, int elementCount)
{
    auto zeroes = Vector<float> {};
    zeroes.resize(elementCount, 0.f);

    return upload(device, zeroes);
}

void allocatePerLayer(Vector<Buffer>& buffers,
                      Device& device,
                      int layers,
                      int elementCount)
{
    buffers.clear();
    buffers.reserve(layers);

    for (auto index = 0; index < layers; ++index)
        buffers.emplace_back(allocate(device, elementCount));
}

// The two product kinds differ in how they read the weight and in how they
// spread the work, so the bind is written once against whichever of them the
// caller picked. The target is a range rather than a buffer because two of its
// call sites write into the middle of a KV cache.
template <typename Program>
void dispatchProduct(Program& program,
                     ComputePass& pass,
                     const Buffer& input,
                     const Buffer& weight,
                     const Buffer& bias,
                     const BufferRange& target,
                     int innerCount,
                     int outputWidth,
                     int rowCount,
                     bool residual)
{
    program.a = input;
    program.b = weight;
    program.bias = bias;
    program.output = target;

    auto shape = TiledMatMulShape::forLinear(rowCount, innerCount, outputWidth);
    shape.residual = residual;

    program.dispatch(pass, shape);
}

template <WeightStorage weightStorage>
void dispatchProduct(SplitLinearProgram<weightStorage>& program,
                     ComputePass& pass,
                     const Buffer& input,
                     const Buffer& weight,
                     const Buffer& bias,
                     const BufferRange& target,
                     int innerCount,
                     int outputWidth,
                     int rowCount,
                     bool residual)
{
    program.input = input;
    program.weights = weight;
    program.bias = bias;
    program.output = target;
    program.innerCount = (std::uint32_t) innerCount;
    program.outputWidth = (std::uint32_t) outputWidth;
    program.rowCount = (std::uint32_t) rowCount;
    program.gelu = 0u;
    program.residual = residual ? 1u : 0u;

    program.dispatch(pass, outputWidth, rowCount);
}
} // namespace

Decoder::Decoder(const DecoderShape& shapeToUse)
    : decoderShape(shapeToUse)
{
}

// Two capacities, and every buffer takes the one it belongs to: the caches are
// the window's, and every per-step intermediate is the step capacity's, so a
// prompt of several tokens and a single token after it are the same buffers at
// different dispatch heights.
//
// **The step capacity is the number to watch.** The gated feed-forward's two
// intermediates are [stepRowCapacity, 2 * intermediate] and
// [stepRowCapacity, intermediate], which at Gemma 2B's 16384 intermediate is
// 1.07 GB and 537 MB if a step is allowed the whole 8192-position window — more
// than the weights of the layer that reads them, for buffers a decode step uses
// one row of. A generation loop sets maxStepRows to its prompt capacity and
// pays for that many rows instead.
void Decoder::prepare(Device& device)
{
    decoderShape.validate();

    embedding.prepare(device);
    normalisation.prepare(device);
    product.prepare(device);
    packedProduct.prepare(device);
    splitProjection.prepare(device);
    packedSplitProjection.prepare(device);
    splitLogits.prepare(device);
    packedSplitLogits.prepare(device);
    rotation.prepare(device);
    gating.prepare(device);
    prefillAttention.prepare(device);
    decodeAttention.prepare(device);

    const auto stepElements = decoderShape.stepElementCount();

    hidden.emplace(allocate(device, stepElements));
    normalised.emplace(allocate(device, stepElements));
    normalisedRows.emplace(allocate(device, stepElements));

    const auto queryElements = decoderShape.stepQueryElementCount();

    queries.emplace(allocate(device, queryElements));
    attended.emplace(allocate(device, queryElements));

    fused.emplace(allocate(device, decoderShape.fusedElementCount()));
    activated.emplace(allocate(device, decoderShape.activatedElementCount()));

    const auto layers = decoderShape.layers;
    const auto cacheElements = decoderShape.cacheElementCount();

    allocatePerLayer(keyCache, device, layers, cacheElements);
    allocatePerLayer(valueCache, device, layers, cacheElements);

    const auto table = makeRotaryTable(
        decoderShape.maxPositions, decoderShape.headWidth, decoderShape.ropeTheta);

    rotaryCosines.emplace(upload(device, table.cosines));
    rotarySines.emplace(upload(device, table.sines));

    zeroQueryBias.emplace(allocateZeroed(device, decoderShape.queryWidth()));
    zeroKeyValueBias.emplace(allocateZeroed(device, decoderShape.kvWidth()));
    zeroWidthBias.emplace(allocateZeroed(device, decoderShape.width));
    zeroFusedBias.emplace(allocateZeroed(device, 2 * decoderShape.intermediate));
    zeroLogitBias.emplace(allocateZeroed(device, decoderShape.vocabularySize));

    decodedPositions = 0;
}

void Decoder::prepare()
{
    prepare(Device::shared());
}

void Decoder::requireMatchingWeights(const DecoderWeights& weights) const
{
    if (!(weights.shape == decoderShape))
        throw ModelError {"the weights were loaded against a different decoder "
                          "shape than this decoder was built for"};
}

void Decoder::requirePrepared() const
{
    if (!hidden)
        throw ModelError {"this decoder has not been prepared: prepare() is what "
                          "compiles its kernels and sizes its caches"};
}

void Decoder::encodeNorm(ComputePass& pass,
                         const Buffer& input,
                         const TensorBuffer& weight,
                         const Buffer& target,
                         int rowCount)
{
    normalisation.input = input;
    normalisation.weight = weight.buffer;
    normalisation.output = target;
    normalisation.rowLength = (std::uint32_t) decoderShape.width;
    normalisation.epsilon = decoderShape.rmsNormEpsilon;

    normalisation.dispatchRows(pass, rowCount);
}

// The one place a weight's storage decides anything: a tensor the loader left
// packed goes to the program that reads packed halves, and one it uploaded as
// floats to the program that subscripts floats. Neither can read the other's
// buffer, which is why the choice is made from the buffer rather than from a
// build-time switch.
//
// The row count decides the other axis. The split form gives every lane of a
// group a share of one output's inner sum, which is what a one-row decode step
// wants and what a prompt cannot afford: it reads the whole weight matrix once
// per row, where the tiled product reads it once per tile of sixty-four.
void Decoder::encodeProduct(ComputePass& pass,
                            const Buffer& input,
                            const TensorBuffer& weight,
                            const Buffer& bias,
                            const BufferRange& target,
                            int innerCount,
                            int outputWidth,
                            int rowCount,
                            bool residual,
                            ProductRole role)
{
    const auto fewRows = rowCount <= splitRowLimit;
    const auto logits = role == ProductRole::Logits;

    if (weight.isPackedHalf() && fewRows)
        dispatchProduct(logits ? packedSplitLogits : packedSplitProjection,
                        pass,
                        input,
                        weight.buffer,
                        bias,
                        target,
                        innerCount,
                        outputWidth,
                        rowCount,
                        residual);
    else if (weight.isPackedHalf())
        dispatchProduct(packedProduct,
                        pass,
                        input,
                        weight.buffer,
                        bias,
                        target,
                        innerCount,
                        outputWidth,
                        rowCount,
                        residual);
    else if (fewRows)
        dispatchProduct(logits ? splitLogits : splitProjection,
                        pass,
                        input,
                        weight.buffer,
                        bias,
                        target,
                        innerCount,
                        outputWidth,
                        rowCount,
                        residual);
    else
        dispatchProduct(product,
                        pass,
                        input,
                        weight.buffer,
                        bias,
                        target,
                        innerCount,
                        outputWidth,
                        rowCount,
                        residual);
}

// Appending to the cache is a bind, not a copy: the key and value projections
// write straight into the rows this step owns, at position * kvWidth floats
// into the layer's buffer, and the attention that follows reads the whole cache
// from row zero.
//
// The range's byte count is not a bound — neither backend can make it one — so
// the row must be inside the buffer before it is bound, and a range at or past
// the end binds nothing at all, silently. step() is where that is checked, once
// for the whole step, before anything is recorded.
BufferRange Decoder::cacheRowsAt(const Buffer& cache, int tokenCount) const
{
    const auto kvWidth = decoderShape.kvWidth();

    return {&cache,
            floatBytes(decodedPositions * kvWidth),
            floatBytes(tokenCount * kvWidth)};
}

// In place on the range it is given, which is what lets the keys be rotated
// where they already lie in the cache: a thread owns both halves of one pair
// and reads both before it stores either, so no other thread touches what it
// wrote. Row r of the range carries position decodedPositions + r, which is
// where the projection above put it.
void Decoder::encodeRotation(ComputePass& pass,
                             const BufferRange& target,
                             int rowStride,
                             int headCount,
                             int rowCount)
{
    rotation.input = target;
    rotation.cosines = *rotaryCosines;
    rotation.sines = *rotarySines;
    rotation.output = target;
    rotation.rowStride = (std::uint32_t) rowStride;
    rotation.headDim = (std::uint32_t) decoderShape.headWidth;
    rotation.headCount = (std::uint32_t) headCount;
    rotation.firstPosition = (std::uint32_t) decodedPositions;

    rotation.dispatch(pass, rowCount, headCount, decoderShape.headWidth);
}

// The queries, the keys and the values, then the rotation on the first two, and
// the attention over every key cached so far.
//
// Two kernels, one per shape the attention runs in. A prompt of several rows
// masks causally by giving row r a key count of decodedPositions + r + 1, so a
// later position is never read rather than read and suppressed; a decode step's
// one row stands at the last position, where nothing is masked at all.
void Decoder::encodeAttention(ComputePass& pass,
                              const DecoderLayerWeights& weights,
                              int layerIndex,
                              int tokenCount)
{
    const auto width = decoderShape.width;
    const auto queryWidth = decoderShape.queryWidth();
    const auto kvWidth = decoderShape.kvWidth();

    const auto keyRows = cacheRowsAt(keyCache[layerIndex], tokenCount);
    const auto valueRows = cacheRowsAt(valueCache[layerIndex], tokenCount);

    encodeProduct(pass,
                  *normalised,
                  weights.query,
                  *zeroQueryBias,
                  BufferRange::of(*queries),
                  width,
                  queryWidth,
                  tokenCount);

    encodeProduct(pass,
                  *normalised,
                  weights.key,
                  *zeroKeyValueBias,
                  keyRows,
                  width,
                  kvWidth,
                  tokenCount);

    encodeProduct(pass,
                  *normalised,
                  weights.value,
                  *zeroKeyValueBias,
                  valueRows,
                  width,
                  kvWidth,
                  tokenCount);

    // The three products above read the same normalised rows and write three
    // buffers nothing else has touched — the queries and this step's rows of
    // the two caches — so they are free to overlap. The rotations read the
    // first two.
    pass.barrier();

    encodeRotation(
        pass, BufferRange::of(*queries), queryWidth, decoderShape.heads, tokenCount);

    encodeRotation(pass, keyRows, kvWidth, decoderShape.kvHeads, tokenCount);

    pass.barrier();

    const auto attentionShape = decoderShape.attentionShape();
    const auto scale = decoderShape.attentionScale();

    if (tokenCount == 1)
    {
        decodeAttention.queries = *queries;
        decodeAttention.keys = keyCache[layerIndex];
        decodeAttention.values = valueCache[layerIndex];
        decodeAttention.output = *attended;

        decodeAttention.dispatch(pass, attentionShape, decodedPositions + 1, scale);

        return;
    }

    prefillAttention.queries = *queries;
    prefillAttention.keys = keyCache[layerIndex];
    prefillAttention.values = valueCache[layerIndex];
    prefillAttention.output = *attended;

    prefillAttention.dispatch(
        pass, attentionShape, tokenCount, decodedPositions, scale);
}

// Pre-norm, which is what Gemma is: each of the two sublayers normalises what
// it reads and adds what it computed to what it was given, so the residual is
// the unnormalised input. The stream is one buffer, added to in place by each
// sublayer's last product, so the layer ends where it started and the next one
// needs to know nothing about the order.
void Decoder::encodeLayer(ComputePass& pass,
                          const DecoderLayerWeights& weights,
                          int layerIndex,
                          int tokenCount)
{
    const auto width = decoderShape.width;
    const auto intermediate = decoderShape.intermediate;

    encodeNorm(pass, *hidden, weights.inputNorm, *normalised, tokenCount);

    pass.barrier();

    encodeAttention(pass, weights, layerIndex, tokenCount);

    pass.barrier();

    encodeProduct(pass,
                  *attended,
                  weights.output,
                  *zeroWidthBias,
                  BufferRange::of(*hidden),
                  decoderShape.queryWidth(),
                  width,
                  tokenCount,
                  true);

    pass.barrier();

    encodeNorm(pass, *hidden, weights.postAttentionNorm, *normalised, tokenCount);

    pass.barrier();

    // One product over the concatenated gate and up weights, so a row of the
    // result holds gate in its first half and up in its second — which is the
    // layout GeGLU folds without a second uniform to say so.
    encodeProduct(pass,
                  *normalised,
                  weights.fusedGateUp,
                  *zeroFusedBias,
                  BufferRange::of(*fused),
                  width,
                  2 * intermediate,
                  tokenCount);

    pass.barrier();

    gating.fused = *fused;
    gating.output = *activated;
    gating.intermediate = (std::uint32_t) intermediate;

    pass.dispatch(gating, intermediate, tokenCount);

    pass.barrier();

    encodeProduct(pass,
                  *activated,
                  weights.down,
                  *zeroWidthBias,
                  BufferRange::of(*hidden),
                  intermediate,
                  width,
                  tokenCount,
                  true);
}

void Decoder::step(ComputePass& pass,
                   const BufferRange& tokens,
                   int tokenCount,
                   const DecoderWeights& weights,
                   const Buffer& logits)
{
    requirePrepared();
    requireMatchingWeights(weights);

    if (tokenCount <= 0)
        throw ModelError {"a decoder step needs at least one token"};

    // The per-step intermediates hold this many rows and no more, and a
    // dispatch taller than the buffer it writes runs off its end silently on
    // both backends. Checked before the window below, since a block wider than
    // the step capacity is a caller feeding a prompt it should have split.
    if (tokenCount > decoderShape.stepRowCapacity())
        throw ModelError {"this step carries " + std::to_string(tokenCount)
                          + " rows and this decoder's intermediates hold "
                          + std::to_string(decoderShape.stepRowCapacity())
                          + "; a longer prompt is fed in blocks"};

    // Before a single dispatch: a step that ran off the end of the window would
    // bind a cache row outside its buffer, and a range past a buffer's end
    // binds nothing at all rather than failing.
    if (decodedPositions + tokenCount > decoderShape.maxPositions)
        throw ModelError {"this step would carry the sequence to "
                          + std::to_string(decodedPositions + tokenCount)
                          + " tokens, past the "
                          + std::to_string(decoderShape.maxPositions)
                          + " this decoder was built for"};

    embedding.tokens = tokens;
    embedding.tokenTable = weights.tokenEmbedding.buffer;
    embedding.output = *hidden;
    embedding.width = (std::uint32_t) decoderShape.width;
    embedding.scale = decoderShape.embeddingScale;

    pass.dispatch(embedding, decoderShape.width, tokenCount);

    for (auto index = 0; index < decoderShape.layers; ++index)
    {
        pass.barrier();
        encodeLayer(pass, weights.layers[index], index, tokenCount);
    }

    pass.barrier();

    encodeNorm(pass, *hidden, weights.finalNorm, *normalisedRows, tokenCount);

    pass.barrier();

    // The logits projection is the embedding itself — config.json ties them, so
    // the checkpoint carries no lm_head and no bias, and what this binds where
    // a bias would go is the vocabulary-wide zero buffer.
    encodeProduct(pass,
                  *normalisedRows,
                  weights.tokenEmbedding,
                  *zeroLogitBias,
                  BufferRange::of(logits),
                  decoderShape.width,
                  decoderShape.vocabularySize,
                  tokenCount,
                  false,
                  ProductRole::Logits);

    decodedPositions += tokenCount;
}
} // namespace HF

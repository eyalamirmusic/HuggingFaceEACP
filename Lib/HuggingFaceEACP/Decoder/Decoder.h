#pragma once

// The module's class and, through what it includes, its umbrella: DecoderShape
// and DecoderWeights come with it, the way Tokenizer.h brings its own pieces.
// Model/ has a Model.h separate from its classes because it has no one class;
// this has one.

#include <HuggingFaceEACP/Decoder/DecoderWeights.h>

#include <optional>

namespace HF
{
// HuggingFace's GemmaForCausalLM.forward with a KV cache, recorded into a
// compute pass the caller opened:
//
//   h = embeddingScale * embed_tokens[token]
//   per layer: n = rmsnorm(h, input_layernorm)
//              q, k, v = n Wqᵀ, n Wkᵀ, n Wvᵀ
//              rope(q), rope(k) at position .. position + rows - 1
//              k and v appended to this layer's cache at those rows
//              h += attention(q over cache rows [0, position + rows)) Woᵀ
//              n = rmsnorm(h, post_attention_layernorm)
//              h += (gelu_tanh(n Wgateᵀ) * (n Wupᵀ)) Wdownᵀ
//   h = rmsnorm(h, model.norm)
//   logits = h · embed_tokensᵀ
//
// A run is a beginSequence and then steps. beginSequence puts the position back
// to zero and allocates nothing — the caches are the shape's capacity and a
// shorter sequence fills a prefix of them. A step appends however many tokens
// it is given and writes their logits, so the prompt a run opens with is one
// call and each token after it is another.
//
// prepare() compiles every kernel, sizes every intermediate and every cache,
// and uploads the rotary tables and the zero biases; the recording calls only
// record. A step is one pass rather than one pass per dispatch: at a few
// microseconds of GPU each, ninety passes would be most of a step, and the KV
// cache is read in the same pass that appended to it.
//
// The ordering inside that pass is spelled out rather than assumed: a
// pass.barrier() sits at every boundary where a stage reads what the stage
// before it wrote, which costs nothing in a serial pass and is the whole
// ordering in a concurrent one. **One place has no barrier and needs none** —
// a layer's three projections read the same normalised rows and write three
// buffers nothing else has touched, the queries and this step's rows of the two
// caches, so they are free to overlap.
//
// One program of each kind serves every layer: the shapes are uniforms, so
// eighteen layers of two norms, four projections and an attention are
// re-bindings of a handful of pipelines rather than pipelines of their own. The
// products are the exception, and are held six ways over three questions: a
// float-weight program and a packed-half one, so a weight the loader left fp16
// is dispatched through the program that reads fp16; the many-row tiled form a
// prompt takes against the few-row split form a decode step takes; and, inside
// that split form, the split count the shape wants — see stepSplitCount and
// logitsSplitCount below.
//
// Argmax is deliberately not here. Greedy sampling is the layer above: which
// tokens are suppressed at which step is generation config, and a decoder that
// sampled would have to hold it.
class Decoder
{
public:
    explicit Decoder(const DecoderShape& shapeToUse);

    void prepare(eacp::GPU::Device& device);
    void prepare();

    const DecoderShape& shape() const { return decoderShape; }

    // Opens a sequence: the position goes back to zero, so the next step is the
    // sequence's first and writes the caches from row zero rather than
    // appending to what the last run left. Nothing is reallocated.
    void beginSequence() { decodedPositions = 0; }

    // Appends tokenCount tokens to the sequence and writes their logits.
    //
    // tokens holds tokenCount uint32 ids, a range so that the slot an Argmax
    // wrote at the end of one step is what the next step embeds, with the host
    // nowhere in between; logits receives
    // tokenCount * shape().logitElementCount() floats, [tokenCount,
    // vocabularySize] row-major. The hidden rows behind them stay readable
    // through hiddenStates() until the next step.
    //
    // A step of more rows than shape().stepRowCapacity(), one that would carry
    // the sequence past maxPositions, one against weights loaded for a
    // different shape, and one on a decoder nothing has prepared are each a
    // ModelError thrown before a single dispatch is recorded — a half-recorded
    // step would leave the caches holding rows the position count does not know
    // about.
    void step(eacp::GPU::ComputePass& pass,
              const eacp::GPU::BufferRange& tokens,
              int tokenCount,
              const DecoderWeights& weights,
              const eacp::GPU::Buffer& logits);

    void step(eacp::GPU::ComputePass& pass,
              const eacp::GPU::Buffer& tokens,
              int tokenCount,
              const DecoderWeights& weights,
              const eacp::GPU::Buffer& logits)
    {
        step(pass, eacp::GPU::BufferRange::of(tokens), tokenCount, weights, logits);
    }

    // How many tokens the sequence holds, which is the position the next one
    // takes and the number of keys already cached.
    int position() const { return decodedPositions; }

    // The last step's rows after the final norm and before the tied projection
    // — [tokenCount, width] row-major, valid until the next step overwrites it.
    // Exposed so a comparison that disagrees bisects to before or after the
    // logits rather than only reporting a wrong vocabulary row.
    const eacp::GPU::Buffer& hiddenStates() const { return *normalisedRows; }

private:
    // Which of the two split counts a product takes, since the two shapes a
    // step projects at are opposite and the row count alone cannot tell them
    // apart in a small test model.
    enum class ProductRole
    {
        Projection,
        Logits
    };

    // Up to how many rows a product splits its inner sum across a group rather
    // than tiling. The split form reads the whole weight matrix once per row
    // where the tiled product reads it once per 64-row tile, so past a handful
    // of rows the extra weight traffic is the whole cost of a bandwidth-bound
    // step — which puts the crossover low. A decode step is one row; three is
    // a speculative triple.
    static constexpr auto splitRowLimit = 3;

    // How many lanes share one output's inner sum in a step's projections.
    // Unmeasured on Gemma's widths: WhisperEACP measured 96 over a 384-wide
    // row, and 2048 wide with 16384 outputs is a different curve. plan.md's
    // fifth step is where both of these get a number rather than a guess.
    static constexpr auto stepSplitCount = 64;

    // The logits projection wants fewer. Its 256,000 outputs fill the device at
    // any split count, so the only thing more lanes buy is a longer fold —
    // WhisperEACP measured 32 the floor of that curve over a 51,864-wide row.
    static constexpr auto logitsSplitCount = 32;

    // A decode step's norms are one row of 2048 with nothing else on the
    // machine, so what is wanted is the widest group that still has work for
    // every lane: 256 is eight elements a lane. Unmeasured here too, and the
    // count is a parameter for the reason RMSNorm.h gives — a prompt step's
    // rows fill the machine on their own, where a wider group only adds
    // barriers.
    static constexpr auto normLanes = 256;

    void requireMatchingWeights(const DecoderWeights& weights) const;
    void requirePrepared() const;

    void encodeLayer(eacp::GPU::ComputePass& pass,
                     const DecoderLayerWeights& weights,
                     int layerIndex,
                     int tokenCount);

    void encodeAttention(eacp::GPU::ComputePass& pass,
                         const DecoderLayerWeights& weights,
                         int layerIndex,
                         int tokenCount);

    void encodeRotation(eacp::GPU::ComputePass& pass,
                        const eacp::GPU::BufferRange& target,
                        int rowStride,
                        int headCount,
                        int rowCount);

    void encodeNorm(eacp::GPU::ComputePass& pass,
                    const eacp::GPU::Buffer& input,
                    const TensorBuffer& weight,
                    const eacp::GPU::Buffer& target,
                    int rowCount);

    // residual adds the result into what the target already holds, which is
    // what follows o_proj and down_proj. The GELU fold the same programs carry
    // is deliberately never asked for here: Gemma's feed-forward is gated, and
    // GeGLU is where its activation happens.
    void encodeProduct(eacp::GPU::ComputePass& pass,
                       const eacp::GPU::Buffer& input,
                       const TensorBuffer& weight,
                       const eacp::GPU::Buffer& bias,
                       const eacp::GPU::BufferRange& target,
                       int innerCount,
                       int outputWidth,
                       int rowCount,
                       bool residual = false,
                       ProductRole role = ProductRole::Projection);

    // Where this step's keys and values are written into a layer's cache: the
    // row the sequence has reached, as a byte offset into the buffer.
    eacp::GPU::BufferRange cacheRowsAt(const eacp::GPU::Buffer& cache,
                                       int tokenCount) const;

    DecoderShape decoderShape;
    int decodedPositions = 0;

    Embed embedding;
    RMSNorm normalisation {normLanes};
    LinearProduct product;
    HalfWeightLinearProduct packedProduct;
    SplitLinear splitProjection {stepSplitCount};
    HalfWeightSplitLinear packedSplitProjection {stepSplitCount};
    SplitLinear splitLogits {logitsSplitCount};
    HalfWeightSplitLinear packedSplitLogits {logitsSplitCount};
    RoPE rotation;
    GeGLU gating;
    MultiQueryPrefillAttention prefillAttention;
    MultiQueryDecodeAttention decodeAttention;

    // The residual stream, which two sublayers add to in place from their last
    // product's store: a layer enters and leaves in hidden, so the next layer
    // reads what this one wrote without a swap the call site would have to keep
    // track of.
    std::optional<eacp::GPU::Buffer> hidden;

    std::optional<eacp::GPU::Buffer> normalised;
    std::optional<eacp::GPU::Buffer> queries;
    std::optional<eacp::GPU::Buffer> attended;
    std::optional<eacp::GPU::Buffer> fused;
    std::optional<eacp::GPU::Buffer> activated;
    std::optional<eacp::GPU::Buffer> normalisedRows;

    // One pair per layer, each grown a row per token and read from row zero by
    // every step.
    Vector<eacp::GPU::Buffer> keyCache;
    Vector<eacp::GPU::Buffer> valueCache;

    // [maxPositions, headWidth / 2] each, built in double precision on the CPU
    // and uploaded once — see RoPE.h on why they are not computed per thread.
    std::optional<eacp::GPU::Buffer> rotaryCosines;
    std::optional<eacp::GPU::Buffer> rotarySines;

    // What every product binds where a model with biases would bind one. Gemma
    // has none anywhere, and a program reads outputWidth of them, so there is
    // one zero buffer per output width a step projects at. Filled once, because
    // neither backend defines what a shader reading an unbound buffer gets and
    // a flag would only guard a read that must not happen at all.
    std::optional<eacp::GPU::Buffer> zeroQueryBias;
    std::optional<eacp::GPU::Buffer> zeroKeyValueBias;
    std::optional<eacp::GPU::Buffer> zeroWidthBias;
    std::optional<eacp::GPU::Buffer> zeroFusedBias;
    std::optional<eacp::GPU::Buffer> zeroLogitBias;
};
} // namespace HF

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
// prepare() compiles every kernel a checkpoint's weights will be dispatched
// through, sizes every intermediate and every cache, and uploads the rotary
// tables and the zero biases; the recording calls only record. A step is one
// pass rather than one pass per dispatch: at a few
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
// products are the exception, and are declared twelve ways over three
// questions: the weight's storage, since a checkpoint may ship F32, fp16 or
// bf16 and the loader may have quantized it to int8 blocks, and nothing about a
// GPU buffer says which; the many-row tiled form a prompt takes against the
// few-row split form a decode step takes; and, inside that split form, the
// split count the shape wants — see stepSplitCount and logitsSplitCount below.
// The embedding gather is declared four ways for the first of those reasons
// alone.
//
// **Only one storage's worth is built.** A checkpoint is in one storage, so
// prepare() takes the weights and compiles the four programs they name; the
// other twelve stay empty optionals. Compiling all sixteen built three whole
// SIMD-group matrix pipelines that nothing would ever dispatch.
//
// Argmax is deliberately not here. Greedy sampling is the layer above: which
// tokens are suppressed at which step is generation config, and a decoder that
// sampled would have to hold it.
class Decoder
{
public:
    explicit Decoder(const DecoderShape& shapeToUse);

    // Compiles the kernels, sizes every intermediate and every cache, and
    // uploads the rotary tables and the zero biases.
    //
    // **The weights are an argument because the products are not one program.**
    // Each of them is held per WeightStorage, and a checkpoint is in one — so
    // compiling all four would build three whole SIMD-group matrix pipelines
    // and three gathers a run never dispatches. What is compiled is
    // weights.storages() and nothing else, and a step against weights in a
    // storage this was not prepared for is a ModelError thrown before anything
    // is recorded.
    //
    // Only the storages are read here, not the shape: a decoder may be prepared
    // against one set of weights and stepped against another of the same
    // storage, and step() is where a shape that disagrees is refused.
    void prepare(eacp::GPU::Device& device, const DecoderWeights& weights);
    void prepare(const DecoderWeights& weights);

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
    // **Measured on Gemma's widths**, which the 64 this started at was not: at
    // records of four weights a lane, over a 6-token prompt and 64 decode
    // tokens in Release, 32 and 64 both gave 83.6 tokens/s, 96 gave 84.1, 128
    // gave 84.6 and 256 gave 85.3, and past that the curve fell off a cliff —
    // 384 at 77.4, 512 at 61.6, 1024 at 27.3.
    //
    // **The record went to sixteen weights and the peak moved with it**, which
    // is the whole reason this is 128 rather than 256. A row of 2048 is 128
    // records of sixteen, so 128 lanes is exactly one apiece: measured on the
    // quantized path, 64 lanes give 139.7 decode tokens/s over 64 tokens, 96
    // give 139.7, 128 give 141.3, 160 give 141.6, 192 give 137.3 and 256 give
    // 122.8, and the packed path is flat across the same range at 85.4 to 87.0.
    // 160 and 128 are a tie to within the run-to-run noise; 128 is the one of
    // them with a reason behind it rather than a peak in it.
    static constexpr auto stepSplitCount = 128;

    // The logits projection wants fewer, and the measurement agrees with the
    // number WhisperEACP arrived at over a 51,864-wide row. Its 256,000 outputs
    // fill the device at any split count, so the only thing more lanes buy is a
    // longer fold: at stepSplitCount 256 the same run gives 85.5 tokens/s at 16
    // lanes, 85.3 at 32, 85.0 at 64, 84.1 at 128 and 84.4 at 256. The top of
    // that curve is flat to within the noise, and 32 is the one point on it
    // where a group is exactly one SIMD group per output — the simdSum fold
    // Linear.h describes, with neither scratch nor a barrier — so it is kept
    // for a reason measurement supports rather than supplies. Re-measured at
    // sixteen-wide records and stepSplitCount 128, and it is flat there too:
    // 141.3 quantized decode tokens/s over 64 at 16 lanes, 141.3 at 32, 139.7
    // at 64.
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

    // Whether the four programs a storage needs were built, and the refusal
    // step() raises when a checkpoint's are not. Both are asked before a step
    // records anything, so a dispatch below can dereference its optional.
    bool isPrepared(WeightStorage storage) const;
    void requirePreparedStorages(const DecoderWeights& weights) const;

    void prepareStorage(eacp::GPU::Device& device, WeightStorage storage);

    void encodeEmbed(eacp::GPU::ComputePass& pass,
                     const eacp::GPU::BufferRange& tokens,
                     const TensorBuffer& table,
                     int tokenCount);

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

    RMSNorm normalisation {normLanes};
    RoPE rotation;
    GeGLU gating;
    MultiQueryPrefillAttention prefillAttention;
    MultiQueryDecodeAttention decodeAttention;

    // The four programs a weight storage needs — the gather, the many-row tiled
    // product, and the split product at each of the two split counts — held
    // empty until prepare() is given a checkpoint that asks for them. A run
    // builds one row of this and leaves the other twelve pipelines uncompiled.
    std::optional<Embed> embedding;
    std::optional<LinearProduct> product;
    std::optional<SplitLinear> splitProjection;
    std::optional<SplitLinear> splitLogits;

    std::optional<HalfWeightEmbed> halfEmbedding;
    std::optional<HalfWeightLinearProduct> halfProduct;
    std::optional<HalfWeightSplitLinear> halfSplitProjection;
    std::optional<HalfWeightSplitLinear> halfSplitLogits;

    std::optional<BFloat16WeightEmbed> bfloatEmbedding;
    std::optional<BFloat16WeightLinearProduct> bfloatProduct;
    std::optional<BFloat16WeightSplitLinear> bfloatSplitProjection;
    std::optional<BFloat16WeightSplitLinear> bfloatSplitLogits;

    std::optional<Int8WeightEmbed> int8Embedding;
    std::optional<Int8WeightLinearProduct> int8Product;
    std::optional<Int8WeightSplitLinear> int8SplitProjection;
    std::optional<Int8WeightSplitLinear> int8SplitLogits;

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

#pragma once

#include <HuggingFaceEACP/Decoder/DecoderShape.h>
#include <HuggingFaceEACP/Model/TensorLoader.h>

namespace HF
{
// Which of the four product programs a buffer is readable by. The loader
// decided it when it uploaded the tensor, and nothing about a GPU::Buffer
// carries it, which is why TensorBuffer holds the buffer and its storage
// together.
WeightStorage weightStorageOf(const TensorBuffer& weight);

// One decoder block's tensors, under the names google/gemma-2b ships them —
// model.layers.<index>.self_attn.q_proj.weight and the rest, spelled once in
// Model/GemmaTensors.h so that nothing here is a convention that could drift
// from the repo it loads.
//
// A block is two pre-normed sublayers: multi-query attention over the tokens
// decoded so far, then the gated feed-forward. Gemma has no bias anywhere, so
// there is no tensor here that Whisper's block would have had a partner for.
//
// **gate_proj and up_proj arrive as one buffer.** The checkpoint ships them
// separately, both [intermediate, width], and they are concatenated at load
// into a single [2 * intermediate, width] weight with the gate rows first. The
// feed-forward is then one product writing [rows, 2 * intermediate], one GeGLU
// dispatch folding a row's two halves into [rows, intermediate], and the down
// product with the residual on its store — where two products would be two
// dispatches over the same activation rows for the same arithmetic, and would
// leave GeGLU.h's secondary form to read rather than its primary one. plan.md's
// fifth step goes one further and folds the multiply into the down product's
// operand read; this is the shape that step starts from.
//
// **Storage.** Every weight a product kernel reads arrives as the checkpoint
// ships it, packed or not: Gemma's own are BF16, a converted repo's may be
// fp16, and the product kernels have a variant per storage that the Decoder
// picks by TensorBuffer::storage. That includes the fused gate-and-up weight,
// which is stacked as raw bytes rather than concatenated through readFloats, so
// a packed pair stays packed — see TensorLoader::loadStackedProjectionWeights.
//
// **Or quantized, when the caller asks.** WeightPrecision::Int8Blocks makes
// every one of those tensors on the way up instead of taking it as it lies:
// thirty-two elements to a block, one fp16 scale each, 1.0625 bytes an element
// against bf16's two. The precision is the caller's because it is a trade
// rather than a fact about the checkpoint — bytes per token against a little
// accuracy per element — and AsShipped stays the default so nothing changes for
// a caller that has not asked.
//
// **The two norm scales are the exception, and are widened to F32.** They are
// [width] each, 8 kB a layer, and the alternative is a packed read in RMSNorm
// for a tensor whose bytes are a rounding error against the projections beside
// it. loadFloatTensor is what widens them.
struct DecoderLayerWeights
{
    DecoderLayerWeights(const ShardedTensors& file,
                        const DecoderShape& shape,
                        int index,
                        WeightPrecision precision = WeightPrecision::AsShipped);

    TensorBuffer inputNorm;
    TensorBuffer query;
    TensorBuffer key;
    TensorBuffer value;
    TensorBuffer output;
    TensorBuffer fusedGateUp;
    TensorBuffer down;
    TensorBuffer postAttentionNorm;
};

// Every decoder tensor of a checkpoint, uploaded and checked against the shape
// the config implies. A name the file does not carry, a rank that is not the
// tensor's, and a dimension that disagrees are each a ModelError naming the
// tensor and the layer — the alternative is a kernel walking a buffer at a
// stride it does not have, which is silent on both backends.
//
// **There is no lm_head.weight.** config.json's tie_word_embeddings is true for
// google/gemma-2b, so the logits projection is the input embedding itself:
// `logits = h · embed_tokensᵀ`, reading the same matrix the gather read. A repo
// that shipped an untied head would be a different architecture rather than a
// variant of this one.
//
// embed_tokens is therefore the tensor bound twice, and both readers take it in
// whatever the checkpoint stored: EmbedProgram and the products are each
// parameterised by WeightStorage, and both widen a packed element through the
// same call. At Gemma's [256000, 2048] that is 1.05 GB bound twice rather than
// 2.10 GB — the largest single saving the packed path makes, and what kept the
// gather from forcing the whole embedding to be widened for the sake of one
// subscript.
struct DecoderWeights
{
    DecoderWeights(const ShardedTensors& file,
                   const DecoderShape& shapeToUse,
                   WeightPrecision precision = WeightPrecision::AsShipped);

    // Every distinct storage the weights above a product or the gather reads
    // are in, which for a real checkpoint is one entry. Decoder::prepare
    // compiles the programs these name and no others — four storages times a
    // tiled product, two split products and a gather is sixteen pipelines, of
    // which a run dispatches four.
    //
    // The norm scales are not in here. They are widened to F32 on the way up
    // and bound to RMSNorm, which has no storage to pick.
    Vector<WeightStorage> storages() const;

    DecoderShape shape;

    // [vocabularySize, width]: the gather's table and the logits projection's
    // weight, which is what ties them.
    TensorBuffer tokenEmbedding;

    // model.norm.weight, the scale of the one normalisation between the last
    // layer and the logits.
    TensorBuffer finalNorm;

    Vector<DecoderLayerWeights> layers;
};
} // namespace HF

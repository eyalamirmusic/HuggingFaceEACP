#pragma once

#include <HuggingFaceEACP/Decoder/DecoderShape.h>
#include <HuggingFaceEACP/Model/ShardedTensors.h>

namespace HF
{
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
// **Storage.** Every weight here reaches the device in the storage the
// checkpoint ships it in: Gemma's own BF16, or the fp16 a converted repo may
// carry, both two values to a word and read there by readBFloat16 or readHalf.
// The products and the embedding gather have a variant per storage and the
// Decoder picks it by TensorBuffer::storage, since nothing about a GPU buffer
// says which one it holds. The norm scales are the exception — RMSNorm
// subscripts floats and has no packed read — so loadFloatTensor widens those,
// which is a row of a weight rather than a matrix.
//
// The fused gate-and-up weight is stacked out of the two tensors' raw bytes
// when both are packed in the same storage, so the largest weight in the layer
// stays packed through the concatenation rather than being widened to be
// joined. A 16-bit pair shares a word, so that needs the gate half to be a
// whole number of words — which every real shape is, the intermediate width
// and the model width both being even — and a half that is not falls back to
// widening both.
struct DecoderLayerWeights
{
    DecoderLayerWeights(const ShardedTensors& file,
                        const DecoderShape& shape,
                        int index);

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
// embed_tokens is therefore the tensor bound twice — as the gather's table and
// as the logits weight — and both readers take it in whatever storage it
// arrived in, so the one buffer serves both and the tie stays exact rather than
// exact up to a conversion. At Gemma's width that is 1.05 GB rather than the
// 2.10 GB a widened copy would be, which is also what keeps it inside the
// 32-bit byte count a GPU::Buffer is described by.
struct DecoderWeights
{
    DecoderWeights(const ShardedTensors& file, const DecoderShape& shapeToUse);

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

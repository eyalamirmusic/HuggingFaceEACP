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
// **Storage.** Gemma's own weights are BF16, which has no shader read yet —
// plan.md's first gap — so SafeTensors widens them and every tensor here
// arrives F32. A converted repo may ship fp16, and a projection weight is the
// one operand that may stay packed: the product kernels have a half-reading
// variant and the Decoder picks it by TensorBuffer::storage. The norm scales
// and the embedding are bound to programs with no packed read at all, so a
// packed one there would be wrong by a factor of two in every index while
// staying silent, and loadFloatTensor refuses it.
//
// The fused gate-and-up weight is the exception on both counts: the
// concatenation happens on the CPU through readFloats, so a packed pair is
// widened on the way in rather than stacked packed. That costs an fp16 repo a
// second copy of the largest weight in the layer, and costs Gemma's own BF16
// checkpoint nothing at all, since every tensor in it is widened anyway.
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
// embed_tokens is therefore the tensor bound twice, and it decides its own
// case. The gather subscripts it as floats and has no packed form; the logits
// projection could read it either way. One buffer cannot be both, so **the
// embedding must be F32** and a packed one is a ModelError naming it — which
// keeps the tie between the two exact rather than exact up to a conversion.
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

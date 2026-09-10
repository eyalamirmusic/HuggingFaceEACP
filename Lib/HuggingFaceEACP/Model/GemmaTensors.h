#pragma once

#include <HuggingFaceEACP/Core/Core.h>
#include <HuggingFaceEACP/Model/GemmaConfig.h>
#include <HuggingFaceEACP/Model/ShardedTensors.h>

#include <string>
#include <string_view>

namespace HF
{
// One weight the decoder needs: what the checkpoint calls it, and the shape
// the config says it has. HuggingFace stores a linear weight as
// [outputs, inputs], which is what every shape below is.
struct TensorEntry
{
    std::string name;
    Vector<int> shape;
};

// Every tensor google/gemma-2b ships, named and shaped from a GemmaConfig. The
// decoder is built from this rather than from string literals spread over the
// layers, and a checkpoint can be checked against it before a byte reaches the
// GPU — which is the difference between a wrong shape failing at load and a
// kernel reading a matrix at the wrong stride.
namespace GemmaTensors
{
inline constexpr auto embedding = "model.embed_tokens.weight";
inline constexpr auto finalNorm = "model.norm.weight";

inline constexpr auto queryProjection = "self_attn.q_proj.weight";
inline constexpr auto keyProjection = "self_attn.k_proj.weight";
inline constexpr auto valueProjection = "self_attn.v_proj.weight";
inline constexpr auto outputProjection = "self_attn.o_proj.weight";
inline constexpr auto gateProjection = "mlp.gate_proj.weight";
inline constexpr auto upProjection = "mlp.up_proj.weight";
inline constexpr auto downProjection = "mlp.down_proj.weight";
inline constexpr auto inputNorm = "input_layernorm.weight";
inline constexpr auto postAttentionNorm = "post_attention_layernorm.weight";

// "model.layers.3.self_attn.q_proj.weight", the one place that spelling lives.
std::string layerTensorName(int layer, std::string_view suffix);

Vector<TensorEntry> forLayer(const GemmaConfig& config, int layer);

// The embedding, every layer in order, then the final norm. Gemma ties the
// output projection to the embedding, so there is no lm_head to name.
Vector<TensorEntry> all(const GemmaConfig& config);

// Every tensor the catalogue names is in the checkpoint, with the shape the
// config implies. Throws a ModelError naming the first that is not.
void checkAgainst(const ShardedTensors& weights, const GemmaConfig& config);
} // namespace GemmaTensors
} // namespace HF

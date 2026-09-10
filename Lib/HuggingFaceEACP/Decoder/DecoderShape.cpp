#include "DecoderShape.h"

#include <HuggingFaceEACP/Model/ModelError.h>

#include <cmath>
#include <string>
#include <string_view>

namespace HF
{
namespace
{
void requirePositive(int value, std::string_view field)
{
    if (value <= 0)
        throw ModelError {"a decoder shape's '" + std::string {field}
                          + "' must be positive"};
}
} // namespace

DecoderShape DecoderShape::fromConfig(const GemmaConfig& config)
{
    auto shape = DecoderShape {};
    shape.width = config.hiddenSize;
    shape.heads = config.attentionHeads;
    shape.kvHeads = config.keyValueHeads;
    shape.headWidth = config.headWidth;
    shape.layers = config.layerCount;
    shape.intermediate = config.intermediateSize;
    shape.vocabularySize = config.vocabularySize;
    shape.maxPositions = config.maxPositions;
    shape.rmsNormEpsilon = (float) config.rmsNormEpsilon;
    shape.ropeTheta = config.ropeTheta;
    shape.embeddingScale = config.embeddingScale();

    shape.validate();

    return shape;
}

void DecoderShape::validate() const
{
    requirePositive(width, "width");
    requirePositive(heads, "heads");
    requirePositive(kvHeads, "kvHeads");
    requirePositive(headWidth, "headWidth");
    requirePositive(layers, "layers");
    requirePositive(intermediate, "intermediate");
    requirePositive(vocabularySize, "vocabularySize");
    requirePositive(maxPositions, "maxPositions");

    if (maxStepRows < 0)
        throw ModelError {"a decoder shape's 'maxStepRows' is a row count or "
                          "zero for the whole window, and this one is "
                          + std::to_string(maxStepRows)};

    // A step cannot carry more rows than the caches have room for, so a step
    // capacity above the window is a pair of numbers that cannot both be meant
    // rather than a larger allocation.
    if (maxStepRows > maxPositions)
        throw ModelError {"a decoder shape's step capacity of "
                          + std::to_string(maxStepRows) + " rows is past the "
                          + std::to_string(maxPositions)
                          + " positions its caches hold"};

    // The attention scores a head four channels at a time, and the rotation
    // pairs channel i with channel i + headWidth / 2 — so a multiple of four
    // is what both want, and it is the stronger of the two conditions.
    if (headWidth % 4 != 0)
        throw ModelError {"a decoder's head width must be a multiple of four, "
                          "since the attention reads it four channels at a time "
                          "and the rotation pairs its halves, and this one is "
                          + std::to_string(headWidth)};

    // The accumulator the attention keeps in threadgroup memory is sized when
    // the kernel is compiled, so a wider head reads past its end rather than
    // failing.
    if (headWidth > MultiQueryAttentionProgram::maxHeadDim)
        throw ModelError {
            "a decoder's head width is at most "
            + std::to_string(MultiQueryAttentionProgram::maxHeadDim)
            + ", which is what the attention's threadgroup accumulator holds, "
              "and this one is "
            + std::to_string(headWidth)};

    // Every query head reads one of the KV heads, so a count that does not
    // divide leaves heads with no key to attend to.
    if (heads % kvHeads != 0)
        throw ModelError {"a decoder's " + std::to_string(kvHeads)
                          + " kv heads do not divide its " + std::to_string(heads)
                          + " query heads"};
}

// D^-0.5, applied to the scores rather than folded into the query projection.
// Gemma 1's own: `self.scaling = self.head_dim**-0.5` in GemmaAttention, and
// the query_pre_attn_scalar that overrides it is a Gemma 2 field this
// architecture does not carry.
float DecoderShape::attentionScale() const
{
    return 1.f / std::sqrt((float) headWidth);
}
} // namespace HF

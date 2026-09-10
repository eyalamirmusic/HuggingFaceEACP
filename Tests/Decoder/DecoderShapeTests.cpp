#include "Common.h"

using namespace nano;
using namespace HF;
using namespace HF::Testing;

namespace
{
// The shape whose validation is under test, built by hand rather than from a
// config so that a field can be made impossible one at a time.
DecoderShape gemmaShape()
{
    return DecoderShape::fromConfig(GemmaConfig {});
}
} // namespace

// No device needed: these are the numbers the whole module is indexed by, and
// getting one of them wrong misplaces every dispatch after it.
auto tDecoderShapeFromConfig = test("Decoder/shapeFromGemmaConfig") = []
{
    const auto shape = gemmaShape();

    check(shape.width == 2048);
    check(shape.heads == 8);
    check(shape.kvHeads == 1);
    check(shape.headWidth == 256);
    check(shape.layers == 18);
    check(shape.intermediate == 16384);
    check(shape.vocabularySize == 256000);
    check(shape.maxPositions == 8192);

    check(shape.rmsNormEpsilon == 1.0e-6f);
    check(shape.ropeTheta == 10000.0);

    // sqrt(2048) unrounded, which is llama.cpp's over an F32 GGUF and what
    // DecoderShape.h says this deliberately is: a bf16 run would round it to
    // 45.25, and that decision belongs to the config rather than to Embed.h.
    check(shape.embeddingScale == std::sqrt(2048.f));

    // head_dim**-0.5, Gemma 1's own. query_pre_attn_scalar is a Gemma 2 field
    // this architecture does not carry.
    check(shape.attentionScale() == 1.f / 16.f);
};

auto tDecoderShapeCounts = test("Decoder/shapeCounts") = []
{
    const auto shape = smallDecoderShape();

    check(shape.queryWidth() == 32);
    check(shape.kvWidth() == 16);
    check(shape.rowElementCount() == 32);
    check(shape.queryRowElementCount() == 32);
    check(shape.logitElementCount() == 64);

    // maxStepRows is zero out of fromConfig, so every per-step intermediate is
    // sized for a prompt filling the whole window — which is what one number
    // serving both capacities used to mean everywhere.
    check(shape.maxStepRows == 0);
    check(shape.stepRowCapacity() == 16);
    check(shape.stepElementCount() == 16 * 32);
    check(shape.stepQueryElementCount() == 16 * 32);
    check(shape.fusedElementCount() == 16 * 96);
    check(shape.activatedElementCount() == 16 * 48);
    check(shape.cacheElementCount() == 16 * 16);

    const auto attention = shape.attentionShape();
    check(attention.heads == 2);
    check(attention.kvHeads == 1);
    check(attention.headDim == 16);
    check(attention.queriesPerKvHead() == 2);
};

// The step capacity, which is what keeps a generation loop from paying for a
// window's worth of intermediates: only the per-step counts move with it, and
// the caches stay the window's.
auto tDecoderShapeStepCapacity = test("Decoder/shapeStepCapacity") = []
{
    auto shape = smallDecoderShape();
    shape.maxStepRows = 4;
    shape.validate();

    check(shape.stepRowCapacity() == 4);
    check(shape.stepElementCount() == 4 * 32);
    check(shape.stepQueryElementCount() == 4 * 32);
    check(shape.fusedElementCount() == 4 * 96);
    check(shape.activatedElementCount() == 4 * 48);

    // The caches are the window's whatever a step may carry, since they hold
    // the whole sequence rather than one step of it.
    check(shape.cacheElementCount() == 16 * 16);

    // Gemma 2B's own numbers, which are what the separation is for: the fused
    // gate-and-up buffer at 512 prompt rows against the 8192 the window would
    // have asked for.
    auto gemma = gemmaShape();
    gemma.maxStepRows = 512;
    check(gemma.fusedElementCount() == 512 * 2 * 16384);
    check(gemmaShape().fusedElementCount() == 8192 * 2 * 16384);
};

// A step capacity past the window is a pair of numbers that cannot both be
// meant: the caches have no row for the step to append to.
auto tDecoderShapeRefusesStepPastWindow =
    test("Decoder/shapeRefusesStepCapacityPastTheWindow") = []
{
    auto past = smallDecoderShape();
    past.maxStepRows = past.maxPositions + 1;
    check(throwsModelError([&] { past.validate(); }));

    auto negative = smallDecoderShape();
    negative.maxStepRows = -1;
    check(throwsModelError([&] { negative.validate(); }));

    // The window itself is allowed, and is what zero already means.
    auto whole = smallDecoderShape();
    whole.maxStepRows = whole.maxPositions;
    whole.validate();
    check(whole.stepRowCapacity() == whole.maxPositions);
};

// Weights are loaded against one of these and dispatched against another, and
// nothing about a GPU buffer says which — so the comparison has to be the
// value's, not a field-by-field one at the call site.
auto tDecoderShapeEquality = test("Decoder/shapeEquality") = []
{
    const auto shape = smallDecoderShape();
    check(shape == smallDecoderShape());

    auto other = shape;
    other.kvHeads += 1;
    check(!(shape == other));

    auto scaled = shape;
    scaled.embeddingScale *= 2.f;
    check(!(shape == scaled));

    // The step capacity participates too, which is right: the weights are
    // checked by the same struct, and two decoders that differ in it hold
    // differently sized intermediates.
    auto blocked = shape;
    blocked.maxStepRows = 4;
    check(!(shape == blocked));
};

// What the kernels below cannot check for themselves, checked once where the
// shape is built: a head the attention's threadgroup accumulator cannot hold,
// a width its four-channel read cannot walk, and a head count the multi-query
// mapping cannot divide.
auto tDecoderShapeRefusesImpossibleHeads =
    test("Decoder/shapeRefusesImpossibleHeads") = []
{
    auto odd = smallDecoderShape();
    odd.headWidth = 18;
    check(throwsModelError([&] { odd.validate(); }));

    auto wide = smallDecoderShape();
    wide.headWidth = MultiQueryAttentionProgram::maxHeadDim + 4;
    check(throwsModelError([&] { wide.validate(); }));

    auto indivisible = smallDecoderShape();
    indivisible.heads = 3;
    indivisible.kvHeads = 2;
    check(throwsModelError([&] { indivisible.validate(); }));

    auto empty = smallDecoderShape();
    empty.layers = 0;
    check(throwsModelError([&] { empty.validate(); }));
};

// fromConfig validates, so a config that parses and describes a model no
// kernel here can run is refused where the shape is built rather than at the
// first dispatch.
auto tDecoderShapeFromConfigValidates = test("Decoder/shapeFromConfigValidates") = []
{
    auto config = smallGemmaConfig();
    config.headWidth = 6;

    check(throwsModelError([&] { return DecoderShape::fromConfig(config); }));
};

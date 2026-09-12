#include "TiledProduct.h"

using namespace nano;
using namespace HF;
using namespace eacp::GPU;

// The register-tiled product, against the scalar reference in TiledProduct.h.
// The SIMD-group product runs through the same checks in
// SimdTiledMatMulTests.cpp, at shapes chosen for its own tiling.

// Nothing here is a multiple of a tile, a slab or a group: a kernel that ran
// its tile past the edge, or counted a partial slab as a whole one, fails.
auto tTiledLinearOddShape = test("Kernels/tiledLinearOddShape") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkLinear<TiledLinear>(5, 7, 3, 100u);
};

// Every extent crosses a tile boundary and none lands on one, and the inner
// count is not a multiple of the slab.
auto tTiledLinearAcrossTiles = test("Kernels/tiledLinearAcrossTiles") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkLinear<TiledLinear>(70, 50, 45, 200u);
};

// A projection shape with every extent a multiple of everything.
auto tTiledLinearWholeTiles = test("Kernels/tiledLinearWholeTiles") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkLinear<TiledLinear>(96, 64, 128, 300u);
};

auto tTiledLinearPackedWeights = test("Kernels/tiledLinearPackedWeights") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkPackedLinear<HalfWeightTiledLinear>(37, 21, 35, 400u);
};

// The bf16 form, which is the one Gemma's own weights are read through on a
// backend with no SIMD-group matrix. Same reference, weights narrowed through
// eacp's host packer first so it sees what the shader widens back.
auto tTiledLinearBFloat16Weights = test("Kernels/tiledLinearBFloat16Weights") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkPackedLinear<BFloat16WeightTiledLinear>(
        37, 21, 35, 400u, TiledProduct::packedBFloat16s);

    TiledProduct::checkPackedLinear<BFloat16WeightTiledLinear>(
        70, 50, 45, 410u, TiledProduct::packedBFloat16s);
};

// The quantized form, which is what a prefill reads once the loader has
// quantized the checkpoint. Same reference, weights taken through the very
// quantizer the loader calls and back through its dequantizer, so what is under
// test is the shader's byte read and its per-block scale.
//
// The shapes are the ones the block format takes: a weight is [columns, inner]
// here, so inner is the contiguous dimension and is a whole number of blocks,
// and the element count is a whole number of scale words. Both are past one
// 32-wide tile, so the product walks several blocks of every row — while a
// staging thread's run of eight lies inside exactly one of them, which is what
// lets the eight come out of a single readInt8x8 against a single scale.
auto tTiledLinearInt8Weights = test("Kernels/tiledLinearInt8Weights") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkPackedLinear<Int8WeightTiledLinear>(
        37, 32, 34, 430u, TiledProduct::quantizedInt8Blocks);

    TiledProduct::checkPackedLinear<Int8WeightTiledLinear>(
        70, 64, 45, 440u, TiledProduct::quantizedInt8Blocks);

    // A prefix of each row, so a staging run of eight straddles the inner
    // extent and the wide quantized read gives way to the loop beside it.
    TiledProduct::checkPackedLinearOverPrefix<Int8WeightTiledLinear>(
        37, 40, 64, 34, 450u, TiledProduct::quantizedInt8Blocks);
};

// Attention scores as a decoder computes them — 35 queries over three heads
// against a cache of 39 keys, scaled and causally masked over the trapezoid
// that leaves, where query 0 stands at position 4.
auto tTiledAttentionScores = test("Kernels/tiledAttentionScores") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkAttentionScores<TiledLinear>(35, 39, 3, 5, 500u);
};

// Gemma's key shape: many query heads against one shared KV head, which is a
// batch stride of zero on the key operand — see checkMultiQueryScores.
auto tTiledMultiQueryScores = test("Kernels/tiledMultiQueryScores") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkMultiQueryScores<TiledLinear>(7, 39, 8, 5, 520u);
};

// The register-tiled pair's softmax, at that kernel's own tile and part
// counts.
auto tTiledSoftmaxAttention = test("Kernels/tiledSoftmaxAttention") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkSoftmaxAttention<MaximaTiledLinear, SoftmaxTiledMatMul>(
        41, 37, 2, 9, 700u);
};

auto tTiledSoftmaxAttentionModelHeads =
    test("Kernels/tiledSoftmaxAttentionModelHeads") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkSoftmaxAttention<MaximaTiledLinear, SoftmaxTiledMatMul>(
        70, 150, 2, 64, 710u);
};

auto tTiledAttentionApply = test("Kernels/tiledAttentionApply") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkAttentionApply<TiledMatMul>(41, 37, 2, 9, 600u);
};

auto tTiledLinearOneProgramTwoShapes =
    test("Kernels/tiledLinearOneProgramTwoShapes") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkTwoShapesThroughOneProgram<TiledLinear>();
};

auto tTiledLinearFoldsGeluAndResidual =
    test("Kernels/tiledLinearFoldsGeluAndResidual") = []
{
    if (!Device::shared().isValid())
        return;

    TiledProduct::checkGeluAndResidual<TiledLinear>(40, 24, 36, 800u);
};

#pragma once

#include "TiledMatMul.h"

namespace HF
{
using eacp::GPU::SimdMatrix;

// The same product TiledMatMulProgram computes — the same TiledMatMulShape,
// the same batch and stride semantics, the same fold on the store — out of
// SIMD-group matrices rather than out of a register block per thread.
//
// Unconditional, unlike WhisperEACP's, which guarded this whole file behind a
// CMake probe for eacp's simdMatrix(). The primitive is on eacp develop, which
// is the only eacp this project fetches, so there is nothing to detect: an
// eacp old enough to lack it would fail to compile the tree rather than
// silently drop to the register-tiled product, which is the better failure.
//
// A threadgroup is 256 threads, which is eight SIMD groups, and it owns a
// 64 x 64 tile of C. The eight groups stand two deep and four across, so each
// holds 32 rows by 16 columns of the tile as eight 8 x 8 accumulator
// fragments; the inner dimension goes by in slabs of 32, staged into one
// threadgroup array as a 64 x 32 block of A and a 32 x 64 block of B, and each
// slab is four multiply-accumulates deep.
//
// The tile of C goes back out through the same array the slabs came in
// through, which is what lets a partial tile be copied out element by element:
// a fragment is loaded and stored whole and has no per-element guard to put on
// one. The store's fold — the scale, the bias, the GELU, the residual and the
// causal mask — happens in that copy-out, which every element passes through
// anyway.
template <OperandLayout bLayout,
          WeightStorage bStorage,
          AFold aFold = AFold::None,
          RowMaxima cMaxima = RowMaxima::None>
struct SimdTiledMatMulProgram final : ComputeProgram
{
    static constexpr auto tile = 64;
    static constexpr auto innerTile = 32;
    static constexpr auto threads = 256;
    static constexpr auto fragment = ComputeProgram::simdMatrixWidth;

    static constexpr auto rowFragments = 4;
    static constexpr auto columnFragments = 2;

    // How many maxima a row of a column tile is reported as: four threads walk
    // a row of the tile, sixteen of its columns each, and each writes what it
    // saw rather than folding the four through threadgroup memory. The reader
    // folds four times as many values out of a buffer four times the size,
    // which is cheaper than a barrier and a second array in the product's hot
    // path.
    static constexpr auto maximaPerRow = 4;
    static constexpr auto tileElements = tile * innerTile + innerTile * tile;
    static constexpr auto columnSlabBase = tile * innerTile;

    // simdgroup_load wants the natural stride, so the A slab carries no
    // bank-conflict pitch the way the register-tiled kernel's does.
    SimdTiledMatMulProgram()
        : ComputeProgram({threads, 1, 1})
    {
        compile();
    }

    void dispatch(ComputePass& pass, const TiledMatMulShape& shape)
    {
        rowCount = (std::uint32_t) shape.rows;
        columnCount = (std::uint32_t) shape.columns;
        innerCount = (std::uint32_t) shape.inner;
        aRowStride = (std::uint32_t) shape.aRowStride;
        aBatchStride = (std::uint32_t) shape.aBatchStride;
        bStride = (std::uint32_t) shape.bStride;
        bBatchStride = (std::uint32_t) shape.bBatchStride;
        cRowStride = (std::uint32_t) shape.cRowStride;
        cBatchStride = (std::uint32_t) shape.cBatchStride;
        scale = shape.scale;
        causal = shape.causal ? 1u : 0u;
        gelu = shape.gelu ? 1u : 0u;
        residual = shape.residual ? 1u : 0u;

        const auto columnTiles = (shape.columns + tile - 1) / tile;
        const auto rowTiles = (shape.rows + tile - 1) / tile;

        maximaStride = (std::uint32_t) (columnTiles * maximaPerRow);
        foldTiles = (std::uint32_t) ((shape.inner + tile - 1) / tile * maximaPerRow);

        pass.dispatch(*this, columnTiles * threads, shape.batches * rowTiles);
    }

    void define() override
    {
        constexpr auto tileWidth = (unsigned) tile;
        constexpr auto slab = (unsigned) innerTile;
        constexpr auto side = (unsigned) fragment;

        auto lane = localPosition().x;
        auto group = groupPosition();
        auto simd = simdGroupIndex();

        auto rowTiles = (rowCount + (tileWidth - 1u)) / tileWidth;
        auto batch = group.y / rowTiles;
        auto m0 = (group.y % rowTiles) * tileWidth;
        auto n0 = group.x * tileWidth;

        auto aBase = batch * aBatchStride;
        auto bBase = batch * bBatchStride;
        auto cBase = batch * cBatchStride;

        auto rowOffset = (simd % 2u) * 32u;
        auto columnOffset = (simd / 2u) * 16u;

        auto staging = shared<Float>(tileElements);
        auto slabStride = unsignedInteger(slab);
        auto tileStride = unsignedInteger(tileWidth);

        // Eight consecutive inner elements per thread, which is 256 threads
        // covering the 64 x 32 slab of A exactly, and the same count covering
        // the transposing store of B when B is contiguous along k.
        auto loadRow = lane / 4u;
        auto loadDepth = (lane % 4u) * 8u;

        auto loadRowIndex = min(m0 + loadRow, rowCount - 1u);
        auto aRow = aBase + loadRowIndex * aRowStride;
        auto bRow = bBase + min(n0 + loadRow, columnCount - 1u) * bStride;

        // One reciprocal per row of the tile, the whole array of them 256
        // bytes, filled after the product.
        auto sums = shared<Float>(aFold == AFold::Softmax ? (unsigned) tile : 1u);

        // The row's maximum, folded out of the column tiles the product that
        // wrote these scores reported one each of, and the sum of what this
        // thread stages, which the store divides the row by.
        //
        // Each of the four threads staging a row folds the row's maxima for
        // itself. Doing it once for the row into threadgroup memory instead
        // costs two barriers, and the reads it saves are all in cache.
        auto foldMaximum = var(0.f);
        auto foldSum = var(0.f);

        if constexpr (aFold == AFold::Softmax)
        {
            auto maximaRow = (batch * rowCount + loadRowIndex) * foldTiles;
            auto largest = var(causalMaskScore);
            auto foldTile = var(0u);

            loop(foldTile.get() < foldTiles,
                 [&]
                 {
                     largest =
                         max(largest.get(), rowMaxima[maximaRow + foldTile.get()]);
                     foldTile += 1u;
                 });

            foldMaximum = largest.get();
        }

        SimdMatrix accumulators[rowFragments * columnFragments];

        for (auto& accumulator: accumulators)
            accumulator = simdMatrix();

        auto k0 = var(0u);

        loop(
            k0.get() < innerCount,
            [&]
            {
                if constexpr (bLayout == OperandLayout::ContiguousK)
                {
                    const auto slotOf = [&](unsigned i)
                    {
                        return columnSlabBase + (loadDepth + i) * tileWidth
                               + loadRow;
                    };

                    for (auto i = 0u; i < 8u; ++i)
                    {
                        auto k = k0.get() + loadDepth + i;
                        auto inside = k < innerCount;
                        auto at = min(k, innerCount - 1u);

                        stageA(staging,
                               loadRow * slab + loadDepth + i,
                               aRow + at,
                               inside,
                               foldMaximum,
                               foldSum);

                        // **The unquantized storages stage their weight here,
                        // beside the A element that shares the loop**, and it
                        // matters: lifting it out into a pass of its own, which
                        // is what the quantized run below has to be, costs the
                        // packed path 6% of a 994-row prefill for arithmetic
                        // that did not change. The interleaving is what the
                        // scheduler wants and it is cheap to keep.
                        if constexpr (bStorage != WeightStorage::Int8Blocks)
                            write(staging,
                                  slotOf(i),
                                  select(inside, weight(bRow + at), 0.f));
                    }

                    if constexpr (bStorage == WeightStorage::Int8Blocks)
                        stageWeightRun(
                            bRow + k0.get() + loadDepth,
                            k0.get() + loadDepth + 8u <= innerCount,
                            [&](unsigned i, const Float& value)
                            { write(staging, slotOf(i), value); },
                            [&](unsigned i)
                            {
                                auto k = k0.get() + loadDepth + i;

                                return select(k < innerCount,
                                              weight(bRow + min(k, innerCount - 1u)),
                                              0.f);
                            });
                }
                else
                {
                    // B read along n instead: one inner row per eight threads,
                    // eight of its columns each, so adjacent threads still
                    // fetch adjacent words.
                    auto slabRow = lane / 8u;
                    auto run = (lane % 8u) * 8u;
                    auto slabK = k0.get() + slabRow;
                    auto slabInside = slabK < innerCount;
                    auto slabBRow = bBase + min(slabK, innerCount - 1u) * bStride;

                    for (auto i = 0u; i < 8u; ++i)
                    {
                        auto k = k0.get() + loadDepth + i;
                        auto inside = k < innerCount;
                        auto at = min(k, innerCount - 1u);

                        stageA(staging,
                               loadRow * slab + loadDepth + i,
                               aRow + at,
                               inside,
                               foldMaximum,
                               foldSum);

                        auto n = min(n0 + run + i, columnCount - 1u);

                        write(staging,
                              columnSlabBase + slabRow * tileWidth + run + i,
                              select(slabInside, weight(slabBRow + n), 0.f));
                    }
                }

                barrier();

                for (auto kk = 0u; kk < slab; kk += side)
                {
                    SimdMatrix left[rowFragments];
                    SimdMatrix right[columnFragments];

                    for (auto i = 0u; i < (unsigned) rowFragments; ++i)
                        left[i] = simdMatrix(
                            staging, (rowOffset + i * side) * slab + kk, slabStride);

                    for (auto j = 0u; j < (unsigned) columnFragments; ++j)
                        right[j] = simdMatrix(staging,
                                              columnSlabBase + kk * tileWidth
                                                  + columnOffset + j * side,
                                              tileStride);

                    for (auto i = 0u; i < (unsigned) rowFragments; ++i)
                        for (auto j = 0u; j < (unsigned) columnFragments; ++j)
                            multiplyAccumulate(accumulators[i * columnFragments + j],
                                               left[i],
                                               right[j]);
                }

                barrier();
                k0 += slab;
            });

        barrier();

        // The partial sums go through the staging array, which is between the
        // last slab and the tile of C.
        if constexpr (aFold == AFold::Softmax)
        {
            write(staging, lane, foldSum.get());
            barrier();

            ifThen(lane < tileWidth,
                   [&]
                   {
                       auto first = 4u * lane;
                       auto total = staging[first] + staging[first + 1u]
                                    + staging[first + 2u] + staging[first + 3u];

                       write(sums, lane, 1.f / total);
                   });

            barrier();
        }

        for (auto i = 0u; i < (unsigned) rowFragments; ++i)
            for (auto j = 0u; j < (unsigned) columnFragments; ++j)
                write(staging,
                      (rowOffset + i * side) * tileWidth + columnOffset + j * side,
                      tileStride,
                      accumulators[i * columnFragments + j]);

        barrier();

        for (auto i = 0u; i < (unsigned) (tile * tile / threads); ++i)
        {
            auto index = lane + i * (unsigned) threads;
            auto m = m0 + index / tileWidth;
            auto n = n0 + index % tileWidth;

            ifThen(m < rowCount && n < columnCount,
                   [&]
                   {
                       auto at = cBase + m * cRowStride + n;
                       auto summed = normalised(staging, index, sums);
                       auto value = scale * summed + bias[n];
                       auto activated = select(gelu != 0u, tanhGelu(value), value);
                       auto carried = select(residual != 0u, output[at], 0.f);
                       auto masked = causal != 0u && n + rowCount > m + columnCount;
                       auto stored =
                           var(select(masked, causalMaskScore, activated + carried));

                       write(output, at, stored.get());

                       // Through the var, not the expression: a read handle
                       // re-materialises after a store to its slot, so a
                       // residual product's second evaluation would read what
                       // the first one just wrote.
                       if constexpr (cMaxima == RowMaxima::PerColumnTile)
                           write(staging, index, stored.get());
                   });
        }

        if constexpr (cMaxima == RowMaxima::PerColumnTile)
        {
            barrier();

            // Four lanes to a row of the tile, sixteen of its columns each,
            // over the values the copy-out put back. A column past the shape
            // holds whatever the accumulator left and is kept out by the same
            // test the store was.
            constexpr auto columnsPerLane = (unsigned) (tile / maximaPerRow);

            auto maximaRow = lane / (unsigned) maximaPerRow;
            auto maximaFirst = (lane % (unsigned) maximaPerRow) * columnsPerLane;
            auto largest = var(causalMaskScore);

            for (auto i = 0u; i < columnsPerLane; ++i)
                largest =
                    max(largest.get(),
                        select(n0 + maximaFirst + i < columnCount,
                               staging[maximaRow * tileWidth + maximaFirst + i],
                               causalMaskScore));

            ifThen(m0 + maximaRow < rowCount,
                   [&]
                   {
                       auto at = (batch * rowCount + m0 + maximaRow) * maximaStride
                                 + group.x * (unsigned) maximaPerRow
                                 + lane % (unsigned) maximaPerRow;

                       write(tileMaxima, at, largest.get());
                   });
        }
    }

    // The A element as it is staged: exponentiated against the row maximum and
    // added into this thread's share of the row's sum when the softmax is
    // being folded in, and the element itself otherwise.
    template <typename Slot, typename Index, typename Inside>
    void stageA(const Shared<Float>& staging,
                const Slot& slot,
                const Index& index,
                const Inside& inside,
                const Var<Float>& maximum,
                Var<Float>& sum)
    {
        if constexpr (aFold == AFold::Softmax)
        {
            auto staged = var(select(inside, exp(a[index] - maximum.get()), 0.f));

            write(staging, slot, staged.get());
            sum += staged.get();
        }
        else
        {
            write(staging, slot, select(inside, a[index], 0.f));
        }
    }

    template <typename Index>
    Float normalised(const Shared<Float>& staging,
                     const Index& index,
                     const Shared<Float>& sums)
    {
        if constexpr (aFold == AFold::Softmax)
            return staging[index] * sums[index / (unsigned) tile];
        else
            return staging[index];
    }

    Float weight(const UInt& index)
    {
        return storedWeight<bStorage>(b, index, scaleBase());
    }

    // TiledMatMulProgram's, at this kernel's own uniforms: past B's columnCount
    // rows of bStride, in halves, and read by the quantized storage alone.
    UInt scaleBase() { return columnCount * bStride / 2u; }

    // TiledMatMulProgram's too, and its comment is where the reasoning is: a
    // quantized run of eight is one eight-byte load and one scale read where
    // the loop beside it is eight of each.
    template <typename Place, typename Narrow>
    void stageWeightRun(const UInt& first,
                        const Bool& whole,
                        Place&& place,
                        Narrow&& narrow)
    {
        const auto oneAtATime = [&]
        {
            for (auto i = 0u; i < 8u; ++i)
                place(i, narrow(i));
        };

        if constexpr (bStorage != WeightStorage::Int8Blocks)
        {
            oneAtATime();
            return;
        }

        ifThen(
            whole && first % 8u == 0u,
            [&]
            {
                auto run = storedWeight8<bStorage>(b, first, scaleBase());
                const Float values[8] = {run.low.x(),
                                         run.low.y(),
                                         run.low.z(),
                                         run.low.w(),
                                         run.high.x(),
                                         run.high.y(),
                                         run.high.z(),
                                         run.high.w()};

                for (auto i = 0u; i < 8u; ++i)
                    place(i, values[i]);
            },
            oneAtATime);
    }

    // TiledMatMulProgram's reason for writing this out rather than declaring
    // it with EACP_SHADER: the maxima bindings belong to one instantiation
    // each.
    void reflectMembers(eacp::GPU::ShaderVisitor& visitor) override
    {
        visitor("a", a);
        visitor("b", b);
        visitor("bias", bias);
        visitor("output", output);
        visitor("rowCount", rowCount);
        visitor("columnCount", columnCount);
        visitor("innerCount", innerCount);
        visitor("aRowStride", aRowStride);
        visitor("aBatchStride", aBatchStride);
        visitor("bStride", bStride);
        visitor("bBatchStride", bBatchStride);
        visitor("cRowStride", cRowStride);
        visitor("cBatchStride", cBatchStride);
        visitor("scale", scale);
        visitor("causal", causal);
        visitor("gelu", gelu);
        visitor("residual", residual);

        if constexpr (aFold == AFold::Softmax)
        {
            visitor("rowMaxima", rowMaxima);
            visitor("foldTiles", foldTiles);
        }

        if constexpr (cMaxima == RowMaxima::PerColumnTile)
        {
            visitor("tileMaxima", tileMaxima);
            visitor("maximaStride", maximaStride);
        }
    }

    Uniform<InputBuffer> a;
    Uniform<InputBuffer> b;
    Uniform<InputBuffer> bias;
    Uniform<OutputBuffer> output;
    Uniform<UInt> rowCount;
    Uniform<UInt> columnCount;
    Uniform<UInt> innerCount;
    Uniform<UInt> aRowStride;
    Uniform<UInt> aBatchStride;
    Uniform<UInt> bStride;
    Uniform<UInt> bBatchStride;
    Uniform<UInt> cRowStride;
    Uniform<UInt> cBatchStride;
    Uniform<Float> scale;
    Uniform<UInt> causal;
    Uniform<UInt> gelu;
    Uniform<UInt> residual;

    // The tile maxima a RowMaxima::PerColumnTile product writes, and the ones
    // an AFold::Softmax product folds. No instantiation is both.
    Uniform<OutputBuffer> tileMaxima;
    Uniform<InputBuffer> rowMaxima;
    Uniform<UInt> maximaStride;
    Uniform<UInt> foldTiles;
};

using SimdTiledLinear =
    SimdTiledMatMulProgram<OperandLayout::ContiguousK, WeightStorage::Float>;
using HalfWeightSimdTiledLinear =
    SimdTiledMatMulProgram<OperandLayout::ContiguousK, WeightStorage::PackedHalf>;
using BFloat16WeightSimdTiledLinear =
    SimdTiledMatMulProgram<OperandLayout::ContiguousK,
                           WeightStorage::PackedBFloat16>;
using Int8WeightSimdTiledLinear =
    SimdTiledMatMulProgram<OperandLayout::ContiguousK, WeightStorage::Int8Blocks>;
using SimdTiledMatMul =
    SimdTiledMatMulProgram<OperandLayout::ContiguousN, WeightStorage::Float>;
using SoftmaxSimdTiledMatMul = SimdTiledMatMulProgram<OperandLayout::ContiguousN,
                                                      WeightStorage::Float,
                                                      AFold::Softmax>;
using MaximaSimdTiledLinear = SimdTiledMatMulProgram<OperandLayout::ContiguousK,
                                                     WeightStorage::Float,
                                                     AFold::None,
                                                     RowMaxima::PerColumnTile>;

// Which program each of the tree's product roles dispatches through. Both
// compute the same TiledMatMulShape, so a role is switched by its alias and
// nothing at the call site moves.
//
// The SIMD-group form is a Metal specialisation. eacp lowers a fragment to 64
// floats of each thread's own wherever there is no wave matrix instruction —
// correct, and doing the arithmetic 32 times over — so the register-tiled form
// is what every other backend gets, and what a role goes back to if it ever
// measures no better on it.
#if defined(__APPLE__)
template <WeightStorage storage>
using LinearProductFor = SimdTiledMatMulProgram<OperandLayout::ContiguousK, storage>;

using AttentionScoresProduct = MaximaSimdTiledLinear;
using SoftmaxAttentionApplyProduct = SoftmaxSimdTiledMatMul;
#else
template <WeightStorage storage>
using LinearProductFor = TiledMatMulProgram<OperandLayout::ContiguousK, storage>;

using AttentionScoresProduct = MaximaTiledLinear;
using SoftmaxAttentionApplyProduct = SoftmaxTiledMatMul;
#endif

// The four storages a checkpoint's projection weights may arrive in, each
// through whichever of the two programs above this backend took.
using LinearProduct = LinearProductFor<WeightStorage::Float>;
using HalfWeightLinearProduct = LinearProductFor<WeightStorage::PackedHalf>;
using BFloat16WeightLinearProduct = LinearProductFor<WeightStorage::PackedBFloat16>;
using Int8WeightLinearProduct = LinearProductFor<WeightStorage::Int8Blocks>;

// The scores product reports its maxima in its own tiling and the apply folds
// them in the apply's, so the two roles have to be tiled the same way. They
// are, both aliases coming out of the same branch above — and this is what
// says so, since neither call site can see it.
static_assert(AttentionScoresProduct::tile == SoftmaxAttentionApplyProduct::tile);
static_assert(AttentionScoresProduct::maximaPerRow
              == SoftmaxAttentionApplyProduct::maximaPerRow);
} // namespace HF

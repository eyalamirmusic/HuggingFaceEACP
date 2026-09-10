#pragma once

#include <HuggingFaceEACP/Core/Core.h>

#include <eacp/GPU/GPU.h>

#include <cmath>
#include <limits>

namespace HF
{
// The EDSL vocabulary these kernels are declared in, brought in by name the way
// Core/Common.h brings in ea_data_structures'. The intrinsics and operators
// need no import: their arguments are eacp::GPU types, so ADL finds them.
using eacp::GPU::ComputePass;
using eacp::GPU::ComputeProgram;
using eacp::GPU::Float;
using eacp::GPU::InputBuffer;
using eacp::GPU::OutputBuffer;
using eacp::GPU::ThreadGroupShape;
using eacp::GPU::UInt;
using eacp::GPU::Uniform;

// Multi-query attention over a KV cache, in the two shapes a decoder-only
// model runs it in: the prompt's many query rows in one dispatch, and the one
// row a decode step has afterwards.
//
// The layouts, over a head width D, H query heads and K key/value heads:
//
//   queries   [rows, H * D]           row-major
//   keys      [maxContext, K * D]     the cache, row-major
//   values    [maxContext, K * D]     the cache, row-major
//   output    [rows, H * D]           row-major
//
// Query head h owns columns [h * D, (h + 1) * D) of every H * D row, and reads
// key/value head h / (H / K) of every K * D cache row. Gemma 2B is H = 8,
// K = 1, D = 256 — every query head over the one cached head — and Whisper's
// plain multi-head attention is K = H, where the mapping is the identity.
//
// Every shape is a uniform rather than a constant: one compiled program serves
// all eighteen layers and both the prompt step and every token step after it.
//
// The softmax lives inside the kernel, streamed rather than materialised: the
// context is walked a tile at a time, and each tile rescales what the earlier
// tiles left by exp(previous maximum - the new one) before adding its own
// share. That is the same rescaling a softmax split across groups does when it
// joins, and it is exact; what it buys here is that no [rows, context] score
// matrix is ever written, which at Gemma's 8192 positions is the difference
// between a kilobyte of threadgroup memory and a megabyte of bandwidth.

// The score an out-of-range lane contributes to a tile's maximum, low enough
// that it can never be the maximum of a tile that holds a real key, and finite
// so that no subtraction of it overflows. Its weight is forced to zero
// independently, so nothing downstream relies on exp() of it underflowing.
inline constexpr auto maskedScore = -1e30f;

// The head decomposition the kernels and their callers both index by. The
// widths are derived rather than given: a query row is H * D wide and a cache
// row K * D by definition, and a shape where they are not is not a shape
// attention has.
struct MultiQueryShape
{
    int heads = 1;
    int kvHeads = 1;
    int headDim = 1;

    int queryWidth() const { return heads * headDim; }
    int kvWidth() const { return kvHeads * headDim; }
    int queriesPerKvHead() const { return heads / kvHeads; }

    float defaultScale() const { return 1.f / std::sqrt((float) headDim); }
};

// What the two kernels share: the uniforms, and the streaming fold itself.
// Only where the query row and its key count come from differs between them,
// so that is all the two bodies below spell.
struct MultiQueryAttentionProgram : ComputeProgram
{
    // The widest head the accumulator below is sized for. Gemma's is exactly
    // this; a wider one is refused by the dispatch helpers rather than read
    // past the end of a threadgroup array whose size the emitted kernel bakes
    // in.
    static constexpr auto maxHeadDim = 256;

    // One group per (row, head), this many threads in it. The stock width,
    // which is what a strided walk steps by and what a group reduction folds.
    static constexpr auto laneCount = ComputeProgram::groupWidth;

    explicit MultiQueryAttentionProgram(ThreadGroupShape shape)
        : ComputeProgram(shape)
    {
    }

    Uniform<InputBuffer> queries;
    Uniform<InputBuffer> keys;
    Uniform<InputBuffer> values;
    Uniform<OutputBuffer> output;
    Uniform<UInt> heads;
    Uniform<UInt> kvHeads;
    Uniform<UInt> headDim;
    Uniform<Float> scale;

protected:
    // One query row against the first keyCount rows of the cache, written into
    // the row'th output row's slice for this head.
    //
    // **The value fold is laid out by column, not by lane.** The other
    // arrangement — a lane per key, each carrying a head-wide row of
    // accumulators for the group to fold at the end — is what WhisperEACP's
    // SingleQueryAttention does, and at a head width of 256 it asks for
    // 64 lanes * 256 floats = 64 kB of threadgroup memory against Metal's
    // 32 kB, which plan.md records as the constraint this kernel is written
    // around. Here the group keeps **one** head-wide accumulator and each lane
    // owns the columns lane, lane + 64, ... of it: 256 floats whatever the lane
    // count is, no lane ever touches another's column, and the fold needs no
    // cross-lane reduction at all. Adjacent lanes read adjacent columns of the
    // same value row, which is the access the memory system serves whole.
    //
    // The accumulator sits in threadgroup memory rather than in registers only
    // because the head width is a uniform: how many columns a lane owns is not
    // known when the kernel is compiled, and a register cannot be indexed by a
    // value that is not.
    void foldContext(const UInt& lane,
                     const UInt& head,
                     const UInt& row,
                     const UInt& keyCount);
};

inline void MultiQueryAttentionProgram::foldContext(const UInt& lane,
                                                    const UInt& head,
                                                    const UInt& row,
                                                    const UInt& keyCount)
{
    constexpr auto lanes = (unsigned) laneCount;

    auto queryStride = heads * headDim;
    auto kvStride = kvHeads * headDim;
    auto kvHead = head / (heads / kvHeads);
    auto queryBase = row * queryStride + head * headDim;
    auto kvBase = kvHead * headDim;
    auto lastKey = keyCount - 1u;

    // A tile's weights, published to the group; and the running unnormalised
    // output row, private to the lanes column by column.
    auto weights = shared<Float>(laneCount);
    auto accumulator = shared<Float>(maxHeadDim);

    auto clearing = var(lane);

    loop(clearing.get() < headDim,
         [&]
         {
             write(accumulator, clearing.get(), constant(0.f));
             clearing += lanes;
         });

    auto runningMax = var(std::numeric_limits<float>::lowest());
    auto runningSum = var(0.f);
    auto tileStart = var(0u);

    loop(tileStart.get() < keyCount,
         [&]
         {
             auto key = var(tileStart.get() + lane);
             auto held = key.get() <= lastKey;

             // Clamped rather than branched around: every lane runs the same
             // product, and the score of a lane past the end is thrown away
             // below. The last tile of a context that is not a whole number of
             // tiles is the only place this costs anything.
             auto keyBase = min(key.get(), lastKey) * kvStride + kvBase;

             auto product = var(0.f);
             auto channel = var(0u);

             loop(channel.get() < headDim,
                  [&]
                  {
                      auto slice = queries.read4((queryBase + channel.get()) / 4u);
                      auto cached = keys.read4((keyBase + channel.get()) / 4u);

                      product += dot(slice, cached);
                      channel += 4u;
                  });

             auto score = var(select(held, scale * product.get(), maskedScore));
             auto tileMax = var(groupMax(score.get()));
             auto largest = var(max(runningMax.get(), tileMax.get()));

             // What the tiles before this one have to be multiplied by now that
             // the maximum has moved. The first tile finds it zero, which is
             // what an empty accumulator wants.
             auto carried = var(exp(runningMax.get() - largest.get()));
             auto weight = var(select(held, exp(score.get() - largest.get()), 0.f));

             write(weights, lane, weight.get());

             // The reduction barriers, which is also what publishes the write
             // above to the rest of the group.
             auto tileSum = var(groupSum(weight.get()));

             runningSum = runningSum.get() * carried.get() + tileSum.get();
             runningMax = largest.get();

             auto tileEnd = min(tileStart.get() + lanes, keyCount);
             auto column = var(lane);

             loop(column.get() < headDim,
                  [&]
                  {
                      auto folded = var(accumulator[column.get()] * carried.get());
                      auto tileKey = var(tileStart.get());

                      loop(tileKey.get() < tileEnd,
                           [&]
                           {
                               auto at =
                                   tileKey.get() * kvStride + kvBase + column.get();

                               folded += weights[tileKey.get() - tileStart.get()]
                                         * values[at];
                               tileKey += 1u;
                           });

                      write(accumulator, column.get(), folded.get());
                      column += lanes;
                  });

             // The reads above finish before the next tile's weights overwrite
             // what they read.
             barrier();
             tileStart += lanes;
         });

    auto storing = var(lane);

    loop(storing.get() < headDim,
         [&]
         {
             write(output,
                   queryBase + storing.get(),
                   accumulator[storing.get()] / runningSum.get());
             storing += lanes;
         });
}

// The prompt: rows query rows standing at positions positionOffset + r, each
// attending to cache entries [0, positionOffset + r]. That bound **is** the
// causal mask — a row's key count stops at its own position, so a later
// position is never read rather than read and suppressed — and positionOffset
// is what lets a prompt be fed in more than one block, or fed against a cache
// that already holds a system prefix.
//
// A group per (row, head) over a 3D grid: x is the lane, y the head, z the
// query row. plan.md records 3D dispatch as already landed in eacp, and this
// is what it buys — the head no longer has to be folded into a dispatch row
// and taken apart again in the body. The grid is exactly one group wide, so
// nothing is rounded up and no thread is out of range, which matters because a
// kernel that barriers has no generated bounds guard.
struct MultiQueryPrefillAttention final : MultiQueryAttentionProgram
{
    MultiQueryPrefillAttention()
        : MultiQueryAttentionProgram({laneCount, 1, 1})
    {
        compile();
    }

    void dispatch(ComputePass& pass,
                  const MultiQueryShape& shape,
                  int rows,
                  int positionOffsetToUse,
                  float scaleToUse);

    void define() override
    {
        auto position = threadPosition3();
        auto queryPosition = positionOffset + position.z;

        foldContext(position.x, position.y, position.z, queryPosition + 1u);
    }

    Uniform<UInt> positionOffset;

    EACP_SHADER(queries,
                keys,
                values,
                output,
                heads,
                kvHeads,
                headDim,
                scale,
                positionOffset)
};

// A decode step: the one query row the model has just projected, against every
// cache entry written so far. Nothing is masked — the row stands at the last
// position, so no cached key is later than it — and the query buffer is a
// single H * D row, which is why row zero is what the fold is given.
//
// A group per head over a 2D grid, x the lane and y the head, since there is
// only ever one row to place.
struct MultiQueryDecodeAttention final : MultiQueryAttentionProgram
{
    MultiQueryDecodeAttention()
        : MultiQueryAttentionProgram({laneCount, 1, 1})
    {
        compile();
    }

    void dispatch(ComputePass& pass,
                  const MultiQueryShape& shape,
                  int contextLengthToUse,
                  float scaleToUse);

    void define() override
    {
        auto position = threadPosition();

        foldContext(position.x, position.y, unsignedInteger(0u), contextLength);
    }

    Uniform<UInt> contextLength;

    EACP_SHADER(
        queries, keys, values, output, heads, kvHeads, headDim, scale, contextLength)
};
} // namespace HF

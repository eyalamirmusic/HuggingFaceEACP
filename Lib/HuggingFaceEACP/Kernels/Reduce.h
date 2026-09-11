#pragma once

#include "KernelTypes.h"

namespace HF
{
// The base of every kernel here that gives a whole group of threads to one row
// rather than one thread: each thread walks a strided share of the row, and
// the group folds what the threads are left holding through groupSum() or
// groupMax() — every thread contributes one value and every thread gets the
// whole group's fold back. A row of 2048 channels or 256,000 logits walked by
// a single thread is the whole cost of a decode step.
//
// A group is the unit, so dispatchRows() dispatches rowCount groups and lanes
// is the stride a thread walks its row at. The group is this program's own
// rather than the one width every kernel shares: the constructor names it and
// lanes reads it back from groupShape(), so a kernel dispatched over one row
// can take the wider group that hides its latency while a kernel dispatched
// over a prompt's worth of rows keeps the stock 64, which already saturates
// memory bandwidth.
//
// A group reduction is a barrier, and a barrier removes the generated bounds
// guard, so a kernel built on this is always dispatched over whole groups and
// bounds itself: every walk stops at the row length, every store is behind a
// lane test or a row test, and every reduction is reached by every thread of
// the group.
struct ReducingProgram : ComputeProgram
{
    explicit ReducingProgram(int laneCount = groupWidth)
        : ComputeProgram({laneCount})
        , lanes((unsigned) groupShape().x)
    {
    }

    void dispatchRows(ComputePass& pass, int rowCount)
    {
        pass.dispatch(*this, rowCount * (int) lanes);
    }

    // The threads one group holds, which is both what a strided walk steps by
    // and how many rows' worth of threads a dispatch asks for.
    const unsigned lanes;

protected:
    // The group's fold, through the SIMD-group intrinsic when this group **is**
    // one SIMD group and through the whole-group one otherwise. A group at or
    // under simdWidth is its own SIMD group, so the two fold the same threads
    // and the narrow one does it with neither threadgroup scratch nor a
    // barrier; a wider group is several, and simdSum there would hand each of
    // them its own answer rather than the row's.
    //
    // The choice is made here rather than at each call site because the lane
    // count is the caller's and every kernel built on this asks the same
    // question of it. Every fold below sits outside the strided walk that feeds
    // it, so control flow at the fold is uniform, which is what both forms
    // require.
    Float foldSum(const Float& value)
    {
        return foldsInOneSimdGroup() ? simdSum(value) : groupSum(value);
    }

    Float foldMax(const Float& value)
    {
        return foldsInOneSimdGroup() ? simdMax(value) : groupMax(value);
    }

    UInt foldMin(const UInt& value)
    {
        return foldsInOneSimdGroup() ? simdMin(value) : groupMin(value);
    }

    bool foldsInOneSimdGroup() const { return lanes <= (unsigned) simdWidth; }
};
} // namespace HF

#include "MultiQueryAttention.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace HF
{
namespace
{
// What both kernels hold their arguments to, checked where the shape arrives
// rather than left to a read past the end of a threadgroup array whose size
// the emitted kernel baked in.
//
// The head width is read four channels at a time, which is what makes the
// scoring loop a quarter of the subscripts it would otherwise be; that is the
// whole of why it has to be a multiple of four. Gemma's 256 is, and so is every
// head width a transformer has ever shipped with.
void checkShape(const MultiQueryShape& shape, int rows, int keyCount)
{
    const auto widthIsHeld =
        shape.headDim > 0 && shape.headDim <= MultiQueryAttentionProgram::maxHeadDim
        && shape.headDim % 4 == 0;

    const auto headsAreHeld =
        shape.heads > 0 && shape.kvHeads > 0 && shape.heads % shape.kvHeads == 0;

    if (!widthIsHeld || !headsAreHeld || rows < 1 || keyCount < 1)
        throw std::invalid_argument {
            "MultiQueryAttention takes heads a whole multiple of kvHeads over a "
            "head width that is a multiple of four and at most "
            + std::to_string(MultiQueryAttentionProgram::maxHeadDim)
            + ", with at least one row over at least one cached key, and was "
              "asked for "
            + std::to_string(shape.heads) + " heads, "
            + std::to_string(shape.kvHeads) + " kv heads, width "
            + std::to_string(shape.headDim) + ", " + std::to_string(rows)
            + " rows over " + std::to_string(keyCount) + " keys"};
}

void setShapeUniforms(MultiQueryAttentionProgram& program,
                      const MultiQueryShape& shape,
                      float scale)
{
    program.heads = (std::uint32_t) shape.heads;
    program.kvHeads = (std::uint32_t) shape.kvHeads;
    program.headDim = (std::uint32_t) shape.headDim;
    program.scale = scale;
}
} // namespace

void MultiQueryPrefillAttention::dispatch(ComputePass& pass,
                                          const MultiQueryShape& shape,
                                          int rows,
                                          int positionOffsetToUse,
                                          float scaleToUse)
{
    checkShape(shape, rows, positionOffsetToUse + 1);

    setShapeUniforms(*this, shape, scaleToUse);
    positionOffset = (std::uint32_t) positionOffsetToUse;

    pass.dispatch(*this, laneCount, shape.heads, rows);
}

void MultiQueryDecodeAttention::dispatch(ComputePass& pass,
                                         const MultiQueryShape& shape,
                                         int contextLengthToUse,
                                         float scaleToUse)
{
    checkShape(shape, 1, contextLengthToUse);

    setShapeUniforms(*this, shape, scaleToUse);
    contextLength = (std::uint32_t) contextLengthToUse;

    pass.dispatch(*this, laneCount, shape.heads);
}
} // namespace HF

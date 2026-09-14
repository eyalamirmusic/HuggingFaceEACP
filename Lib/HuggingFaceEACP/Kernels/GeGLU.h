#pragma once

#include "Gelu.h"
#include "KernelTypes.h"

namespace HF
{
// The gate of Gemma's feed-forward: gelu(gate(x)) * up(x), elementwise, which
// with the down projection either side of it is the whole MLP.
//
//   output[r, c] = tanhGelu(gate[r, c]) * up[r, c]
//
// The activation is the tanh GELU and not the exact one, since that is what
// Gemma's config names and what its weights were fitted against — see Gelu.h.
//
// Two forms, because the product before this can be dispatched two ways. The
// concatenated one is primary: gate_proj [16384, 2048] and up_proj [16384,
// 2048] stacked into one [32768, 2048] weight is a single product writing
// [rows, 32768], which is one dispatch instead of two over the same activation
// rows, and the halves of a row are then the two operands. The separate one is
// what two products leave, and is here because it is fifteen lines.
//
// One thread per output element over a 2D grid: dispatch(kernel, intermediate,
// rowCount). Out of place in both forms — the output is [rows, intermediate]
// and the concatenated input is [rows, 2 * intermediate], so writing into the
// input's first half would land on a row another thread has not read yet.
//
// **This stays a dispatch of its own, and that was measured.** Folding it into
// a neighbouring product would save a decode step 18 layers of one 128 KB read
// and one 64 KB write — 3.5 MB against the 5.0 GB of weights the same step
// reads, which is 0.07% of its traffic. Removing the stage outright, which is
// the ceiling any fusion could reach, measures 95.4 tokens/s against 95.1 over
// 16 decode tokens and 86.0 against 85.5 over 64: a third of a percent and
// half a percent, at the edge of the run-to-run noise. Neither fold is worth
// the kernel it would cost, and the two are not equally priced either. Folding
// into the down product's operand read is the cheap one to write and the
// expensive one to run, since that product's 2048 outputs each walk the whole
// 16384-wide row — it would evaluate the activation 2048 times per element
// instead of once. Folding into the gate-and-up product's store is the one
// that is actually free at run time, and it needs a product kernel that
// computes two dot products per output and folds them, in both the split and
// the tiled form and in each of the three weight storages. A step at memory
// bandwidth has no arithmetic or dispatch cost left to save — the same
// conclusion the attention lane-count experiment reached from the other
// direction.

// The concatenated form: gate is columns [0, intermediate) of a row and up is
// columns [intermediate, 2 * intermediate), so the input's row stride is twice
// the output's and no second uniform is needed to say it.
struct GeGLU final : ComputeProgram
{
    GeGLU() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto column = position.x;
        auto row = position.y;

        auto base = row * intermediate * 2u;

        auto gate = fused[base + column];
        auto up = fused[base + intermediate + column];

        write(output, row * intermediate + column, tanhGelu(gate) * up);
    }

    Uniform<InputBuffer> fused;
    Uniform<OutputBuffer> output;
    Uniform<UInt> intermediate;

    EACP_SHADER(fused, output, intermediate)
};

// The same product of the same two numbers, with the halves in buffers of
// their own: two [rows, intermediate] operands walked at the output's stride.
struct SplitGeGLU final : ComputeProgram
{
    SplitGeGLU() { compile(); }

    void define() override
    {
        auto position = threadPosition();
        auto column = position.x;
        auto row = position.y;

        auto at = row * intermediate + column;

        write(output, at, tanhGelu(gate[at]) * up[at]);
    }

    Uniform<InputBuffer> gate;
    Uniform<InputBuffer> up;
    Uniform<OutputBuffer> output;
    Uniform<UInt> intermediate;

    EACP_SHADER(gate, up, output, intermediate)
};
} // namespace HF

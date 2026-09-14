#include "Common.h"

using namespace nano;
using namespace HF;
using namespace eacp::GPU;

namespace
{
// out[r, c] = tanhGelu(gate[r, c]) * up[r, c], from the definition and in
// double.
Vector<float> geGluReference(const Vector<float>& gate,
                             const Vector<float>& up,
                             int rows,
                             int intermediate)
{
    auto expected = sized(rows * intermediate);

    for (auto row = 0; row < rows; ++row)
        for (auto column = 0; column < intermediate; ++column)
        {
            const auto at = row * intermediate + column;
            expected[at] = (float) (tanhGeluReference(gate[at]) * up[at]);
        }

    return expected;
}

// The two halves of a row laid out the way one product over the concatenated
// gate and up weights writes them: gate first, then up, per row.
Vector<float> concatenated(const Vector<float>& gate,
                           const Vector<float>& up,
                           int rows,
                           int intermediate)
{
    auto fused = sized(rows * intermediate * 2);

    for (auto row = 0; row < rows; ++row)
        for (auto column = 0; column < intermediate; ++column)
        {
            const auto at = row * intermediate + column;

            fused[row * intermediate * 2 + column] = gate[at];
            fused[row * intermediate * 2 + intermediate + column] = up[at];
        }

    return fused;
}

Vector<float> runGeGLU(const Vector<float>& gate,
                       const Vector<float>& up,
                       int rows,
                       int intermediate)
{
    auto fusedBuffer = storageOf(concatenated(gate, up, rows, intermediate));
    auto output = outputFor(rows * intermediate);

    auto kernel = GeGLU {};
    kernel.fused = fusedBuffer;
    kernel.output = output;
    kernel.intermediate = (unsigned) intermediate;

    return runOverGrid(kernel, output, intermediate, rows);
}

Vector<float> runSplitGeGLU(const Vector<float>& gate,
                            const Vector<float>& up,
                            int rows,
                            int intermediate)
{
    auto gateBuffer = storageOf(gate);
    auto upBuffer = storageOf(up);
    auto output = outputFor(rows * intermediate);

    auto kernel = SplitGeGLU {};
    kernel.gate = gateBuffer;
    kernel.up = upBuffer;
    kernel.output = output;
    kernel.intermediate = (unsigned) intermediate;

    return runOverGrid(kernel, output, intermediate, rows);
}

// The activation and one multiply in float32 against the same in double.
constexpr auto tolerance = 1e-6;

// Not a multiple of the 8 x 8 dispatch group, so a kernel that leaned on the
// grid landing exactly on the matrix fails here.
constexpr auto rows = 5;
constexpr auto intermediate = 19;
} // namespace

auto tGeGluMatchesCpu = test("Kernels/geGluMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    auto gate = spreadValues(rows * intermediate, 111u, 4.f);
    auto up = spreadValues(rows * intermediate, 222u, 4.f);

    auto result = runGeGLU(gate, up, rows, intermediate);
    auto expected = geGluReference(gate, up, rows, intermediate);

    check(result.size() == expected.size());

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], tolerance));
};

// The two operands are not interchangeable — the activation goes on the gate
// and the up half is multiplied in raw — so swapping them has to change the
// answer. An up half of all ones is what isolates the activation: the output
// is then the GELU of the gate, and a kernel that had activated the up half
// instead returns the gate itself.
auto tGeGluActivatesOnlyTheGate = test("Kernels/geGluActivatesOnlyTheGate") = []
{
    if (!Device::shared().isValid())
        return;

    auto gate = spreadValues(rows * intermediate, 333u, 4.f);
    auto ones = sized(rows * intermediate);

    for (auto i = 0; i < ones.size(); ++i)
        ones[i] = 1.f;

    auto activated = runGeGLU(gate, ones, rows, intermediate);

    for (auto i = 0; i < activated.size(); ++i)
        check(isClose(activated[i], tanhGeluReference(gate[i]), tolerance));

    // The same numbers the other way round: gelu(1) * gate, which is not
    // gelu(gate) for any gate but the fixed points.
    auto swapped = runGeGLU(ones, gate, rows, intermediate);

    for (auto i = 0; i < swapped.size(); ++i)
        check(isClose(swapped[i], tanhGeluReference(1.0) * gate[i], tolerance));
};

// A zero up half zeroes the output whatever the gate held, which is the case a
// kernel that read the wrong half of the row cannot produce.
auto tGeGluZeroUpHalfZeroesTheOutput =
    test("Kernels/geGluZeroUpHalfZeroesTheOutput") = []
{
    if (!Device::shared().isValid())
        return;

    auto gate = spreadValues(rows * intermediate, 444u, 6.f);
    auto result = runGeGLU(gate, zeroes(rows * intermediate), rows, intermediate);

    for (auto i = 0; i < result.size(); ++i)
        check(result[i] == 0.f);
};

// Every row reads its own two halves. With each row's gate held at a constant
// of its own and the up half at one, the output row is that constant's GELU
// repeated — so a kernel whose row stride was the output's rather than twice
// it comes back holding another row's numbers.
auto tGeGluKeepsRowsApart = test("Kernels/geGluKeepsRowsApart") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto manyRows = 7;

    auto gate = sized(manyRows * intermediate);
    auto ones = sized(manyRows * intermediate);

    for (auto row = 0; row < manyRows; ++row)
        for (auto column = 0; column < intermediate; ++column)
        {
            gate[row * intermediate + column] = (float) row - 3.f;
            ones[row * intermediate + column] = 1.f;
        }

    auto result = runGeGLU(gate, ones, manyRows, intermediate);

    for (auto row = 0; row < manyRows; ++row)
    {
        const auto expected = tanhGeluReference((double) row - 3.0);

        for (auto column = 0; column < intermediate; ++column)
            check(isClose(result[row * intermediate + column], expected, tolerance));
    }
};

// The separate-buffer form answers the same numbers as the concatenated one,
// which is what says the two are one op with two operand layouts rather than
// two ops.
auto tSplitGeGluMatchesConcatenated =
    test("Kernels/splitGeGluMatchesConcatenated") = []
{
    if (!Device::shared().isValid())
        return;

    auto gate = spreadValues(rows * intermediate, 555u, 4.f);
    auto up = spreadValues(rows * intermediate, 666u, 4.f);

    auto fusedResult = runGeGLU(gate, up, rows, intermediate);
    auto splitResult = runSplitGeGLU(gate, up, rows, intermediate);
    auto expected = geGluReference(gate, up, rows, intermediate);

    for (auto i = 0; i < splitResult.size(); ++i)
    {
        check(splitResult[i] == fusedResult[i]);
        check(isClose(splitResult[i], expected[i], tolerance));
    }
};

// One pipeline, two widths — the model's own 16384 alongside a ragged one,
// since the intermediate width is a uniform and not something the pipeline was
// compiled around. A row of 16384 is also 128 kB of concatenated input, which
// is where an index that overflowed a narrower type would show.
auto tGeGluOneProgramTwoWidths = test("Kernels/geGluOneProgramTwoWidths") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto modelIntermediate = 16384;

    for (const auto width: {intermediate, modelIntermediate})
    {
        auto gate = spreadValues(2 * width, 777u, 4.f);
        auto up = spreadValues(2 * width, 888u, 4.f);

        auto result = runGeGLU(gate, up, 2, width);
        auto expected = geGluReference(gate, up, 2, width);

        for (auto i = 0; i < result.size(); ++i)
            check(isClose(result[i], expected[i], tolerance));
    }
};

// The gate's tails through the kernel Gemma's feed-forward actually dispatches.
// GeluTests has the same check on the standalone activation; this one is here
// because it is this kernel the model runs, and because the gate of a real
// layer does reach thirty — which asks tanh for an argument of about 990, where
// Metal's own returns a NaN under the compile options eacp builds a library
// with. saturatingTanh is what answers it, and one NaN in the residual stream
// is the whole rest of the sequence.
//
// The up operand is deliberately not one: a gate tail multiplied by zero would
// hide a NaN rather than propagate it.
auto tGeGluGateTailsStayFinite = test("Kernels/geGluGateTailsStayFinite") = []
{
    if (!Device::shared().isValid())
        return;

    const float tails[] = {-1000.f, -100.f, -30.f, -10.f, 0.f, 10.f, 30.f, 1000.f};
    constexpr auto tailCount = (int) (sizeof(tails) / sizeof(tails[0]));

    auto gate = sized(tailCount);
    auto up = sized(tailCount);

    for (auto i = 0; i < tailCount; ++i)
    {
        gate[i] = tails[i];
        up[i] = 1.f + 0.5f * (float) i;
    }

    auto result = runGeGLU(gate, up, 1, tailCount);
    auto splitResult = runSplitGeGLU(gate, up, 1, tailCount);
    auto expected = geGluReference(gate, up, 1, tailCount);

    for (auto i = 0; i < tailCount; ++i)
    {
        check(std::isfinite(result[i]));
        check(result[i] == splitResult[i]);
        check(isClose(result[i], expected[i], tolerance));
    }
};

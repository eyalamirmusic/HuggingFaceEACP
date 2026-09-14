#include "Common.h"

#include <algorithm>

using namespace nano;
using namespace HF;
using namespace eacp::GPU;

namespace
{
// A sweep dense enough to land on the region the two GELUs disagree most
// about, which is |x| between one and three rather than at either tail.
Vector<float> sweptValues(int count, double from, double to)
{
    auto values = sized(count);

    for (auto i = 0; i < count; ++i)
        values[i] = (float) (from + (to - from) * i / (count - 1));

    return values;
}

template <typename Kernel>
Vector<float> runGelu(const Vector<float>& input)
{
    auto values = storageOf(input);

    auto kernel = Kernel {};
    kernel.values = values;

    return runOverRows(kernel, values, input.size(), input.size());
}

constexpr auto sweepCount = 513;
} // namespace

// The kernel Gemma runs, against the same expression in double precision. The
// tolerance is float32's own: the kernel and the reference evaluate the same
// closed form, so what separates them is eacp's tanh against libm's and the
// rounding of a cube.
auto tTanhGeluMatchesCpu = test("Kernels/tanhGeluMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    auto input = sweptValues(sweepCount, -8.0, 8.0);
    auto result = runGelu<Gelu>(input);

    for (auto i = 0; i < sweepCount; ++i)
        check(isClose(result[i], tanhGeluReference(input[i]), 1e-6));
};

auto tExactGeluMatchesCpu = test("Kernels/exactGeluMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    auto input = sweptValues(sweepCount, -8.0, 8.0);
    auto result = runGelu<ExactGelu>(input);

    for (auto i = 0; i < sweepCount; ++i)
        check(isClose(result[i], exactGeluReference(input[i]), 1e-6));
};

// The two forms are different functions, not one function to a tolerance, and
// this is the number that says so: over the range a transformer's activations
// occupy they part by about 1e-3, which is three orders of magnitude above the
// float32 error of either. So a decoder that ran the exact GELU where Gemma
// wants the tanh one would not fail by rounding — it would compute something
// else, quietly.
auto tTheTwoGelusAreDifferentFunctions =
    test("Kernels/theTwoGelusAreDifferentFunctions") = []
{
    if (!Device::shared().isValid())
        return;

    auto input = sweptValues(sweepCount, -4.0, 4.0);
    auto tanhResult = runGelu<Gelu>(input);
    auto exactResult = runGelu<ExactGelu>(input);

    auto worstGap = 0.0;
    auto worstTanhError = 0.0;
    auto worstExactError = 0.0;

    for (auto i = 0; i < sweepCount; ++i)
    {
        worstGap =
            std::max(worstGap, std::abs((double) tanhResult[i] - exactResult[i]));

        worstTanhError =
            std::max(worstTanhError,
                     std::abs((double) tanhResult[i] - tanhGeluReference(input[i])));

        worstExactError = std::max(
            worstExactError,
            std::abs((double) exactResult[i] - exactGeluReference(input[i])));
    }

    check(worstGap > 1e-4);
    check(worstTanhError < worstGap / 100.0);
    check(worstExactError < worstGap / 100.0);
};

// Both tails: GELU is the identity far to the right and zero far to the left,
// and neither may arrive as a NaN.
//
// This is the check that caught the one real hazard in the tanh form. Its
// argument grows as the cube of the input, so an activation of 30 asks tanh
// for 990 — and Metal's tanh, under the compile options eacp builds a library
// with, answers that with a NaN rather than with one. Every input at or past
// 30 came back as not-a-number before the argument was clamped: by hand in
// Gelu.h at first, and by eacp's saturatingTanh since. A transformer's
// activations do reach 30, and one NaN in the residual stream is the whole
// rest of the sequence, so this case is not a corner.
auto tGeluTailsStayFinite = test("Kernels/geluTailsStayFinite") = []
{
    if (!Device::shared().isValid())
        return;

    const float tails[] = {-1000.f, -100.f, -30.f, 0.f, 30.f, 100.f, 1000.f};
    constexpr auto tailCount = (int) (sizeof(tails) / sizeof(tails[0]));

    auto input = sized(tailCount);

    for (auto i = 0; i < tailCount; ++i)
        input[i] = tails[i];

    auto tanhResult = runGelu<Gelu>(input);
    auto exactResult = runGelu<ExactGelu>(input);

    for (auto i = 0; i < tailCount; ++i)
    {
        const auto expected = input[i] > 0.f ? (double) input[i] : 0.0;

        check(std::isfinite(tanhResult[i]));
        check(std::isfinite(exactResult[i]));
        check(isClose(tanhResult[i], expected, 1e-6));
        check(isClose(exactResult[i], expected, 1e-6));
    }
};

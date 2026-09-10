#include "Common.h"

#include <algorithm>

using namespace nano;
using namespace HF;
using namespace eacp::GPU;

namespace
{
Vector<float>
    softmaxReference(const Vector<float>& input, int rowCount, int rowLength)
{
    auto expected = sized(rowCount * rowLength);

    for (auto row = 0; row < rowCount; ++row)
    {
        auto base = row * rowLength;
        auto largest = (double) input[base];

        for (auto i = 1; i < rowLength; ++i)
            largest = std::max(largest, (double) input[base + i]);

        auto total = 0.0;

        for (auto i = 0; i < rowLength; ++i)
            total += std::exp(input[base + i] - largest);

        for (auto i = 0; i < rowLength; ++i)
            expected[base + i] =
                (float) (std::exp(input[base + i] - largest) / total);
    }

    return expected;
}

Vector<float> runSoftmax(const Vector<float>& input, int rowCount, int rowLength)
{
    auto values = storageOf(input);

    auto kernel = Softmax {};
    kernel.values = values;
    kernel.rowLength = (unsigned) rowLength;

    return runGroupPerRow(kernel, values, rowCount, rowCount * rowLength);
}

constexpr auto rowCount = 5;
constexpr auto rowLength = 29;
} // namespace

auto tSoftmaxMatchesCpu = test("Kernels/softmaxMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    auto input = spreadValues(rowCount * rowLength, 90210u, 6.f);
    auto result = runSoftmax(input, rowCount, rowLength);
    auto expected = softmaxReference(input, rowCount, rowLength);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));

    for (auto row = 0; row < rowCount; ++row)
    {
        auto total = 0.0;

        for (auto i = 0; i < rowLength; ++i)
            total += result[row * rowLength + i];

        check(isClose((float) total, 1.0, 1e-5));
    }
};

// The row a naive softmax loses: exp overflows float32 somewhere past 88, so
// every one of these logits exponentiates to infinity and the ratio comes back
// as a NaN. Subtracting the row maximum first leaves the same distribution and
// the largest term at exactly one.
auto tSoftmaxSurvivesLargeLogits = test("Kernels/softmaxSurvivesLargeLogits") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto wideRowLength = 16;

    auto input = sized(2 * wideRowLength);

    for (auto i = 0; i < wideRowLength; ++i)
    {
        input[i] = 1000.f + (float) i;
        input[wideRowLength + i] = -1000.f - (float) i;
    }

    auto result = runSoftmax(input, 2, wideRowLength);
    auto expected = softmaxReference(input, 2, wideRowLength);

    for (auto i = 0; i < result.size(); ++i)
    {
        check(std::isfinite(result[i]));
        check(isClose(result[i], expected[i], 1e-5));
    }
};

// A masked row: the sentinel a causal product stores has to exponentiate to
// exactly zero against any real maximum, which is the whole reason the value
// is what it is rather than -infinity. Half the row masked, and the surviving
// half has to come out as the softmax of itself.
auto tSoftmaxTurnsMaskedScoresIntoZero =
    test("Kernels/softmaxTurnsMaskedScoresIntoZero") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto keys = 24;
    constexpr auto live = 10;

    auto input = spreadValues(keys, 31337u, 8.f);

    for (auto i = live; i < keys; ++i)
        input[i] = causalMaskScore;

    auto result = runSoftmax(input, 1, keys);

    auto largest = (double) input[0];

    for (auto i = 1; i < live; ++i)
        largest = std::max(largest, (double) input[i]);

    auto total = 0.0;

    for (auto i = 0; i < live; ++i)
        total += std::exp(input[i] - largest);

    for (auto i = 0; i < live; ++i)
        check(isClose(result[i], std::exp(input[i] - largest) / total, 1e-5));

    for (auto i = live; i < keys; ++i)
        check(result[i] == 0.f);
};

// A row far longer than a group, so every lane walks many elements and the
// strided shares interleave, and a count that is not a multiple of the group
// so the last share is ragged.
auto tSoftmaxRowsLongerThanAGroup = test("Kernels/softmaxRowsLongerThanAGroup") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto longRowLength = 1500;
    constexpr auto longRowCount = 3;

    auto input = spreadValues(longRowCount * longRowLength, 424242u, 4.f);
    auto result = runSoftmax(input, longRowCount, longRowLength);
    auto expected = softmaxReference(input, longRowCount, longRowLength);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));

    for (auto row = 0; row < longRowCount; ++row)
    {
        auto total = 0.0;

        for (auto i = 0; i < longRowLength; ++i)
            total += result[row * longRowLength + i];

        check(isClose((float) total, 1.0, 1e-5));
    }
};

// A row of equal logits is the one case the answer can be written down without
// computing it, and the one a missing normalisation still gets wrong.
auto tSoftmaxUniformRowIsUniform = test("Kernels/softmaxUniformRowIsUniform") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto flatRowLength = 40;

    auto input = sized(flatRowLength);

    for (auto i = 0; i < flatRowLength; ++i)
        input[i] = -7.25f;

    auto result = runSoftmax(input, 1, flatRowLength);

    for (auto i = 0; i < flatRowLength; ++i)
        check(isClose(result[i], 1.0 / flatRowLength, 1e-6));
};

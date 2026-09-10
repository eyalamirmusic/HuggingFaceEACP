#include "Common.h"

using namespace nano;
using namespace HF;
using namespace eacp::GPU;

namespace
{
// The reference, in double so that what it disagrees with the kernel about is
// the kernel and not the CPU's own float rounding. No mean subtraction, no
// bias, and the scale offset by one — which is Gemma's normalisation and not a
// LayerNorm with a term dropped.
Vector<float> rmsNormReference(const Vector<float>& input,
                               const Vector<float>& weight,
                               int rowCount,
                               int rowLength,
                               float epsilon)
{
    auto expected = sized(rowCount * rowLength);

    for (auto row = 0; row < rowCount; ++row)
    {
        auto base = row * rowLength;
        auto squares = 0.0;

        for (auto i = 0; i < rowLength; ++i)
            squares += (double) input[base + i] * input[base + i];

        auto scale = 1.0 / std::sqrt(squares / rowLength + epsilon);

        for (auto i = 0; i < rowLength; ++i)
            expected[base + i] =
                (float) (input[base + i] * scale * (1.0 + weight[i]));
    }

    return expected;
}

// Every row with statistics of its own, so a kernel that normalised the whole
// buffer at once, or reused one row's scale for the next, would not survive.
Vector<float> rowsWithDifferentScales(int rowCount, int rowLength)
{
    auto values = spreadValues(rowCount * rowLength, 20260910u, 4.f);

    for (auto row = 0; row < rowCount; ++row)
        for (auto i = 0; i < rowLength; ++i)
            values[row * rowLength + i] *= (float) (row + 1) * 0.75f;

    return values;
}

Vector<float> runRMSNorm(const Vector<float>& input,
                         const Vector<float>& weight,
                         int rowCount,
                         int rowLength,
                         int lanes = RMSNorm::groupWidth)
{
    auto inputBuffer = storageOf(input);
    auto weightBuffer = storageOf(weight);
    auto output = outputFor(rowCount * rowLength);

    auto kernel = RMSNorm {lanes};
    kernel.input = inputBuffer;
    kernel.weight = weightBuffer;
    kernel.output = output;
    kernel.rowLength = (unsigned) rowLength;

    return runGroupPerRow(kernel, output, rowCount, rowCount * rowLength);
}

// The tolerance every check here states: the reduction is 2048 float32 squares
// against a double sum, so the scale itself carries the accumulation's error
// and the store multiplies it through.
constexpr auto tolerance = 1e-4;

constexpr auto rowCount = 6;

// Not a multiple of the group, so a lane's strided share of the row is ragged
// and a walk that rounded up runs past the row.
constexpr auto raggedLength = 37;

// Gemma's own width.
constexpr auto modelWidth = 2048;
} // namespace

auto tRMSNormMatchesCpu = test("Kernels/rmsNormMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    for (const auto length: {raggedLength, 200, modelWidth})
    {
        auto input = rowsWithDifferentScales(rowCount, length);
        auto weight = spreadValues(length, 11u, 1.5f);

        auto result = runRMSNorm(input, weight, rowCount, length);
        auto expected =
            rmsNormReference(input, weight, rowCount, length, RMSNorm::gemmaEpsilon);

        check(result.size() == expected.size());

        for (auto i = 0; i < result.size(); ++i)
            check(isClose(result[i], expected[i], tolerance));
    }
};

// The `1 +` is the part a LayerNorm does not have, and a weight of exactly
// zero is what shows whether it is there: the answer is then the normalised
// row itself, and a kernel that multiplied by w rather than by 1 + w returns
// zeroes.
auto tRMSNormOffsetsTheScaleByOne = test("Kernels/rmsNormOffsetsTheScaleByOne") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto length = 64;

    auto input = spreadValues(length, 909u, 3.f);
    auto result = runRMSNorm(input, zeroes(length), 1, length);

    auto squares = 0.0;

    for (auto i = 0; i < length; ++i)
        squares += (double) input[i] * input[i];

    auto scale = 1.0 / std::sqrt(squares / length + RMSNorm::gemmaEpsilon);

    for (auto i = 0; i < length; ++i)
        check(isClose(result[i], input[i] * scale, tolerance));

    // The same row through a weight of -1, where 1 + w is zero: the output is
    // zero exactly, which no scale can produce by accident.
    auto minusOne = sized(length);

    for (auto i = 0; i < length; ++i)
        minusOne[i] = -1.f;

    auto zeroed = runRMSNorm(input, minusOne, 1, length);

    for (auto i = 0; i < length; ++i)
        check(zeroed[i] == 0.f);
};

// A row of zeroes: the mean square is zero, and epsilon is the only thing
// between rsqrt and a division by it. The answer is zero, finitely.
auto tRMSNormZeroRowStaysFinite = test("Kernels/rmsNormZeroRowStaysFinite") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto length = 32;

    auto weight = spreadValues(length, 33u, 2.f);
    auto result = runRMSNorm(zeroes(length), weight, 1, length);

    for (auto i = 0; i < length; ++i)
    {
        check(std::isfinite(result[i]));
        check(result[i] == 0.f);
    }
};

// A row of a single repeated magnitude is the one case the scale can be
// written down rather than accumulated: the mean square is that magnitude
// squared, so the answer is the sign pattern times (1 + w), and no reduction
// error can hide in it.
auto tRMSNormOfAUnitRowIsTheScale = test("Kernels/rmsNormOfAUnitRowIsTheScale") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto length = 48;

    auto input = sized(length);

    for (auto i = 0; i < length; ++i)
        input[i] = i % 2 == 0 ? 2.f : -2.f;

    auto weight = spreadValues(length, 77u, 1.f);
    auto result = runRMSNorm(input, weight, 1, length);

    const auto scale = 1.0 / std::sqrt(4.0 + RMSNorm::gemmaEpsilon);

    for (auto i = 0; i < length; ++i)
        check(isClose(result[i], input[i] * scale * (1.0 + weight[i]), tolerance));
};

// One compiled pipeline, two widths, two dispatches — eighteen layers of two
// norms each go through one program, and a kernel that had baked the width in
// gets the second dispatch wrong. The wide row is the model's, the narrow one
// is not a multiple of the group.
auto tRMSNormOneProgramTwoWidths = test("Kernels/rmsNormOneProgramTwoWidths") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    auto narrow = rowsWithDifferentScales(rowCount, raggedLength);
    auto wide = rowsWithDifferentScales(rowCount, modelWidth);
    auto narrowWeight = spreadValues(raggedLength, 55u, 1.25f);
    auto wideWeight = spreadValues(modelWidth, 77u, 1.25f);

    auto narrowInput = storageOf(narrow);
    auto wideInput = storageOf(wide);
    auto narrowWeightBuffer = storageOf(narrowWeight);
    auto wideWeightBuffer = storageOf(wideWeight);
    auto narrowOutput = outputFor(rowCount * raggedLength);
    auto wideOutput = outputFor(rowCount * modelWidth);

    auto kernel = RMSNorm {};
    kernel.prepare();

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();

        kernel.input = narrowInput;
        kernel.weight = narrowWeightBuffer;
        kernel.output = narrowOutput;
        kernel.rowLength = (unsigned) raggedLength;
        kernel.dispatchRows(pass, rowCount);

        kernel.input = wideInput;
        kernel.weight = wideWeightBuffer;
        kernel.output = wideOutput;
        kernel.rowLength = (unsigned) modelWidth;
        kernel.dispatchRows(pass, rowCount);
    }

    commands.commit();

    auto narrowResult = readBack(narrowOutput, rowCount * raggedLength);
    auto wideResult = readBack(wideOutput, rowCount * modelWidth);

    auto narrowExpected = rmsNormReference(
        narrow, narrowWeight, rowCount, raggedLength, RMSNorm::gemmaEpsilon);

    auto wideExpected = rmsNormReference(
        wide, wideWeight, rowCount, modelWidth, RMSNorm::gemmaEpsilon);

    for (auto i = 0; i < narrowResult.size(); ++i)
        check(isClose(narrowResult[i], narrowExpected[i], tolerance));

    for (auto i = 0; i < wideResult.size(); ++i)
        check(isClose(wideResult[i], wideExpected[i], tolerance));
};

// A group wider than the stock one, and wider than some of the rows it is
// asked to normalise here, so the lanes past the row contribute nothing to the
// sum but still have to reach the reduction. The answer is the same scalar
// reference the stock group is held to; a lane count that changed it would be
// a fold that only some threads arrived at.
auto tRMSNormInAWideGroup = test("Kernels/rmsNormInAWideGroup") = []
{
    if (!Device::shared().isValid())
        return;

    for (const auto lanes: {32, 192, 256})
    {
        for (const auto length: {raggedLength, modelWidth})
        {
            auto input = rowsWithDifferentScales(rowCount, length);
            auto weight = spreadValues(length, 33u, 1.5f);

            auto result = runRMSNorm(input, weight, rowCount, length, lanes);
            auto expected = rmsNormReference(
                input, weight, rowCount, length, RMSNorm::gemmaEpsilon);

            for (auto i = 0; i < result.size(); ++i)
                check(isClose(result[i], expected[i], tolerance));
        }
    }
};

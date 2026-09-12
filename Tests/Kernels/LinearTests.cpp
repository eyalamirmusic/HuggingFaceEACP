#include "Common.h"
#include "TiledProduct.h"

#include <cmath>
#include <cstdint>
#include <cstring>

using namespace nano;
using namespace HF;
using namespace eacp::GPU;

namespace
{
// output[row, o] = bias[o] + sum over k of input[row, k] * weight[o, k], which
// is nn.Linear's own indexing rather than a transpose of the matmul reference:
// the whole point of this kernel is that the weight is read in the layout a
// repo ships it in.
Vector<float> linearReference(const Vector<float>& input,
                              const Vector<float>& weight,
                              const Vector<float>& bias,
                              int rowCount,
                              int innerCount,
                              int outputWidth)
{
    auto expected = sized(rowCount * outputWidth);

    for (auto row = 0; row < rowCount; ++row)
    {
        for (auto column = 0; column < outputWidth; ++column)
        {
            auto total = 0.0;

            for (auto step = 0; step < innerCount; ++step)
                total += (double) input[row * innerCount + step]
                         * weight[column * innerCount + step];

            expected[row * outputWidth + column] = (float) (total + bias[column]);
        }
    }

    return expected;
}

Vector<float> runLinear(const Vector<float>& input,
                        const Vector<float>& weight,
                        const Vector<float>& bias,
                        int rowCount,
                        int innerCount,
                        int outputWidth)
{
    auto inputBuffer = storageOf(input);
    auto weightBuffer = storageOf(weight);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(rowCount * outputWidth);

    auto kernel = Linear {};
    kernel.input = inputBuffer;
    kernel.weights = weightBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.innerCount = (unsigned) innerCount;
    kernel.outputWidth = (unsigned) outputWidth;

    return runOverGrid(kernel, output, outputWidth, rowCount);
}

void checkLinearMatchesCpu(int rowCount,
                           int innerCount,
                           int outputWidth,
                           const Vector<float>& bias)
{
    auto input = spreadValues(rowCount * innerCount, 314159u, 2.f);
    auto weight = spreadValues(outputWidth * innerCount, 271828u, 3.f);

    auto result = runLinear(input, weight, bias, rowCount, innerCount, outputWidth);
    auto expected =
        linearReference(input, weight, bias, rowCount, innerCount, outputWidth);

    check(result.size() == expected.size());

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
}

// fp16 with a normal exponent, both signs, and a mantissa walking the whole
// field. Widening one is lossless, so the reference below is the same
// arithmetic the GPU does and the tolerance is float32 accumulation's rather
// than the storage format's.
std::uint16_t halfPattern(int index)
{
    const auto sign = index % 2 == 0 ? 0x0000u : 0x8000u;
    const auto mantissa = (unsigned) ((index * 137) % 1024);

    return (std::uint16_t) (sign | 0x3800u | mantissa);
}

float widenedHalf(std::uint16_t bits)
{
    const auto sign = (bits & 0x8000u) != 0 ? -1.f : 1.f;
    const auto exponent = (int) ((bits >> 10) & 0x1Fu) - 15;
    const auto mantissa = 1.f + (float) (bits & 0x3FFu) / 1024.f;

    return sign * mantissa * std::pow(2.f, (float) exponent);
}

Buffer packedHalfBuffer(int count)
{
    auto bytes = Vector<std::uint8_t> {};
    bytes.resize(((count + 1) / 2) * 4);

    for (auto index = 0; index < count; ++index)
    {
        const auto bits = halfPattern(index);
        std::memcpy(bytes.data() + index * 2, &bits, sizeof(bits));
    }

    return Device::shared().makeBuffer(
        bytes.data(), bytes.size(), BufferUsage::Storage);
}

Vector<float> widenedHalves(int count)
{
    auto values = sized(count);

    for (auto index = 0; index < count; ++index)
        values[index] = widenedHalf(halfPattern(index));

    return values;
}

// None of the three equal, none a multiple of the 8 x 8 dispatch group, so a
// kernel that confused the weight's output stride for its input stride, or
// leaned on the grid landing exactly on the matrix, fails here rather than on
// the first real model.
constexpr auto rowCount = 5;
constexpr auto innerCount = 7;
constexpr auto outputWidth = 3;
} // namespace

auto tLinearMatchesCpu = test("Kernels/linearMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    checkLinearMatchesCpu(
        rowCount, innerCount, outputWidth, spreadValues(outputWidth, 161803u, 1.f));
};

// A shape wide enough that a thread reaching one element past its row, or an
// accumulator that never got reset between output elements, shows up as a
// number rather than as a crash — and with the zero bias every Gemma
// projection binds.
auto tLinearWiderShapeWithoutBias = test("Kernels/linearWiderShapeWithoutBias") = []
{
    if (!Device::shared().isValid())
        return;

    checkLinearMatchesCpu(13, 41, 17, zeroes(17));
};

// The weight in this layout applied to the identity input is the weight's own
// transpose, so a swapped pair of strides is visible by eye rather than
// through a tolerance.
auto tLinearOfIdentityIsTheTransposedWeight =
    test("Kernels/linearOfIdentityIsTheTransposedWeight") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto width = 9;
    constexpr auto outputs = 6;

    auto identity = zeroes(width * width);

    for (auto row = 0; row < width; ++row)
        identity[row * width + row] = 1.f;

    auto weight = spreadValues(outputs * width, 555u, 5.f);
    auto result =
        runLinear(identity, weight, zeroes(outputs), width, width, outputs);

    for (auto row = 0; row < width; ++row)
        for (auto column = 0; column < outputs; ++column)
            check(result[row * outputs + column] == weight[column * width + row]);
};

// One pipeline, two shapes, two dispatches — what a transformer block needs of
// it, where the same program runs the 2048-wide projections and the 16384-wide
// feed forward.
auto tLinearOneProgramTwoShapes = test("Kernels/linearOneProgramTwoShapes") = []
{
    auto& device = Device::shared();

    if (!device.isValid())
        return;

    constexpr auto secondRows = 4;
    constexpr auto secondInner = 11;
    constexpr auto secondWidth = 6;

    auto firstInput = spreadValues(rowCount * innerCount, 1001u, 2.f);
    auto firstWeight = spreadValues(outputWidth * innerCount, 1002u, 2.f);
    auto firstBias = spreadValues(outputWidth, 1003u, 1.f);

    auto secondInput = spreadValues(secondRows * secondInner, 2001u, 2.f);
    auto secondWeight = spreadValues(secondWidth * secondInner, 2002u, 2.f);
    auto secondBias = spreadValues(secondWidth, 2003u, 1.f);

    auto firstInputBuffer = storageOf(firstInput);
    auto firstWeightBuffer = storageOf(firstWeight);
    auto firstBiasBuffer = storageOf(firstBias);
    auto firstOutput = outputFor(rowCount * outputWidth);

    auto secondInputBuffer = storageOf(secondInput);
    auto secondWeightBuffer = storageOf(secondWeight);
    auto secondBiasBuffer = storageOf(secondBias);
    auto secondOutput = outputFor(secondRows * secondWidth);

    auto kernel = Linear {};
    kernel.prepare();

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();

        kernel.input = firstInputBuffer;
        kernel.weights = firstWeightBuffer;
        kernel.bias = firstBiasBuffer;
        kernel.output = firstOutput;
        kernel.innerCount = (unsigned) innerCount;
        kernel.outputWidth = (unsigned) outputWidth;
        pass.dispatch(kernel, outputWidth, rowCount);

        kernel.input = secondInputBuffer;
        kernel.weights = secondWeightBuffer;
        kernel.bias = secondBiasBuffer;
        kernel.output = secondOutput;
        kernel.innerCount = (unsigned) secondInner;
        kernel.outputWidth = (unsigned) secondWidth;
        pass.dispatch(kernel, secondWidth, secondRows);
    }

    commands.commit();

    auto firstResult = readBack(firstOutput, rowCount * outputWidth);
    auto secondResult = readBack(secondOutput, secondRows * secondWidth);

    auto firstExpected = linearReference(
        firstInput, firstWeight, firstBias, rowCount, innerCount, outputWidth);

    auto secondExpected = linearReference(
        secondInput, secondWeight, secondBias, secondRows, secondInner, secondWidth);

    for (auto i = 0; i < firstResult.size(); ++i)
        check(isClose(firstResult[i], firstExpected[i], 1e-5));

    for (auto i = 0; i < secondResult.size(); ++i)
        check(isClose(secondResult[i], secondExpected[i], 1e-5));
};

// The packed-half variant over the same shapes, against a reference driven by
// the widened halves: fp16 to fp32 is exact, so the two have to agree to
// float32 accumulation and not to fp16's three digits. An odd element count,
// so the last output's thread reads the half in the word the padding
// completes.
auto tHalfWeightLinearMatchesCpu = test("Kernels/halfWeightLinearMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto inner = 7;
    constexpr auto outputs = 3;
    constexpr auto rows = 5;

    auto input = spreadValues(rows * inner, 424242u, 2.f);
    auto bias = spreadValues(outputs, 242424u, 1.f);
    auto weight = widenedHalves(outputs * inner);

    auto inputBuffer = storageOf(input);
    auto weightBuffer = packedHalfBuffer(outputs * inner);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(rows * outputs);

    auto kernel = HalfWeightLinear {};
    kernel.input = inputBuffer;
    kernel.weights = weightBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.innerCount = (unsigned) inner;
    kernel.outputWidth = (unsigned) outputs;

    auto result = runOverGrid(kernel, output, outputs, rows);
    auto expected = linearReference(input, weight, bias, rows, inner, outputs);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

// The bf16 variant, which is what Gemma's own projections are read through. The
// weights are narrowed by eacp's host packer and the reference runs on what
// that leaves, so the disagreement under test is the shader's widening and not
// the format's precision. An odd element count again, for the padding half.
auto tBFloat16WeightLinearMatchesCpu =
    test("Kernels/bfloat16WeightLinearMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto inner = 7;
    constexpr auto outputs = 3;
    constexpr auto rows = 5;

    auto input = spreadValues(rows * inner, 424242u, 2.f);
    auto bias = spreadValues(outputs, 242424u, 1.f);
    auto values = spreadValues(outputs * inner, 909090u, 3.f);

    auto widened = Vector<float> {};
    auto packed = TiledProduct::packedBFloat16s(values, widened);

    auto inputBuffer = storageOf(input);
    auto weightBuffer = storageOf(packed);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(rows * outputs);

    auto kernel = BFloat16WeightLinear {};
    kernel.input = inputBuffer;
    kernel.weights = weightBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.innerCount = (unsigned) inner;
    kernel.outputWidth = (unsigned) outputs;

    auto result = runOverGrid(kernel, output, outputs, rows);
    auto expected = linearReference(input, widened, bias, rows, inner, outputs);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

// The quantized form of the same product. The weights go through the very
// quantizer the loader calls and the reference reads them back through its
// dequantizer, so what is under test is the shader's byte read and its
// per-block scale rather than the format's accuracy — and the tolerance is the
// exact path's, because against the dequantized weights this arithmetic is
// exact too.
//
// An inner extent of 32 is one whole block to a row, which is what the format
// asks of a weight's contiguous dimension, and six outputs make the table a
// whole number of scale words.
auto tInt8WeightLinearMatchesCpu = test("Kernels/int8WeightLinearMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto inner = 32;
    constexpr auto outputs = 6;
    constexpr auto rows = 5;

    auto input = spreadValues(rows * inner, 424242u, 2.f);
    auto bias = spreadValues(outputs, 242424u, 1.f);
    auto values = spreadValues(outputs * inner, 919191u, 3.f);

    auto widened = Vector<float> {};
    auto packed = TiledProduct::quantizedInt8Blocks(values, widened);

    auto inputBuffer = storageOf(input);
    auto weightBuffer = storageOf(packed);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(rows * outputs);

    auto kernel = Int8WeightLinear {};
    kernel.input = inputBuffer;
    kernel.weights = weightBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.innerCount = (unsigned) inner;
    kernel.outputWidth = (unsigned) outputs;

    auto result = runOverGrid(kernel, output, outputs, rows);
    auto expected = linearReference(input, widened, bias, rows, inner, outputs);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
};

namespace
{
// SplitLinear's own runner: it decides its grid from the shape, so this
// dispatches through the kernel rather than over a grid the test names, and
// reads back the buffer the caller bound — which for a residual store is the
// one it was seeded with.
template <typename Kernel>
Vector<float> runSplitLinear(Kernel& kernel,
                             const eacp::GPU::Buffer& output,
                             int outputWidth,
                             int rowCount)
{
    kernel.prepare();

    auto commands = Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        kernel.dispatch(pass, outputWidth, rowCount);
    }

    commands.commit();
    return readBack(output, rowCount * outputWidth);
}

// The reference the split form answers to, with the two folded stages applied
// where the kernel applies them: the tanh GELU on the result, and the residual
// added to whatever the target already held.
Vector<float> splitReference(const Vector<float>& input,
                             const Vector<float>& weight,
                             const Vector<float>& bias,
                             const Vector<float>& carried,
                             int rowCount,
                             int innerCount,
                             int outputWidth,
                             bool gelu,
                             bool residual)
{
    auto expected =
        linearReference(input, weight, bias, rowCount, innerCount, outputWidth);

    for (auto index = 0; index < expected.size(); ++index)
    {
        auto value = (double) expected[index];

        if (gelu)
            value = tanhGeluReference(value);

        expected[index] = (float) (value + (residual ? carried[index] : 0.f));
    }

    return expected;
}

void checkSplitLinearMatchesCpu(int splitCount,
                                int rowCount,
                                int innerCount,
                                int outputWidth,
                                bool gelu = false,
                                bool residual = false)
{
    auto input = spreadValues(rowCount * innerCount, 314159u, 2.f);
    auto weight = spreadValues(outputWidth * innerCount, 271828u, 3.f);
    auto bias = spreadValues(outputWidth, 161803u, 1.f);
    auto carried = spreadValues(rowCount * outputWidth, 141421u, 1.f);

    auto inputBuffer = storageOf(input);
    auto weightBuffer = storageOf(weight);
    auto biasBuffer = storageOf(bias);
    auto output = storageOf(carried);

    auto kernel = SplitLinear {splitCount};
    kernel.input = inputBuffer;
    kernel.weights = weightBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.innerCount = (unsigned) innerCount;
    kernel.outputWidth = (unsigned) outputWidth;
    kernel.rowCount = (unsigned) rowCount;
    kernel.gelu = gelu ? 1u : 0u;
    kernel.residual = residual ? 1u : 0u;

    auto result = runSplitLinear(kernel, output, outputWidth, rowCount);
    auto expected = splitReference(input,
                                   weight,
                                   bias,
                                   carried,
                                   rowCount,
                                   innerCount,
                                   outputWidth,
                                   gelu,
                                   residual);

    check(result.size() == expected.size());

    // The inner sum's own error, not the answer's: a 2048-term product's
    // partial sums reach far past what it cancels to. See dotProductTolerance.
    const auto tolerance = TiledProduct::dotProductTolerance(innerCount);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], tolerance));
}
} // namespace

// The split form at the split counts that pick out its two folds: below the
// group width, where a group holds several outputs and folds them through a
// shared array, and at and above it, where a group holds one and folds it with
// groupSum — with 32 for the simdSum fold, and the decoder's own two, 128 in a
// projection and 32 in the logits.
//
// **And at every record width the loop picks between.** A lane's run is
// sixteen weights, eight, four or one, the widest of them the row divides by,
// so the four inner extents here are 16, 24, 12 and 7 — one for each branch —
// beside the model's own 2048, which takes the sixteen-wide one.
//
// The model's row width is in here at the shape a decode step has it — 2048 in,
// one row — with a modest column count rather than the logits' 256,000, since
// what a wide output tests is the dispatch height and that is the same
// arithmetic at 384 as at a quarter of a million.
auto tSplitLinearMatchesCpu = test("Kernels/splitLinearMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    for (const auto splits: {8, 32, 64, 96, 128})
    {
        checkSplitLinearMatchesCpu(splits, rowCount, innerCount, outputWidth);

        for (const auto inner: {16, 24, 12, 7})
            checkSplitLinearMatchesCpu(splits, 3, inner, 5);

        checkSplitLinearMatchesCpu(splits, 1, 2048, 384);
        checkSplitLinearMatchesCpu(splits, 2, 2048, 256);
    }
};

// The two stages folded into the store, and the reason the fold has to happen
// on exactly one lane of the group: the residual is a read-modify-write, so a
// kernel that stored it from every lane holding the folded value would add it
// as many times as the group has lanes.
auto tSplitLinearFoldsGeluAndResidual =
    test("Kernels/splitLinearFoldsGeluAndResidual") = []
{
    if (!Device::shared().isValid())
        return;

    for (const auto splits: {8, 96})
    {
        checkSplitLinearMatchesCpu(splits, 2, 384, 1536, true, false);
        checkSplitLinearMatchesCpu(splits, 2, 1536, 384, false, true);
        checkSplitLinearMatchesCpu(splits, 3, 41, 17, true, true);
    }
};

namespace
{
void checkHalfSplitLinear(int splitCount, int rows, int inner, int outputs)
{
    auto input = spreadValues(rows * inner, 424242u, 2.f);
    auto bias = spreadValues(outputs, 242424u, 1.f);
    auto weight = widenedHalves(outputs * inner);

    auto inputBuffer = storageOf(input);
    auto weightBuffer = packedHalfBuffer(outputs * inner);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(rows * outputs);

    auto kernel = HalfWeightSplitLinear {splitCount};
    kernel.input = inputBuffer;
    kernel.weights = weightBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.innerCount = (unsigned) inner;
    kernel.outputWidth = (unsigned) outputs;
    kernel.rowCount = (unsigned) rows;
    kernel.gelu = 0u;
    kernel.residual = 0u;

    auto result = runSplitLinear(kernel, output, outputs, rows);
    auto expected = linearReference(input, weight, bias, rows, inner, outputs);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-5));
}
} // namespace

// The packed-half split form, which is what a logits projection over a packed
// embedding would run, over the same widened halves the dense one is checked
// against — and over the same four inner extents, one for each record width the
// loop picks between.
auto tHalfWeightSplitLinearMatchesCpu =
    test("Kernels/halfWeightSplitLinearMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    for (const auto inner: {16, 24, 12, 8, 7})
        checkHalfSplitLinear(32, 2, inner, 5);
};

namespace
{
void checkBFloat16SplitLinear(int splitCount, int rows, int inner, int outputs)
{
    auto input = spreadValues(rows * inner, 515151u, 2.f);
    auto bias = spreadValues(outputs, 626262u, 1.f);
    auto values = spreadValues(outputs * inner, 737373u, 3.f);

    auto widened = Vector<float> {};
    auto packed = TiledProduct::packedBFloat16s(values, widened);

    auto inputBuffer = storageOf(input);
    auto weightBuffer = storageOf(packed);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(rows * outputs);

    auto kernel = BFloat16WeightSplitLinear {splitCount};
    kernel.input = inputBuffer;
    kernel.weights = weightBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.innerCount = (unsigned) inner;
    kernel.outputWidth = (unsigned) outputs;
    kernel.rowCount = (unsigned) rows;
    kernel.gelu = 0u;
    kernel.residual = 0u;

    auto result = runSplitLinear(kernel, output, outputs, rows);
    auto expected = linearReference(input, widened, bias, rows, inner, outputs);

    const auto tolerance = TiledProduct::dotProductTolerance(inner);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], tolerance));
}
} // namespace

// The bf16 split form, which is the product a decode step spends nearly all its
// bandwidth in — every projection and the logits row against the tied
// embedding. All four of the kernel's inner loops are covered, which for bf16
// is one, two or four `readBFloat16x4` of two words each and then the
// element-at-a-time read: 16 takes the sixteen-wide record, 24 the eight-wide,
// 12 the four-wide and 7 and 11 the single elements.
//
// The model's own decode shape is in here too — one row, 2048 in — since that
// is the only place the wide bf16 read is on the hot path.
auto tBFloat16WeightSplitLinearMatchesCpu =
    test("Kernels/bfloat16WeightSplitLinearMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    for (const auto splits: {8, 32, 64, 96, 128})
    {
        for (const auto inner: {16, 24, 12, 8, 11})
            checkBFloat16SplitLinear(splits, 2, inner, 5);

        checkBFloat16SplitLinear(splits, 5, 7, 3);
        checkBFloat16SplitLinear(splits, 1, 2048, 384);
    }
};

namespace
{
void checkInt8SplitLinear(int splitCount, int rows, int inner, int outputs)
{
    auto input = spreadValues(rows * inner, 515151u, 2.f);
    auto bias = spreadValues(outputs, 626262u, 1.f);
    auto values = spreadValues(outputs * inner, 747474u, 3.f);

    auto widened = Vector<float> {};
    auto packed = TiledProduct::quantizedInt8Blocks(values, widened);

    auto inputBuffer = storageOf(input);
    auto weightBuffer = storageOf(packed);
    auto biasBuffer = storageOf(bias);
    auto output = outputFor(rows * outputs);

    auto kernel = Int8WeightSplitLinear {splitCount};
    kernel.input = inputBuffer;
    kernel.weights = weightBuffer;
    kernel.bias = biasBuffer;
    kernel.output = output;
    kernel.innerCount = (unsigned) inner;
    kernel.outputWidth = (unsigned) outputs;
    kernel.rowCount = (unsigned) rows;
    kernel.gelu = 0u;
    kernel.residual = 0u;

    auto result = runSplitLinear(kernel, output, outputs, rows);
    auto expected = linearReference(input, widened, bias, rows, inner, outputs);

    const auto tolerance = TiledProduct::dotProductTolerance(inner);

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], tolerance));
}
} // namespace

// The quantized split form, which is the product a decode step spends nearly
// all its bandwidth in once the weights are quantized — every projection and
// the logits row against the tied embedding.
//
// **Only the sixteen-wide branch is reachable here**, and deliberately: the
// block format asks that a weight's contiguous dimension be a whole number of
// thirty-twos, so an inner extent it accepts always divides by sixteen and the
// three narrower loops beside it can never be the one a quantized weight takes.
// The model's own decode shape is in the list for the reason the bf16 case has
// it — one row, 2048 in, which is where the sixteen-wide read is on the hot
// path.
//
// Every split count is covered because each leaves a lane a different run: at
// 128 lanes over 2048 a lane reads exactly one record of sixteen, which is the
// decoder's own shape and the one the count was chosen for; at 256 it reads one
// and half the lanes read none; at 8 it walks sixteen records and fetches
// sixteen scales, one block apart each time.
auto tInt8WeightSplitLinearMatchesCpu =
    test("Kernels/int8WeightSplitLinearMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    for (const auto splits: {8, 32, 64, 96, 128, 256})
    {
        checkInt8SplitLinear(splits, 2, 32, 6);
        checkInt8SplitLinear(splits, 5, 64, 3);
        checkInt8SplitLinear(splits, 1, 2048, 384);
    }
};

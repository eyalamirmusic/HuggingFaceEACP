#include "Common.h"

using namespace nano;
using namespace HF;
using namespace eacp::GPU;

namespace
{
// The first index holding the row's largest unsuppressed logit, scanned
// forwards so that an equal value never displaces an earlier index — which is
// what torch.argmax documents and what the kernel has to agree with.
Vector<std::uint32_t> argmaxReference(const Vector<float>& logits,
                                      const Vector<float>& mask,
                                      int rowCount,
                                      int rowLength)
{
    auto expected = unsignedSized(rowCount);

    for (auto row = 0; row < rowCount; ++row)
    {
        auto base = row * rowLength;
        auto best = -1;

        for (auto index = 0; index < rowLength; ++index)
        {
            if (mask[index] != 0.f)
                continue;

            if (best < 0 || logits[base + index] > logits[base + best])
                best = index;
        }

        expected[row] = (std::uint32_t) (best < 0 ? 0 : best);
    }

    return expected;
}

Vector<std::uint32_t> runArgmax(const Vector<float>& logits,
                                const Vector<float>& mask,
                                int rowCount,
                                int rowLength)
{
    auto& device = Device::shared();

    auto logitBuffer = storageOf(logits);
    auto maskBuffer = storageOf(mask);
    auto indices = device.makeBuffer((int) sizeof(std::uint32_t) * rowCount,
                                     BufferUsage::Storage);

    auto kernel = Argmax {};
    kernel.prepare(device, rowCount, rowLength);

    auto commands = device.makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        kernel.encode(pass,
                      BufferRange::of(logitBuffer),
                      maskBuffer,
                      indices,
                      rowCount,
                      rowLength);
    }

    commands.commit();
    return readBackUnsigned(indices, rowCount);
}

Vector<float> suppressing(int rowLength, std::initializer_list<int> tokens)
{
    auto values = zeroes(rowLength);

    for (auto token: tokens)
        values[token] = 1.f;

    return values;
}

// A prime, so nothing in the scan lines up with the 64-wide group and a thread
// that walked a rounded-up row runs off its own end.
constexpr auto rowLength = 37;

// Gemma's vocabulary, which is what a real step reduces over: five hundred
// groups of candidates folded to one, five times what Whisper's kernel was
// dispatched at.
constexpr auto vocabulary = 256000;

// Rows whose answers can be written down: the maximum at the first index, at
// the middle, at the last, and a row where the largest value occurs twice.
Vector<float> handBuiltRows()
{
    auto values = sized(4 * rowLength);

    for (auto i = 0; i < rowLength; ++i)
    {
        values[i] = -(float) i;
        values[rowLength + i] = -std::abs((float) i - 18.f);
        values[2 * rowLength + i] = (float) i;
        values[3 * rowLength + i] = i == 5 || i == 29 ? 4.f : (float) i * 0.01f;
    }

    return values;
}
} // namespace

auto tArgmaxFindsTheMaximum = test("Kernels/argmaxFindsTheMaximum") = []
{
    if (!Device::shared().isValid())
        return;

    auto logits = handBuiltRows();
    auto result = runArgmax(logits, zeroes(rowLength), 4, rowLength);

    check(result.size() == 4);
    check(result[0] == 0u);
    check(result[1] == 18u);
    check(result[2] == (std::uint32_t) (rowLength - 1));
    check(result[3] == 5u);
};

auto tArgmaxMatchesCpu = test("Kernels/argmaxMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto rowCount = 11;

    auto logits = spreadValues(rowCount * rowLength, 606u, 8.f);
    auto mask = suppressing(rowLength, {0, 1, 2, 17, 36});

    auto result = runArgmax(logits, mask, rowCount, rowLength);
    auto expected = argmaxReference(logits, mask, rowCount, rowLength);

    check(result.size() == expected.size());

    for (auto row = 0; row < result.size(); ++row)
    {
        check(result[row] == expected[row]);
        check(mask[(int) result[row]] == 0.f);
    }
};

// Suppressing the winner has to hand the row to the runner-up rather than to
// the next index or to nothing: a kernel that masked after the reduction, or
// that clamped a suppressed score instead of skipping it, gets this wrong.
auto tArgmaxSkipsTheSuppressedMaximum =
    test("Kernels/argmaxSkipsTheSuppressedMaximum") = []
{
    if (!Device::shared().isValid())
        return;

    auto logits = sized(rowLength);

    for (auto i = 0; i < rowLength; ++i)
        logits[i] = (float) i * 0.5f;

    logits[7] = 100.f;
    logits[23] = 50.f;

    auto open = runArgmax(logits, zeroes(rowLength), 1, rowLength);
    check(open[0] == 7u);

    auto withoutFirst = runArgmax(logits, suppressing(rowLength, {7}), 1, rowLength);
    check(withoutFirst[0] == 23u);

    auto withoutBoth =
        runArgmax(logits, suppressing(rowLength, {7, 23}), 1, rowLength);
    check(withoutBoth[0] == (std::uint32_t) (rowLength - 1));
};

// One token left standing, and it holds the row's smallest logit: the answer
// is that token whatever the arithmetic says, which is the case a mask applied
// as a bias rather than as an exclusion cannot produce.
auto tArgmaxTakesTheOnlyTokenLeft = test("Kernels/argmaxTakesTheOnlyTokenLeft") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto survivor = 11;

    auto logits = sized(rowLength);

    for (auto i = 0; i < rowLength; ++i)
        logits[i] = 1000.f - (float) i;

    logits[survivor] = -1000.f;

    auto mask = sized(rowLength);

    for (auto i = 0; i < rowLength; ++i)
        mask[i] = i == survivor ? 0.f : 1.f;

    auto result = runArgmax(logits, mask, 1, rowLength);
    check(result[0] == (std::uint32_t) survivor);
};

// The shape a Gemma decode step is: one row of 256,000 logits, which is the
// width that had to be confirmed rather than assumed — the two-dispatch split
// spreads it over five hundred groups, and the second stage folds five hundred
// candidates with sixty-four lanes, so both stages are walking many elements
// per thread rather than one.
//
// The winner sits at neither end and the runner-up in a different group
// entirely, so a fold that only ever compared within a group, or that kept the
// last candidate rather than the best, comes back with something else.
auto tArgmaxOverTheVocabulary = test("Kernels/argmaxOverTheVocabulary") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto winner = 137425;
    constexpr auto runnerUp = 318;

    check(winner != 0 && winner != vocabulary - 1);

    auto logits = spreadValues(vocabulary, 256000u, 4.f);
    logits[winner] = 900.f;
    logits[runnerUp] = 800.f;

    auto open = runArgmax(logits, zeroes(vocabulary), 1, vocabulary);
    check(open[0] == (std::uint32_t) winner);

    auto masked = suppressing(vocabulary, {winner});
    auto result = runArgmax(logits, masked, 1, vocabulary);
    check(result[0] == (std::uint32_t) runnerUp);

    auto expected = argmaxReference(logits, masked, 1, vocabulary);
    check(result[0] == expected[0]);
};

// The last index of the whole vocabulary, which is where an unsigned index
// that overflowed a narrower type, or a partial group whose share of the row
// stopped short, would lose the answer.
auto tArgmaxFindsTheLastToken = test("Kernels/argmaxFindsTheLastToken") = []
{
    if (!Device::shared().isValid())
        return;

    auto logits = spreadValues(vocabulary, 4242u, 4.f);
    logits[vocabulary - 1] = 900.f;

    auto result = runArgmax(logits, zeroes(vocabulary), 1, vocabulary);
    check(result[0] == (std::uint32_t) (vocabulary - 1));
};

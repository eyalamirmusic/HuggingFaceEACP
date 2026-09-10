#include "Common.h"

using namespace nano;
using namespace HF;
using namespace eacp::GPU;

namespace
{
// The rotation from its definition, in double, over the same layout the kernel
// walks — the angle recomputed here rather than read from the table, so a
// table built wrong and a kernel reading it wrong do not agree with each
// other.
Vector<float> ropeReference(const Vector<float>& input,
                            int rows,
                            int heads,
                            int headDim,
                            int firstPosition,
                            double theta)
{
    const auto half = headDim / 2;
    const auto rowStride = heads * headDim;

    auto expected = input;

    for (auto row = 0; row < rows; ++row)
        for (auto head = 0; head < heads; ++head)
            for (auto channel = 0; channel < half; ++channel)
            {
                const auto inverseFrequency =
                    1.0 / std::pow(theta, 2.0 * channel / (double) headDim);
                const auto angle = (double) (firstPosition + row) * inverseFrequency;

                const auto low = row * rowStride + head * headDim + channel;
                const auto high = low + half;

                const auto x = (double) input[low];
                const auto y = (double) input[high];

                expected[low] = (float) (x * std::cos(angle) - y * std::sin(angle));
                expected[high] = (float) (y * std::cos(angle) + x * std::sin(angle));
            }

    return expected;
}

struct RotaryBuffers
{
    Buffer cosines;
    Buffer sines;
};

RotaryBuffers uploadTable(const RotaryTable& table)
{
    return {storageOf(table.cosines), storageOf(table.sines)};
}

// In place: the input buffer is bound to both slots, which is what the model
// does after a projection writes q or k. The two elements a thread stores are
// the two it read, so nothing else has to be true for that to be safe.
Vector<float> runRoPE(const Vector<float>& input,
                      const RotaryBuffers& table,
                      int rows,
                      int heads,
                      int headDim,
                      int firstPosition)
{
    auto values = storageOf(input);

    auto kernel = RoPE {};
    kernel.input = values;
    kernel.cosines = table.cosines;
    kernel.sines = table.sines;
    kernel.output = values;
    kernel.rowStride = (unsigned) (heads * headDim);
    kernel.headDim = (unsigned) headDim;
    kernel.headCount = (unsigned) heads;
    kernel.firstPosition = (unsigned) firstPosition;
    kernel.prepare();

    auto commands = Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        kernel.dispatch(pass, rows, heads, headDim);
    }

    commands.commit();
    return readBack(values, input.size());
}

// The tolerance every check here states. The rotation is two products and a
// difference of float32 numbers whose cosine and sine came in already rounded,
// so what is left is a couple of ulps against the double reference.
constexpr auto tolerance = 1e-6;

// Gemma's own: 256-wide heads, eight of them on the query side and one on the
// key side, over a context of 8192.
constexpr auto headDim = 256;
constexpr auto queryHeads = 8;
constexpr auto maxPositions = 8192;
} // namespace

auto tRoPEMatchesCpu = test("Kernels/ropeMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto rows = 5;
    constexpr auto heads = 3;
    constexpr auto width = 8;
    constexpr auto positions = 64;

    auto table = uploadTable(makeRotaryTable(positions, width));
    auto input = spreadValues(rows * heads * width, 12345u, 2.f);

    for (const auto firstPosition: {0, 17})
    {
        auto result = runRoPE(input, table, rows, heads, width, firstPosition);
        auto expected =
            ropeReference(input, rows, heads, width, firstPosition, gemmaRopeTheta);

        check(result.size() == expected.size());

        for (auto i = 0; i < result.size(); ++i)
            check(isClose(result[i], expected[i], tolerance));
    }
};

// Position zero is the identity: every angle is zero, so cos is one and sin is
// zero and the buffer comes back bit for bit. A kernel that had the pair the
// wrong way round, or that rotated by a neighbouring channel, still returns
// the input here — which is why this is the first check and not the only one.
auto tRoPEAtPositionZeroIsTheIdentity =
    test("Kernels/ropeAtPositionZeroIsTheIdentity") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto heads = 2;
    constexpr auto width = 16;

    auto table = uploadTable(makeRotaryTable(8, width));
    auto input = spreadValues(heads * width, 4321u, 3.f);
    auto result = runRoPE(input, table, 1, heads, width, 0);

    for (auto i = 0; i < result.size(); ++i)
        check(result[i] == input[i]);
};

// Rotate-half, spelled out on a single head at position 1 with the lowest
// frequency channel, whose inverse frequency is exactly 1: the pair
// (x[0], x[half]) turns by one radian and nothing else touches it. A kernel
// that paired adjacent channels instead moves x[1], which is asserted to be
// untouched by any rotation but its own.
auto tRoPERotatesTheHalfPair = test("Kernels/ropeRotatesTheHalfPair") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto width = 8;
    constexpr auto half = width / 2;

    auto table = uploadTable(makeRotaryTable(4, width));

    auto input = zeroes(width);
    input[0] = 1.f;
    input[half] = 0.f;

    auto result = runRoPE(input, table, 1, 1, width, 1);

    check(isClose(result[0], std::cos(1.0), tolerance));
    check(isClose(result[half], std::sin(1.0), tolerance));

    for (auto i = 1; i < half; ++i)
    {
        check(result[i] == 0.f);
        check(result[half + i] == 0.f);
    }
};

// The property the whole scheme exists for: after rotating a query at p and a
// key at q, their dot product depends on p - q and not on p and q separately.
// So shifting both by the same amount leaves every score where it was, which
// is what makes a KV cache's absolute positions arbitrary.
//
// A relative-position property is also the one thing a sign error in the
// rotation cannot fake: flipping the sine's sign on both operands still gives
// a shift-invariant score, so the check below is run alongside the reference
// comparison above rather than instead of it.
auto tRoPEPreservesRelativePosition =
    test("Kernels/ropePreservesRelativePosition") = []
{
    if (!Device::shared().isValid())
        return;

    auto table = uploadTable(makeRotaryTable(maxPositions, headDim));

    // One query head and one key head, so the dot product below is over the
    // whole of each buffer.
    auto query = spreadValues(headDim, 2024u, 1.f);
    auto key = spreadValues(headDim, 2025u, 1.f);

    const auto dotAt = [&](int queryPosition, int keyPosition)
    {
        auto rotatedQuery = runRoPE(query, table, 1, 1, headDim, queryPosition);
        auto rotatedKey = runRoPE(key, table, 1, 1, headDim, keyPosition);

        auto total = 0.0;

        for (auto i = 0; i < headDim; ++i)
            total += (double) rotatedQuery[i] * rotatedKey[i];

        return total;
    };

    // Two positions eleven apart, then the same gap moved far up the context.
    const auto near = dotAt(11, 0);

    for (const auto shift: {1, 40, 1000, maxPositions - 12})
        check(isClose((float) dotAt(11 + shift, shift), near, 1e-5));

    // A different gap is a different score, or the dependence on relative
    // position would be no dependence at all.
    check(std::abs(dotAt(12, 0) - near) > 1e-3);
};

// Gemma's shapes through one pipeline: eight query heads and the one key head
// multi-query gives it, over the same table and at the same position. The
// heads are independent, so the key's rotation has to equal the rotation the
// first query head would get from the same numbers.
auto tRoPEOneProgramQueryAndKeyHeads =
    test("Kernels/ropeOneProgramQueryAndKeyHeads") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto rows = 3;
    constexpr auto firstPosition = 500;

    auto table = uploadTable(makeRotaryTable(maxPositions, headDim));

    auto queries = spreadValues(rows * queryHeads * headDim, 606u, 2.f);
    auto keys = sized(rows * headDim);

    // The key head is a copy of each row's first query head, so the two
    // dispatches have to agree element for element despite differing head
    // counts and row strides.
    for (auto row = 0; row < rows; ++row)
        for (auto i = 0; i < headDim; ++i)
            keys[row * headDim + i] = queries[row * queryHeads * headDim + i];

    auto rotatedQueries =
        runRoPE(queries, table, rows, queryHeads, headDim, firstPosition);
    auto rotatedKeys = runRoPE(keys, table, rows, 1, headDim, firstPosition);

    auto expected = ropeReference(
        queries, rows, queryHeads, headDim, firstPosition, gemmaRopeTheta);

    for (auto i = 0; i < rotatedQueries.size(); ++i)
        check(isClose(rotatedQueries[i], expected[i], tolerance));

    for (auto row = 0; row < rows; ++row)
        for (auto i = 0; i < headDim; ++i)
            check(rotatedKeys[row * headDim + i]
                  == rotatedQueries[row * queryHeads * headDim + i]);
};

// The rotation is a rotation, so it preserves the length of every pair and
// therefore of the whole head — at a position far enough up the context that a
// table built in float32 would have lost digits to the pow.
auto tRoPEPreservesNorm = test("Kernels/ropePreservesNorm") = []
{
    if (!Device::shared().isValid())
        return;

    auto table = uploadTable(makeRotaryTable(maxPositions, headDim));
    auto input = spreadValues(headDim, 8191u, 2.f);

    auto before = 0.0;

    for (auto i = 0; i < headDim; ++i)
        before += (double) input[i] * input[i];

    for (const auto position: {1, 4096, maxPositions - 1})
    {
        auto result = runRoPE(input, table, 1, 1, headDim, position);
        auto after = 0.0;

        for (auto i = 0; i < headDim; ++i)
            after += (double) result[i] * result[i];

        check(isClose((float) after, before, 1e-6));
    }
};

// The table itself, against the definition: the first channel is angle zero
// whatever the position, the last channel's inverse frequency is
// theta^-((headDim - 2) / headDim), and every entry is a cosine and a sine of
// the same angle.
auto tRotaryTableMatchesDefinition =
    test("Kernels/rotaryTableMatchesDefinition") = []
{
    constexpr auto positions = 64;
    constexpr auto width = 16;
    constexpr auto half = width / 2;

    auto table = makeRotaryTable(positions, width);

    check(table.cosines.size() == positions * half);
    check(table.sines.size() == positions * half);

    for (auto position = 0; position < positions; ++position)
    {
        // Channel zero is inverse frequency 1, so its angle is the position.
        check(isClose(
            table.cosines[position * half], std::cos((double) position), tolerance));

        for (auto channel = 0; channel < half; ++channel)
        {
            const auto angle =
                (double) position
                / std::pow(gemmaRopeTheta, 2.0 * channel / (double) width);

            const auto at = position * half + channel;

            check(isClose(table.cosines[at], std::cos(angle), tolerance));
            check(isClose(table.sines[at], std::sin(angle), tolerance));
        }
    }
};

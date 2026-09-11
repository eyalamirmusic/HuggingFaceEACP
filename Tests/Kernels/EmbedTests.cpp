#include "Common.h"
#include "TiledProduct.h"

using namespace nano;
using namespace HF;
using namespace eacp::GPU;

namespace
{
// output[t, c] = scale * tokenTable[token[t], c], written from the definition
// rather than from the kernel's index arithmetic.
Vector<float> embedReference(const Vector<std::uint32_t>& tokens,
                             const Vector<float>& tokenTable,
                             int width,
                             float scale)
{
    auto expected = sized(tokens.size() * width);

    for (auto step = 0; step < tokens.size(); ++step)
        for (auto channel = 0; channel < width; ++channel)
            expected[step * width + channel] =
                scale * tokenTable[(int) tokens[step] * width + channel];

    return expected;
}

Vector<float> runEmbed(const Vector<std::uint32_t>& tokens,
                       const Vector<float>& tokenTable,
                       int width,
                       float scale)
{
    auto tokenBuffer = storageOf(tokens);
    auto tokenTableBuffer = storageOf(tokenTable);
    auto output = outputFor(tokens.size() * width);

    auto kernel = Embed {};
    kernel.tokens = tokenBuffer;
    kernel.tokenTable = tokenTableBuffer;
    kernel.output = output;
    kernel.width = (unsigned) width;
    kernel.scale = scale;

    return runOverGrid(kernel, output, width, tokens.size());
}

Vector<std::uint32_t> tokensOf(std::initializer_list<std::uint32_t> ids)
{
    auto values = unsignedSized((int) ids.size());
    auto at = 0;

    for (auto id: ids)
        values[at++] = id;

    return values;
}

// A table whose every element says which row and which column it is, so a
// gather that read the right number out of the wrong row is a wrong number
// rather than a wrong tolerance. Every value here is an integer plus a quarter
// of one, so float32 holds it exactly up to a vocabulary of eight million.
Vector<float> countedTable(int rowCount, int width, float rowScale)
{
    auto values = sized(rowCount * width);

    for (auto row = 0; row < rowCount; ++row)
        for (auto column = 0; column < width; ++column)
            values[row * width + column] =
                rowScale * (float) row + 0.25f * (float) column;

    return values;
}

// Gemma's vocabulary, which is the number that decides whether a token id
// survives the trip: every id below 2^23 has a float bit pattern with an
// exponent field of zero, so the whole vocabulary arrives as what a float
// would call a subnormal.
constexpr auto vocabulary = 256000;

// sqrt(2048) rounded through bfloat16, which is the factor Gemma 2B's
// embedding is actually multiplied by — see Embed.h. Exactly representable in
// float32, so the assertions below can be equalities.
constexpr auto gemmaScale = 45.25f;
} // namespace

auto tEmbedMatchesCpu = test("Kernels/embedMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto width = 5;
    constexpr auto smallVocabulary = 13;

    auto tokens = tokensOf({7, 0, 12, 3, 3, 11});
    auto tokenTable = spreadValues(smallVocabulary * width, 4242u, 2.f);

    auto result = runEmbed(tokens, tokenTable, width, gemmaScale);
    auto expected = embedReference(tokens, tokenTable, width, gemmaScale);

    check(result.size() == expected.size());

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-6));
};

// The scale is the whole difference between this kernel and a plain gather,
// and it is a uniform so the caller can hand over the bf16-rounded constant
// the reference implementation uses rather than sqrt(width) as a float. Three
// factors through one pipeline: one that changes nothing, Gemma's own, and one
// that is neither.
auto tEmbedAppliesTheScale = test("Kernels/embedAppliesTheScale") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto width = 4;
    constexpr auto smallVocabulary = 9;

    auto tokens = tokensOf({5, 2, 8});
    auto tokenTable = countedTable(smallVocabulary, width, 2.f);

    for (const auto scale: {1.f, gemmaScale, -0.5f})
    {
        auto result = runEmbed(tokens, tokenTable, width, scale);

        for (auto step = 0; step < tokens.size(); ++step)
            for (auto column = 0; column < width; ++column)
            {
                const auto gathered =
                    2.f * (float) tokens[step] + 0.25f * (float) column;

                check(result[step * width + column] == scale * gathered);
            }
    }

    // sqrt(2048) is 45.254833, and the bf16 constant Gemma multiplies by is
    // 45.25. The two are a thousand float ulps apart, which is why the number
    // is the caller's and not this kernel's.
    check(gemmaScale != (float) std::sqrt(2048.0));
};

// The id itself, over Gemma's whole vocabulary rather than a toy one. The
// table is built so the answer is the token id as a float and nothing else, so
// an id that lost a bit on the way through the buffer — the bitcast reading a
// subnormal, a backend flushing one to zero — comes back as row zero's value
// and fails by 255999 rather than by a tolerance.
auto tEmbedReadsWholeVocabularyExactly =
    test("Kernels/embedReadsWholeVocabularyExactly") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto width = 3;

    // Gemma's own specials at the bottom — pad 0, eos 1, bos 2 — then the
    // interior and the last row of the table.
    auto tokens = tokensOf({0, 1, 2, 220, 65535, 65536, vocabulary - 1});
    auto tokenTable = countedTable(vocabulary, width, 1.f);

    auto result = runEmbed(tokens, tokenTable, width, 1.f);

    for (auto step = 0; step < tokens.size(); ++step)
        for (auto column = 0; column < width; ++column)
            check(result[step * width + column]
                  == (float) tokens[step] + 0.25f * (float) column);
};

// The packed table, which is what the tied embedding is once the checkpoint's
// bf16 rows go up as they lie: the gather has to widen exactly what the logits
// product widens, since the two read the same buffer. The reference runs on the
// narrowed values, so what this asserts is the gather's indexing — a row picked
// out of a buffer whose elements are half the width its index counts in.
//
// An odd model width, so a row starts in the high half of a word as often as in
// the low one and a gather that assumed rows begin on word boundaries fails.
auto tBFloat16WeightEmbedMatchesCpu =
    test("Kernels/bfloat16WeightEmbedMatchesCpu") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto width = 5;
    constexpr auto smallVocabulary = 13;

    auto tokens = tokensOf({7, 0, 12, 3, 3, 11});
    auto values = spreadValues(smallVocabulary * width, 4242u, 2.f);

    auto widened = Vector<float> {};
    auto packed = TiledProduct::packedBFloat16s(values, widened);

    auto tokenBuffer = storageOf(tokens);
    auto tableBuffer = storageOf(packed);
    auto output = outputFor(tokens.size() * width);

    auto kernel = BFloat16WeightEmbed {};
    kernel.tokens = tokenBuffer;
    kernel.tokenTable = tableBuffer;
    kernel.output = output;
    kernel.width = (unsigned) width;
    kernel.scale = gemmaScale;

    auto result = runOverGrid(kernel, output, width, tokens.size());
    auto expected = embedReference(tokens, widened, width, gemmaScale);

    check(result.size() == expected.size());

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], 1e-6));
};

// The shape a decode step has: one token, the model's own width, through the
// same pipeline the prompt's many rows went through.
auto tEmbedOneRowAtModelWidth = test("Kernels/embedOneRowAtModelWidth") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto width = 2048;
    constexpr auto smallVocabulary = 6;

    auto tokenTable = spreadValues(smallVocabulary * width, 777u, 2.f);
    auto many = tokensOf({4, 1, 5});
    auto one = tokensOf({4});

    auto manyResult = runEmbed(many, tokenTable, width, gemmaScale);
    auto oneResult = runEmbed(one, tokenTable, width, gemmaScale);

    auto expected = embedReference(one, tokenTable, width, gemmaScale);

    for (auto i = 0; i < width; ++i)
    {
        check(isClose(oneResult[i], expected[i], 1e-6));

        // The same token in both dispatches: nothing about the row count
        // reaches the gather.
        check(oneResult[i] == manyResult[i]);
    }
};

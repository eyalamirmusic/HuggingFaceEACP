#include <HuggingFaceEACP/Sampling/Sampling.h>

#include <NanoTest/NanoTest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

using namespace nano;
using namespace HF;

namespace
{
// Logits whose softmax is exactly the probabilities asked for, so a frequency
// check has a number to compare against that the code under test did not
// produce.
Vector<float> logitsForProbabilities(std::initializer_list<double> probabilities)
{
    auto logits = Vector<float> {};

    for (auto probability: probabilities)
        logits.add((float) std::log(probability));

    return logits;
}

// The distribution the sampler is supposed to be drawing from, in double and
// spelled out here rather than reached for through the module, which is what
// makes it a reference and not a restatement.
Vector<double> referenceSoftmax(const Vector<float>& logits, double temperature)
{
    auto weights = Vector<double> {};
    weights.resize(logits.size());

    auto largest = std::numeric_limits<double>::lowest();

    for (auto logit: logits)
        largest = std::max(largest, (double) logit / temperature);

    auto total = 0.0;

    for (auto index = 0; index < logits.size(); ++index)
    {
        weights[index] = std::exp((double) logits[index] / temperature - largest);
        total += weights[index];
    }

    for (auto& weight: weights)
        weight /= total;

    return weights;
}

Vector<int> drawCounts(Sampler& sampler, const Vector<float>& logits, int draws)
{
    auto counts = Vector<int> {};
    counts.resize(logits.size(), 0);

    for (auto draw = 0; draw < draws; ++draw)
        ++counts[sampler.sample(logits)];

    return counts;
}

// Long enough that two seeds agreeing on all of it would be the generator
// repeating rather than two draws coinciding.
constexpr auto sequenceLength = 200;

Vector<TokenId> drawSequence(SamplingOptions options, const Vector<float>& logits)
{
    auto sampler = Sampler {options};
    auto tokens = Vector<TokenId> {};

    for (auto draw = 0; draw < sequenceLength; ++draw)
        tokens.add(sampler.sample(logits));

    return tokens;
}

bool frequenciesMatch(const Vector<int>& counts,
                      const Vector<double>& expected,
                      int draws,
                      double tolerance)
{
    for (auto index = 0; index < counts.size(); ++index)
    {
        const auto frequency = (double) counts[index] / (double) draws;

        if (std::abs(frequency - expected[index]) > tolerance)
            return false;
    }

    return true;
}

// Constructing one and throwing it away is the whole assertion: the options
// are checked in the constructor, so a Sampler never exists when they describe
// no distribution.
bool refusesOptions(SamplingOptions options)
{
    try
    {
        Sampler {options};
    }
    catch (const SamplingError&)
    {
        return true;
    }

    return false;
}

bool refusesRow(Sampler& sampler,
                const Vector<float>& logits,
                const Vector<float>& mask)
{
    try
    {
        sampler.sample(logits, mask);
    }
    catch (const SamplingError&)
    {
        return true;
    }

    return false;
}
} // namespace

// --- Greedy, which is Argmax's rule on the CPU ---------------------------

auto tGreedyPicksTheLargest = test("Sampler/greedyPicksTheLargest") = []
{
    const auto logits = Vector<float> {0.1f, 2.5f, -1.f, 2.4f};
    auto sampler = Sampler {};

    check(sampler.options().temperature == 0.f);
    check(Sampler::argmax(logits) == 1);
    check(sampler.sample(logits) == 1);
};

auto tGreedyTakesTheLowestIndexOnATie = test("Sampler/greedyTie") = []
{
    const auto logits = Vector<float> {1.f, 3.f, 3.f, 2.f, 3.f};
    auto sampler = Sampler {};

    check(Sampler::argmax(logits) == 1);
    check(sampler.sample(logits) == 1);
};

auto tMaskTakesTokensOutOfTheRunning = test("Sampler/mask") = []
{
    const auto logits = Vector<float> {0.f, 5.f, 4.f, 1.f};
    const auto withoutTheLargest = Vector<float> {0.f, 1.f, 0.f, 0.f};
    const auto withoutTheTopTwo = Vector<float> {0.f, 1.f, 1.f, 0.f};

    check(Sampler::argmax(logits, withoutTheLargest) == 2);
    check(Sampler::argmax(logits, withoutTheTopTwo) == 3);

    // A sampler with a temperature has to honour the same mask, and a
    // suppressed token is out before the draw rather than merely unlikely.
    auto sampler = Sampler {SamplingOptions {.temperature = 2.f, .seed = 5}};
    auto avoided = true;

    for (auto draw = 0; draw < 10000; ++draw)
    {
        const auto token = sampler.sample(logits, withoutTheTopTwo);
        avoided = avoided && token != 1 && token != 2;
    }

    check(avoided);
};

auto tAllSuppressedGivesTokenZero = test("Sampler/allSuppressed") = []
{
    const auto logits = Vector<float> {0.f, 5.f, 4.f, 1.f};
    const auto everything = Vector<float> {1.f, 1.f, 1.f, 1.f};
    auto sampler = Sampler {SamplingOptions {.temperature = 1.f, .seed = 11}};

    check(Sampler::argmax(logits, everything) == 0);
    check(Sampler {}.sample(logits, everything) == 0);
    check(sampler.sample(logits, everything) == 0);
};

// --- Each cut, at its degenerate setting, is greedy again ----------------

auto tEachCutCanReduceToGreedy = test("Sampler/reducesToGreedy") = []
{
    const auto logits = Vector<float> {0.5f, 3.f, 2.f, 2.99f, -4.f};
    const auto expected = Sampler::argmax(logits);
    auto alwaysGreedy = true;

    for (auto seed = std::uint64_t {0}; seed < 8; ++seed)
    {
        auto zeroTemperature =
            Sampler {SamplingOptions {.temperature = 0.f, .seed = seed}};
        auto oneOfK = Sampler {SamplingOptions {.topK = 1, .seed = seed}};
        auto tinyMass = Sampler {SamplingOptions {.topP = 1e-6f, .seed = seed}};

        for (auto draw = 0; draw < 100; ++draw)
        {
            alwaysGreedy = alwaysGreedy && zeroTemperature.sample(logits) == expected
                           && oneOfK.sample(logits) == expected
                           && tinyMass.sample(logits) == expected;
        }
    }

    check(alwaysGreedy);
};

// --- The cuts ------------------------------------------------------------

auto tTopKNeverLeavesTheKLargest = test("Sampler/topK") = []
{
    const auto logits = Vector<float> {1.f, 4.f, 2.f, 3.5f, 0.f, 3.9f, -2.f, 2.2f};
    auto sampler = Sampler {SamplingOptions {.topK = 3, .seed = 7}};
    const auto counts = drawCounts(sampler, logits, 10000);

    check(counts[0] == 0);
    check(counts[2] == 0);
    check(counts[4] == 0);
    check(counts[6] == 0);
    check(counts[7] == 0);

    // All three that survive are drawn, so k is a cut rather than a tie-break.
    check(counts[1] > 0);
    check(counts[3] > 0);
    check(counts[5] > 0);
};

auto tTopPKeepsTheBoundaryToken = test("Sampler/topP") = []
{
    const auto logits = logitsForProbabilities({0.4, 0.3, 0.2, 0.1});
    auto sampler = Sampler {SamplingOptions {.topP = 0.5f, .seed = 3}};

    constexpr auto draws = 20000;
    const auto counts = drawCounts(sampler, logits, draws);

    // 0.4 alone does not reach 0.5, so the token that carries the running mass
    // across p is kept and the two below it are not.
    check(counts[0] > 0);
    check(counts[1] > 0);
    check(counts[2] == 0);
    check(counts[3] == 0);

    // What survives is renormalised over the kept mass of 0.7.
    const auto kept = Vector<double> {0.4 / 0.7, 0.3 / 0.7, 0.0, 0.0};
    check(frequenciesMatch(counts, kept, draws, 0.02));
};

// --- The draw itself -----------------------------------------------------

auto tFrequenciesFollowTheSoftmax = test("Sampler/frequencies") = []
{
    const auto logits = logitsForProbabilities({0.5, 0.25, 0.15, 0.1});
    auto sampler = Sampler {SamplingOptions {.seed = 12345}};

    constexpr auto draws = 20000;
    const auto counts = drawCounts(sampler, logits, draws);

    check(frequenciesMatch(counts, referenceSoftmax(logits, 1.0), draws, 0.02));
};

auto tTemperatureScalesTheDistribution = test("Sampler/temperature") = []
{
    const auto logits = Vector<float> {0.f, 1.f, 2.f, -1.f};
    constexpr auto draws = 20000;
    auto matched = true;

    for (auto temperature: {0.5f, 1.f, 2.f})
    {
        auto sampler =
            Sampler {SamplingOptions {.temperature = temperature, .seed = 808}};
        const auto counts = drawCounts(sampler, logits, draws);
        const auto expected = referenceSoftmax(logits, (double) temperature);

        matched = matched && frequenciesMatch(counts, expected, draws, 0.02);
    }

    check(matched);
};

// The engine is the standard's own and the draw is built out of its bits by
// hand, so a seed names one token sequence everywhere rather than one per
// standard library.
auto tASeedReproducesItsSequence = test("Sampler/seeds") = []
{
    const auto logits = logitsForProbabilities({0.5, 0.25, 0.15, 0.1});
    const auto first = drawSequence(SamplingOptions {.seed = 4242}, logits);
    const auto again = drawSequence(SamplingOptions {.seed = 4242}, logits);
    const auto other = drawSequence(SamplingOptions {.seed = 4243}, logits);

    check(first == again);
    check(first != other);

    // The cuts are on the same footing: top-k orders what it kept before the
    // draw walks it, so a seed still names one sequence.
    const auto cut = SamplingOptions {.topK = 3, .seed = 77};

    check(drawSequence(cut, logits) == drawSequence(cut, logits));

    auto sampler = Sampler {SamplingOptions {.seed = 1}};
    sampler.sample(logits);
    sampler.reseed(4242);

    auto reseeded = Vector<TokenId> {};

    for (auto draw = 0; draw < sequenceLength; ++draw)
        reseeded.add(sampler.sample(logits));

    check(sampler.options().seed == 4242);
    check(reseeded == first);
};

// --- What is refused -----------------------------------------------------

auto tOptionsAreCheckedInTheConstructor = test("Sampler/optionValidation") = []
{
    constexpr auto notANumber = std::numeric_limits<float>::quiet_NaN();

    check(refusesOptions(SamplingOptions {.temperature = -1.f}));
    check(refusesOptions(SamplingOptions {.temperature = notANumber}));
    check(refusesOptions(SamplingOptions {.topK = -1}));
    check(refusesOptions(SamplingOptions {.topP = 0.f}));
    check(refusesOptions(SamplingOptions {.topP = -0.5f}));
    check(refusesOptions(SamplingOptions {.topP = 1.5f}));
    check(refusesOptions(SamplingOptions {.topP = notANumber}));

    const auto everythingOn =
        SamplingOptions {.temperature = 0.7f, .topK = 40, .topP = 0.95f, .seed = 1};

    check(!refusesOptions(SamplingOptions {}));
    check(!refusesOptions(SamplingOptions::greedy()));
    check(!refusesOptions(everythingOn));
};

auto tDegenerateRowsAreRefused = test("Sampler/degenerateRows") = []
{
    const auto nothing = Vector<float> {};
    const auto logits = Vector<float> {1.f, 2.f};
    const auto tooShort = Vector<float> {0.f};
    auto sampler = Sampler {SamplingOptions {.seed = 2}};
    auto greedy = Sampler {};

    check(refusesRow(sampler, nothing, nothing));
    check(refusesRow(greedy, nothing, nothing));
    check(refusesRow(sampler, logits, tooShort));
    check(refusesRow(greedy, logits, tooShort));
    check(!refusesRow(sampler, logits, nothing));
};

// --- Gemma's own width ---------------------------------------------------

auto tVocabularyWidthRow = test("Sampler/vocabularyWidthRow") = []
{
    constexpr auto vocabularySize = 256000;
    constexpr auto keep = 40;

    auto logits = Vector<float> {};
    logits.resize(vocabularySize);

    // A deterministic spread rather than a generator, so the shape check is
    // the same row on every run.
    for (auto token = 0; token < vocabularySize; ++token)
        logits[token] = std::sin((float) token * 0.7f) * 6.f;

    const auto ranksBefore = [&](TokenId left, TokenId right)
    {
        if (logits[left] != logits[right])
            return logits[left] > logits[right];

        return left < right;
    };

    auto ranked = Vector<TokenId> {};
    ranked.resize(vocabularySize);
    std::iota(ranked.begin(), ranked.end(), 0);
    std::partial_sort(
        ranked.begin(), ranked.begin() + keep, ranked.end(), ranksBefore);
    ranked.resize(keep);

    auto sampler =
        Sampler {SamplingOptions {.temperature = 0.8f, .topK = keep, .seed = 99}};
    auto insideTheCut = true;

    for (auto draw = 0; draw < 200; ++draw)
        insideTheCut = insideTheCut && ranked.contains(sampler.sample(logits));

    check(insideTheCut);
};

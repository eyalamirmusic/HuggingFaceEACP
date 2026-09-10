#include <HuggingFaceEACP/Sampling/Sampler.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace HF
{
namespace
{
bool isSuppressed(Span<const float> mask, TokenId token)
{
    return !mask.empty() && mask[token] != 0.f;
}

void requireRow(Span<const float> logits, Span<const float> mask)
{
    if (logits.empty())
        throw SamplingError {"A row of logits to sample from cannot be empty"};

    if (!mask.empty() && mask.size() != logits.size())
        throw SamplingError {
            "A sampling mask must have one entry per vocabulary token"};
}

// Spelled as negated comparisons so that a NaN, which compares false against
// everything, is refused rather than let through to produce a NaN token.
SamplingOptions validated(SamplingOptions options)
{
    if (!(options.temperature >= 0.f))
        throw SamplingError {"A sampling temperature must be zero or positive"};

    if (options.topK < 0)
        throw SamplingError {"A top-k count cannot be negative"};

    if (!(options.topP > 0.f && options.topP <= 1.f))
        throw SamplingError {"A top-p mass must lie inside (0, 1]"};

    return options;
}
} // namespace

SamplingOptions SamplingOptions::greedy()
{
    return SamplingOptions {.temperature = 0.f};
}

Sampler::Sampler(SamplingOptions options)
    : samplingOptions(validated(options))
    , generator(samplingOptions.seed)
{
}

void Sampler::reseed(std::uint64_t seed)
{
    samplingOptions.seed = seed;
    generator.seed(seed);
}

TokenId Sampler::argmax(Span<const float> logits, Span<const float> mask)
{
    requireRow(logits, mask);

    auto best = invalidTokenId;

    for (auto token = 0; token < logits.size(); ++token)
    {
        if (isSuppressed(mask, token))
            continue;

        if (best == invalidTokenId || logits[token] > logits[best])
            best = token;
    }

    return best == invalidTokenId ? 0 : best;
}

TokenId Sampler::sample(Span<const float> logits, Span<const float> mask)
{
    requireRow(logits, mask);

    if (samplingOptions.temperature == 0.f)
        return argmax(logits, mask);

    collectCandidates(logits, mask);

    if (candidates.empty())
        return 0;

    applyTopK();
    applyTopP();
    softmaxOverCandidates();

    return drawFromCandidates();
}

bool Sampler::ranksBefore(const Candidate& left, const Candidate& right)
{
    if (left.score != right.score)
        return left.score > right.score;

    return left.token < right.token;
}

void Sampler::collectCandidates(Span<const float> logits, Span<const float> mask)
{
    candidates.clear();
    candidates.reserveAtLeast(logits.size());

    const auto inverseTemperature = 1.0 / (double) samplingOptions.temperature;

    for (auto token = 0; token < logits.size(); ++token)
    {
        if (isSuppressed(mask, token))
            continue;

        const auto score = (double) logits[token] * inverseTemperature;

        candidates.add(Candidate {token, score});
    }
}

void Sampler::applyTopK()
{
    const auto keep = samplingOptions.topK;

    if (keep == 0 || keep >= candidates.size())
        return;

    // A partial selection rather than a sort: at 256,000 entries a full sort
    // per token is the expensive half of a decode step, and only the k best
    // are wanted.
    std::nth_element(candidates.begin(),
                     candidates.begin() + keep,
                     candidates.end(),
                     ranksBefore);

    candidates.resize(keep);

    // nth_element leaves the k it kept in an order the standard does not fix,
    // and the draw walks them accumulating mass — so without this a seed would
    // name one token on libc++ and another on libstdc++. Ordering k of them
    // costs nothing beside the selection that found them, and it is what keeps
    // the whole sampler's output a function of the seed alone.
    std::sort(candidates.begin(), candidates.end(), ranksBefore);
}

void Sampler::applyTopP()
{
    if (samplingOptions.topP >= 1.f)
        return;

    std::sort(candidates.begin(), candidates.end(), ranksBefore);
    softmaxOverCandidates();

    const auto wanted = (double) samplingOptions.topP;
    auto mass = 0.0;
    auto kept = 0;

    while (kept < candidates.size() && mass < wanted)
    {
        mass += probabilities[kept];
        ++kept;
    }

    candidates.resize(kept);
}

void Sampler::softmaxOverCandidates()
{
    probabilities.resize(candidates.size());

    auto largest = std::numeric_limits<double>::lowest();

    for (const auto& candidate: candidates)
        largest = std::max(largest, candidate.score);

    auto total = 0.0;

    for (auto index = 0; index < candidates.size(); ++index)
    {
        // The maximum comes off first, so a large logit over a small
        // temperature exponentiates to something finite rather than to
        // infinity over infinity.
        probabilities[index] = std::exp(candidates[index].score - largest);
        total += probabilities[index];
    }

    for (auto index = 0; index < candidates.size(); ++index)
        probabilities[index] /= total;
}

TokenId Sampler::drawFromCandidates()
{
    const auto target = uniform();
    auto mass = 0.0;

    for (auto index = 0; index < candidates.size(); ++index)
    {
        mass += probabilities[index];

        if (target < mass)
            return candidates[index].token;
    }

    // The draw is below one and the probabilities sum to one, so the loop only
    // runs off the end when rounding has left the total a hair short.
    return candidates.back().token;
}

double Sampler::uniform()
{
    return (double) (generator() >> 11) * 0x1.0p-53;
}
} // namespace HF

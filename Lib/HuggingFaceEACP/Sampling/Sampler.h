#pragma once

// The CPU half of a generation step: one token drawn from one row of logits,
// which is plan.md's "1 MB logits readback per token" and nothing more than
// that yet.
//
// Kernels/Argmax.h is the on-GPU sibling and the two agree wherever they
// overlap — the largest logit wins, an exact tie goes to the lowest index, a
// suppressed token is out of the running before anything is compared, and a
// row whose every token is suppressed comes out as token zero. A decoder can
// therefore move greedy decoding onto the device without the tokens changing.

#include <HuggingFaceEACP/Core/Core.h>
#include <HuggingFaceEACP/Sampling/SamplingError.h>

#include <cstdint>
#include <random>

namespace HF
{
// The four numbers a generation_config carries, applied in Hugging Face
// `generate`'s own processor order: temperature, then top-k, then top-p, then
// a softmax over whatever survived and one draw from it.
//
// The order is stated because it changes the answer and because HF's is the
// one the rest of this project is checked against. Top-k is indifferent to it
// — dividing by a positive temperature preserves the ranking, so the same k
// tokens are chosen either way — but top-p is not: a high temperature flattens
// the distribution, so more tokens are needed to reach p, and cutting before
// the scaling would cut a different set. llama.cpp's default sampler chain
// runs the temperature last, after top-k and top-p, so its p is a cut of the
// unscaled distribution; the same numbers there do not mean the same thing.
struct SamplingOptions
{
    // Zero is greedy: the largest logit, and the lowest index on a tie.
    float temperature = 1.f;

    // Zero is off. Otherwise only the k largest logits stay in the running.
    int topK = 0;

    // One is off. Otherwise the smallest set of tokens whose probability mass
    // reaches p, taken in descending order, with the boundary token that
    // carries the running mass across p kept rather than dropped — which is
    // Hugging Face's rule, and why p never selects an empty set.
    float topP = 1.f;

    // The generator's seed, so a run is reproducible.
    std::uint64_t seed = 0;

    static SamplingOptions greedy();
};

class Sampler
{
public:
    explicit Sampler(SamplingOptions options = SamplingOptions::greedy());

    // One token from one row of logits. The mask, when it is not empty, is one
    // float per vocabulary entry — the layout Argmax's mask buffer has — where
    // a nonzero value suppresses that token, and it is applied before the
    // temperature and both cuts.
    TokenId sample(Span<const float> logits, Span<const float> mask = {});

    const SamplingOptions& options() const { return samplingOptions; }

    void reseed(std::uint64_t seed);

    // The greedy draw without a Sampler to hold it, since a decoder that reads
    // its logits back wants exactly this and no state.
    static TokenId argmax(Span<const float> logits, Span<const float> mask = {});

private:
    struct Candidate
    {
        TokenId token = invalidTokenId;
        double score = 0.0;
    };

    // Descending by score and, on an exact tie, the lower token first, so the
    // head of the order is the token a greedy draw would have taken and no two
    // candidates ever compare equal — which is what makes nth_element pick the
    // same k on every standard library.
    static bool ranksBefore(const Candidate& left, const Candidate& right);

    void collectCandidates(Span<const float> logits, Span<const float> mask);
    void applyTopK();
    void applyTopP();
    void softmaxOverCandidates();
    TokenId drawFromCandidates();

    // std::mt19937_64 produces the standard's own sequence, but
    // std::uniform_real_distribution and std::discrete_distribution do not:
    // their algorithms are implementation-defined, so a fixed seed would give
    // one token sequence on libc++ and another on libstdc++, and every test
    // that pins a seed would only hold on the machine that wrote it. The top
    // 53 bits of one engine draw scaled by 2^-53 is a value every
    // implementation agrees on.
    double uniform();

    SamplingOptions samplingOptions;
    std::mt19937_64 generator;

    // Kept between steps rather than built per token: at Gemma's 256,000
    // entries these are a few megabytes, and a step after the first allocates
    // nothing.
    Vector<Candidate> candidates;
    Vector<double> probabilities;
};
} // namespace HF

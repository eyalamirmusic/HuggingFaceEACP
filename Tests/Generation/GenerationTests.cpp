#include "Common.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// The generation loop over the synthetic checkpoint, with a tokenizer.json of
// the same sixty-four tokens written beside it — so Gemma::load, prepare,
// generate and generateText all run end to end without a download.
//
// What every comparison here is against is a loop the test runs itself:
// Decoder::step, the last logits row read back, and Sampler::argmax on the
// host. That is the readback the on-device Argmax exists to avoid, so the two
// agreeing says both that the kernel picks what the CPU would and that the
// pipelining — two command buffers in the air, the token never leaving the
// device between steps — changes nothing about the tokens.

using namespace nano;
using namespace HF;
using namespace HF::Testing;
using namespace eacp::GPU;

namespace
{
// Seven ids, none of them zero, none repeated, and none of them the
// end-of-sequence token, so a run only stops where the model puts it. Past
// Decoder::splitRowLimit, so an unblocked prefill takes the tiled product.
std::vector<int> promptTokens()
{
    return {5, 41, 3, 17, 28, 9, 60};
}

Vector<TokenId> asTokens(const std::vector<int>& ids)
{
    auto tokens = Vector<TokenId> {};

    for (auto id: ids)
        tokens.add(id);

    return tokens;
}

// A loaded and prepared Gemma over a directory, since every test below wants
// one and only two of them want it configured differently.
void prepareOver(Gemma& gemma, const SyntheticModel& model, int promptCapacity)
{
    gemma.load(model.path());
    gemma.setPromptCapacity(promptCapacity);
    gemma.prepare();
}

std::string tokenList(Span<const TokenId> tokens)
{
    auto text = std::string {};

    for (auto index = 0; index < tokens.size(); ++index)
        text += (index == 0 ? "" : " ") + std::to_string(tokens[index]);

    return text;
}

// Which token's embedding row the end-of-sequence row is replaced by, and what
// it is multiplied by first.
struct EndOfSequenceBoost
{
    TokenId borrowed = invalidTokenId;
    float factor = 0.f;
};

// The largest logit in a row that is not the end-of-sequence token's, which is
// what a boosted end-of-sequence logit has to beat at one step and lose to at
// every step before it.
float largestOtherLogit(const Vector<float>& row, TokenId endOfSequence)
{
    auto largest = std::numeric_limits<float>::lowest();

    for (auto index = 0; index < row.size(); ++index)
        if (index != endOfSequence)
            largest = std::max(largest, row[index]);

    return largest;
}

// A logit is the hidden row dotted with a token's embedding row, so putting
// `factor * e(u)` where the end-of-sequence row was makes the end-of-sequence
// logit `factor * logits[u]` at every step and changes nothing else about the
// model — the row is never gathered, since the prompt does not hold that token
// and a run stops the moment it is produced.
//
// So the search is for a token whose logit is relatively strongest at the step
// the run should stop at: the factor then has to clear best/logits[u] there and
// stay under it everywhere earlier, and the interval between the two is what
// says such a token was found. Searched rather than picked, because a seeded
// random checkpoint promises nothing about which token that is — and this one
// is degenerate enough to repeat a single token, so borrowing the winner's own
// row could never have worked.
EndOfSequenceBoost chooseEndOfSequenceBoost(const std::vector<Vector<float>>& rows,
                                            int stopStep,
                                            TokenId endOfSequence)
{
    for (auto token = 0; token < rows[0].size(); ++token)
    {
        if (token == endOfSequence)
            continue;

        auto mustExceed = 0.0;
        auto mustStayUnder = std::numeric_limits<double>::max();
        auto usable = true;

        for (auto step = 0; step <= stopStep && usable; ++step)
        {
            const auto& row = rows[(std::size_t) step];
            const auto value = (double) row[token];

            // A negative or zero logit gives the factor no direction to push
            // in, since the sign of the product would flip with it.
            usable = value > 0.0;

            if (!usable)
                break;

            const auto ratio =
                (double) largestOtherLogit(row, endOfSequence) / value;

            if (step == stopStep)
                mustExceed = ratio;
            else
                mustStayUnder = std::min(mustStayUnder, ratio);
        }

        // Five per cent past the bound it has to clear, so the win is a margin
        // rather than a tie in float32.
        const auto factor = mustExceed * 1.05;

        if (usable && factor < mustStayUnder)
            return {token, (float) factor};
    }

    return {};
}
} // namespace

// The load half, which touches no device: the four files a directory has to
// carry, and the vocabulary that must fit the embedding the model gathers from.
auto tGenerationLoads = test("Generation/loadsAModelDirectory") = []
{
    const auto model = SyntheticModel {"generation-load"};

    auto gemma = Gemma {};
    check(!gemma.isLoaded());

    gemma.load(model.path());

    check(gemma.isLoaded());
    check(!gemma.isPrepared());
    check(gemma.config().vocabularySize == 64);
    check(gemma.config().endOfSequenceToken == 1);
    check(gemma.config().beginningOfSequenceToken == 2);
    check(gemma.tokenizer().vocabularySize() == 64);
    check(gemma.tokenizer().eos() == 1);
    check(gemma.tokenizer().bos() == 2);

    // The window's own bound until a run says otherwise, which is what zero
    // means: a prompt's length is not known until generate is called.
    check(gemma.maximumTokens() == 0);
    check(gemma.sampling().temperature == 0.f);
};

// A directory the checkpoint is in but the tokenizer is not, which ModelFiles
// records without requiring and generation cannot do without.
auto tGenerationNeedsATokenizer = test("Generation/needsATokenizer") = []
{
    const auto checkpoint = SyntheticCheckpoint {"generation-no-tokenizer"};

    auto gemma = Gemma {};
    check(throwsModelError([&] { gemma.load(checkpoint.path()); }));
    check(!gemma.isLoaded());
};

// Greedy on the device against the same greedy on the host, token for token.
//
// **The synthetic model repeats one token**, which is what a checkpoint of
// seeded random weights does: after the prompt it lands on an attractor and
// stays there. So this says the two argmaxes agree and the pipelining changes
// nothing, and it is not what would catch a run embedding the wrong sequence
// slot — a model whose answer does not depend on its input cannot tell that
// apart. The end-of-sequence test below is what does: its boost is calibrated
// to win at one step by five per cent, which only holds if the hidden state at
// that step is the one a correctly fed loop produces.
auto tGreedyMatchesReference = test("Generation/greedyMatchesAReferenceLoop") = []
{
    if (!Device::shared().isValid())
        return;

    const auto model = SyntheticModel {"generation-greedy"};
    const auto prompt = promptTokens();

    auto gemma = Gemma {};
    prepareOver(gemma, model, 16);

    const auto limit = 6;
    gemma.setMaximumTokens(limit);

    const auto promptIds = asTokens(prompt);
    const auto generated = gemma.generate(promptIds);

    const auto expected = referenceGenerate(model.checkpoint(),
                                            gemma.shape(),
                                            prompt,
                                            limit,
                                            gemma.config().endOfSequenceToken);

    check(sameTokens(generated, expected.tokens));
    check(generated.size() > 0);

    // One prefill block and one step per token after the first, with the last
    // stepsInFlight - 1 of them computed past the end.
    check(gemma.lastStepCount() >= 1 + generated.size() - 1);
    check(gemma.lastPrefillSeconds() > 0.0);

    std::cout << "  greedy: " << tokenList(generated) << "\n";
};

// The second path, which is the same tokens through the readback: a
// temperature just above zero with top-k one leaves exactly the largest logit
// in the running, so the draw is forced onto the token the argmax would have
// taken and the two paths have to agree.
//
// Temperature zero is not the way in — it is the on-device path by definition,
// and Gemma reads it as such — so the smallest thing that is not zero is what
// makes the CPU loop run at all.
auto tSampledPathMatchesGreedy = test("Generation/sampledPathMatchesGreedy") = []
{
    if (!Device::shared().isValid())
        return;

    const auto model = SyntheticModel {"generation-sampled"};
    const auto prompt = promptTokens();
    const auto promptIds = asTokens(prompt);
    const auto limit = 6;

    auto greedy = Gemma {};
    prepareOver(greedy, model, 16);
    greedy.setMaximumTokens(limit);

    const auto greedyTokens = greedy.generate(promptIds);

    auto sampled = Gemma {};
    prepareOver(sampled, model, 16);
    sampled.setMaximumTokens(limit);
    sampled.setSampling(SamplingOptions {.temperature = 1.0e-3f, .topK = 1});

    const auto sampledTokens = sampled.generate(promptIds);

    check(sameTokens(greedyTokens, sampledTokens));
    check(sampledTokens.size() > 0);

    // A real temperature, twice at the same seed: every run reseeds from the
    // options rather than carrying the generator on, so the second run is the
    // first even though it followed it.
    sampled.setSampling(SamplingOptions {.temperature = 1.f, .seed = 4242});

    const auto drawn = sampled.generate(promptIds);
    const auto again = sampled.generate(promptIds);

    check(sameTokens(drawn, again));

    std::cout << "  readback: " << tokenList(sampledTokens) << ", at "
              << "temperature 1: " << tokenList(drawn) << "\n";
};

// The end-of-sequence token, made reachable rather than hoped for.
//
// The logits are the hidden row against the tied embedding, so a token's logit
// is that row dotted with that token's embedding row — which means replacing
// the end-of-sequence row with a multiple of the row of whichever token wins at
// some step makes its logit that multiple of the winning logit, and nothing
// else about the model moves. The end-of-sequence row is never gathered, since
// the prompt does not hold it and a run stops the moment it is produced.
//
// The multiple is searched for rather than picked, because the sign of a logit
// is not something a seeded random checkpoint promises: the one taken is the
// first that wins at the chosen step and loses at every step before it.
auto tEndOfSequenceStops = test("Generation/endOfSequenceStopsAndIsExcluded") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto endOfSequence = TokenId {1};
    constexpr auto stopStep = 3;
    constexpr auto limit = 8;

    const auto prompt = promptTokens();

    const auto probe = SyntheticModel {"generation-eos-probe"};
    const auto plain = referenceGenerate(
        probe.checkpoint(), probe.shape(), prompt, limit, endOfSequence);

    check((int) plain.rows.size() > stopStep);

    const auto boost = chooseEndOfSequenceBoost(plain.rows, stopStep, endOfSequence);

    check(boost.borrowed != invalidTokenId);

    const auto borrowed = boost.borrowed;
    const auto factor = boost.factor;

    const auto edit = [borrowed, factor](Vector<SyntheticTensor>& tensors)
    {
        auto* embedding = findTensor(tensors, GemmaTensors::embedding);
        const auto width = embedding->shape[1];

        for (auto channel = 0; channel < width; ++channel)
            embedding->values[endOfSequence * width + channel] =
                factor * embedding->values[borrowed * width + channel];
    };

    const auto model = SyntheticModel {"generation-eos", edit};
    const auto expected = referenceGenerate(
        model.checkpoint(), model.shape(), prompt, limit, endOfSequence);

    // The construction has to have worked, or the assertions below would pass
    // over a run that simply ran out of tokens.
    check(expected.stoppedOnEndOfSequence);
    check(expected.tokens.size() == stopStep);

    auto gemma = Gemma {};
    prepareOver(gemma, model, 16);
    gemma.setMaximumTokens(limit);

    const auto promptIds = asTokens(prompt);
    const auto generated = gemma.generate(promptIds);

    check(sameTokens(generated, expected.tokens));
    check(generated.size() == stopStep);

    for (auto index = 0; index < generated.size(); ++index)
        check(generated[index] != endOfSequence);

    std::cout << "  stopped on <eos> after " << generated.size() << " tokens\n";
};

// The two bounds that are not the model's: the count a caller set, and the
// window the sequence has to stay inside.
auto tLimitsStopTheRun = test("Generation/limitsStopTheRun") = []
{
    if (!Device::shared().isValid())
        return;

    const auto model = SyntheticModel {"generation-limits"};
    const auto prompt = promptTokens();
    const auto promptIds = asTokens(prompt);

    auto gemma = Gemma {};
    prepareOver(gemma, model, 16);

    gemma.setMaximumTokens(3);
    check(gemma.generate(promptIds).size() == 3);

    gemma.setMaximumTokens(1);
    check(gemma.generate(promptIds).size() == 1);

    // Zero puts the window's own bound back, which for a seven-token prompt in
    // a sixteen-position window is nine.
    gemma.setMaximumTokens(0);
    check(gemma.generate(promptIds).size() <= 9);

    check(throwsModelError([&] { gemma.setMaximumTokens(-1); }));
};

// The sequence never passes the window: the prompt and everything after it
// share the maxPositions the caches hold, so a prompt one short of the window
// has room for exactly one token and a prompt filling it has room for none.
auto tWindowBoundsTheRun = test("Generation/windowBoundsTheSequence") = []
{
    if (!Device::shared().isValid())
        return;

    const auto model = SyntheticModel {"generation-window"};

    auto gemma = Gemma {};
    prepareOver(gemma, model, 8);

    const auto window = gemma.shape().maxPositions;

    auto longPrompt = std::vector<int> {};

    for (auto index = 0; index < window; ++index)
        longPrompt.push_back(2 + (index % 40));

    const auto oneShort = asTokens(
        std::vector<int> {longPrompt.begin(), longPrompt.begin() + window - 1});

    check(gemma.generate(oneShort).size() <= 1);

    const auto whole = asTokens(longPrompt);
    check(gemma.generate(whole).size() == 0);

    // And a prompt past the window at all, which is refused rather than
    // truncated.
    longPrompt.push_back(7);
    const auto tooLong = asTokens(longPrompt);
    check(throwsModelError([&] { gemma.generate(tooLong); }));
};

// A prompt longer than one prefill block, which is what the decoder's step
// capacity made possible: the blocks append to the same caches at the same
// positions, so the tokens have to be the tokens one call would have produced.
auto tBlockedPrefillMatches = test("Generation/blockedPrefillMatchesOneBlock") = []
{
    if (!Device::shared().isValid())
        return;

    const auto model = SyntheticModel {"generation-blocks"};
    const auto prompt = promptTokens();
    const auto promptIds = asTokens(prompt);
    const auto limit = 5;

    auto whole = Gemma {};
    prepareOver(whole, model, 16);
    whole.setMaximumTokens(limit);

    const auto unblocked = whole.generate(promptIds);
    check(whole.lastStepCount() == 1 + limit - 1);

    auto blocked = Gemma {};
    prepareOver(blocked, model, 3);
    blocked.setMaximumTokens(limit);

    check(blocked.shape().stepRowCapacity() == 3);

    const auto inBlocks = blocked.generate(promptIds);

    // Seven tokens in blocks of three is three blocks, and the generated steps
    // after them are one row each.
    check(blocked.lastStepCount() == 3 + limit - 1);
    check(sameTokens(unblocked, inBlocks));
    check(inBlocks.size() == limit);

    std::cout << "  seven tokens in blocks of three: " << tokenList(inBlocks)
              << "\n";
};

// The callback, which is what a streaming caller reads a run through: once per
// generated token, in order, and never for the end-of-sequence token that was
// dropped.
auto tOnTokenFires = test("Generation/onTokenFiresOncePerToken") = []
{
    if (!Device::shared().isValid())
        return;

    const auto model = SyntheticModel {"generation-callback"};
    const auto promptIds = asTokens(promptTokens());

    auto gemma = Gemma {};
    prepareOver(gemma, model, 16);
    gemma.setMaximumTokens(4);

    auto streamed = Vector<TokenId> {};
    gemma.onToken = [&streamed](TokenId token) { streamed.add(token); };

    const auto generated = gemma.generate(promptIds);

    check(sameTokens(streamed, generated));
    check(streamed.size() == 4);

    // And the default is a no-op rather than a null, so a run that was never
    // given one calls it all the same.
    auto quiet = Gemma {};
    prepareOver(quiet, model, 16);
    quiet.setMaximumTokens(2);
    check(quiet.generate(promptIds).size() == 2);
};

// The string in and the string out, through the synthetic vocabulary: the
// prompt encoded with the beginning-of-sequence token in front, and the
// continuation decoded back without it.
auto tGenerateTextRoundTrips = test("Generation/generateTextRoundTrips") = []
{
    if (!Device::shared().isValid())
        return;

    const auto model = SyntheticModel {"generation-text"};

    auto gemma = Gemma {};
    prepareOver(gemma, model, 16);
    gemma.setMaximumTokens(4);

    const auto& vocabulary = gemma.tokenizer();

    // The fixture spells these, so a prompt that tokenizes to <unk> would be
    // testing the vocabulary rather than the loop.
    const auto encoded = vocabulary.encodeWithBos("the cat");
    check(encoded.size() >= 2);
    check(encoded[0] == vocabulary.bos());
    check(vocabulary.decode(encoded) == "the cat");

    const auto tokens = gemma.generateFromText("the cat");
    const auto text = gemma.generateText("the cat");

    check(tokens.size() == 4);
    check(text == gemma.textForTokens(tokens));

    // The two calls are the same run twice, since beginSequence puts the
    // decoder back to zero and the weights never moved.
    const auto again = gemma.generateFromText("the cat");
    check(sameTokens(tokens, again));

    std::cout << "  \"the cat\" -> \"" << text << "\"\n";
};

// A run before prepare, and a prompt with no tokens in it: the two ways a
// caller can ask for something that has no answer.
auto tGenerationRefusesTheImpossible = test("Generation/refusesTheImpossible") = []
{
    const auto model = SyntheticModel {"generation-refusals"};

    auto unprepared = Gemma {};
    unprepared.load(model.path());

    const auto promptIds = asTokens(promptTokens());
    check(throwsModelError([&] { unprepared.generate(promptIds); }));
    check(throwsModelError([&] { unprepared.shape(); }));

    auto unloaded = Gemma {};
    check(throwsModelError([&] { unloaded.tokenizer(); }));
    check(throwsModelError([&] { unloaded.prepare(); }));

    if (!Device::shared().isValid())
        return;

    auto gemma = Gemma {};
    prepareOver(gemma, model, 16);

    check(throwsModelError([&] { gemma.generate({}); }));

    // A token the embedding has no row for, which would be a gather outside the
    // weights rather than a wrong-looking answer.
    const auto outside = asTokens({2, 999});
    check(throwsModelError([&] { gemma.generate(outside); }));

    check(throwsModelError([&] { gemma.setPromptCapacity(0); }));
};

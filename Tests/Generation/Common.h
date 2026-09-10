#pragma once

#include <HuggingFaceEACP/Generation/Gemma.h>

// The decoder suite's own headers by path rather than by a second copy of what
// they hold: Tests/Decoder/Common.h is where the synthetic checkpoint and its
// scratch directory live, and ReferenceDecoder.h beside it is where the scalar
// forward pass and the DecoderRun that drives one step at a time do. The same
// borrowing Tests/Decoder does from Tests/Model and Tests/Kernels, and for the
// same reason: these are test headers, so nothing but this include path depends
// on them.
#include "../Decoder/ReferenceDecoder.h"

#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace HF::Testing
{
// ---------------------------------------------------------------------------
// A tokenizer.json of exactly the synthetic model's width.
// ---------------------------------------------------------------------------

// The metaspace character, U+2581, spelled as bytes so the source encoding
// cannot change what it means — the same spelling Metaspace.h uses.
inline constexpr auto metaspaceCharacter = "\xe2\x96\x81";

// Whitespace-separated words, which is how the lists below stay one line each
// rather than one word a line: a piece of this vocabulary never holds a space,
// since a space is the metaspace character by the time BPE sees one.
inline Vector<std::string_view> wordsOf(std::string_view text)
{
    auto words = Vector<std::string_view> {};

    for (auto at = text.find_first_not_of(' '); at != std::string_view::npos;
         at = text.find_first_not_of(' ', at))
    {
        const auto end = text.find(' ', at);
        words.add(text.substr(at, end - at));
        at = end;

        if (end == std::string_view::npos)
            break;
    }

    return words;
}

// The whole pieces the vocabulary is built out of, in the order that gives each
// its id: `▁the`, `▁cat` and the rest are the metaspace character in front of
// one of these, and are added separately.
inline constexpr auto syntheticWords = "the cat dog run sit ball and is on a";

// The two-character pieces those words merge through, and the trailing pairs
// that exist as pieces without ever being the way a word is reached.
inline constexpr auto syntheticStems = "th ca do ru si ba ll an";
inline constexpr auto syntheticTails = "he at og un nd it";

// Gemma's shape at sixty-four tokens, which is what smallGemmaConfig's
// embedding holds: the four added tokens at the ids the real file gives them,
// the metaspace character, the twenty-six letters, the stems, the words, the
// same words behind a metaspace, and the trailing pairs. Byte fallback is off,
// since 256 byte pieces do not fit in a vocabulary this size and an unknown
// character has <unk> to go to.
//
// The normalizer and the decoder are MiniTokenizer.json's, so what is under
// test here is the loop rather than a second reading of the tokenizer.
inline Vector<std::string> syntheticVocabulary()
{
    auto pieces = Vector<std::string> {};

    for (auto piece: wordsOf("<pad> <eos> <bos> <unk>"))
        pieces.add(std::string {piece});

    pieces.add(metaspaceCharacter);

    for (auto letter = 'a'; letter <= 'z'; ++letter)
        pieces.add(std::string {letter});

    for (auto stem: wordsOf(syntheticStems))
        pieces.add(std::string {stem});

    for (auto word: wordsOf(syntheticWords))
        if (word.size() > 1)
            pieces.add(std::string {word});

    for (auto word: wordsOf(syntheticWords))
        pieces.add(metaspaceCharacter + std::string {word});

    for (auto tail: wordsOf(syntheticTails))
        pieces.add(std::string {tail});

    return pieces;
}

// In rank order, which is what decides a word rather than the vocabulary's, in
// three groups.
//
// The word-internal merges come first, then the ones that join a word to the
// metaspace character, then the trailing pairs — `he`, `at`, `og`, `un`, `nd`,
// `it` — which are pieces of the vocabulary and must never outrank the first
// group: `he` firing first would strand `t` in `the`, and `at` would strand `c`
// in `cat`. That the metaspace joins come after `a` + `n` is the same rule from
// the other side, since `▁a` firing first would strand `nd` in `▁and`.
inline Vector<std::string> syntheticMerges()
{
    auto merges = Vector<std::string> {};

    auto add = [&](std::string_view left, std::string_view right)
    {
        merges.add("[\"" + std::string {left} + "\",\"" + std::string {right}
                   + "\"]");
    };

    // A merge is a pair, spelled `left|right` so that a whole group of them
    // fits on a line the way the words above do.
    auto addPair = [&](std::string_view text)
    { add(text.substr(0, text.find('|')), text.substr(text.find('|') + 1)); };

    for (auto text: wordsOf("t|h th|e c|a ca|t d|o do|g r|u ru|n s|i si|t b|a "
                            "l|l ba|ll a|n an|d i|s o|n"))
        addPair(text);

    for (auto word: wordsOf(syntheticWords))
        add(metaspaceCharacter, word);

    for (auto text: wordsOf("h|e a|t o|g u|n n|d i|t"))
        addPair(text);

    return merges;
}

inline std::string syntheticTokenizerJson()
{
    const auto pieces = syntheticVocabulary();
    const auto merges = syntheticMerges();

    auto added = std::string {};
    auto vocabulary = std::string {};

    for (auto index = 0; index < pieces.size(); ++index)
    {
        if (index > 0)
            vocabulary += ",";

        vocabulary += "\"" + pieces[index] + "\":" + std::to_string(index);

        if (index < 4)
            added += std::string {added.empty() ? "" : ","}
                     + "{\"id\":" + std::to_string(index) + ",\"content\":\""
                     + pieces[index] + "\",\"special\":true}";
    }

    auto merged = std::string {};

    for (auto index = 0; index < merges.size(); ++index)
        merged += std::string {index == 0 ? "" : ","} + merges[index];

    return "{\"version\":\"1.0\",\"added_tokens\":[" + added
           + "],\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\" "
             "\"},\"content\":\"\xe2\x96\x81\"},\"pre_tokenizer\":null,"
             "\"post_processor\":null,\"decoder\":{\"type\":\"Sequence\","
             "\"decoders\":[{\"type\":\"Replace\",\"pattern\":{\"String\":"
             "\"\xe2\x96\x81\"},\"content\":\" \"},{\"type\":\"Fuse\"}]},"
             "\"model\":{\"type\":\"BPE\",\"unk_token\":\"<unk>\","
             "\"byte_fallback\":false,\"fuse_unk\":true,\"vocab\":{"
           + vocabulary + "},\"merges\":[" + merged + "]}}";
}

// ---------------------------------------------------------------------------
// A whole model directory a Gemma can be loaded from.
// ---------------------------------------------------------------------------

// The decoder suite's synthetic checkpoint with a tokenizer.json beside it, so
// Gemma::load(directory) has every file it asks for. The tokenizer is written
// after the checkpoint rather than by it, since nothing in Tests/Decoder needs
// a vocabulary and ModelFiles is re-read by the load.
class SyntheticModel
{
public:
    explicit SyntheticModel(
        std::string_view name, TensorEdit edit = [](Vector<SyntheticTensor>&) {})
        : synthetic(name, Sharding::Single, std::move(edit))
    {
        auto out = std::ofstream {synthetic.path() / ModelFileNames::tokenizerJson,
                                  std::ios::binary | std::ios::trunc};

        const auto json = syntheticTokenizerJson();
        out.write(json.data(), (std::streamsize) json.size());
    }

    const SyntheticCheckpoint& checkpoint() const { return synthetic; }
    const std::filesystem::path& path() const { return synthetic.path(); }
    const GemmaConfig& config() const { return synthetic.config(); }
    DecoderShape shape() const { return synthetic.shape(); }

    // The shape a Gemma built over this checkpoint has: the config's window,
    // and a step capacity of whatever the run's prompt capacity was.
    DecoderShape shapeWithStepRows(int stepRows) const
    {
        auto shape = synthetic.shape();
        shape.maxStepRows = std::min(stepRows, shape.maxPositions);

        return shape;
    }

private:
    SyntheticCheckpoint synthetic;
};

// ---------------------------------------------------------------------------
// The loop a test runs itself, to compare the one under test against.
// ---------------------------------------------------------------------------

inline Vector<float>
    lastRowOf(const Vector<float>& logits, int rowCount, int rowLength)
{
    auto row = Vector<float> {};
    row.resize(rowLength);

    for (auto index = 0; index < rowLength; ++index)
        row[index] = logits[(rowCount - 1) * rowLength + index];

    return row;
}

// One step at a time through DecoderRun, with Sampler::argmax on a row that has
// been read back — the readback path the on-device Argmax exists to avoid, so
// the two agreeing is the whole point of comparing them.
//
// The prompt is fed in blocks of the shape's own step capacity, which is what a
// Gemma does, so the same helper serves the blocked and the unblocked runs.
struct ReferenceGeneration
{
    Vector<TokenId> tokens;
    bool stoppedOnEndOfSequence = false;

    // Every last-row logit vector the run produced, first the prompt's and then
    // one per step, which is what a test needs to reason about what the model
    // would have chosen next.
    std::vector<Vector<float>> rows;
};

inline ReferenceGeneration referenceGenerate(const SyntheticCheckpoint& checkpoint,
                                             const DecoderShape& shape,
                                             const std::vector<int>& prompt,
                                             int tokenLimit,
                                             TokenId endOfSequence)
{
    const auto rowLength = shape.logitElementCount();
    const auto capacity = shape.stepRowCapacity();

    auto run = DecoderRun {shape, checkpoint.weights()};
    auto result = StepResult {};
    auto rowCount = 0;

    for (auto first = 0; first < (int) prompt.size(); first += capacity)
    {
        const auto rows = std::min(capacity, (int) prompt.size() - first);

        result = run.step(std::vector<int> {prompt.begin() + first,
                                            prompt.begin() + first + rows});
        rowCount = rows;
    }

    auto generation = ReferenceGeneration {};

    for (auto step = 0; step < tokenLimit; ++step)
    {
        auto row = lastRowOf(result.logits, rowCount, rowLength);
        const auto token = Sampler::argmax(row);

        generation.rows.push_back(std::move(row));

        if (token == endOfSequence)
        {
            generation.stoppedOnEndOfSequence = true;
            break;
        }

        generation.tokens.add(token);

        if (step + 1 >= tokenLimit)
            break;

        result = run.step({token});
        rowCount = 1;
    }

    return generation;
}

inline bool sameTokens(Span<const TokenId> left, Span<const TokenId> right)
{
    if (left.size() != right.size())
        return false;

    for (auto index = 0; index < left.size(); ++index)
        if (left[index] != right[index])
            return false;

    return true;
}

inline SyntheticTensor* findTensor(Vector<SyntheticTensor>& tensors,
                                   std::string_view name)
{
    for (auto& tensor: tensors)
        if (tensor.name == name)
            return &tensor;

    return nullptr;
}
} // namespace HF::Testing

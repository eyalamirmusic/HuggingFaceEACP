#include "Common.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// Our tokenizer against llama.cpp's, over the same vocabulary.
//
// Unlike WhisperEACP's version of this test, both sides here run the same
// algorithm on the same data: SentencePiece BPE over gemma's vocabulary, ours
// out of tokenizer.json and llama.cpp's out of the GGUF's tokenizer section,
// which the conversion writes from the same file. So exact agreement is the
// assertion rather than a count, and a disagreement is a bug on one side.
//
// The cases are chosen for the places the two could part: the metaspace
// substitution around leading, trailing and repeated spaces, whether a dummy
// prefix goes in front of a segment, digits (gemma splits them one per token),
// and byte fallback for anything with no piece of its own.

using namespace nano;
using namespace HF;
using namespace HF::Testing;

namespace
{
// Spelled as bytes rather than as literals, the way Tests/Tokenizer does, so
// a case means the same thing whatever the compiler decides the source
// encoding is: "café", "日本語", and a grinning face.
const auto frenchWord = std::string {"caf\xc3\xa9"};
const auto japaneseWord = std::string {"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e"};
const auto grinningFace = std::string {"\xf0\x9f\x98\x80"};

const auto textCases = std::vector<std::string> {
    "",
    "Hello world",
    " leading space",
    "trailing space ",
    "three   spaces   between",
    "The quick brown fox jumps over the lazy dog.",
    "Well... it's 3:45 p.m. -- isn't it?",
    "In 1969 there were 365 days and 12 months.",
    "0123456789",
    frenchWord,
    japaneseWord,
    grinningFace,
    "\n",
    "line one\nline two\n",
    "\tone tab",
    "The capital of France is",
    // The probe the decoder and generation tests run on. It is fed to the
    // model through our tokenizer on one side and llama.cpp's on the other, so
    // the three witnesses only compare if this case agrees.
    "Q: What is the capital of France?\nA:",
};

bool canRun()
{
    return hasOracleModel() && hasOurTokenizer();
}

bool sameTokens(Span<const TokenId> ours, Span<const TokenId> theirs)
{
    return std::equal(ours.begin(), ours.end(), theirs.begin(), theirs.end());
}
} // namespace

// encodeWithBos rather than encode, because add_special is how llama.cpp
// prepends BOS and there is no way to ask it for the merges alone. Both sides
// are therefore asked for exactly what generation feeds the model.
auto tEncodeAgreesWithTheOracle = test("Oracle/Tokenizer/encodeAgrees") = []
{
    if (!canRun())
        return;

    auto& oracle = sharedOracle();
    check(oracle.isValid(), "the GGUF loaded");

    if (!oracle.isValid())
        return;

    const auto& tokenizer = ourTokenizer();

    check(tokenizer.vocabularySize() == oracle.vocabularySize(),
          "the vocabularies are one size");
    check(tokenizer.bos() == oracle.bos(), "bos");
    check(tokenizer.eos() == oracle.eos(), "eos");

    for (const auto& text: textCases)
    {
        const auto ours = tokenizer.encodeWithBos(text);
        const auto theirs = oracle.tokenize(text, true);

        if (!sameTokens(ours, theirs))
            std::cout << "  differ on \"" << text << "\": ours " << spelled(ours)
                      << ", llama.cpp " << spelled(theirs) << "\n";

        check(sameTokens(ours, theirs), "the encodings agree");
    }
};

// Decode is where the two sides come back into the same alphabet: a piece is
// still spelled with U+2581 on both, and turning that back into a space is the
// step this compares. The round trip is asserted separately because it is the
// stronger claim — byte fallback makes every input representable, so a
// decoded encoding that is not the input is a bug whatever the reference says.
auto tDecodeAgreesWithTheOracle = test("Oracle/Tokenizer/decodeAgrees") = []
{
    if (!canRun())
        return;

    auto& oracle = sharedOracle();
    check(oracle.isValid(), "the GGUF loaded");

    if (!oracle.isValid())
        return;

    const auto& tokenizer = ourTokenizer();

    for (const auto& text: textCases)
    {
        const auto tokens = tokenizer.encode(text);
        const auto ours = tokenizer.decode(tokens);
        const auto theirs = oracle.detokenize(tokens);

        if (ours != theirs)
            std::cout << "  differ on " << spelled(tokens) << ": ours \"" << ours
                      << "\", llama.cpp \"" << theirs << "\"\n";

        check(ours == theirs, "the decodings agree");
        check(ours == text, "ours round trips");

        // BOS is a control token, so neither side renders it: the same text
        // has to come back out of the sequence generation actually runs on.
        const auto withBos = tokenizer.encodeWithBos(text);

        check(tokenizer.decode(withBos) == text, "ours drops bos");
        check(oracle.detokenize(withBos) == text, "llama.cpp drops bos");
    }
};

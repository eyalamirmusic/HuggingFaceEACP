#include "../Support/GemmaModel.h"

#include <HuggingFaceEACP/Tokenizer/Tokenizer.h>

#include <Miro/Json.h>
#include <NanoTest/NanoTest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>

using namespace nano;
using namespace HF;

namespace
{
// Spelled as bytes rather than as literals so the tests mean the same thing
// whatever the compiler decides the source encoding is.
constexpr auto eAcute = std::string_view {"\xc3\xa9"};
constexpr auto grinningFace = std::string_view {"\xf0\x9f\x98\x80"};
constexpr auto replacementCharacter = std::string_view {"\xef\xbf\xbd"};
constexpr auto metaspace = std::string_view {"\xe2\x96\x81"};

std::string joined(std::initializer_list<std::string_view> parts)
{
    auto text = std::string {};

    for (auto part: parts)
        text += part;

    return text;
}

bool encodesTo(const Tokenizer& tokenizer,
               std::string_view text,
               std::initializer_list<TokenId> expected)
{
    const auto tokens = tokenizer.encode(text);
    return std::equal(
        tokens.begin(), tokens.end(), expected.begin(), expected.end());
}

// The tokens are named rather than passed straight through: a Span cannot bind
// to a temporary Vector, which is EA::Span refusing to dangle.
bool roundTrips(const Tokenizer& tokenizer, std::string_view text)
{
    const auto tokens = tokenizer.encode(text);
    return tokenizer.decode(tokens) == text;
}

std::filesystem::path fixturePath()
{
    return std::filesystem::path {HF_TOKENIZER_FIXTURE_DIR} / "MiniTokenizer.json";
}

// Gemma's shape at a readable size: the six added tokens at the ids the real
// file gives them, all 256 byte pieces at 6 + the byte, and enough pieces and
// merges for `hello`, `world`, `the` and `cat` to be reachable.
const Tokenizer& fixtureTokenizer()
{
    static const auto tokenizer = Tokenizer::fromFile(fixturePath());
    return tokenizer;
}

TokenId byteToken(std::uint8_t byte)
{
    return 6 + byte;
}

// The three metaspace properties, in the places a tokenizer.json can express
// them. The replacement is an underscore rather than U+2581 — which is a
// property of the file too, so a vocabulary spelled with one proves it is read
// rather than assumed, and keeps this source ASCII. The merges are pairs
// rather than strings, which is the other form the fast tokenizer writes them
// in.
std::string metaspaceJson(std::string_view normalizer, std::string_view preTokenizer)
{
    auto text = std::string {R"({"added_tokens": [], "normalizer": )"};

    text += normalizer;
    text += R"(, "pre_tokenizer": )";
    text += preTokenizer;
    text += R"(, "model": {"type": "BPE", "unk_token": "<unk>",)"
            R"( "byte_fallback": true, "fuse_unk": true,)"
            R"( "vocab": {"<unk>": 0, "_": 1, "a": 2, "b": 3,)"
            R"( "_a": 4, "_b": 5, "_a_b": 6},)"
            R"( "merges": [["_", "a"], ["_", "b"], ["_a", "_b"]]}})";

    return text;
}

bool fileExists(const std::filesystem::path& path)
{
    auto error = std::error_code {};
    return !path.empty() && std::filesystem::is_regular_file(path, error);
}

// The real tokenizer.json is 17.5 MB and never a commit, so it comes from the
// gemma-2b the build fetched, or from the GEMMA_MODEL_DIR override — which is
// what Testing::gemmaModelDirectory() resolves. The test that wants one still
// returns early when the fetch was off, the way the GPU tests do without a
// device.
std::filesystem::path realTokenizerPath()
{
    return Testing::gemmaModelDirectory() / ModelFileNames::tokenizerJson;
}
} // namespace

// --- The byte pieces ----------------------------------------------------

auto tByteNamesRoundTrip = test("Tokenizer/byteNamesRoundTrip") = []
{
    for (auto byte = 0; byte < 256; ++byte)
    {
        const auto name = ByteFallback::nameForByte((std::uint8_t) byte);

        check(name.size() == 6);
        check(ByteFallback::byteForName(name) == byte);
    }

    check(ByteFallback::nameForByte(0) == "<0x00>");
    check(ByteFallback::nameForByte(10) == "<0x0A>");
    check(ByteFallback::nameForByte(255) == "<0xFF>");

    // Everything that is not one of the 256 answers "not a byte piece", which
    // is what keeps decoding from reading an ordinary piece as a byte.
    check(ByteFallback::byteForName("hello") == -1);
    check(ByteFallback::byteForName("<0xGG>") == -1);
    check(ByteFallback::byteForName("<0x1>") == -1);
    check(ByteFallback::byteForName("<unk>") == -1);
};

// --- The committed fixture ----------------------------------------------

auto tFixtureLoads = test("Tokenizer/fixtureLoads") = []
{
    const auto& tokenizer = fixtureTokenizer();

    check(tokenizer.vocabularySize() == 289);

    check(tokenizer.pad() == 0);
    check(tokenizer.eos() == 1);
    check(tokenizer.bos() == 2);
    check(tokenizer.unk() == 3);
    check(tokenizer.specials().startOfTurn == 4);
    check(tokenizer.specials().endOfTurn == 5);

    check(tokenizer.tokenForText(joined({metaspace, "the"})) == 285);
    check(tokenizer.textForToken(285) == joined({metaspace, "the"}));
    check(tokenizer.tokenForText("nonesuch") == invalidTokenId);

    // A SentencePiece vocabulary has to spell all 256 bytes, or byte fallback
    // has nothing to fall back to.
    for (auto byte = 0; byte < 256; ++byte)
    {
        const auto name = ByteFallback::nameForByte((std::uint8_t) byte);
        check(tokenizer.tokenForText(name) == byteToken((std::uint8_t) byte));
    }
};

// The merges are ranked, so `hello` is hell + o rather than any of the other
// ways its characters could come together, and `world` needs four of them.
auto tAppliesMergesInRankOrder = test("Tokenizer/appliesMergesInRankOrder") = []
{
    const auto& tokenizer = fixtureTokenizer();

    check(encodesTo(tokenizer, "hello world", {276, 282}));
    check(encodesTo(tokenizer, "the cat", {284, 288}));
    check(encodesTo(tokenizer, "cat", {287}));
    check(encodesTo(tokenizer, "the", {284}));
};

// A space is U+2581 and part of the piece that follows it, which is the whole
// of what makes this not GPT-2's byte-level BPE. A run of them leaves the
// standalone piece behind.
auto tSpacesBecomeMetaspace = test("Tokenizer/spacesBecomeMetaspace") = []
{
    const auto& tokenizer = fixtureTokenizer();

    check(encodesTo(tokenizer, " the", {285}));
    check(encodesTo(tokenizer, "the  cat", {284, 262, 288}));
    check(roundTrips(tokenizer, "the  cat"));
    check(roundTrips(tokenizer, " the cat "));

    // Not prepended, which is what this fixture's normalizer says and what
    // gemma-2b is believed to say. The check is here so that pointing the
    // suite at a file that does prepend fails loudly rather than quietly.
    check(!tokenizer.metaspace().prependsDummyPrefix);
    check(!tokenizer.metaspace().splitsBeforeReplacement);
    check(tokenizer.metaspace().replacement == metaspace);
};

auto tEncodesEmptyInput = test("Tokenizer/encodesEmptyInput") = []
{
    const auto& tokenizer = fixtureTokenizer();

    check(encodesTo(tokenizer, "", {}));
    check(roundTrips(tokenizer, ""));
};

// --- Byte fallback ------------------------------------------------------

auto tUnknownCharactersFallBackToBytes =
    test("Tokenizer/unknownCharactersFallBackToBytes") = []
{
    const auto& tokenizer = fixtureTokenizer();

    // `z` and `p` have no piece of their own; `a` does.
    check(encodesTo(tokenizer, "zap", {byteToken('z'), 263, byteToken('p')}));
    check(roundTrips(tokenizer, "zap"));

    // Never the unknown token: a vocabulary with all 256 byte pieces has
    // nothing left for <unk> to stand for.
    for (auto token: tokenizer.encode("zap"))
        check(token != tokenizer.unk());
};

auto tMultiByteCharactersRoundTrip =
    test("Tokenizer/multiByteCharactersRoundTrip") = []
{
    const auto& tokenizer = fixtureTokenizer();

    check(encodesTo(tokenizer, eAcute, {byteToken(0xC3), byteToken(0xA9)}));
    check(encodesTo(
        tokenizer,
        grinningFace,
        {byteToken(0xF0), byteToken(0x9F), byteToken(0x98), byteToken(0x80)}));

    check(roundTrips(tokenizer, eAcute));
    check(roundTrips(tokenizer, grinningFace));
    check(roundTrips(tokenizer, joined({"the caf", eAcute, " ", grinningFace})));

    // Bytes that are not valid UTF-8 on their own still reach a byte piece
    // each, because the character loop keeps an undecodable byte as itself.
    check(encodesTo(tokenizer, "\xff\xfe", {byteToken(0xFF), byteToken(0xFE)}));
};

// A generation cut off inside a character, or a single byte piece read on its
// own, has to come back as text rather than as a crash or as invalid UTF-8.
auto tDecodesIncompleteByteRuns = test("Tokenizer/decodesIncompleteByteRuns") = []
{
    const auto& tokenizer = fixtureTokenizer();

    const auto loneByte = Vector<TokenId> {byteToken(0xC3)};
    check(tokenizer.decode(loneByte) == replacementCharacter);

    const auto truncatedEmoji =
        Vector<TokenId> {byteToken(0xF0), byteToken(0x9F), byteToken(0x98)};
    check(tokenizer.decode(truncatedEmoji)
          == joined(
              {replacementCharacter, replacementCharacter, replacementCharacter}));

    // A complete character on either side of the broken one still decodes.
    const auto around = Vector<TokenId> {284, byteToken(0xC3), 288};
    check(tokenizer.decode(around) == joined({"the", replacementCharacter, " cat"}));

    const auto outOfRange = Vector<TokenId> {-1, 284, 100000};
    check(tokenizer.decode(outOfRange) == "the");
};

// --- Added tokens -------------------------------------------------------

auto tSpecialTokensAreMatchedVerbatim =
    test("Tokenizer/specialTokensAreMatchedVerbatim") = []
{
    const auto& tokenizer = fixtureTokenizer();
    const auto text = std::string {"<bos>hello world<eos>"};

    check(encodesTo(tokenizer, text, {2, 276, 282, 1}));

    const auto tokens = tokenizer.encode(text);
    check(tokenizer.decode(tokens) == "hello world");
    check(tokenizer.decodeKeepingSpecialTokens(tokens) == text);

    const auto turn = std::string {"<start_of_turn>the cat<end_of_turn>"};
    check(encodesTo(tokenizer, turn, {4, 284, 288, 5}));

    check(tokenizer.isSpecial(2));
    check(!tokenizer.isSpecial(284));
};

// Ordinary text never produces one: `pad` is the pieces its characters spell,
// not <pad>, and a `<...>` the file never added is ordinary text.
auto tOrdinaryTextNeverProducesSpecials =
    test("Tokenizer/ordinaryTextNeverProducesSpecials") = []
{
    const auto& tokenizer = fixtureTokenizer();

    check(encodesTo(tokenizer, "pad", {byteToken('p'), 263, 265}));

    for (auto token: tokenizer.encode("the cat pad"))
        check(!tokenizer.isSpecial(token));

    check(roundTrips(tokenizer, "<nope>"));
};

auto tEncodeWithBos = test("Tokenizer/encodeWithBos") = []
{
    const auto& tokenizer = fixtureTokenizer();
    const auto tokens = tokenizer.encodeWithBos("hello world");

    check(tokens.size() == 3);
    check(tokens[0] == 2);
    check(tokens[1] == 276);
    check(tokens[2] == 282);

    check(tokenizer.decode(tokens) == "hello world");
};

// --- What the JSON decides ----------------------------------------------

// The dummy prefix and the word split vary between the SentencePiece models
// that share this layer, so both are read from the JSON rather than assumed —
// gemma-2b's own file turns both off. Same text, same vocabulary, three
// spellings.
auto tMetaspaceOptionsComeFromTheJson =
    test("Tokenizer/metaspaceOptionsComeFromTheJson") = []
{
    const auto replaceSpaces =
        std::string_view {R"({"type": "Replace",)"
                          R"( "pattern": {"String": " "}, "content": "_"})"};

    const auto plain = Tokenizer::fromJsonText(metaspaceJson(replaceSpaces, "null"));

    check(plain.metaspace().replacement == "_");
    check(!plain.metaspace().prependsDummyPrefix);
    check(encodesTo(plain, "a b", {2, 5}));

    auto prependingNormalizer =
        std::string {R"({"type": "Sequence", "normalizers": [)"
                     R"({"type": "Prepend", "prepend": "_"}, )"};
    prependingNormalizer += replaceSpaces;
    prependingNormalizer += "]}";

    const auto prepending =
        Tokenizer::fromJsonText(metaspaceJson(prependingNormalizer, "null"));

    check(prepending.metaspace().prependsDummyPrefix);
    check(encodesTo(prepending, "a b", {6}));

    // And comes off again, so the prefix is invisible to a caller.
    const auto prepended = prepending.encode("a b");
    check(prepending.decode(prepended) == "a b");

    // A Metaspace pre-tokenizer says all three at once, and its split is what
    // keeps a merge from spanning a word boundary.
    const auto splitting = Tokenizer::fromJsonText(
        metaspaceJson("null",
                      R"({"type": "Metaspace", "replacement": "_",)"
                      R"( "prepend_scheme": "always", "split": true})"));

    check(splitting.metaspace().prependsDummyPrefix);
    check(splitting.metaspace().splitsBeforeReplacement);
    check(encodesTo(splitting, "a b", {4, 5}));
};

auto tMalformedJsonThrows = test("Tokenizer/malformedJsonThrows") = []
{
    auto threw = false;

    try
    {
        Tokenizer::fromJsonText("{ not json");
    }
    catch (const Miro::Json::ParseError&)
    {
        threw = true;
    }

    check(threw);

    threw = false;

    try
    {
        Tokenizer::fromJsonText(R"({"model": {"vocab": {}}})");
    }
    catch (const TokenizerError&)
    {
        threw = true;
    }

    check(threw);
};

// --- The real vocabulary, when one is available -------------------------

// plan.md asks for the parse time of the 17.5 MB tokenizer.json to be
// measured, because it decides whether this project reads the fast-tokenizer
// form at all or goes to tokenizer.model instead. Printed rather than
// asserted: it is a number to look at, not a threshold to fail on.
auto tRealTokenizerParseTime = test("Tokenizer/realTokenizerParseTime") = []
{
    const auto path = realTokenizerPath();

    if (!fileExists(path))
        return;

    const auto start = std::chrono::steady_clock::now();
    const auto tokenizer = Tokenizer::fromFile(path);
    const auto elapsed =
        std::chrono::duration<double> {std::chrono::steady_clock::now() - start};

    std::cout << "  " << path.string() << "\n  parsed in " << elapsed.count()
              << " s, " << tokenizer.vocabularySize() << " tokens, dummy prefix "
              << (tokenizer.metaspace().prependsDummyPrefix ? "on" : "off") << "\n";

    check(tokenizer.pad() == 0);
    check(tokenizer.eos() == 1);
    check(tokenizer.bos() == 2);
    check(tokenizer.unk() == 3);
    check(tokenizer.vocabularySize() >= 256000);

    // Every byte reaches a piece that stands for exactly it, which is what
    // byte fallback is for — and not that every byte has a `<0xNN>` piece,
    // which the real vocabulary turns out not to hold. 255 of the 256 are
    // there at 217..472; id 226, where `<0x09>` would sit, is a literal tab
    // instead, so a tab goes through its own piece and the other 255 through
    // the fallback.
    //
    // A byte below 0x80 is a character on its own, so the claim there is the
    // whole round trip. One above is not — a lone continuation or lead byte is
    // not UTF-8, and decode replaces an undecodable run with U+FFFD rather
    // than handing back invalid text, which Tokenizer/decodesIncompleteByteRuns
    // is about — so the claim is that it reaches its own byte piece, and the
    // characters below are where such bytes come back through it.
    for (auto byte = 0; byte < 256; ++byte)
    {
        const auto one = std::string(1, (char) byte);

        if (byte < 0x80)
        {
            check(roundTrips(tokenizer, one));
            continue;
        }

        const auto tokens = tokenizer.encode(one);

        check(tokens.size() == 1);
        check(tokens.size() == 1
              && tokenizer.textForToken(tokens[0])
                     == ByteFallback::nameForByte((std::uint8_t) byte));
    }

    const auto sentence =
        std::string_view {"The quick brown fox jumps over the lazy dog."};
    const auto tokens = tokenizer.encodeWithBos(sentence);

    check(tokens.size() > 1);
    check(tokens[0] == tokenizer.bos());
    check(tokenizer.decode(tokens) == sentence);
    check(roundTrips(tokenizer, joined({"caf", eAcute, " ", grinningFace})));
};

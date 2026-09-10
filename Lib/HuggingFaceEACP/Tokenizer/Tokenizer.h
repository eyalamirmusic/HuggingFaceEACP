#pragma once

// Gemma's SentencePiece BPE, read from the fast-tokenizer form in the model's
// tokenizer.json: a vocabulary of pieces, the merges that build them out of
// single characters, the byte pieces a character with no piece of its own
// falls back to, and the added tokens that steer generation.
//
// Not GPT-2's byte-level BPE, so there is no printable alphabet in front of
// BPE and no pre-tokenizer pattern: a space is U+2581 (see Metaspace.h) and
// everything else reaches the merge engine as itself.

#include <HuggingFaceEACP/Core/Core.h>
#include <HuggingFaceEACP/Tokenizer/ByteFallback.h>
#include <HuggingFaceEACP/Tokenizer/Metaspace.h>
#include <HuggingFaceEACP/Tokenizer/SpecialTokens.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace HF
{
class TokenizerError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

class Tokenizer
{
public:
    static Tokenizer fromJsonText(std::string_view jsonText);
    static Tokenizer fromFile(const std::filesystem::path& path);

    Vector<TokenId> encode(std::string_view text) const;

    // What generation feeds the model: the same tokens behind <bos>, which
    // gemma-2b is trained to see and its generation_config asks for.
    Vector<TokenId> encodeWithBos(std::string_view text) const;

    std::string decode(Span<const TokenId> tokens) const;
    std::string decodeKeepingSpecialTokens(Span<const TokenId> tokens) const;

    int vocabularySize() const;

    TokenId bos() const { return specialTokens.beginningOfSequence; }
    TokenId eos() const { return specialTokens.endOfSequence; }
    TokenId pad() const { return specialTokens.padding; }
    TokenId unk() const { return specialTokens.unknown; }

    bool isSpecial(TokenId token) const;

    // The vocabulary's own spelling of a piece, still in the normalized
    // alphabet: `▁the`, not ` the`.
    std::string_view textForToken(TokenId token) const;
    TokenId tokenForText(std::string_view tokenText) const;

    const SpecialTokens& specials() const { return specialTokens; }
    const Metaspace& metaspace() const { return metaspaceOptions; }

private:
    // Transparent, so a std::string_view looks a piece up without first being
    // copied into a std::string.
    struct StringHash
    {
        using is_transparent = void;

        std::size_t operator()(std::string_view text) const
        {
            return std::hash<std::string_view> {}(text);
        }
    };

    using TokenIdByText =
        std::unordered_map<std::string, TokenId, StringHash, std::equal_to<>>;

    struct Merge
    {
        int rank = 0;
        TokenId merged = invalidTokenId;
    };

    struct AddedToken
    {
        std::string content;
        TokenId token = invalidTokenId;
    };

    void addToken(std::string text, TokenId token, bool special);
    void addMerge(std::string_view leftText, std::string_view rightText, int rank);
    void indexAddedTokens();

    TokenId addedTokenAt(std::string_view text, std::size_t position) const;
    void encodeSegment(std::string_view segment, Vector<TokenId>& tokens) const;
    void encodeWord(std::string_view word, Vector<TokenId>& tokens) const;
    Vector<TokenId> charactersOf(std::string_view word) const;
    void appendCharacter(std::string_view character, Vector<TokenId>& ids) const;
    void appendUnknown(Vector<TokenId>& ids) const;

    std::string decodeTokens(Span<const TokenId> tokens, bool keepSpecial) const;

    Vector<std::string> tokenTexts;
    Vector<std::uint8_t> specialFlags;
    TokenIdByText tokensByText;
    std::unordered_map<std::uint64_t, Merge> merges;

    // The 256 byte pieces by byte value, so encoding never formats a `<0xNN>`
    // name to look one up.
    Array<TokenId, 256> byteTokens;

    // Sorted longest first inside each bucket, so the first content that
    // matches at a position is the longest one that does.
    Vector<AddedToken> addedTokens;
    Array<Vector<int>, 256> addedTokensByFirstByte;

    SpecialTokens specialTokens;
    Metaspace metaspaceOptions;
    bool usesByteFallback = false;
    bool fusesUnknown = false;
};
} // namespace HF

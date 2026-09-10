#include "Tokenizer.h"

#include <Miro/Json.h>
#include <Miro/Unicode.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <queue>
#include <utility>
#include <vector>

namespace HF
{
namespace
{
using Miro::Json::Value;

const Value* optionalMember(const Value& parent, const char* key)
{
    if (!parent.isObject())
        return nullptr;

    return Miro::Json::find(parent.asObject(), key);
}

const Value& memberOrThrow(const Value& parent, const char* key)
{
    if (!parent.isObject())
        throw TokenizerError(std::string("expected an object holding '") + key
                             + "'");

    const auto* found = optionalMember(parent, key);

    if (found == nullptr)
        throw TokenizerError(std::string("tokenizer.json has no '") + key + "'");

    return *found;
}

std::string_view
    stringOr(const Value& parent, const char* key, std::string_view fallback)
{
    const auto* found = optionalMember(parent, key);

    if (found == nullptr || !found->isString())
        return fallback;

    return found->asString();
}

bool flagOr(const Value& parent, const char* key, bool fallback)
{
    const auto* found = optionalMember(parent, key);

    if (found == nullptr || !found->isBool())
        return fallback;

    return found->asBool();
}

TokenId asTokenId(const Value& value)
{
    if (!value.isNumber())
        throw TokenizerError("expected a token id");

    return static_cast<TokenId>(value.asNumber());
}

std::pair<std::string_view, std::string_view> splitMergeRule(const Value& entry)
{
    if (entry.isArray())
    {
        const auto& pair = entry.asArray();

        if (pair.size() != 2 || !pair[0].isString() || !pair[1].isString())
            throw TokenizerError("a merge rule is not a pair of strings");

        return {pair[0].asString(), pair[1].asString()};
    }

    if (!entry.isString())
        throw TokenizerError("a merge rule is neither a string nor a pair");

    const auto rule = std::string_view {entry.asString()};
    const auto separator = rule.find(' ');

    if (separator == std::string_view::npos)
        throw TokenizerError("merge rule '" + std::string(rule) + "' has no space");

    return {rule.substr(0, separator), rule.substr(separator + 1)};
}

std::uint64_t mergeKey(TokenId left, TokenId right)
{
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32)
           | static_cast<std::uint32_t>(right);
}

std::string readWholeFile(const std::filesystem::path& path)
{
    auto stream = std::ifstream {path, std::ios::binary};

    if (!stream)
        throw TokenizerError("cannot open '" + path.string() + "'");

    return std::string {std::istreambuf_iterator<char> {stream},
                        std::istreambuf_iterator<char> {}};
}

// A normalizer or a pre-tokenizer is either one component or a Sequence of
// them, and the two spell their children under different keys.
void forEachComponent(const Value* node,
                      const std::function<void(const Value&)>& visit)
{
    if (node == nullptr || !node->isObject())
        return;

    for (const auto* key: {"normalizers", "pretokenizers"})
    {
        const auto* children = optionalMember(*node, key);

        if (children == nullptr || !children->isArray())
            continue;

        for (const auto& child: children->asArray())
            forEachComponent(&child, visit);

        return;
    }

    visit(*node);
}

// The three metaspace properties this project could not confirm against the
// real gemma-2b tokenizer.json, each read from wherever the fast tokenizer
// expresses it. Check every one of them once GEMMA_MODEL_DIR points at a
// downloaded model: what is here is HuggingFace's own defaults, not a guess
// about which of them Gemma uses.
void readMetaspaceComponent(const Value& component, Metaspace& metaspace)
{
    const auto type = stringOr(component, "type", {});

    // A Prepend normalizer is one of the two ways the fast tokenizer spells
    // SentencePiece's dummy prefix, and it carries the string it prepends.
    if (type == "Prepend")
    {
        metaspace.prependsDummyPrefix = true;

        if (const auto prepended = stringOr(component, "prepend", {});
            !prepended.empty())
            metaspace.replacement = prepended;

        return;
    }

    // `Replace(" " -> "▁")` is how a space becomes a metaspace character
    // without a Metaspace component. A Replace of any other pattern is some
    // other normalization and is not ours to read.
    if (type == "Replace")
    {
        const auto* pattern = optionalMember(component, "pattern");

        if (pattern == nullptr || stringOr(*pattern, "String", {}) != " ")
            return;

        if (const auto content = stringOr(component, "content", {});
            !content.empty())
            metaspace.replacement = content;

        return;
    }

    // The other spelling: one Metaspace component carrying all three. Its
    // HuggingFace defaults are prepend_scheme "always" and split true, so a
    // Metaspace that names neither asks for both.
    if (type == "Metaspace")
    {
        if (const auto replacement = stringOr(component, "replacement", {});
            !replacement.empty())
            metaspace.replacement = replacement;

        if (const auto scheme = stringOr(component, "prepend_scheme", {});
            !scheme.empty())
            metaspace.prependsDummyPrefix = scheme != "never";
        else
            metaspace.prependsDummyPrefix =
                flagOr(component, "add_prefix_space", true);

        metaspace.splitsBeforeReplacement = flagOr(component, "split", true);
    }
}

Metaspace readMetaspace(const Value& root)
{
    auto metaspace = Metaspace {};

    const auto read = [&](const Value& component)
    { readMetaspaceComponent(component, metaspace); };

    forEachComponent(optionalMember(root, "normalizer"), read);
    forEachComponent(optionalMember(root, "pre_tokenizer"), read);

    return metaspace;
}
} // namespace

Tokenizer Tokenizer::fromFile(const std::filesystem::path& path)
{
    return fromJsonText(readWholeFile(path));
}

Tokenizer Tokenizer::fromJsonText(std::string_view jsonText)
{
    const auto root = Miro::Json::parse(jsonText);
    const auto& model = memberOrThrow(root, "model");
    const auto& vocabValue = memberOrThrow(model, "vocab");
    const auto& mergesValue = memberOrThrow(model, "merges");

    if (!vocabValue.isObject())
        throw TokenizerError("model.vocab is not an object");

    if (!mergesValue.isArray())
        throw TokenizerError("model.merges is not an array");

    const auto& vocab = vocabValue.asObject();
    const auto* added = optionalMember(root, "added_tokens");
    const auto hasAddedTokens = added != nullptr && added->isArray();

    auto highestId = TokenId {-1};

    for (const auto& [text, id]: vocab)
        highestId = std::max(highestId, asTokenId(id));

    if (hasAddedTokens)
    {
        for (const auto& entry: added->asArray())
            highestId = std::max(highestId, asTokenId(memberOrThrow(entry, "id")));
    }

    auto tokenizer = Tokenizer {};

    // Zero is <pad>'s id and not "no such piece", so the table says so itself
    // before anything looks a byte up in it.
    tokenizer.byteTokens.fill(invalidTokenId);

    tokenizer.tokenTexts.resize(highestId + 1);
    tokenizer.specialFlags.resize(highestId + 1);
    tokenizer.tokensByText.reserve((std::size_t) (highestId + 1));

    tokenizer.usesByteFallback = flagOr(model, "byte_fallback", false);
    tokenizer.fusesUnknown = flagOr(model, "fuse_unk", false);
    tokenizer.metaspaceOptions = readMetaspace(root);

    for (const auto& [text, id]: vocab)
        tokenizer.addToken(text, asTokenId(id), false);

    if (hasAddedTokens)
    {
        for (const auto& entry: added->asArray())
        {
            const auto id = asTokenId(memberOrThrow(entry, "id"));
            const auto& content = memberOrThrow(entry, "content").asString();

            // Every added token is matched verbatim; only the ones the file
            // marks special are the ones decode() leaves out.
            tokenizer.addToken(content, id, flagOr(entry, "special", true));
            tokenizer.addedTokens.add(AddedToken {content, id});
            assignSpecialToken(tokenizer.specialTokens, content, id);
        }
    }

    tokenizer.indexAddedTokens();

    // Found by reading the vocabulary rather than by looking `<0xNN>` up in
    // it, so a file that spells the digits in lower case is read as well.
    for (auto id = 0; id < tokenizer.tokenTexts.size(); ++id)
    {
        if (const auto byte = ByteFallback::byteForName(tokenizer.tokenTexts[id]);
            byte >= 0)
            tokenizer.byteTokens[byte] = id;
    }

    // model.unk_token names the piece byte fallback falls back to, and it is
    // the vocabulary's own spelling rather than an added token's.
    if (const auto unknown = stringOr(model, "unk_token", {}); !unknown.empty())
    {
        if (const auto id = tokenizer.tokenForText(unknown); id != invalidTokenId)
            tokenizer.specialTokens.unknown = id;
    }

    const auto& mergeRules = mergesValue.asArray();
    tokenizer.merges.reserve((std::size_t) mergeRules.size());

    for (auto rank = 0; rank < mergeRules.size(); ++rank)
    {
        const auto [leftText, rightText] = splitMergeRule(mergeRules[rank]);
        tokenizer.addMerge(leftText, rightText, rank);
    }

    return tokenizer;
}

void Tokenizer::addToken(std::string text, TokenId token, bool special)
{
    if (token < 0 || text.empty())
        return;

    if (token >= tokenTexts.size())
    {
        tokenTexts.resize(token + 1);
        specialFlags.resize(token + 1);
    }

    tokenTexts[token] = std::move(text);
    specialFlags[token] = special ? 1 : 0;
    tokensByText.insert_or_assign(tokenTexts[token], token);
}

void Tokenizer::addMerge(std::string_view leftText,
                         std::string_view rightText,
                         int rank)
{
    auto mergedText = std::string {leftText};
    mergedText += rightText;

    const auto left = tokenForText(leftText);
    const auto right = tokenForText(rightText);
    const auto merged = tokenForText(mergedText);

    if (left == invalidTokenId || right == invalidTokenId
        || merged == invalidTokenId)
        return;

    merges.emplace(mergeKey(left, right), Merge {rank, merged});
}

void Tokenizer::indexAddedTokens()
{
    for (auto index = 0; index < addedTokens.size(); ++index)
    {
        const auto& content = addedTokens[index].content;

        if (!content.empty())
            addedTokensByFirstByte[(unsigned char) content.front()].add(index);
    }

    // Longest first, so the first content that matches at a position is the
    // longest one that does and no added token is ever a prefix of the match.
    for (auto& bucket: addedTokensByFirstByte)
    {
        std::sort(bucket.begin(),
                  bucket.end(),
                  [this](int left, int right)
                  {
                      return addedTokens[left].content.size()
                             > addedTokens[right].content.size();
                  });
    }
}

int Tokenizer::vocabularySize() const
{
    return tokenTexts.size();
}

bool Tokenizer::isSpecial(TokenId token) const
{
    return token >= 0 && token < specialFlags.size() && specialFlags[token] != 0;
}

std::string_view Tokenizer::textForToken(TokenId token) const
{
    if (token < 0 || token >= tokenTexts.size())
        return {};

    return tokenTexts[token];
}

TokenId Tokenizer::tokenForText(std::string_view tokenText) const
{
    const auto found = tokensByText.find(tokenText);
    return found == tokensByText.end() ? invalidTokenId : found->second;
}

TokenId Tokenizer::addedTokenAt(std::string_view text, std::size_t position) const
{
    const auto firstByte = (unsigned char) text[position];

    for (auto index: addedTokensByFirstByte[firstByte])
    {
        const auto& candidate = addedTokens[index];

        if (text.compare(position, candidate.content.size(), candidate.content) == 0)
            return candidate.token;
    }

    return invalidTokenId;
}

void Tokenizer::appendUnknown(Vector<TokenId>& ids) const
{
    if (specialTokens.unknown == invalidTokenId)
        return;

    // fuse_unk: a run of characters the vocabulary has nothing at all for is
    // one unknown token rather than one each.
    if (fusesUnknown && !ids.empty() && ids.back() == specialTokens.unknown)
        return;

    ids.add(specialTokens.unknown);
}

void Tokenizer::appendCharacter(std::string_view character,
                                Vector<TokenId>& ids) const
{
    if (const auto id = tokenForText(character); id != invalidTokenId)
    {
        ids.add(id);
        return;
    }

    if (!usesByteFallback)
    {
        appendUnknown(ids);
        return;
    }

    // All of the character's bytes or none of them, as HuggingFace's BPE does
    // it: a vocabulary missing one of the 256 byte pieces spells the whole
    // character as unknown rather than half of it as bytes.
    for (auto byte: character)
    {
        if (byteTokens[(unsigned char) byte] == invalidTokenId)
        {
            appendUnknown(ids);
            return;
        }
    }

    for (auto byte: character)
        ids.add(byteTokens[(unsigned char) byte]);
}

// One symbol per character, and byte fallback before the merges rather than
// after them: the byte pieces are ordinary pieces, so a merge rule naming one
// applies to it like any other.
Vector<TokenId> Tokenizer::charactersOf(std::string_view word) const
{
    auto ids = Vector<TokenId> {};

    for (auto position = std::size_t {}; position < word.size();)
    {
        const auto decoded = Miro::Unicode::decodeUtf8(word, position);

        // A byte that starts no valid sequence is one symbol of its own, and
        // byte fallback is what then spells it.
        const auto length = decoded.valid ? (std::size_t) decoded.byteLength : 1;

        appendCharacter(word.substr(position, length), ids);
        position += length;
    }

    return ids;
}

void Tokenizer::encodeWord(std::string_view word, Vector<TokenId>& tokens) const
{
    struct Symbol
    {
        TokenId token = invalidTokenId;
        int previous = -1;
        int next = -1;
        bool alive = true;
    };

    struct Candidate
    {
        int rank = 0;
        int left = 0;
        TokenId leftToken = invalidTokenId;
        TokenId rightToken = invalidTokenId;

        bool operator>(const Candidate& other) const
        {
            return rank != other.rank ? rank > other.rank : left > other.left;
        }
    };

    const auto characters = charactersOf(word);

    if (characters.empty())
        return;

    auto symbols = Vector<Symbol> {};
    symbols.reserve(characters.size());

    for (auto index = 0; index < characters.size(); ++index)
        symbols.add(Symbol {characters[index], index - 1, index + 1, true});

    symbols.back().next = -1;

    auto queue =
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> {};

    const auto pushCandidate = [&](int left)
    {
        if (left < 0 || symbols[left].next < 0)
            return;

        const auto right = symbols[left].next;
        const auto found =
            merges.find(mergeKey(symbols[left].token, symbols[right].token));

        if (found == merges.end())
            return;

        queue.push(Candidate {
            found->second.rank, left, symbols[left].token, symbols[right].token});
    };

    for (auto index = 0; index < symbols.size(); ++index)
        pushCandidate(index);

    while (!queue.empty())
    {
        const auto candidate = queue.top();
        queue.pop();

        auto& left = symbols[candidate.left];

        if (!left.alive || left.token != candidate.leftToken || left.next < 0)
            continue;

        const auto rightIndex = left.next;
        auto& right = symbols[rightIndex];

        if (right.token != candidate.rightToken)
            continue;

        const auto found = merges.find(mergeKey(left.token, right.token));

        if (found == merges.end())
            continue;

        left.token = found->second.merged;
        left.next = right.next;
        right.alive = false;

        if (left.next >= 0)
            symbols[left.next].previous = candidate.left;

        pushCandidate(candidate.left);
        pushCandidate(left.previous);
    }

    for (auto index = 0; index >= 0; index = symbols[index].next)
    {
        if (symbols[index].token != invalidTokenId)
            tokens.add(symbols[index].token);
    }
}

// Normalization is per segment rather than over the whole input, which is
// where HuggingFace applies it: a dummy prefix therefore lands after every
// added token as well as at the start.
void Tokenizer::encodeSegment(std::string_view segment,
                              Vector<TokenId>& tokens) const
{
    const auto normalized = metaspaceOptions.normalize(segment);

    for (auto word: metaspaceOptions.split(normalized))
        encodeWord(word, tokens);
}

Vector<TokenId> Tokenizer::encode(std::string_view text) const
{
    auto tokens = Vector<TokenId> {};
    auto segmentStart = std::size_t {};
    auto position = std::size_t {};

    while (position < text.size())
    {
        const auto added = addedTokenAt(text, position);

        if (added == invalidTokenId)
        {
            ++position;
            continue;
        }

        if (position > segmentStart)
            encodeSegment(text.substr(segmentStart, position - segmentStart),
                          tokens);

        tokens.add(added);
        position += textForToken(added).size();
        segmentStart = position;
    }

    if (segmentStart < text.size())
        encodeSegment(text.substr(segmentStart), tokens);

    return tokens;
}

Vector<TokenId> Tokenizer::encodeWithBos(std::string_view text) const
{
    auto tokens = Vector<TokenId> {};

    if (specialTokens.beginningOfSequence != invalidTokenId)
        tokens.add(specialTokens.beginningOfSequence);

    for (auto token: encode(text))
        tokens.add(token);

    return tokens;
}

std::string Tokenizer::decodeTokens(Span<const TokenId> tokens,
                                    bool keepSpecial) const
{
    auto text = std::string {};
    auto byteRun = std::string {};

    const auto flushByteRun = [&]
    {
        if (byteRun.empty())
            return;

        ByteFallback::appendRunAsText(text, byteRun);
        byteRun.clear();
    };

    for (auto token: tokens)
    {
        const auto tokenText = textForToken(token);

        if (tokenText.empty())
            continue;

        // Byte pieces first: a run of them is one string, and only the whole
        // run knows whether it spells a character.
        if (const auto byte = ByteFallback::byteForName(tokenText); byte >= 0)
        {
            byteRun += static_cast<char>(byte);
            continue;
        }

        flushByteRun();

        if (isSpecial(token))
        {
            if (keepSpecial)
                text += tokenText;

            continue;
        }

        text += metaspaceOptions.restoreSpaces(tokenText);
    }

    flushByteRun();

    // The dummy prefix is a space that was never in the input, so it comes off
    // again — and only from the very front, which is where encoding put the
    // first of them.
    if (metaspaceOptions.prependsDummyPrefix && text.starts_with(' '))
        text.erase(0, 1);

    return text;
}

std::string Tokenizer::decode(Span<const TokenId> tokens) const
{
    return decodeTokens(tokens, false);
}

std::string Tokenizer::decodeKeepingSpecialTokens(Span<const TokenId> tokens) const
{
    return decodeTokens(tokens, true);
}
} // namespace HF

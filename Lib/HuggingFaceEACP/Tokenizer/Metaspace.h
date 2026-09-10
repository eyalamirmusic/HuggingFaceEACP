#pragma once

#include <HuggingFaceEACP/Core/Core.h>

#include <string>
#include <string_view>

// SentencePiece's metaspace layer, which is what Gemma has instead of GPT-2's
// byte-level alphabet: a space becomes U+2581 before BPE sees the text, and
// becomes a space again on the way out. Nothing else is touched, so every
// other byte reaches BPE as itself and a character with no piece of its own
// leaves through ByteFallback.
//
// Every field here is read out of the model's tokenizer.json, because the
// three that vary are exactly the ones this project could not confirm without
// the gated file: see Tokenizer::fromJsonText for where each is read from, and
// check them against the real tokenizer.json once GEMMA_MODEL_DIR points at a
// downloaded gemma-2b.
namespace HF
{
struct Metaspace
{
    // U+2581 LOWER ONE EIGHTH BLOCK, spelled as bytes so the source encoding
    // cannot change what it means.
    std::string replacement = "\xe2\x96\x81";

    // Whether one replacement is prepended to the text, so that a first word
    // is spelled the same as a word after a space. SentencePiece calls it the
    // dummy prefix; Llama's tokenizer.json asks for it and Gemma is believed
    // not to, which is why the default here is off.
    bool prependsDummyPrefix = false;

    // Whether BPE is run on each replacement-led word separately rather than
    // on the whole text at once. Only a merge rule that spans a word boundary
    // can tell the two apart, and only a Metaspace pre-tokenizer asks for it —
    // with a null pre_tokenizer, which is what Gemma is believed to have, the
    // whole segment is one word.
    bool splitsBeforeReplacement = false;

    std::string normalize(std::string_view text) const;

    // The words BPE is run on. The views point into the caller's normalized
    // text, which therefore has to outlive them.
    Vector<std::string_view> split(std::string_view normalizedText) const;

    std::string restoreSpaces(std::string_view tokenText) const;
};
} // namespace HF

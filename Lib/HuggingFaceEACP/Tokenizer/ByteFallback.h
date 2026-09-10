#pragma once

#include <HuggingFaceEACP/Core/Core.h>

#include <cstdint>
#include <string>
#include <string_view>

// The `<0x00>` .. `<0xFF>` pieces a SentencePiece vocabulary carries so that a
// character it has no piece of its own for still encodes: the character's
// UTF-8 bytes become one piece each. Gemma's tokenizer.json asks for this with
// `"byte_fallback": true`, and it is why the vocabulary needs no byte-level
// alphabet of the kind GPT-2 puts in front of BPE.
namespace HF::ByteFallback
{
std::string nameForByte(std::uint8_t byte);

// -1 for a piece that is not one of the 256, so the same call answers both
// "which byte is this" and "is this a byte piece at all".
int byteForName(std::string_view tokenText);

// A run of decoded byte pieces, as one string. A run that is not valid UTF-8 —
// a generation cut off in the middle of a character, or a lone byte piece —
// becomes one U+FFFD per byte, which is what HuggingFace's ByteFallback
// decoder emits and is the reason a truncated run cannot crash a caller that
// treats the result as text.
void appendRunAsText(std::string& text, std::string_view byteRun);
} // namespace HF::ByteFallback

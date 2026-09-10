#pragma once

#include <HuggingFaceEACP/Core/Core.h>

#include <string_view>

namespace HF
{
// The tokens that steer generation, named rather than numbered. Every field is
// read out of the model's tokenizer.json rather than hard-coded: gemma-2b's
// generation_config gives BOS 2, EOS 1 and PAD 0, and the numbers below are
// what the file says they are.
//
// The turn tokens exist in the -it vocabularies and are the whole of the chat
// prompt format; a base gemma-2b leaves them at invalidTokenId.
struct SpecialTokens
{
    TokenId padding = invalidTokenId;
    TokenId endOfSequence = invalidTokenId;
    TokenId beginningOfSequence = invalidTokenId;
    TokenId unknown = invalidTokenId;
    TokenId startOfTurn = invalidTokenId;
    TokenId endOfTurn = invalidTokenId;
};

// Sorts one added_tokens entry into the field its content names. Everything
// else is left alone: Gemma's added_tokens also carries a hundred <unusedN>
// placeholders, which are added tokens without being any of these.
void assignSpecialToken(SpecialTokens& tokens,
                        std::string_view content,
                        TokenId token);
} // namespace HF

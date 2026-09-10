#pragma once

namespace HF
{
// A token's index into the vocabulary — what the embed kernel gathers by, what
// the argmax writes, and what sampling draws.
//
// It lives in Core rather than beside SpecialTokens, where it started, because
// a module that never reads a tokenizer.json still has to name the type: the
// sampler turns a row of logits into one of these and has no business linking
// a vocabulary to say so.
using TokenId = int;

inline constexpr auto invalidTokenId = TokenId {-1};
} // namespace HF

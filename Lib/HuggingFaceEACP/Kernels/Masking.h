#pragma once

namespace HF
{
// The score a causally masked key gets, chosen so that a softmax over the row
// turns it into a weight of exactly zero and nothing else.
//
// The softmax subtracts the row maximum before exponentiating, so what has to
// underflow is sentinel - rowMaximum, and exp of anything below about -104 is
// zero in float32 — this leaves twenty-eight orders of magnitude of margin
// over that, whatever the real scores are. It is finite rather than -infinity
// because -inf - -inf is a NaN, and it stops eight orders of magnitude short
// of -FLT_MAX so that the subtraction cannot overflow to -infinity either.
//
// It is never the row maximum: query i sits at absolute position
// keyCount - queryCount + i, which is a key index in range, so every row has
// its own diagonal unmasked and therefore at least one real score in it.
//
// A header of its own rather than a constant inside an attention kernel,
// because the products in TiledMatMul.h mask on the store and fold the maxima
// of what they stored, and they must agree with whatever attention kernel
// reads them about which value means "not a key".
inline constexpr auto causalMaskScore = -1e30f;
} // namespace HF

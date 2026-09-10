#pragma once

#include <stdexcept>

namespace HF
{
// Every way a draw can be asked for and refused: a row with no logits in it, a
// mask that is not the vocabulary's width, and options that describe no
// distribution at all — a negative temperature, a negative k, a p outside
// (0, 1].
//
// A type of this module's own rather than a bare std::invalid_argument,
// because that is what the rest of this tree does: ModelError for the
// checkpoint, TokenizerError for the vocabulary, this for the draw, so a
// caller catches exactly one thing per layer.
class SamplingError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
} // namespace HF

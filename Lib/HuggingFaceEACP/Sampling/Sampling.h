#pragma once

// What a row of logits becomes: the temperature, the two cuts and the draw
// that turn a decoder step's output into the next token, on the CPU behind a
// readback.
//
// The greedy path is here as well as on the device, in Kernels/Argmax.h, and
// the two are written to agree. This module owns the rest of plan.md's step 6,
// and the generation loop above it owns the BOS in front and the EOS that
// stops it.

#include <HuggingFaceEACP/Core/Core.h>
#include <HuggingFaceEACP/Sampling/Sampler.h>
#include <HuggingFaceEACP/Sampling/SamplingError.h>

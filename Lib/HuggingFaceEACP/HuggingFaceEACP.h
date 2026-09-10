#pragma once

// HuggingFaceEACP: Hugging Face models on eacp's GPU compute stack.
//
// Every layer of a model is authored as an eacp::GPU::ComputeProgram — a C++
// struct whose body is written in eacp's shader EDSL rather than in a shading
// language — so one source emits MSL on Apple and HLSL on Windows, and Metal
// and D3D12 both come free. `plan.md` carries which model comes first and what
// it needs.

#include "Core/Core.h"
#include "Decoder/Decoder.h"

// Generation has one class, so Gemma.h doubles as its umbrella the way
// Decoder/Decoder.h does — there is no Generation/Generation.h to include.
#include "Generation/Gemma.h"
#include "Kernels/Kernels.h"
#include "Model/Model.h"
#include "Sampling/Sampling.h"
#include "Tokenizer/Tokenizer.h"

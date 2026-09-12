#pragma once

// The op set a decoder-only transformer is assembled from, each a
// GPU::ComputeProgram in its own header. Shapes are uniforms rather than
// Gemma 2B's own numbers: these are what the decoder gets written out of, and
// a matmul that only knows a width of 2048 is one that gets rewritten for 7B.
//
// The outer extent — how many rows, how many elements, how many output rows —
// comes from the dispatch and never appears as a uniform, since the generated
// bounds guard already has it. The inner extents, which are what a kernel
// walks its buffers at, are uniforms.
//
// Each began as the simplest kernel that is correct — a thread per row or per
// output element, reducing serially — which is the version a scalar CPU
// reference is checked against without argument. The ones a decode step spends
// its time in give a group of threads to a row (Reduce.h) or split an inner
// sum across one (SplitLinear), and keep agreeing with the same references.
//
// Most of these carry over from WhisperEACP, which is a template here and
// never a dependency. Four are Gemma's own: RMSNorm, which is the only
// normalisation it has; RoPE, which is the only way position reaches it;
// GeGLU, which is its feed-forward's gate; and the multi-query attention,
// whose 256-wide head is what WhisperEACP's single-query kernel could not
// hold.

#include "Add.h"
#include "Argmax.h"
#include "Embed.h"
#include "GeGLU.h"
#include "Gelu.h"
#include "Int8Blocks.h"
#include "KernelTypes.h"
#include "Linear.h"
#include "Masking.h"
#include "MatMul.h"
#include "MultiQueryAttention.h"
#include "RMSNorm.h"
#include "Reduce.h"
#include "RoPE.h"
#include "SimdTiledMatMul.h"
#include "Softmax.h"
#include "TiledMatMul.h"
#include "WeightStorage.h"

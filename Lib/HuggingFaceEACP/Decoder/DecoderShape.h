#pragma once

#include <HuggingFaceEACP/Kernels/Kernels.h>
#include <HuggingFaceEACP/Model/GemmaConfig.h>

namespace HF
{
// Every extent one decoder is built against. All of them reach the kernels as
// uniforms, so this is a description of a model rather than a set of numbers
// compiled in: fromConfig reads gemma-2b's out of config.json, and a test
// writes its own small ones.
//
// Three of the fields are constants of the architecture rather than counts,
// and are here for the same reason the counts are — a kernel takes each as a
// uniform, so a checkpoint that names a different one runs without a
// recompile. rmsNormEpsilon is config.json's rms_norm_eps, ropeTheta its
// rope_theta, and embeddingScale the factor Gemma multiplies its input
// embedding by.
//
// **embeddingScale is sqrt(hiddenSize) as a float**, which is 45.254833 at
// 2048. Hugging Face builds that constant in the model's own dtype, so a bf16
// run rounds it to 45.25; llama.cpp over an F32 GGUF — the oracle plan.md's
// third step compares against — uses sqrtf(n_embd) unrounded, and so does
// this. The number is the config's rather than the kernel's precisely so that
// a run which wants the bf16-rounded one sets it here and nothing in Kernels/
// moves.
//
// **The two capacities are separate, and that is the whole memory budget.**
// maxPositions is how many tokens the KV cache holds, and maxStepRows is how
// many rows one step() may carry. A generation run makes them wildly different:
// the caches want the full window, and a step is the prompt once and then a
// single row for every token after it. Sizing the per-step intermediates at the
// window would put the gated feed-forward's pair at 1.07 GB and 537 MB for
// gemma-2b's 8192 positions — more than the weights of the layer that reads
// them, for buffers a decode step uses one row of.
//
// Zero, the default, means "the same as maxPositions", so fromConfig and a
// decoder that never says otherwise behave exactly as they did when one number
// served both. A decoder built for a shorter window is separately what lets the
// real weights be checked at a size a machine can afford, the way WhisperEACP's
// crossPositions is a parameter rather than 1500.
struct DecoderShape
{
    int width = 0;
    int heads = 0;
    int kvHeads = 0;
    int headWidth = 0;
    int layers = 0;
    int intermediate = 0;
    int vocabularySize = 0;
    int maxPositions = 0;
    int maxStepRows = 0;

    float rmsNormEpsilon = RMSNorm::gemmaEpsilon;
    double ropeTheta = gemmaRopeTheta;
    float embeddingScale = 1.f;

    static DecoderShape fromConfig(const GemmaConfig& config);

    // Weights are loaded against one of these and dispatched against another,
    // and nothing about a GPU buffer says which — so the two are compared
    // outright before the first dispatch.
    friend bool operator==(const DecoderShape&, const DecoderShape&) = default;

    // What the kernels below this cannot check for themselves, checked once
    // where the shape is built. A ModelError naming the field, since the
    // alternative is a threadgroup array read past its end or a rotation that
    // pairs a channel with itself.
    void validate() const;

    // The concatenated query heads, which is what q_proj writes and o_proj
    // reads, and the shared KV heads, which is one cache row. Neither is width
    // in general: Gemma 2B's eight heads of 256 are 2048 wide by coincidence
    // of the config, and its one KV head is 256.
    int queryWidth() const { return heads * headWidth; }
    int kvWidth() const { return kvHeads * headWidth; }

    // One row of each of the widths a step walks. A step of n tokens is n of
    // them: the prompt a sequence opens with is decoded in one call, so a step
    // is not a single token and none of the counts below is a buffer size.
    int rowElementCount() const { return width; }
    int queryRowElementCount() const { return queryWidth(); }
    int logitElementCount() const { return vocabularySize; }

    // How many rows one step() may carry, with zero resolved to the window.
    int stepRowCapacity() const
    {
        return maxStepRows > 0 ? maxStepRows : maxPositions;
    }

    // What every per-token intermediate is sized at, once: the largest step the
    // decoder was built for, so the same buffers serve a prompt of many rows and
    // the single token after it at different dispatch heights.
    int stepElementCount() const { return stepRowCapacity() * width; }
    int stepQueryElementCount() const { return stepRowCapacity() * queryWidth(); }

    // The gated feed-forward's two intermediates: gate and up are one product
    // over the concatenated weight, so the buffer it writes is twice as wide
    // as the one GeGLU folds it down to.
    int fusedElementCount() const { return stepRowCapacity() * 2 * intermediate; }
    int activatedElementCount() const { return stepRowCapacity() * intermediate; }

    // One layer's K or V cache, sized for the whole run rather than for one
    // step: it grows a row of kvWidth per token up to maxPositions.
    int cacheElementCount() const { return maxPositions * kvWidth(); }

    MultiQueryShape attentionShape() const { return {heads, kvHeads, headWidth}; }

    float attentionScale() const;
};
} // namespace HF

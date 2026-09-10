#pragma once

// llama.cpp's C API behind one RAII handle: a model, a context and the
// vocabulary that hangs off the model, freed in that order whatever a test
// does with them.
//
// What it reads is gemma-2b.gguf, the F32 conversion google/gemma-2b ships
// beside the safetensors. F32 on both sides is what makes this tier worth a
// dependency — the reference holds the same weights at the same width, so a
// logit that disagrees is arithmetic of ours rather than a conversion, which
// is not a claim the fp16 GGML files most models ship can support.
//
// Everything skips on a missing file, the same shape as a GPU test returning
// early when Device::shared().isValid() is false: the GGUF is 10 GB, and the
// ungated mirror the build fetches has no copy of it, so it is a manual
// download out of Google's own gated repo and never a commit.

#include <HuggingFaceEACP/Core/Core.h>
#include <HuggingFaceEACP/Tokenizer/SpecialTokens.h>

#include <filesystem>
#include <string>
#include <string_view>

struct llama_context;
struct llama_model;
struct llama_vocab;

namespace HF::Testing
{
class LlamaOracle
{
public:
    // Longer than any prompt this suite has and short enough that the KV
    // cache costs nothing. A caller that wants a longer sequence says so, and
    // pays for it in the cache rather than in the comparison.
    static constexpr auto defaultContextSize = 512;

    static bool isAvailable();

    // gemma-2b.gguf in the model directory the tests resolve — the
    // GEMMA_MODEL_DIR override, else the one the build assembled, which is a
    // mirror without the GGUF. Named separately from isAvailable() so a
    // failure can say which file it wanted.
    static std::filesystem::path ggufPath();

    static LlamaOracle load(int contextSize = defaultContextSize);

    explicit LlamaOracle(const std::filesystem::path& ggufFile,
                         int contextSize = defaultContextSize);

    LlamaOracle(LlamaOracle&& other) noexcept;
    LlamaOracle& operator=(LlamaOracle&& other) noexcept;

    LlamaOracle(const LlamaOracle&) = delete;
    LlamaOracle& operator=(const LlamaOracle&) = delete;

    ~LlamaOracle();

    bool isValid() const;

    Vector<TokenId> tokenize(std::string_view text, bool addBos) const;
    std::string detokenize(Span<const TokenId> tokens) const;

    // [tokens.size(), vocabularySize()] row-major, over one batch from an
    // emptied KV cache: row i is the distribution over the token that follows
    // tokens[i], so the last row is what a generation step would sample from.
    Vector<float> logits(Span<const TokenId> tokens);

    // The greedy continuation of a prompt, the prompt's own tokens not
    // included, stopping at end of generation or after maxTokens. Token by
    // token through the KV cache, which is the loop plan.md's step 4 builds.
    Vector<TokenId> greedy(Span<const TokenId> prompt, int maxTokens);

    int vocabularySize() const;
    TokenId bos() const;
    TokenId eos() const;

    // The hyperparameters the GGUF's own header carries, which is the other
    // half of plan.md's "confirm every number against config.json": the
    // conversion read them out of the same config, so a disagreement is a
    // checkpoint that is not the model this was written for.
    //
    // llama.cpp exposes some of them as accessors and the rest only as GGUF
    // metadata, so both are here — the accessors below, and metadata() for a
    // key llama.cpp has no getter for.
    int embeddingWidth() const;
    int layerCount() const;
    int attentionHeads() const;
    int keyValueHeads() const;
    int trainingContext() const;

    // One GGUF metadata value as the string llama.cpp stores it under, empty
    // when the file does not carry that key — a conversion is free to leave
    // out anything that took its default, so a caller checks before it
    // compares. The gemma keys are "gemma.attention.key_length",
    // "gemma.feed_forward_length", "gemma.rope.freq_base" and
    // "gemma.attention.layer_norm_rms_epsilon".
    std::string metadata(const char* key) const;

    // What llama.cpp calls this model, for a printed line.
    std::string description() const;

    int contextSize() const { return contextLength; }

private:
    void release();
    void clearMemory();
    bool decode(Span<const TokenId> tokens, int firstPosition, bool everyRow);
    TokenId argmaxOfLastRow() const;

    llama_model* model = nullptr;
    llama_context* context = nullptr;
    const llama_vocab* vocabulary = nullptr;
    int contextLength = defaultContextSize;
};
} // namespace HF::Testing

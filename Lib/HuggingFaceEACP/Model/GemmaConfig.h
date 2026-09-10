#pragma once

#include <HuggingFaceEACP/Core/Core.h>
#include <HuggingFaceEACP/Model/ModelError.h>
#include <HuggingFaceEACP/Model/ModelFiles.h>

#include <filesystem>
#include <string>
#include <string_view>

namespace HF
{
// generation_config.json: the three token ids the decode loop needs, kept
// apart from the shapes because a checkpoint may ship it and may not, and
// because config.json carries its own copy of the same three.
//
// eos_token_id is a number in gemma-2b and an array in some of the instruction
// tuned repos, so both parse and an array's first entry is the one taken.
struct GenerationConfig
{
    static constexpr auto noToken = -1;

    int beginningOfSequenceToken = noToken;
    int endOfSequenceToken = noToken;
    int padToken = noToken;

    static GenerationConfig fromFile(const std::filesystem::path& path);
    static GenerationConfig fromJson(std::string_view text);
};

// config.json for google/gemma-2b: the shapes every kernel above is built
// against, and the token ids the loop stops on.
//
// Every field has Gemma 2B's own value as its default, so a config that omits
// one still describes the model it came from. The defaults were plan.md's
// remembered numbers and are now the file's: `Model/Checkpoint/config` reads
// the fetched checkpoint and every one of them holds. A real config.json still
// overrides each. What is not defaulted is model_type: a config for some other
// architecture parses into shapes that mean nothing, so it is an error
// instead.
struct GemmaConfig
{
    static constexpr auto noToken = -1;

    int vocabularySize = 256000;
    int hiddenSize = 2048;
    int intermediateSize = 16384;
    int layerCount = 18;
    int attentionHeads = 8;
    int keyValueHeads = 1;
    int headWidth = 256;
    int maxPositions = 8192;

    double rmsNormEpsilon = 1.0e-6;
    double ropeTheta = 10000.0;

    // `hidden_activation` is the current key and `hidden_act` the one older
    // transformers wrote; either is read, and the newer one wins.
    std::string hiddenActivation = "gelu_pytorch_tanh";

    int beginningOfSequenceToken = 2;
    int endOfSequenceToken = 1;
    int padToken = 0;

    bool tieWordEmbeddings = true;

    // 8 query heads over 1 shared KV head for Gemma 2B, which is what makes
    // the attention multi-query rather than merely grouped.
    int queryHeadsPerKeyValueHead() const;

    bool isMultiQuery() const;

    // The width of the concatenated heads, which is what q_proj and o_proj are
    // shaped by and is not hiddenSize in general.
    int queryWidth() const;
    int keyValueWidth() const;

    // The input embedding is scaled by sqrt(hiddenSize) before the first
    // layer, which is Gemma's own normalisation and not an eacp detail.
    float embeddingScale() const;

    static GemmaConfig fromFile(const std::filesystem::path& path);
    static GemmaConfig fromJson(std::string_view text);

    // config.json, plus generation_config.json when the directory has one: the
    // generation file is the authority on the three token ids, since a repo
    // that disagrees with itself disagrees in config.json's copy.
    static GemmaConfig fromDirectory(const std::filesystem::path& directory);
    static GemmaConfig fromModelFiles(const ModelFiles& files);

    void apply(const GenerationConfig& generation);
};
} // namespace HF

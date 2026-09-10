#pragma once

#include <HuggingFaceEACP/Core/Core.h>
#include <HuggingFaceEACP/Model/ModelError.h>

#include <filesystem>
#include <string_view>

namespace HF
{
// What a HuggingFace Gemma repo is called on disk. The weights are either one
// model.safetensors or an index naming shards; a repo never has both.
namespace ModelFileNames
{
inline constexpr auto config = "config.json";
inline constexpr auto generationConfig = "generation_config.json";
inline constexpr auto shardIndex = "model.safetensors.index.json";
inline constexpr auto weights = "model.safetensors";
inline constexpr auto tokenizerJson = "tokenizer.json";
inline constexpr auto tokenizerModel = "tokenizer.model";
} // namespace ModelFileNames

// google/gemma-2b is a gated repo: downloading it needs an account that
// accepted Google's terms and a token, so nothing here fetches anything. This
// is plan.md's fourth gap and its answer — the user downloads the repo once by
// hand and points this variable at the directory.
inline constexpr auto modelDirectoryVariable = "GEMMA_MODEL_DIR";

// A model directory that has been looked at: every path below either names a
// file that exists or is empty, so a caller checks a path rather than the
// filesystem. Locating is the whole job; nothing here opens a byte.
//
// config.json and the weights are required, because a directory without them
// is not a checkpoint. The tokenizer files are recorded and not required: a
// caller loading weights to test a kernel has no use for them, and the
// tokenizer's own error is better than one from here.
struct ModelFiles
{
    std::filesystem::path directory;
    std::filesystem::path config;
    std::filesystem::path generationConfig;
    std::filesystem::path shardIndex;
    Vector<std::filesystem::path> shards;
    std::filesystem::path tokenizerJson;
    std::filesystem::path tokenizerModel;

    bool isSharded() const;
    bool hasGenerationConfig() const;
    bool hasTokenizer() const;

    // Throws a ModelError naming the directory and the file it wanted.
    static ModelFiles fromDirectory(const std::filesystem::path& directory);

    // Empty when GEMMA_MODEL_DIR is unset or set to nothing, which is what a
    // test skipping on the absence of the real model asks about.
    static std::filesystem::path directoryFromEnvironment();
    static bool hasDirectoryInEnvironment();

    // The same checks as fromDirectory, plus one for the variable itself.
    static ModelFiles fromEnvironment();
};
} // namespace HF

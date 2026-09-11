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

// The F32 GGUF, converted out of the safetensors beside it. Nothing in this
// library reads it — it is llama.cpp's format, and llama.cpp is the oracle
// Tests/Oracle checks our logits and our tokens against.
inline constexpr auto ggufModel = "gemma-2b.gguf";
} // namespace ModelFileNames

// The explicit override, and nothing's default. The build fetches gemma-2b
// from an ungated mirror and copies it beside every binary that asks — see
// Model/CMakeLists.txt and Gemma::loadBundled — so a run needs nothing set.
// This variable is how a caller names a checkpoint of its own instead: the
// directory holding gemma-2b.gguf beside the safetensors that Tests/Oracle
// reads, or any other Gemma directory. Set, it wins over what the build
// copied.
inline constexpr auto modelDirectoryVariable = "GEMMA_MODEL_DIR";

// A model directory that has been looked at: every path below either names a
// file that exists or is empty, so a caller checks a path rather than the
// filesystem. Locating is the whole job; nothing here opens a byte.
//
// config.json and the weights are required, because a directory without them
// is not a checkpoint. The tokenizer files and the GGUF are recorded and not
// required: a caller loading weights to test a kernel has no use for them, and
// the tokenizer's own error is better than one from here.
struct ModelFiles
{
    std::filesystem::path directory;
    std::filesystem::path config;
    std::filesystem::path generationConfig;
    std::filesystem::path shardIndex;
    Vector<std::filesystem::path> shards;
    std::filesystem::path tokenizerJson;
    std::filesystem::path tokenizerModel;
    std::filesystem::path ggufModel;

    bool isSharded() const;
    bool hasGenerationConfig() const;
    bool hasTokenizer() const;
    bool hasGgufModel() const;

    // Throws a ModelError naming the directory and the file it wanted.
    static ModelFiles fromDirectory(const std::filesystem::path& directory);

    // Empty when GEMMA_MODEL_DIR is unset or set to nothing, which is a run
    // taking the model the build assembled rather than one of its own.
    static std::filesystem::path directoryFromEnvironment();
};
} // namespace HF

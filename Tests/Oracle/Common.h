#pragma once

// What every test in this directory skips on, and the two things they all
// load: llama.cpp over gemma-2b.gguf on one side, our own modules over the
// files beside it on the other.
//
// The GGUF is a 10 GB manual download from a gated repo, so on a machine
// without one every test here returns early — the same shape as a GPU test
// returning early when Device::shared().isValid() is false. The skip says so
// once rather than per test, because a silent pass and a real pass look alike.

#include "LlamaOracle.h"

#include <HuggingFaceEACP/Model/ModelFiles.h>
#include <HuggingFaceEACP/Tokenizer/Tokenizer.h>

#include <NanoTest/NanoTest.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace HF::Testing
{
// Spelled out of the directory rather than through ModelFiles::fromEnvironment
// because that one requires a whole checkpoint: it throws on a directory with
// no config.json, and the tokenizer comparison below has no use for the
// weights. The names still come from ModelFileNames, so there is one spelling
// of each file in the project.
inline std::filesystem::path modelFile(std::string_view name)
{
    const auto directory = ModelFiles::directoryFromEnvironment();
    return directory.empty() ? std::filesystem::path {} : directory / name;
}

inline bool hasModelFile(std::string_view name)
{
    const auto path = modelFile(name);

    if (path.empty())
        return false;

    auto error = std::error_code {};
    return std::filesystem::is_regular_file(path, error);
}

inline void announceSkip()
{
    static const auto announced = []
    {
        std::cout << "  no " << ModelFileNames::ggufModel << " under "
                  << modelDirectoryVariable << ": the llama.cpp oracle tests skip\n";

        return true;
    }();

    (void) announced;
}

inline bool hasOracleModel()
{
    if (LlamaOracle::isAvailable())
        return true;

    announceSkip();
    return false;
}

inline bool hasOurTokenizer()
{
    return hasModelFile(ModelFileNames::tokenizerJson);
}

// 17.5 MB of JSON through Miro, which is seconds in Debug, so it is read once
// for the whole executable rather than once per test.
inline const Tokenizer& ourTokenizer()
{
    static const auto tokenizer =
        Tokenizer::fromFile(modelFile(ModelFileNames::tokenizerJson));

    return tokenizer;
}

// One oracle for the whole executable. gemma-2b.gguf is 10 GB of F32 off
// disk, so a load per test would dominate the run — and every test here reads
// the reference without changing it, since logits() and greedy() both start by
// emptying the KV cache.
inline LlamaOracle& sharedOracle()
{
    static auto oracle = LlamaOracle::load();
    return oracle;
}

inline std::string spelled(Span<const TokenId> tokens)
{
    auto text = std::string {};

    for (const auto token: tokens)
        text += (text.empty() ? "" : ", ") + std::to_string(token);

    return "[" + text + "]";
}
} // namespace HF::Testing

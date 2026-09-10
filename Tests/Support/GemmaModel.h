#pragma once

#include <HuggingFaceEACP/Model/ModelFiles.h>

#include <filesystem>
#include <system_error>

// Which gemma-2b a test runs against, resolved the way every binary here does:
// the GEMMA_MODEL_DIR override when it is set, and otherwise the directory the
// build assembled out of the fetched files, which reaches every test target as
// HF_EACP_GEMMA_MODEL_DIR from Tests/CMakeLists.txt.
//
// A test still asks hasGemmaModel() before using it rather than assuming the
// path is populated: the fetch is an option, and a build configured with
// -DHF_EACP_FETCH_MODEL=OFF has the path without the files.
namespace HF::Testing
{
inline std::filesystem::path gemmaModelDirectory()
{
    const auto named = ModelFiles::directoryFromEnvironment();

    return named.empty() ? std::filesystem::path {HF_EACP_GEMMA_MODEL_DIR} : named;
}

inline bool hasGemmaModel()
{
    const auto directory = gemmaModelDirectory();

    if (directory.empty())
        return false;

    const auto isFile = [](const std::filesystem::path& path)
    {
        auto error = std::error_code {};
        return std::filesystem::is_regular_file(path, error);
    };

    const auto hasWeights = isFile(directory / ModelFileNames::weights)
                            || isFile(directory / ModelFileNames::shardIndex);

    return isFile(directory / ModelFileNames::config) && hasWeights;
}

inline ModelFiles gemmaModelFiles()
{
    return ModelFiles::fromDirectory(gemmaModelDirectory());
}
} // namespace HF::Testing

#include "ModelFiles.h"

#include "ShardIndex.h"

#include <cstdlib>

namespace HF
{
namespace
{
bool isFile(const std::filesystem::path& path)
{
    auto error = std::error_code {};
    return std::filesystem::is_regular_file(path, error);
}

std::filesystem::path fileIfPresent(const std::filesystem::path& directory,
                                    std::string_view name)
{
    const auto path = directory / name;
    return isFile(path) ? path : std::filesystem::path {};
}

std::filesystem::path requireFile(const std::filesystem::path& directory,
                                  std::string_view name)
{
    const auto path = directory / name;

    if (!isFile(path))
        throw ModelError {"the model directory '" + directory.string() + "' has no "
                          + std::string {name}};

    return path;
}

Vector<std::filesystem::path> shardsFromIndex(const std::filesystem::path& directory,
                                              const std::filesystem::path& index)
{
    // Named, because shardFileNames() hands back a reference into it and a
    // range-for over a temporary's member does not extend that temporary's
    // life until C++23.
    const auto parsed = ShardIndex::fromFile(index);

    auto shards = Vector<std::filesystem::path> {};

    for (const auto& fileName: parsed.shardFileNames())
        shards.add(requireFile(directory, fileName));

    return shards;
}
} // namespace

bool ModelFiles::isSharded() const
{
    return !shardIndex.empty();
}

bool ModelFiles::hasGenerationConfig() const
{
    return !generationConfig.empty();
}

bool ModelFiles::hasTokenizer() const
{
    return !tokenizerJson.empty() || !tokenizerModel.empty();
}

bool ModelFiles::hasGgufModel() const
{
    return !ggufModel.empty();
}

ModelFiles ModelFiles::fromDirectory(const std::filesystem::path& directory)
{
    auto error = std::error_code {};

    if (!std::filesystem::is_directory(directory, error))
        throw ModelError {"'" + directory.string() + "' is not a directory"};

    auto files = ModelFiles {};
    files.directory = directory;
    files.config = requireFile(directory, ModelFileNames::config);
    files.generationConfig =
        fileIfPresent(directory, ModelFileNames::generationConfig);
    files.shardIndex = fileIfPresent(directory, ModelFileNames::shardIndex);
    files.tokenizerJson = fileIfPresent(directory, ModelFileNames::tokenizerJson);
    files.tokenizerModel = fileIfPresent(directory, ModelFileNames::tokenizerModel);
    files.ggufModel = fileIfPresent(directory, ModelFileNames::ggufModel);

    if (files.isSharded())
        files.shards = shardsFromIndex(directory, files.shardIndex);
    else
        files.shards.add(requireFile(directory, ModelFileNames::weights));

    return files;
}

std::filesystem::path ModelFiles::directoryFromEnvironment()
{
    const auto* value = std::getenv(modelDirectoryVariable);

    if (value == nullptr || *value == '\0')
        return {};

    return {value};
}
} // namespace HF

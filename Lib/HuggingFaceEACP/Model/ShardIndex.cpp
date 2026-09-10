#include "ShardIndex.h"

#include "ModelIO.h"

namespace HF
{
namespace
{
constexpr auto indexName = "model.safetensors.index.json";

// A shard file name reaches the filesystem as a sibling of the index, so a
// path separator or a parent step in it would read a file the directory does
// not hold. The format's own names are "model-00001-of-00002.safetensors".
void requirePlainFileName(const std::string& fileName, const std::string& tensor)
{
    const auto isPlain = !fileName.empty() && fileName.find('/') == std::string::npos
                         && fileName.find('\\') == std::string::npos
                         && fileName != "." && fileName != "..";

    if (!isPlain)
        throw ModelError {std::string {indexName} + " maps '" + tensor + "' to '"
                          + fileName + "', which is not a plain file name"};
}
} // namespace

ShardIndex ShardIndex::fromFile(const std::filesystem::path& path)
{
    const auto bytes = ModelIO::readFileBytes(path);
    return fromJson(ModelIO::textOf(bytes));
}

ShardIndex ShardIndex::fromJson(std::string_view text)
{
    const auto parsed = ModelIO::parseObject(text, indexName);
    const auto& object = parsed.asObject();

    auto index = ShardIndex {};

    const auto& map = ModelIO::field(object, "weight_map", indexName);

    if (!map.isObject())
        throw ModelError {std::string {indexName} + " weight_map is not an object"};

    for (const auto& [tensor, shard]: map.asObject())
    {
        if (!shard.isString())
            throw ModelError {std::string {indexName} + " maps '" + tensor
                              + "' to something that is not a file name"};

        const auto& fileName = shard.asString();
        requirePlainFileName(fileName, tensor);

        index.weightMap.emplace(tensor, fileName);

        if (!index.shardNames.contains(fileName))
            index.shardNames.add(fileName);
    }

    if (index.shardNames.empty())
        throw ModelError {std::string {indexName} + " names no shards"};

    if (const auto* metadata = Miro::Json::find(object, "metadata"))
    {
        if (!metadata->isObject())
            throw ModelError {std::string {indexName}
                              + " metadata is not an object"};

        // Read as an int64 rather than through intFieldOr: gemma-2b's is
        // 5,012,344,832, which is four times what an int describes.
        if (const auto* size = Miro::Json::find(metadata->asObject(), "total_size"))
            index.totalByteCount = ModelIO::asInteger(*size, "metadata total_size");
    }

    return index;
}

const std::string* ShardIndex::findShard(std::string_view name) const
{
    const auto found = weightMap.find(name);
    return found == weightMap.end() ? nullptr : &found->second;
}

const std::string& ShardIndex::shardFor(std::string_view name) const
{
    if (const auto* found = findShard(name))
        return *found;

    throw ModelError {std::string {indexName} + " has no tensor named '"
                      + std::string {name} + "'"};
}

bool ShardIndex::contains(std::string_view name) const
{
    return findShard(name) != nullptr;
}

Vector<std::string> ShardIndex::names() const
{
    auto found = Vector<std::string> {};
    found.reserve(static_cast<int>(weightMap.size()));

    for (const auto& [tensor, shard]: weightMap)
        found.add(tensor);

    return found;
}

int ShardIndex::tensorCount() const
{
    return static_cast<int>(weightMap.size());
}

const Vector<std::string>& ShardIndex::shardFileNames() const
{
    return shardNames;
}

std::int64_t ShardIndex::totalSize() const
{
    return totalByteCount;
}
} // namespace HF

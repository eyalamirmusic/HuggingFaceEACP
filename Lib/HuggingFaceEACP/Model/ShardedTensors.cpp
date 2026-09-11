#include "ShardedTensors.h"

namespace HF
{
ShardedTensors ShardedTensors::fromFile(const std::filesystem::path& weights)
{
    auto loaded = ShardedTensors {};
    loaded.directory = weights.parent_path();
    loaded.singleShardName = weights.filename().string();

    return loaded;
}

ShardedTensors ShardedTensors::fromModelFiles(const ModelFiles& files)
{
    if (!files.isSharded())
        return fromFile(files.shards.front());

    auto loaded = ShardedTensors {};
    loaded.directory = files.directory;
    loaded.index = ShardIndex::fromFile(files.shardIndex);

    return loaded;
}

ShardedTensors ShardedTensors::fromDirectory(const std::filesystem::path& directory)
{
    return fromModelFiles(ModelFiles::fromDirectory(directory));
}

const SafeTensors& ShardedTensors::mapShard(const std::string& fileName) const
{
    if (const auto found = shards.find(fileName); found != shards.end())
        return found->second;

    auto mapped = SafeTensors::fromFile(directory / fileName);

    return shards.emplace(fileName, std::move(mapped)).first->second;
}

const SafeTensors& ShardedTensors::shardFor(std::string_view name) const
{
    if (!index)
        return mapShard(singleShardName);

    return mapShard(index->shardFor(name));
}

Vector<std::string> ShardedTensors::names() const
{
    if (index)
        return index->names();

    return mapShard(singleShardName).names();
}

int ShardedTensors::tensorCount() const
{
    if (index)
        return index->tensorCount();

    return mapShard(singleShardName).tensors().size();
}

bool ShardedTensors::contains(std::string_view name) const
{
    return find(name) != nullptr;
}

const TensorInfo* ShardedTensors::find(std::string_view name) const
{
    if (index && !index->contains(name))
        return nullptr;

    return shardFor(name).find(name);
}

const TensorInfo& ShardedTensors::info(std::string_view name) const
{
    return shardFor(name).info(name);
}

Span<const std::uint8_t> ShardedTensors::rawBytes(std::string_view name) const
{
    return shardFor(name).rawBytes(name);
}

Vector<float> ShardedTensors::readFloats(std::string_view name) const
{
    return shardFor(name).readFloats(name);
}

void ShardedTensors::readFloats(std::string_view name, Span<float> destination) const
{
    shardFor(name).readFloats(name, destination);
}

TensorBuffer ShardedTensors::makeBuffer(std::string_view name) const
{
    return shardFor(name).makeBuffer(name);
}

TensorBuffer ShardedTensors::makeWidenedBuffer(std::string_view name) const
{
    return shardFor(name).makeWidenedBuffer(name);
}

const SafeTensors& ShardedTensors::shardHolding(std::string_view name) const
{
    return shardFor(name);
}

std::string ShardedTensors::shardNameFor(std::string_view name) const
{
    if (!index)
        return {};

    return index->shardFor(name);
}

int ShardedTensors::shardCount() const
{
    return index ? index->shardFileNames().size() : 1;
}

int ShardedTensors::mappedShardCount() const
{
    return static_cast<int>(shards.size());
}
} // namespace HF

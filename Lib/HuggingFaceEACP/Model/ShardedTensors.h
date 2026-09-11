#pragma once

#include <HuggingFaceEACP/Model/ModelFiles.h>
#include <HuggingFaceEACP/Model/SafeTensors.h>
#include <HuggingFaceEACP/Model/ShardIndex.h>

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace HF
{
// One checkpoint, however many files it came in: the same lookup surface as
// SafeTensors, over a directory rather than over a file. A name resolves
// through the index to a shard, and that shard is mapped the first time
// something in it is asked for and never again.
//
// Lazily, because that is what makes the two-shard shape cheap: reading the
// 67 MB shard costs nothing on the 4.95 GB one, and a checkpoint whose tensors
// are all in one shard never touches the other. Mapping is not free even
// though it copies nothing — it is a syscall and an address-space reservation
// per shard — and a load that walks every layer maps each exactly once.
//
// A directory with a single model.safetensors and no index works the same way,
// with one unnamed shard and every lookup going to it.
class ShardedTensors
{
public:
    static ShardedTensors fromDirectory(const std::filesystem::path& directory);
    static ShardedTensors fromModelFiles(const ModelFiles& files);

    // One file, for a checkpoint that never had an index and for a test that
    // writes its own.
    static ShardedTensors fromFile(const std::filesystem::path& weights);

    // Sorted. From the index when there is one, which is what lets a caller
    // enumerate a checkpoint without mapping a byte of it.
    Vector<std::string> names() const;
    int tensorCount() const;

    bool contains(std::string_view name) const;
    const TensorInfo* find(std::string_view name) const;
    const TensorInfo& info(std::string_view name) const;

    Span<const std::uint8_t> rawBytes(std::string_view name) const;
    Vector<float> readFloats(std::string_view name) const;
    void readFloats(std::string_view name, Span<float> destination) const;
    TensorBuffer makeBuffer(std::string_view name) const;
    TensorBuffer makeWidenedBuffer(std::string_view name) const;

    // The shard the tensor lives in, mapped if it was not already.
    const SafeTensors& shardHolding(std::string_view name) const;

    // What the index says the tensor's file is called, empty for the
    // single-file case that has no index to say it.
    std::string shardNameFor(std::string_view name) const;

    int shardCount() const;
    int mappedShardCount() const;

private:
    const SafeTensors& mapShard(const std::string& fileName) const;
    const SafeTensors& shardFor(std::string_view name) const;

    std::filesystem::path directory;
    std::optional<ShardIndex> index;

    // The one file's name when there is no index, so both cases resolve to a
    // name the map below is keyed by.
    std::string singleShardName;

    // Mutable because mapping is what "lazily" means, and const lookup is what
    // every caller wants. std::map, so a reference handed out stays valid when
    // the next shard is mapped.
    mutable std::map<std::string, SafeTensors, std::less<>> shards;
};
} // namespace HF

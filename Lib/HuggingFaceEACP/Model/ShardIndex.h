#pragma once

#include <HuggingFaceEACP/Core/Core.h>
#include <HuggingFaceEACP/Model/ModelError.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>

namespace HF
{
// model.safetensors.index.json: the `weight_map` that says which shard file
// each tensor lives in, and a `metadata` block whose `total_size` is the whole
// checkpoint in bytes.
//
// A repo small enough to fit one file ships no index at all, which is why
// ShardedTensors treats this as optional rather than as the way in.
class ShardIndex
{
public:
    static ShardIndex fromFile(const std::filesystem::path& path);
    static ShardIndex fromJson(std::string_view text);

    // Null when the index does not list that tensor, which is the one thing a
    // caller can do about a name the checkpoint has never heard of.
    const std::string* findShard(std::string_view name) const;
    const std::string& shardFor(std::string_view name) const;

    bool contains(std::string_view name) const;

    // Sorted, since the weight map is read out of an ordered map.
    Vector<std::string> names() const;
    int tensorCount() const;

    // Each shard once, in the order the weight map first names it.
    const Vector<std::string>& shardFileNames() const;

    // Zero when the index carries no metadata, so a caller reporting it says
    // "unknown" rather than "empty".
    std::int64_t totalSize() const;

private:
    std::map<std::string, std::string, std::less<>> weightMap;
    Vector<std::string> shardNames;
    std::int64_t totalByteCount = 0;
};
} // namespace HF

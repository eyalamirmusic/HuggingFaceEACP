#include "Common.h"

using namespace nano;
using namespace HF;
using namespace HF::Testing;

namespace
{
constexpr auto firstShardName = "model-00001-of-00002.safetensors";
constexpr auto secondShardName = "model-00002-of-00002.safetensors";

// Two tensors in the first shard and one in the second, so a lookup that went
// to the wrong file would find either nothing or the wrong values.
Vector<std::uint8_t> firstShard()
{
    return assemble(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
                    R"("b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}})",
                    toBytes<float>({1.0f, 2.0f, 3.0f, 4.0f}));
}

Vector<std::uint8_t> secondShard()
{
    return assemble(R"({"c":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
                    toBytes<float>({5.0f, 6.0f}));
}

std::string indexJson()
{
    return std::string {R"({"metadata":{"total_size":24},"weight_map":{)"}
           + R"("a":")" + firstShardName + R"(",)" + R"("b":")" + firstShardName
           + R"(",)" + R"("c":")" + secondShardName + R"("}})";
}

const auto gemmaConfigJson = std::string {R"({"model_type":"gemma"})"};

// A whole model directory: config.json, the index, and the two shards it
// names. Written rather than fetched, which is the only way this suite runs
// on a machine with no copy of a gated 5 GB repo.
class ShardedModel
{
public:
    explicit ShardedModel(std::string_view name)
        : scratch(name)
    {
        scratch.writeText(ModelFileNames::config, gemmaConfigJson);
        scratch.writeText(ModelFileNames::shardIndex, indexJson());
        scratch.write(firstShardName, firstShard());
        scratch.write(secondShardName, secondShard());
    }

    const std::filesystem::path& path() const { return scratch.path(); }

private:
    ScratchDirectory scratch;
};
} // namespace

auto tIndexParses = test("Model/Shards/indexParses") = []
{
    const auto index = ShardIndex::fromJson(indexJson());

    check(index.tensorCount() == 3);
    check(index.totalSize() == 24);
    check(index.shardFileNames().size() == 2);
    check(index.shardFor("a") == firstShardName);
    check(index.shardFor("b") == firstShardName);
    check(index.shardFor("c") == secondShardName);
    check(index.findShard("d") == nullptr);
    check(throwsModelError([&] { return index.shardFor("d"); }));
};

auto tIndexRejectsMalformed = test("Model/Shards/indexRejectsMalformed") = []
{
    check(throwsModelError([] { return ShardIndex::fromJson("{}"); }));
    check(throwsModelError(
        [] { return ShardIndex::fromJson(R"({"weight_map":[]})"); }));
    check(throwsModelError(
        [] { return ShardIndex::fromJson(R"({"weight_map":{}})"); }));
    check(throwsModelError(
        [] { return ShardIndex::fromJson(R"({"weight_map":{"a":7}})"); }));

    // A shard name is joined to the model directory, so anything with a path
    // in it would read a file the directory does not hold.
    check(throwsModelError(
        [] { return ShardIndex::fromJson(R"({"weight_map":{"a":"../x"}})"); }));
};

auto tLookupLandsInTheRightShard = test("Model/Shards/lookupLandsInRightShard") = []
{
    const auto model = ShardedModel {"shards-lookup"};
    const auto weights = ShardedTensors::fromDirectory(model.path());

    check(weights.shardCount() == 2);
    check(weights.tensorCount() == 3);
    check(weights.shardNameFor("a") == firstShardName);
    check(weights.shardNameFor("c") == secondShardName);

    check(nearlyEqual(weights.readFloats("a")[0], 1.0f));
    check(nearlyEqual(weights.readFloats("b")[0], 3.0f));
    check(nearlyEqual(weights.readFloats("c")[0], 5.0f));

    check(weights.info("c").dimension(0) == 2);

    const auto names = weights.names();
    check(names.size() == 3);
    check(names[0] == "a");
    check(names[2] == "c");
};

// The index is what makes a name resolvable at all, so a name it does not
// carry is an error rather than a walk over every shard looking for it.
auto tUnknownNameErrors = test("Model/Shards/unknownNameErrors") = []
{
    const auto model = ShardedModel {"shards-unknown"};
    const auto weights = ShardedTensors::fromDirectory(model.path());

    check(!weights.contains("d"));
    check(weights.find("d") == nullptr);
    check(throwsModelError([&] { return weights.info("d"); }));
    check(throwsModelError([&] { return weights.readFloats("d"); }));
};

// Mapping is a syscall and an address-space reservation per shard, and a load
// that walks every layer would otherwise pay it once per tensor.
auto tShardsAreMappedOnceEach = test("Model/Shards/mappedOnceEach") = []
{
    const auto model = ShardedModel {"shards-lazy"};
    const auto weights = ShardedTensors::fromDirectory(model.path());

    check(weights.mappedShardCount() == 0);
    check(weights.names().size() == 3);
    check(weights.mappedShardCount() == 0);

    weights.readFloats("a");
    check(weights.mappedShardCount() == 1);

    weights.readFloats("b");
    check(weights.mappedShardCount() == 1);

    weights.readFloats("c");
    check(weights.mappedShardCount() == 2);

    weights.readFloats("a");
    check(weights.mappedShardCount() == 2);
};

// A repo small enough to fit one file ships no index, and has to load the same
// way as one that does.
auto tSingleFileNeedsNoIndex = test("Model/Shards/singleFileNeedsNoIndex") = []
{
    const auto scratch = ScratchDirectory {"shards-single"};
    scratch.writeText(ModelFileNames::config, gemmaConfigJson);
    scratch.write(ModelFileNames::weights, firstShard());

    const auto weights = ShardedTensors::fromDirectory(scratch.path());

    check(weights.shardCount() == 1);
    check(weights.tensorCount() == 2);
    check(weights.shardNameFor("a").empty());
    check(nearlyEqual(weights.readFloats("b")[1], 4.0f));
    check(!weights.contains("c"));
    check(throwsModelError([&] { return weights.info("c"); }));
};

auto tModelFilesFindsTheShards = test("Model/Files/findsTheShards") = []
{
    const auto model = ShardedModel {"files-sharded"};
    const auto files = ModelFiles::fromDirectory(model.path());

    check(files.isSharded());
    check(files.shards.size() == 2);
    check(files.shards[0].filename().string() == firstShardName);
    check(files.shards[1].filename().string() == secondShardName);
    check(files.config.filename().string() == ModelFileNames::config);
    check(!files.hasGenerationConfig());
    check(!files.hasTokenizer());
};

auto tModelFilesNamesWhatIsMissing = test("Model/Files/namesWhatIsMissing") = []
{
    check(throwsModelError(
        [] { return ModelFiles::fromDirectory("no/such/directory"); }));

    const auto empty = ScratchDirectory {"files-empty"};
    check(throwsModelError([&] { return ModelFiles::fromDirectory(empty.path()); }));

    // config.json and no weights beside it.
    const auto configOnly = ScratchDirectory {"files-config-only"};
    configOnly.writeText(ModelFileNames::config, gemmaConfigJson);
    check(throwsModelError(
        [&] { return ModelFiles::fromDirectory(configOnly.path()); }));

    // An index naming a shard the directory does not have, which is the one
    // failure a directory listing alone would not catch.
    const auto shardless = ScratchDirectory {"files-shardless"};
    shardless.writeText(ModelFileNames::config, gemmaConfigJson);
    shardless.writeText(ModelFileNames::shardIndex, indexJson());
    shardless.write(firstShardName, firstShard());
    check(throwsModelError([&]
                           { return ModelFiles::fromDirectory(shardless.path()); }));
};

auto tModelFilesReadsTheEnvironment = test("Model/Files/readsTheEnvironment") = []
{
    // GEMMA_MODEL_DIR is unset on a machine with no manual download, and that
    // is a clear error rather than a search of somewhere plausible.
    if (!ModelFiles::hasDirectoryInEnvironment())
    {
        check(ModelFiles::directoryFromEnvironment().empty());
        check(throwsModelError([] { return ModelFiles::fromEnvironment(); }));
        return;
    }

    check(!ModelFiles::directoryFromEnvironment().empty());
};

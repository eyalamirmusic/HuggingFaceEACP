#include "Common.h"

// The real google/gemma-2b files, which are a manual download and never a
// commit: the repo is gated, so nothing in this build fetches them. Point
// GEMMA_MODEL_DIR at a directory that has them and every test here runs;
// leave it unset, as it is on this machine, and every one returns early — the
// same shape as a GPU test returning early on an invalid device.
//
// **plan.md's Gemma numbers are the released config as remembered rather than
// read, and this suite is where that gets settled.** The shapes below are
// taken from the checkpoint's own config.json, not from plan.md's table, so a
// run against a real download tells us whether the two agree: a mismatch here
// is a correction to plan.md and to GemmaConfig's defaults, not a bug in the
// loader.

using namespace nano;
using namespace HF;
using namespace HF::Testing;

namespace
{
bool hasRealCheckpoint()
{
    return ModelFiles::hasDirectoryInEnvironment();
}
} // namespace

auto tCheckpointFilesAreThere = test("Model/Checkpoint/filesAreThere") = []
{
    if (!hasRealCheckpoint())
        return;

    const auto files = ModelFiles::fromEnvironment();

    check(!files.config.empty());
    check(!files.shards.empty());
    check(files.hasTokenizer());

    // gemma-2b ships two shards and an index; a repo that has been merged into
    // one file is legal too, which is why this is not an equality.
    check(files.shards.size() >= 1);
};

auto tCheckpointConfig = test("Model/Checkpoint/config") = []
{
    if (!hasRealCheckpoint())
        return;

    const auto config = GemmaConfig::fromModelFiles(ModelFiles::fromEnvironment());

    check(config.vocabularySize > 0);
    check(config.hiddenSize > 0);
    check(config.layerCount > 0);
    check(config.attentionHeads > 0);
    check(config.keyValueHeads > 0);
    check(config.headWidth > 0);
    check(config.attentionHeads % config.keyValueHeads == 0);
    check(config.rmsNormEpsilon > 0.0);
    check(config.ropeTheta > 0.0);

    // What plan.md says gemma-2b is. A failure here is plan.md being wrong
    // about the model rather than this module being wrong about the file.
    check(config.vocabularySize == 256000);
    check(config.hiddenSize == 2048);
    check(config.intermediateSize == 16384);
    check(config.layerCount == 18);
    check(config.attentionHeads == 8);
    check(config.keyValueHeads == 1);
    check(config.headWidth == 256);
    check(config.maxPositions == 8192);
    check(config.beginningOfSequenceToken == 2);
    check(config.endOfSequenceToken == 1);
    check(config.padToken == 0);
    check(config.tieWordEmbeddings);
};

// The whole point of the catalogue: every tensor the decoder will ask for is
// in the checkpoint, with the shape the config implies, before any of it is
// uploaded.
auto tCheckpointMatchesTheCatalogue =
    test("Model/Checkpoint/matchesTheCatalogue") = []
{
    if (!hasRealCheckpoint())
        return;

    const auto files = ModelFiles::fromEnvironment();
    const auto config = GemmaConfig::fromModelFiles(files);
    const auto weights = ShardedTensors::fromModelFiles(files);

    GemmaTensors::checkAgainst(weights, config);

    // Every shard the index names carries something, so a checkpoint whose
    // second file was never read would show up as a shard mapped zero times.
    check(weights.mappedShardCount() == weights.shardCount());
};

// Gemma ships BF16, which has no shader read yet — plan.md's first eacp gap.
// The loader widens it on the CPU, and this is where a real checkpoint says
// what it actually holds.
auto tCheckpointStorageIsBFloat = test("Model/Checkpoint/storageIsBFloat") = []
{
    if (!hasRealCheckpoint())
        return;

    const auto files = ModelFiles::fromEnvironment();
    const auto config = GemmaConfig::fromModelFiles(files);
    const auto weights = ShardedTensors::fromModelFiles(files);

    const auto& embedding = weights.info(GemmaTensors::embedding);

    check(isFloatingPoint(embedding.type));
    check(embedding.elementCount()
          == static_cast<std::int64_t>(config.vocabularySize) * config.hiddenSize);

    // A widened embedding is 2.10 GB against the 2.15 GB an int describes,
    // which is plan.md's second gap and the reason this is worth asserting.
    const auto widenedBytes =
        embedding.elementCount() * static_cast<std::int64_t>(sizeof(float));

    check(widenedBytes < static_cast<std::int64_t>(2) * 1024 * 1024 * 1024);
};

// One tensor read all the way through, so the mapping, the offsets and the
// widening are exercised against a real 5 GB file rather than a written one.
auto tCheckpointReadsATensor = test("Model/Checkpoint/readsATensor") = []
{
    if (!hasRealCheckpoint())
        return;

    const auto files = ModelFiles::fromEnvironment();
    const auto config = GemmaConfig::fromModelFiles(files);
    const auto weights = ShardedTensors::fromModelFiles(files);

    const auto norm = weights.readFloats(GemmaTensors::finalNorm);
    check(norm.size() == config.hiddenSize);

    // RMSNorm's scale is 1 + w, so the stored weights sit around zero and a
    // read that landed on the wrong bytes would not.
    for (auto value: norm)
        check(value > -4.0f && value < 4.0f);
};

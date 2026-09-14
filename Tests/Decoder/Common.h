#pragma once

#include <HuggingFaceEACP/Decoder/Decoder.h>

// The two suites this one is built on top of, by path rather than by a second
// copy of what they hold: Tests/Model/Common.h is where ScratchDirectory and
// the safetensors byte assembly live, and Tests/Kernels/Common.h is where the
// spread of inputs, the readback plumbing and the double-precision tanh GELU
// do. Neither module is a dependency of hf-decoder; these are its test suites'
// headers, included the way WhisperEACP's Tests/Decoder/Common.h includes the
// encoder's.
#include "../Kernels/Common.h"
#include "../Kernels/TiledProduct.h"
#include "../Model/Common.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace HF::Testing
{
// Small enough that a scalar reference in doubles runs the whole forward pass
// instantly, and lopsided enough that a confused stride shows up: the query
// width happens to be the model width, as it is in Gemma 2B, but the KV width
// is not, no extent is a multiple of the 8 x 8 dispatch group, and the head
// width is neither.
//
// The head width is 16 because the attention reads a head four channels at a
// time and the rotation pairs its halves — a shape a real model has and a
// shape the kernels refuse are not the same thing, and the refusals are tested
// through DecoderShape::validate rather than by feeding one here.
inline GemmaConfig smallGemmaConfig()
{
    auto config = GemmaConfig {};
    config.vocabularySize = 64;
    config.hiddenSize = 32;
    config.intermediateSize = 48;
    config.layerCount = 2;
    config.attentionHeads = 2;
    config.keyValueHeads = 1;
    config.headWidth = 16;
    config.maxPositions = 16;
    config.rmsNormEpsilon = 1.0e-6;
    config.ropeTheta = 10000.0;

    return config;
}

inline DecoderShape smallDecoderShape()
{
    return DecoderShape::fromConfig(smallGemmaConfig());
}

// The same small shape with a feed-forward width the int8 blocks divide. 48 is
// not a whole number of thirty-twos, so down_proj's [width, intermediate] rows
// are exactly what TensorLoader refuses to quantize — which every real Gemma
// tensor passes and this one does not, and is why the quantized tiers run at 64
// instead. Nothing else about the shape moves, so the two are the same model at
// two feed-forward widths rather than two models.
inline GemmaConfig smallQuantizableGemmaConfig()
{
    auto config = smallGemmaConfig();
    config.intermediateSize = 64;

    return config;
}

// ---------------------------------------------------------------------------
// A checkpoint written to disk, so the shard index, the config reader and the
// safetensors header are all under test alongside the arithmetic.
// ---------------------------------------------------------------------------

// One tensor before it is written: the name the checkpoint gives it, the shape
// its header will claim, the floats its blob will hold, and what the blob
// stores them as. F32 by default and BF16 for the checkpoints written the way
// google/gemma-2b itself ships — see asBFloat16 below.
struct SyntheticTensor
{
    std::string name;
    Vector<int> shape;
    Vector<float> values;
    TensorType type = TensorType::F32;
};

// A safetensors file assembled in memory — eight bytes of header length, the
// JSON naming each tensor's dtype, shape and byte range, then the blob those
// ranges are relative to.
class TensorFileWriter
{
public:
    void add(const SyntheticTensor& tensor)
    {
        const auto begin = (std::uint64_t) blob.size();
        appendValues(tensor);

        if (!header.empty())
            header += ",";

        header += entryText(tensor, begin, (std::uint64_t) blob.size());
    }

    Vector<std::uint8_t> bytes() const { return assemble("{" + header + "}", blob); }

private:
    static std::string shapeText(const Vector<int>& shape)
    {
        auto text = std::string {"["};

        for (auto axis = 0; axis < shape.size(); ++axis)
            text += (axis == 0 ? "" : ",") + std::to_string(shape[axis]);

        return text + "]";
    }

    static std::string entryText(const SyntheticTensor& tensor,
                                 std::uint64_t begin,
                                 std::uint64_t end)
    {
        return "\"" + tensor.name + "\":{\"dtype\":\""
               + std::string {tensorTypeName(tensor.type)}
               + "\",\"shape\":" + shapeText(tensor.shape) + ",\"data_offsets\":["
               + std::to_string(begin) + "," + std::to_string(end) + "]}";
    }

    void appendValues(const SyntheticTensor& tensor)
    {
        if (tensor.type != TensorType::BF16)
        {
            appendFloats(tensor.values);
            return;
        }

        // Little-endian, which is what the reader asserts the format is, and
        // through eacp's own converter so the rounding is the one a shader
        // reading this back would widen.
        for (auto value: tensor.values)
        {
            const auto bits = eacp::GPU::bfloat16FromFloat(value);

            blob.add((std::uint8_t) (bits & 0xFFu));
            blob.add((std::uint8_t) (bits >> 8));
        }
    }

    void appendFloats(const Vector<float>& values)
    {
        const auto at = blob.size();
        blob.resize(at + (int) sizeof(float) * values.size());
        std::memcpy(blob.data() + at,
                    values.data(),
                    sizeof(float) * (std::size_t) values.size());
    }

    std::string header;
    Vector<std::uint8_t> blob;
};

// 1 / sqrt(fan in), which is what keeps a product of a normalised row and one
// of these O(1): a feed-forward whose activations came out at thirty would be
// testing the GELU's saturation clamp rather than the decoder.
inline float fanInRange(int innerCount)
{
    return 1.f / std::sqrt((float) innerCount);
}

// Every tensor google/gemma-2b ships, at the small config's shapes and filled
// from mt19937 — a generator whose sequence the standard specifies rather than
// an implementation, so the same numbers reach the GPU and the reference on
// every machine.
//
// The norm scales are small and centred on zero because that is where a real
// Gemma's are: GemmaRMSNorm applies `1 + w`, so the checkpoint stores the
// offset rather than the scale.
inline Vector<SyntheticTensor> syntheticGemmaTensors(const GemmaConfig& config)
{
    const auto width = config.hiddenSize;
    const auto queries = config.queryWidth();
    const auto keyValues = config.keyValueWidth();
    const auto intermediate = config.intermediateSize;

    auto tensors = Vector<SyntheticTensor> {};
    auto seed = 7100u;

    auto add = [&](std::string name, Vector<int> shape, float range)
    {
        auto count = 1;

        for (auto axis = 0; axis < shape.size(); ++axis)
            count *= shape[axis];

        tensors.add(
            {std::move(name), std::move(shape), spreadValues(count, seed, range)});
        ++seed;
    };

    add(GemmaTensors::embedding, {config.vocabularySize, width}, 0.2f);

    for (auto layer = 0; layer < config.layerCount; ++layer)
    {
        auto named = [&](std::string_view suffix)
        { return GemmaTensors::layerTensorName(layer, suffix); };

        add(named(GemmaTensors::inputNorm), {width}, 0.1f);
        add(named(GemmaTensors::queryProjection),
            {queries, width},
            fanInRange(width));
        add(named(GemmaTensors::keyProjection),
            {keyValues, width},
            fanInRange(width));
        add(named(GemmaTensors::valueProjection),
            {keyValues, width},
            fanInRange(width));
        add(named(GemmaTensors::outputProjection),
            {width, queries},
            fanInRange(queries));
        add(named(GemmaTensors::gateProjection),
            {intermediate, width},
            fanInRange(width));
        add(named(GemmaTensors::upProjection),
            {intermediate, width},
            fanInRange(width));
        add(named(GemmaTensors::downProjection),
            {width, intermediate},
            fanInRange(intermediate));
        add(named(GemmaTensors::postAttentionNorm), {width}, 0.1f);
    }

    add(GemmaTensors::finalNorm, {width}, 0.1f);

    return tensors;
}

inline std::string configJson(const GemmaConfig& config)
{
    auto field = [](std::string_view key, int value)
    { return "\"" + std::string {key} + "\":" + std::to_string(value) + ","; };

    return "{\"model_type\":\"gemma\"," + field("vocab_size", config.vocabularySize)
           + field("hidden_size", config.hiddenSize)
           + field("intermediate_size", config.intermediateSize)
           + field("num_hidden_layers", config.layerCount)
           + field("num_attention_heads", config.attentionHeads)
           + field("num_key_value_heads", config.keyValueHeads)
           + field("head_dim", config.headWidth)
           + field("max_position_embeddings", config.maxPositions)
           + "\"rms_norm_eps\":1e-6,\"rope_theta\":10000.0,"
             "\"hidden_activation\":\"gelu_pytorch_tanh\","
             "\"bos_token_id\":2,\"eos_token_id\":1,\"pad_token_id\":0,"
             "\"tie_word_embeddings\":true}";
}

// Whether the checkpoint is written as one model.safetensors or as two shards
// and the index that says which of them a name is in. Both are shapes a
// HuggingFace repo comes in, and gemma-2b itself is the second.
enum class Sharding
{
    Single,
    TwoShards
};

// What a test does to the catalogue before it is written, which is how the
// checkpoints that must be refused are built: drop a tensor, or give one a
// shape the config does not imply. The default edits nothing, so the ordinary
// case names no callback at all.
using TensorEdit = std::function<void(Vector<SyntheticTensor>&)>;

// Every tensor written as BF16, which is what google/gemma-2b itself ships and
// what the loader now uploads still packed. The values are narrowed on the way
// into the blob, so readFloats hands the reference exactly the numbers
// readBFloat16 hands the GPU and the two are compared on equal terms.
inline TensorEdit asBFloat16()
{
    return [](Vector<SyntheticTensor>& tensors)
    {
        for (auto& tensor: tensors)
            tensor.type = TensorType::BF16;
    };
}

// A packed bf16 buffer read back as the values a shader widens it to, for a
// test that has to look at what was uploaded rather than at what came out of a
// kernel. The buffer's elements are words holding two, low half first.
inline Vector<float> readBackBFloat16(const eacp::GPU::Buffer& buffer, int count)
{
    const auto words = readBack(buffer, (count + 1) / 2);
    auto values = sized(count);

    for (auto index = 0; index < count; ++index)
    {
        auto word = std::uint32_t {};
        std::memcpy(&word, &words[index / 2], sizeof(word));

        const auto half = index % 2 == 0 ? word & 0xFFFFu : word >> 16;
        values[index] = eacp::GPU::bfloat16ToFloat((std::uint16_t) half);
    }

    return values;
}

// A whole model directory of the test's own: config.json, the weights, and —
// for the sharded variant — the index naming the two files they are split
// across. Removed when the test that made it ends, so nothing is left behind
// whether it passed or threw.
class SyntheticCheckpoint
{
public:
    explicit SyntheticCheckpoint(
        std::string_view name,
        Sharding sharding = Sharding::Single,
        TensorEdit edit = [](Vector<SyntheticTensor>&) {},
        GemmaConfig configToUse = smallGemmaConfig())
        : scratch(name)
        , modelConfig(configToUse)
    {
        write(sharding, edit);

        files = ModelFiles::fromDirectory(scratch.path());
        modelConfig = GemmaConfig::fromModelFiles(files);
        tensors.emplace(ShardedTensors::fromModelFiles(files));
    }

    const GemmaConfig& config() const { return modelConfig; }
    const ShardedTensors& weights() const { return *tensors; }
    const ModelFiles& modelFiles() const { return files; }
    const std::filesystem::path& path() const { return scratch.path(); }

    DecoderShape shape() const { return DecoderShape::fromConfig(modelConfig); }

private:
    static constexpr auto firstShardName = "model-00001-of-00002.safetensors";
    static constexpr auto secondShardName = "model-00002-of-00002.safetensors";

    void write(Sharding sharding, const TensorEdit& edit)
    {
        scratch.writeText(ModelFileNames::config, configJson(modelConfig));

        auto all = syntheticGemmaTensors(modelConfig);
        edit(all);

        if (sharding == Sharding::Single)
        {
            auto writer = TensorFileWriter {};

            for (const auto& tensor: all)
                writer.add(tensor);

            scratch.write(ModelFileNames::weights, writer.bytes());
            return;
        }

        // Split so that neither shard holds a whole model: the embedding and
        // the first layer in one, everything after it in the other, which is
        // the shape gemma-2b's own index has.
        const auto boundary = 1 + all.size() / 2;

        auto first = TensorFileWriter {};
        auto second = TensorFileWriter {};
        auto index = std::string {};

        for (auto at = 0; at < all.size(); ++at)
        {
            const auto inFirst = at < boundary;
            (inFirst ? first : second).add(all[at]);

            index += (at == 0 ? "" : ",") + std::string {"\""} + all[at].name
                     + "\":\"" + (inFirst ? firstShardName : secondShardName) + "\"";
        }

        scratch.write(firstShardName, first.bytes());
        scratch.write(secondShardName, second.bytes());
        scratch.writeText(ModelFileNames::shardIndex,
                          "{\"weight_map\":{" + index + "}}");
    }

    ScratchDirectory scratch;
    GemmaConfig modelConfig;
    ModelFiles files;
    std::optional<ShardedTensors> tensors;
};

// The largest |actual - expected| over 1 + |expected| across two runs of rows,
// which is an absolute bound at outputs this size and a relative one past it.
// Reported as a number rather than only as a pass, because a tolerance nobody
// has measured against is a tolerance that was guessed.
inline double worstError(const Vector<float>& actual,
                         const std::vector<double>& expected)
{
    auto worst = 0.0;

    for (auto index = 0; index < actual.size(); ++index)
    {
        const auto reference = expected[(std::size_t) index];
        const auto error = std::abs((double) actual[index] - reference)
                           / (1.0 + std::abs(reference));

        worst = std::max(worst, error);
    }

    return worst;
}

// The rows of a whole-sequence reference one step is answerable for, since a
// step after the first produces the tail of a sequence the reference computed
// in one go.
inline std::vector<double>
    rowsOf(const std::vector<double>& all, int firstRow, int rowCount, int rowLength)
{
    const auto begin = all.begin() + (std::ptrdiff_t) (firstRow * rowLength);
    return {begin, begin + (std::ptrdiff_t) (rowCount * rowLength)};
}
} // namespace HF::Testing

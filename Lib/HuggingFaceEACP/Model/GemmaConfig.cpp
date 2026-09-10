#include "GemmaConfig.h"

#include "ModelIO.h"

#include <cmath>
#include <limits>

namespace HF
{
namespace
{
constexpr auto configName = "config.json";
constexpr auto generationName = "generation_config.json";

void requirePositive(int value, std::string_view field)
{
    if (value <= 0)
        throw ModelError {std::string {configName} + " field '" + std::string {field}
                          + "' must be positive"};
}

void requireGemma(const Miro::Json::Object& object)
{
    const auto type =
        ModelIO::stringFieldOr(object, "model_type", "gemma", configName);

    if (type != "gemma")
        throw ModelError {std::string {configName} + " describes a '" + type
                          + "' model, not a gemma one"};
}

// A token id that HuggingFace writes either as a number or as a list of them.
// The list form is how a repo says "any of these ends a sequence"; the loop
// here stops on one, so the first is what it takes.
int tokenFieldOr(const Miro::Json::Object& object,
                 std::string_view key,
                 int fallback,
                 std::string_view what)
{
    const auto* found = Miro::Json::find(object, key);

    if (found == nullptr || found->isNull())
        return fallback;

    if (!found->isArray())
        return ModelIO::intFieldOr(object, key, fallback, what);

    const auto& ids = found->asArray();

    if (ids.empty())
        return fallback;

    const auto id = ModelIO::asInteger(ids[0], what);

    if (id < std::numeric_limits<int>::min() || id > std::numeric_limits<int>::max())
        throw ModelError {std::string {what} + " token id does not fit in an int"};

    return static_cast<int>(id);
}

std::string activationOf(const Miro::Json::Object& object,
                         const std::string& fallback)
{
    const auto older =
        ModelIO::stringFieldOr(object, "hidden_act", fallback, configName);

    return ModelIO::stringFieldOr(object, "hidden_activation", older, configName);
}
} // namespace

GenerationConfig GenerationConfig::fromFile(const std::filesystem::path& path)
{
    const auto bytes = ModelIO::readFileBytes(path);
    return fromJson(ModelIO::textOf(bytes));
}

GenerationConfig GenerationConfig::fromJson(std::string_view text)
{
    const auto parsed = ModelIO::parseObject(text, generationName);
    const auto& object = parsed.asObject();

    auto config = GenerationConfig {};

    config.beginningOfSequenceToken =
        tokenFieldOr(object, "bos_token_id", noToken, generationName);
    config.endOfSequenceToken =
        tokenFieldOr(object, "eos_token_id", noToken, generationName);
    config.padToken = tokenFieldOr(object, "pad_token_id", noToken, generationName);

    return config;
}

int GemmaConfig::queryHeadsPerKeyValueHead() const
{
    return attentionHeads / keyValueHeads;
}

bool GemmaConfig::isMultiQuery() const
{
    return keyValueHeads == 1 && attentionHeads > 1;
}

int GemmaConfig::queryWidth() const
{
    return attentionHeads * headWidth;
}

int GemmaConfig::keyValueWidth() const
{
    return keyValueHeads * headWidth;
}

float GemmaConfig::embeddingScale() const
{
    return std::sqrt(static_cast<float>(hiddenSize));
}

GemmaConfig GemmaConfig::fromFile(const std::filesystem::path& path)
{
    const auto bytes = ModelIO::readFileBytes(path);
    return fromJson(ModelIO::textOf(bytes));
}

GemmaConfig GemmaConfig::fromJson(std::string_view text)
{
    const auto parsed = ModelIO::parseObject(text, configName);
    const auto& object = parsed.asObject();

    requireGemma(object);

    auto config = GemmaConfig {};

    config.vocabularySize =
        ModelIO::intFieldOr(object, "vocab_size", config.vocabularySize, configName);
    config.hiddenSize =
        ModelIO::intFieldOr(object, "hidden_size", config.hiddenSize, configName);
    config.intermediateSize = ModelIO::intFieldOr(
        object, "intermediate_size", config.intermediateSize, configName);
    config.layerCount = ModelIO::intFieldOr(
        object, "num_hidden_layers", config.layerCount, configName);
    config.attentionHeads = ModelIO::intFieldOr(
        object, "num_attention_heads", config.attentionHeads, configName);
    config.keyValueHeads = ModelIO::intFieldOr(
        object, "num_key_value_heads", config.keyValueHeads, configName);
    config.headWidth =
        ModelIO::intFieldOr(object, "head_dim", config.headWidth, configName);
    config.maxPositions = ModelIO::intFieldOr(
        object, "max_position_embeddings", config.maxPositions, configName);

    config.rmsNormEpsilon = ModelIO::doubleFieldOr(
        object, "rms_norm_eps", config.rmsNormEpsilon, configName);
    config.ropeTheta =
        ModelIO::doubleFieldOr(object, "rope_theta", config.ropeTheta, configName);

    config.hiddenActivation = activationOf(object, config.hiddenActivation);

    config.beginningOfSequenceToken = tokenFieldOr(
        object, "bos_token_id", config.beginningOfSequenceToken, configName);
    config.endOfSequenceToken =
        tokenFieldOr(object, "eos_token_id", config.endOfSequenceToken, configName);
    config.padToken =
        tokenFieldOr(object, "pad_token_id", config.padToken, configName);

    config.tieWordEmbeddings = ModelIO::boolFieldOr(
        object, "tie_word_embeddings", config.tieWordEmbeddings, configName);

    requirePositive(config.vocabularySize, "vocab_size");
    requirePositive(config.hiddenSize, "hidden_size");
    requirePositive(config.intermediateSize, "intermediate_size");
    requirePositive(config.layerCount, "num_hidden_layers");
    requirePositive(config.attentionHeads, "num_attention_heads");
    requirePositive(config.keyValueHeads, "num_key_value_heads");
    requirePositive(config.headWidth, "head_dim");
    requirePositive(config.maxPositions, "max_position_embeddings");

    if (config.rmsNormEpsilon <= 0.0)
        throw ModelError {std::string {configName}
                          + " field 'rms_norm_eps' must be positive"};

    if (config.ropeTheta <= 0.0)
        throw ModelError {std::string {configName}
                          + " field 'rope_theta' must be positive"};

    // Every query head reads one of the KV heads, so a count that does not
    // divide leaves heads with no key to attend to.
    if (config.attentionHeads % config.keyValueHeads != 0)
        throw ModelError {std::string {configName}
                          + " num_key_value_heads does not divide "
                            "num_attention_heads"};

    return config;
}

void GemmaConfig::apply(const GenerationConfig& generation)
{
    if (generation.beginningOfSequenceToken != GenerationConfig::noToken)
        beginningOfSequenceToken = generation.beginningOfSequenceToken;

    if (generation.endOfSequenceToken != GenerationConfig::noToken)
        endOfSequenceToken = generation.endOfSequenceToken;

    if (generation.padToken != GenerationConfig::noToken)
        padToken = generation.padToken;
}

GemmaConfig GemmaConfig::fromModelFiles(const ModelFiles& files)
{
    auto config = fromFile(files.config);

    if (files.hasGenerationConfig())
        config.apply(GenerationConfig::fromFile(files.generationConfig));

    return config;
}

GemmaConfig GemmaConfig::fromDirectory(const std::filesystem::path& directory)
{
    return fromModelFiles(ModelFiles::fromDirectory(directory));
}
} // namespace HF

#include "Common.h"

using namespace nano;
using namespace HF;
using namespace HF::Testing;

namespace
{
// Overrides come first because Miro's parser keeps the first occurrence of a
// duplicate key, so a field named here is the one that wins.
std::string gemmaConfigJson(std::string_view overrides = {})
{
    auto text = std::string {"{"};

    if (!overrides.empty())
        text += std::string {overrides} + ",";

    return text
           + R"("model_type":"gemma",)"
             R"("vocab_size":256000,"hidden_size":2048,)"
             R"("intermediate_size":16384,"num_hidden_layers":18,)"
             R"("num_attention_heads":8,"num_key_value_heads":1,)"
             R"("head_dim":256,"max_position_embeddings":8192,)"
             R"("rms_norm_eps":1e-06,"rope_theta":10000.0,)"
             R"("hidden_activation":"gelu_pytorch_tanh",)"
             R"("bos_token_id":2,"eos_token_id":1,"pad_token_id":0,)"
             R"("tie_word_embeddings":true})";
}
} // namespace

auto tConfigShapes = test("Model/Config/shapes") = []
{
    const auto config = GemmaConfig::fromJson(gemmaConfigJson());

    check(config.vocabularySize == 256000);
    check(config.hiddenSize == 2048);
    check(config.intermediateSize == 16384);
    check(config.layerCount == 18);
    check(config.attentionHeads == 8);
    check(config.keyValueHeads == 1);
    check(config.headWidth == 256);
    check(config.maxPositions == 8192);
    check(nearlyEqual(static_cast<float>(config.rmsNormEpsilon), 1.0e-6f, 1.0e-9f));
    check(nearlyEqual(static_cast<float>(config.ropeTheta), 10000.0f, 1.0e-3f));
    check(config.hiddenActivation == "gelu_pytorch_tanh");
    check(config.beginningOfSequenceToken == 2);
    check(config.endOfSequenceToken == 1);
    check(config.padToken == 0);
    check(config.tieWordEmbeddings);

    check(config.queryWidth() == 2048);
    check(config.keyValueWidth() == 256);
    check(config.queryHeadsPerKeyValueHead() == 8);
    check(config.isMultiQuery());
    check(nearlyEqual(config.embeddingScale(), 45.254834f, 1.0e-4f));
};

// A config that omits a field still describes the model it came from, because
// every default is Gemma 2B's own value. plan.md's numbers are unconfirmed
// against the real config.json, so these are the defaults under test rather
// than the truth about the checkpoint.
auto tConfigDefaults = test("Model/Config/defaults") = []
{
    const auto bare = GemmaConfig::fromJson(R"({"model_type":"gemma"})");
    const auto full = GemmaConfig::fromJson(gemmaConfigJson());

    check(bare.vocabularySize == full.vocabularySize);
    check(bare.hiddenSize == full.hiddenSize);
    check(bare.intermediateSize == full.intermediateSize);
    check(bare.layerCount == full.layerCount);
    check(bare.attentionHeads == full.attentionHeads);
    check(bare.keyValueHeads == full.keyValueHeads);
    check(bare.headWidth == full.headWidth);
    check(bare.maxPositions == full.maxPositions);
    check(bare.hiddenActivation == full.hiddenActivation);
    check(bare.beginningOfSequenceToken == full.beginningOfSequenceToken);
    check(bare.endOfSequenceToken == full.endOfSequenceToken);
    check(bare.padToken == full.padToken);
    check(bare.tieWordEmbeddings == full.tieWordEmbeddings);

    // model_type itself is the one thing that may be absent, since a file with
    // no architecture named is not a file about another architecture.
    check(GemmaConfig::fromJson("{}").hiddenSize == 2048);
};

// `hidden_act` is what older transformers wrote and `hidden_activation` what
// it writes now. A checkpoint may carry either, or both.
auto tConfigActivationKeys = test("Model/Config/activationKeys") = []
{
    const auto older =
        GemmaConfig::fromJson(R"({"model_type":"gemma","hidden_act":"gelu"})");
    check(older.hiddenActivation == "gelu");

    const auto newer = GemmaConfig::fromJson(
        R"({"model_type":"gemma","hidden_activation":"gelu_pytorch_tanh"})");
    check(newer.hiddenActivation == "gelu_pytorch_tanh");

    const auto both = GemmaConfig::fromJson(
        R"({"model_type":"gemma","hidden_activation":"gelu_pytorch_tanh",)"
        R"("hidden_act":"gelu"})");
    check(both.hiddenActivation == "gelu_pytorch_tanh");

    // And the shape the real gemma-2b config.json is in: an explicit null on
    // the newer key beside a string on the older one. transformers writes that
    // when the newer key was never set, and reads it as Gemma's own
    // gelu_pytorch_tanh with `hidden_act` ignored — so the null is the
    // default, not a missing key deferring to "gelu".
    const auto nulled = GemmaConfig::fromJson(
        R"({"model_type":"gemma","hidden_activation":null,"hidden_act":"gelu"})");
    check(nulled.hiddenActivation == "gelu_pytorch_tanh");
};

// A decoder built for gemma and pointed at some other architecture's config
// would otherwise get shapes that parse and mean nothing.
auto tConfigRejectsAnotherArchitecture =
    test("Model/Config/rejectsAnotherArchitecture") = []
{
    check(throwsModelError(
        [] { return GemmaConfig::fromJson(R"({"model_type":"llama"})"); }));

    check(throwsModelError(
        []
        { return GemmaConfig::fromJson(gemmaConfigJson(R"("model_type":"t5")")); }));
};

auto tConfigRejectsBadShapes = test("Model/Config/rejectsBadShapes") = []
{
    check(throwsModelError([] { return GemmaConfig::fromJson("not json"); }));
    check(throwsModelError([] { return GemmaConfig::fromJson("[]"); }));

    check(throwsModelError(
        []
        { return GemmaConfig::fromJson(gemmaConfigJson(R"("hidden_size":0)")); }));

    check(throwsModelError(
        []
        {
            return GemmaConfig::fromJson(
                gemmaConfigJson(R"("num_hidden_layers":-1)"));
        }));

    check(throwsModelError(
        []
        { return GemmaConfig::fromJson(gemmaConfigJson(R"("rms_norm_eps":0)")); }));

    check(throwsModelError(
        []
        { return GemmaConfig::fromJson(gemmaConfigJson(R"("rope_theta":-1)")); }));

    check(throwsModelError(
        []
        {
            return GemmaConfig::fromJson(gemmaConfigJson(R"("hidden_size":"2048")"));
        }));

    check(throwsModelError(
        []
        {
            return GemmaConfig::fromJson(gemmaConfigJson(R"("hidden_size":2048.5)"));
        }));

    // Three query heads over two KV heads leaves a head with nothing to read.
    check(throwsModelError(
        []
        {
            return GemmaConfig::fromJson(gemmaConfigJson(
                R"("num_attention_heads":3,"num_key_value_heads":2)"));
        }));
};

auto tConfigMissingFile = test("Model/Config/missingFile") = []
{ check(throwsModelError([] { return GemmaConfig::fromFile("no/such.json"); })); };

auto tGenerationConfig = test("Model/Config/generationConfig") = []
{
    const auto config = GenerationConfig::fromJson(
        R"({"bos_token_id":2,"eos_token_id":1,"pad_token_id":0})");

    check(config.beginningOfSequenceToken == 2);
    check(config.endOfSequenceToken == 1);
    check(config.padToken == 0);

    const auto bare = GenerationConfig::fromJson("{}");
    check(bare.endOfSequenceToken == GenerationConfig::noToken);

    // The instruction-tuned repos write a list of end tokens; the loop here
    // stops on one, so the first is what it takes.
    const auto listed =
        GenerationConfig::fromJson(R"({"eos_token_id":[1,107],"bos_token_id":2})");
    check(listed.endOfSequenceToken == 1);
    check(listed.padToken == GenerationConfig::noToken);
};

// generation_config.json is the authority on the token ids when the two files
// disagree, and a config with no generation file beside it keeps its own.
auto tGenerationConfigOverrides = test("Model/Config/generationOverrides") = []
{
    auto config = GemmaConfig::fromJson(gemmaConfigJson());

    auto generation = GenerationConfig {};
    generation.endOfSequenceToken = 107;
    config.apply(generation);

    check(config.endOfSequenceToken == 107);
    check(config.beginningOfSequenceToken == 2);
    check(config.padToken == 0);
};

auto tConfigFromDirectory = test("Model/Config/fromDirectory") = []
{
    const auto scratch = ScratchDirectory {"config-directory"};
    scratch.writeText(ModelFileNames::config, gemmaConfigJson());
    scratch.write(ModelFileNames::weights,
                  assemble(R"({"w":{"dtype":"F32","shape":[1],)"
                           R"("data_offsets":[0,4]}})",
                           toBytes<float>({1.0f})));

    check(GemmaConfig::fromDirectory(scratch.path()).endOfSequenceToken == 1);

    scratch.writeText(ModelFileNames::generationConfig,
                      R"({"eos_token_id":107,"pad_token_id":3})");

    const auto merged = GemmaConfig::fromDirectory(scratch.path());
    check(merged.endOfSequenceToken == 107);
    check(merged.padToken == 3);
    check(merged.beginningOfSequenceToken == 2);
};

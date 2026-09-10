#pragma once

#include "Common.h"

#include <algorithm>
#include <limits>

// HuggingFace's GemmaForCausalLM.forward in double precision, written from the
// definition rather than out of the kernels' own references — the chain is what
// this checks, and a reference assembled from the same pieces the decoder is
// assembled from could not check it.
//
// Full attention rather than a cached one: every step of a sequence recomputes
// every key and value from the tokens before it, so a KV cache that dropped or
// misplaced a row has nothing here agreeing with it.
namespace HF::Testing
{
struct ReferenceLayer
{
    Vector<float> inputNorm;
    Vector<float> query;
    Vector<float> key;
    Vector<float> value;
    Vector<float> output;
    Vector<float> gate;
    Vector<float> up;
    Vector<float> down;
    Vector<float> postAttentionNorm;
};

struct ReferenceModel
{
    Vector<float> embedding;
    Vector<float> finalNorm;
    std::vector<ReferenceLayer> layers;
};

// Read back through readFloats, which widens whatever the file holds — so the
// reference runs on exactly the numbers the kernels read, and the gate and up
// weights are the two the decoder concatenated rather than a copy of them.
inline ReferenceModel readReferenceModel(const ShardedTensors& file,
                                         const DecoderShape& shape)
{
    auto model = ReferenceModel {};
    model.embedding = file.readFloats(GemmaTensors::embedding);
    model.finalNorm = file.readFloats(GemmaTensors::finalNorm);

    for (auto index = 0; index < shape.layers; ++index)
    {
        auto read = [&](std::string_view suffix)
        { return file.readFloats(GemmaTensors::layerTensorName(index, suffix)); };

        model.layers.push_back({read(GemmaTensors::inputNorm),
                                read(GemmaTensors::queryProjection),
                                read(GemmaTensors::keyProjection),
                                read(GemmaTensors::valueProjection),
                                read(GemmaTensors::outputProjection),
                                read(GemmaTensors::gateProjection),
                                read(GemmaTensors::upProjection),
                                read(GemmaTensors::downProjection),
                                read(GemmaTensors::postAttentionNorm)});
    }

    return model;
}

// x * rsqrt(mean(x^2) + epsilon) * (1 + w), which is Gemma's only
// normalisation: no mean subtraction, no bias, and the scale offset by one
// because GemmaRMSNorm stores the offset rather than the scale.
inline std::vector<double> referenceRmsNorm(const std::vector<double>& rows,
                                            int rowCount,
                                            int rowLength,
                                            const Vector<float>& weight,
                                            double epsilon)
{
    auto out = std::vector<double>((std::size_t) (rowCount * rowLength));

    for (auto row = 0; row < rowCount; ++row)
    {
        const auto base = (std::size_t) (row * rowLength);
        auto squares = 0.0;

        for (auto column = 0; column < rowLength; ++column)
            squares += rows[base + (std::size_t) column]
                       * rows[base + (std::size_t) column];

        const auto scale = 1.0 / std::sqrt(squares / rowLength + epsilon);

        for (auto column = 0; column < rowLength; ++column)
            out[base + (std::size_t) column] =
                rows[base + (std::size_t) column] * scale * (1.0 + weight[column]);
    }

    return out;
}

// y = x Wᵀ, with W stored [outputWidth, innerCount] the way nn.Linear stores
// it. Gemma has no bias anywhere, so there is no term for one.
inline std::vector<double> referenceLinear(const std::vector<double>& rows,
                                           int rowCount,
                                           int innerCount,
                                           const Vector<float>& weight,
                                           int outputWidth)
{
    auto out = std::vector<double>((std::size_t) (rowCount * outputWidth));

    for (auto row = 0; row < rowCount; ++row)
    {
        for (auto column = 0; column < outputWidth; ++column)
        {
            auto total = 0.0;

            for (auto inner = 0; inner < innerCount; ++inner)
                total += rows[(std::size_t) (row * innerCount + inner)]
                         * weight[column * innerCount + inner];

            out[(std::size_t) (row * outputWidth + column)] = total;
        }
    }

    return out;
}

// Rotate-half, in place: channel i is turned against channel i + headWidth / 2
// through the angle position * theta^(-2i/headWidth). Not pairwise — the two
// orderings give different vectors from the same weights, and a model run
// under the wrong one produces plausible nonsense rather than an error.
inline void referenceRotate(std::vector<double>& rows,
                            int rowCount,
                            int rowStride,
                            int headCount,
                            int headWidth,
                            int firstPosition,
                            double theta)
{
    const auto half = headWidth / 2;

    for (auto row = 0; row < rowCount; ++row)
    {
        const auto position = (double) (firstPosition + row);

        for (auto head = 0; head < headCount; ++head)
        {
            const auto base = row * rowStride + head * headWidth;

            for (auto pair = 0; pair < half; ++pair)
            {
                const auto exponent = 2.0 * pair / (double) headWidth;
                const auto angle = position / std::pow(theta, exponent);
                const auto cosine = std::cos(angle);
                const auto sine = std::sin(angle);

                const auto low = (std::size_t) (base + pair);
                const auto high = (std::size_t) (base + pair + half);
                const auto x = rows[low];
                const auto y = rows[high];

                rows[low] = x * cosine - y * sine;
                rows[high] = y * cosine + x * sine;
            }
        }
    }
}

// Multi-query attention over the whole sequence, causally: query row t reads
// keys 0 through t and no further, and query head h reads KV head
// h / (heads / kvHeads) — which for Gemma 2B is the one shared head for all
// eight. Masking here is exclusion rather than a sentinel, so a later key takes
// no part in the maximum, contributes nothing to the sum and carries no weight.
inline std::vector<double> referenceAttention(const std::vector<double>& queries,
                                              const std::vector<double>& keys,
                                              const std::vector<double>& values,
                                              const DecoderShape& shape,
                                              int rowCount)
{
    const auto queryWidth = shape.queryWidth();
    const auto kvWidth = shape.kvWidth();
    const auto headWidth = shape.headWidth;
    const auto perKvHead = shape.heads / shape.kvHeads;
    const auto scale = 1.0 / std::sqrt((double) headWidth);

    auto out = std::vector<double>((std::size_t) (rowCount * queryWidth));
    auto weights = std::vector<double>((std::size_t) rowCount);

    for (auto row = 0; row < rowCount; ++row)
    {
        const auto keyCount = row + 1;

        for (auto head = 0; head < shape.heads; ++head)
        {
            const auto queryColumn = head * headWidth;
            const auto kvColumn = head / perKvHead * headWidth;
            auto largest = -std::numeric_limits<double>::infinity();

            for (auto key = 0; key < keyCount; ++key)
            {
                auto total = 0.0;

                for (auto channel = 0; channel < headWidth; ++channel)
                    total +=
                        queries[(std::size_t) (row * queryWidth + queryColumn
                                               + channel)]
                        * keys[(std::size_t) (key * kvWidth + kvColumn + channel)];

                weights[(std::size_t) key] = scale * total;
                largest = std::max(largest, weights[(std::size_t) key]);
            }

            auto denominator = 0.0;

            for (auto key = 0; key < keyCount; ++key)
            {
                weights[(std::size_t) key] =
                    std::exp(weights[(std::size_t) key] - largest);
                denominator += weights[(std::size_t) key];
            }

            for (auto channel = 0; channel < headWidth; ++channel)
            {
                auto weighted = 0.0;

                for (auto key = 0; key < keyCount; ++key)
                    weighted +=
                        weights[(std::size_t) key]
                        * values[(std::size_t) (key * kvWidth + kvColumn + channel)];

                out[(std::size_t) (row * queryWidth + queryColumn + channel)] =
                    weighted / denominator;
            }
        }
    }

    return out;
}

// Both halves of the forward pass, so a GPU decoder can be bisected at either:
// the rows model.norm produces and the vocabulary-wide row the tied embedding
// turns each of them into.
struct ReferenceDecoding
{
    std::vector<double> hidden;
    std::vector<double> logits;
};

// The whole sequence from position zero, which is what the KV cache has to
// reproduce however the tokens are fed:
//
//   h = embeddingScale * embed_tokens[token]
//   per layer: n = rmsnorm(h, input_layernorm)
//              q, k, v = n Wqᵀ, n Wkᵀ, n Wvᵀ, then rope on q and k
//              h += attention(q, k, v) Woᵀ
//              n = rmsnorm(h, post_attention_layernorm)
//              h += (gelu_tanh(n Wgateᵀ) * (n Wupᵀ)) Wdownᵀ
//   h = rmsnorm(h, model.norm)
//   logits = h · embed_tokensᵀ
inline ReferenceDecoding referenceDecode(const ReferenceModel& model,
                                         const DecoderShape& shape,
                                         const std::vector<int>& tokens)
{
    const auto width = shape.width;
    const auto rowCount = (int) tokens.size();
    const auto epsilon = (double) shape.rmsNormEpsilon;

    auto hidden = std::vector<double>((std::size_t) (rowCount * width));

    for (auto row = 0; row < rowCount; ++row)
        for (auto column = 0; column < width; ++column)
            hidden[(std::size_t) (row * width + column)] =
                (double) shape.embeddingScale
                * model.embedding[tokens[(std::size_t) row] * width + column];

    for (const auto& layer: model.layers)
    {
        auto normalised =
            referenceRmsNorm(hidden, rowCount, width, layer.inputNorm, epsilon);

        auto queries = referenceLinear(
            normalised, rowCount, width, layer.query, shape.queryWidth());
        auto keys =
            referenceLinear(normalised, rowCount, width, layer.key, shape.kvWidth());
        const auto values = referenceLinear(
            normalised, rowCount, width, layer.value, shape.kvWidth());

        referenceRotate(queries,
                        rowCount,
                        shape.queryWidth(),
                        shape.heads,
                        shape.headWidth,
                        0,
                        shape.ropeTheta);

        referenceRotate(keys,
                        rowCount,
                        shape.kvWidth(),
                        shape.kvHeads,
                        shape.headWidth,
                        0,
                        shape.ropeTheta);

        const auto attended =
            referenceAttention(queries, keys, values, shape, rowCount);

        const auto projected = referenceLinear(
            attended, rowCount, shape.queryWidth(), layer.output, width);

        for (auto index = 0u; index < hidden.size(); ++index)
            hidden[index] += projected[index];

        normalised = referenceRmsNorm(
            hidden, rowCount, width, layer.postAttentionNorm, epsilon);

        const auto gate = referenceLinear(
            normalised, rowCount, width, layer.gate, shape.intermediate);
        const auto up = referenceLinear(
            normalised, rowCount, width, layer.up, shape.intermediate);

        auto activated = std::vector<double>(gate.size());

        for (auto index = 0u; index < activated.size(); ++index)
            activated[index] = tanhGeluReference(gate[index]) * up[index];

        const auto down = referenceLinear(
            activated, rowCount, shape.intermediate, layer.down, width);

        for (auto index = 0u; index < hidden.size(); ++index)
            hidden[index] += down[index];
    }

    auto decoding = ReferenceDecoding {};
    decoding.hidden =
        referenceRmsNorm(hidden, rowCount, width, model.finalNorm, epsilon);

    decoding.logits = referenceLinear(
        decoding.hidden, rowCount, width, model.embedding, shape.vocabularySize);

    return decoding;
}

// ---------------------------------------------------------------------------
// Running the GPU decoder, one step at a time the way a sampler would.
// ---------------------------------------------------------------------------

// One step's outputs, read back together: the rows after the final norm and
// the vocabulary rows the tied projection turned them into. Comparing both is
// what lets a disagreement bisect to before or after the logits.
struct StepResult
{
    Vector<float> hidden;
    Vector<float> logits;
};

// A decoder, its weights and the sequence state. Each call is its own command
// buffer and its own commit, because a test reads what it wrote before deciding
// what to feed next — a real run records several steps before it needs a number
// back.
class DecoderRun
{
public:
    DecoderRun(const DecoderShape& shapeToUse, const ShardedTensors& file)
        : decoderShape(shapeToUse)
        , weights(file, shapeToUse)
        , decoder(shapeToUse)
    {
        decoder.prepare(eacp::GPU::Device::shared());
    }

    StepResult step(const std::vector<int>& tokens)
    {
        const auto count = (int) tokens.size();
        const auto logits = outputFor(count * decoderShape.logitElementCount());

        record(tokens, logits);

        return {readBack(decoder.hiddenStates(), count * decoderShape.width),
                readBack(logits, count * decoderShape.logitElementCount())};
    }

    // The step that must not record: the logits buffer is filled with a value
    // no logit could be, and a step that threw has to leave it there.
    bool stepThrowsLeavingLogitsUntouched(const std::vector<int>& tokens)
    {
        constexpr auto poison = -12345.f;

        const auto count = (int) tokens.size();
        const auto elements = count * decoderShape.logitElementCount();

        auto filled = sized(elements);

        for (auto index = 0; index < elements; ++index)
            filled[index] = poison;

        const auto logits = storageOf(filled);
        const auto before = decoder.position();
        const auto threw = throwsModelError([&] { record(tokens, logits); });

        if (!threw || decoder.position() != before)
            return false;

        const auto after = readBack(logits, elements);

        for (auto index = 0; index < elements; ++index)
            if (after[index] != poison)
                return false;

        return true;
    }

    void beginSequence() { decoder.beginSequence(); }
    int position() const { return decoder.position(); }
    Decoder& gpuDecoder() { return decoder; }
    const DecoderWeights& loaded() const { return weights; }

private:
    void record(const std::vector<int>& tokens, const eacp::GPU::Buffer& logits)
    {
        auto ids = unsignedSized((int) tokens.size());

        for (auto index = 0; index < ids.size(); ++index)
            ids[index] = (std::uint32_t) tokens[(std::size_t) index];

        const auto tokenBuffer = storageOf(ids);
        auto commands = eacp::GPU::Device::shared().makeCommandBuffer();

        {
            auto pass = commands.beginCompute();
            decoder.step(pass, tokenBuffer, ids.size(), weights, logits);
        }

        commands.commit();
    }

    DecoderShape decoderShape;
    DecoderWeights weights;
    Decoder decoder;
};
} // namespace HF::Testing

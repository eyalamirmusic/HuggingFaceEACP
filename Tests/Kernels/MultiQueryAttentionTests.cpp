#include "ReferenceAttention.h"

using namespace nano;
using namespace HF;
using namespace HF::Reference;
using namespace eacp::GPU;

namespace
{
// Gemma 2B's attention: eight query heads of 256 over the one cached KV head.
constexpr auto gemmaShape = MultiQueryShape {8, 1, 256};

// The identity case of the head-to-KV-head mapping, which is what Whisper's
// plain multi-head attention is and what a bug in the mapping's division would
// still get right if it were only ever tested all-to-one.
constexpr auto multiHeadShape = MultiQueryShape {2, 2, 256};

// How far a kernel folding 256 channels and up to a thousand keys in float32
// may sit from the same fold in double precision, as |actual - expected| over
// 1 + |expected| — so at outputs this size it is an absolute bound. Every case
// below still passes at 2e-7, which is where the streaming rescale really
// lands; the order of magnitude over it is the margin a second backend gets.
constexpr auto tolerance = 2e-6;

// A cache long enough for the shapes below, filled once so that every test
// reads the same numbers: a cache the model would have written is not special
// in any way a kernel can see.
struct AttentionInputs
{
    AttentionInputs(const MultiQueryShape& shapeToUse, int rows, int cacheRows)
        : shape(shapeToUse)
        , queries(spreadValues(rows * shape.queryWidth(), 4801u, 1.f))
        , keys(spreadValues(cacheRows * shape.kvWidth(), 4802u, 1.f))
        , values(spreadValues(cacheRows * shape.kvWidth(), 4803u, 1.f))
    {
    }

    MultiQueryShape shape;
    Vector<float> queries;
    Vector<float> keys;
    Vector<float> values;
};

// The decode kernel over a cache of contextLength entries, reading back the one
// H * D output row.
Vector<float> runDecode(const AttentionInputs& inputs, int contextLength)
{
    auto queryBuffer = storageOf(inputs.queries);
    auto keyBuffer = storageOf(inputs.keys);
    auto valueBuffer = storageOf(inputs.values);
    auto output = outputFor(inputs.shape.queryWidth());

    auto kernel = MultiQueryDecodeAttention {};
    kernel.queries = queryBuffer;
    kernel.keys = keyBuffer;
    kernel.values = valueBuffer;
    kernel.output = output;
    kernel.prepare();

    auto commands = Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        kernel.dispatch(
            pass, inputs.shape, contextLength, inputs.shape.defaultScale());
    }

    commands.commit();
    return readBack(output, inputs.shape.queryWidth());
}

// The prefill kernel over rows query rows standing at positionOffset onwards.
Vector<float> runPrefill(const AttentionInputs& inputs, int rows, int offset)
{
    auto queryBuffer = storageOf(inputs.queries);
    auto keyBuffer = storageOf(inputs.keys);
    auto valueBuffer = storageOf(inputs.values);
    auto output = outputFor(rows * inputs.shape.queryWidth());

    auto kernel = MultiQueryPrefillAttention {};
    kernel.queries = queryBuffer;
    kernel.keys = keyBuffer;
    kernel.values = valueBuffer;
    kernel.output = output;
    kernel.prepare();

    auto commands = Device::shared().makeCommandBuffer();

    {
        auto pass = commands.beginCompute();
        kernel.dispatch(
            pass, inputs.shape, rows, offset, inputs.shape.defaultScale());
    }

    commands.commit();
    return readBack(output, rows * inputs.shape.queryWidth());
}

void checkAgainstReference(const Vector<float>& result,
                           const AttentionInputs& inputs,
                           int rows,
                           int offset)
{
    auto expected = attentionReference(inputs.queries,
                                       inputs.keys,
                                       inputs.values,
                                       inputs.shape,
                                       rows,
                                       offset,
                                       (double) inputs.shape.defaultScale());

    check(result.size() == expected.size());

    for (auto i = 0; i < result.size(); ++i)
        check(isClose(result[i], expected[i], tolerance));
}
} // namespace

// The decode step at the context lengths that are the kernel's edges: one key,
// fewer than a group has lanes, exactly one group's worth, one more than that,
// and a thousand — which is fifteen full tiles and a partial one, so the
// streaming rescale is exercised as well as the tail.
auto tDecodeMatchesReference = test("Kernels/multiQueryDecodeMatchesReference") = []
{
    if (!Device::shared().isValid())
        return;

    for (auto contextLength: {1, 7, 64, 65, 1000})
    {
        auto inputs = AttentionInputs {gemmaShape, 1, contextLength};
        auto result = runDecode(inputs, contextLength);

        checkAgainstReference(result, inputs, 1, contextLength - 1);
    }
};

// The prompt, at a fresh cache and against one that already holds earlier
// tokens. 33 rows is deliberately not a multiple of anything the kernel counts
// in.
auto tPrefillMatchesReference =
    test("Kernels/multiQueryPrefillMatchesReference") = []
{
    if (!Device::shared().isValid())
        return;

    for (auto offset: {0, 71})
    {
        for (auto rows: {1, 5, 33})
        {
            auto inputs = AttentionInputs {gemmaShape, rows, offset + rows};
            auto result = runPrefill(inputs, rows, offset);

            checkAgainstReference(result, inputs, rows, offset);
        }
    }
};

// Causality, asserted rather than inferred: the cache carries eight entries
// past the last query's position, every one of them poisoned with a key that
// would dominate any softmax it reached and a value that would swamp any mean
// it landed in. The reference never looks at them, so agreeing with it is the
// statement that the kernel did not either.
auto tPrefillIsCausal = test("Kernels/multiQueryPrefillIsCausal") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto rows = 9;
    constexpr auto offset = 5;
    constexpr auto poisoned = 8;

    auto inputs = AttentionInputs {gemmaShape, rows, offset + rows + poisoned};

    for (auto key = offset + rows; key < offset + rows + poisoned; ++key)
    {
        for (auto channel = 0; channel < inputs.shape.kvWidth(); ++channel)
        {
            inputs.keys[key * inputs.shape.kvWidth() + channel] = 40.f;
            inputs.values[key * inputs.shape.kvWidth() + channel] = 1000.f;
        }
    }

    auto result = runPrefill(inputs, rows, offset);
    checkAgainstReference(result, inputs, rows, offset);
};

// The mapping's identity case on both kernels: with as many KV heads as query
// heads, head h reads cache head h, which is the multi-head attention Whisper
// runs and the shape a division that got the arithmetic backwards would fail.
auto tMultiHeadSanity = test("Kernels/multiQueryMultiHeadSanity") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto contextLength = 37;

    auto decodeInputs = AttentionInputs {multiHeadShape, 1, contextLength};
    checkAgainstReference(
        runDecode(decodeInputs, contextLength), decodeInputs, 1, contextLength - 1);

    constexpr auto rows = 5;
    constexpr auto offset = 3;

    auto prefillInputs = AttentionInputs {multiHeadShape, rows, offset + rows};
    checkAgainstReference(
        runPrefill(prefillInputs, rows, offset), prefillInputs, rows, offset);
};

// The property the generation loop rests on: feeding a prompt in one dispatch
// and feeding it a token at a time have to agree, or a model's first token
// after the prompt is not the token it would have decoded. Row r of the prefill
// is the decode step over the r + 1 cache entries that exist when that token is
// the one being generated.
auto tPrefillMatchesDecodeSteps =
    test("Kernels/multiQueryPrefillMatchesDecodeSteps") = []
{
    if (!Device::shared().isValid())
        return;

    constexpr auto rows = 12;
    const auto width = gemmaShape.queryWidth();

    auto inputs = AttentionInputs {gemmaShape, rows, rows};
    auto prefilled = runPrefill(inputs, rows, 0);

    for (auto row = 0; row < rows; ++row)
    {
        auto step = inputs;
        step.queries = sized(width);

        for (auto channel = 0; channel < width; ++channel)
            step.queries[channel] = inputs.queries[row * width + channel];

        auto decoded = runDecode(step, row + 1);

        for (auto channel = 0; channel < width; ++channel)
            check(isClose(prefilled[row * width + channel],
                          (double) decoded[channel],
                          tolerance));
    }
};

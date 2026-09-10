#pragma once

#include <HuggingFaceEACP/Kernels/MultiQueryAttention.h>

#include <eacp/Core/App/App.h>
#include <eacp/GPU/GPU.h>

#include <NanoTest/NanoTest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

// The scalar reference the attention kernels are checked against, and the host
// plumbing the check needs, in a namespace of their own: the Kernels tests
// share a Common.h, and a reference this specific belongs beside the one test
// that asserts against it rather than in it.
namespace HF::Reference
{
inline Vector<float> sized(int elementCount)
{
    auto values = Vector<float>();
    values.resize(elementCount);
    return values;
}

// Values in [-range, range], negatives included, from a generator whose
// sequence the standard specifies rather than an implementation: the same
// numbers reach the GPU and the reference on every machine, so a disagreement
// is the kernel.
inline Vector<float> spreadValues(int count, unsigned seed, float range)
{
    auto engine = std::mt19937 {seed};
    auto values = sized(count);

    for (auto i = 0; i < count; ++i)
        values[i] = range * ((float) (engine() % 2001u) / 1000.f - 1.f);

    return values;
}

inline eacp::GPU::Buffer storageOf(const Vector<float>& values)
{
    return eacp::GPU::Device::shared().makeBuffer(values.data(),
                                                  (int) sizeof(float)
                                                      * values.size(),
                                                  eacp::GPU::BufferUsage::Storage);
}

inline eacp::GPU::Buffer outputFor(int elementCount)
{
    return eacp::GPU::Device::shared().makeBuffer((int) sizeof(float)
                                                  * elementCount);
}

inline Vector<float> readBack(const eacp::GPU::Buffer& buffer, int elementCount)
{
    auto values = sized(elementCount);
    buffer.read(values.data(), (int) sizeof(float) * elementCount);
    return values;
}

inline bool isClose(float actual, double expected, double tolerance)
{
    return std::abs((double) actual - expected)
           <= tolerance * (1.0 + std::abs(expected));
}

// Multi-query attention in doubles, written straight from the definition:
// softmax(q . K^T * scale) V per head, with query row r standing at position
// positionOffset + r and reading cache entries [0, positionOffset + r].
//
// Causal masking here is exclusion rather than a sentinel — a later key takes
// no part in the maximum, contributes nothing to the sum and carries no weight
// — which is exactly what the kernel's per-row key count does, arrived at from
// the other end.
inline Vector<double> attentionReference(const Vector<float>& queries,
                                         const Vector<float>& keys,
                                         const Vector<float>& values,
                                         const MultiQueryShape& shape,
                                         int rows,
                                         int positionOffset,
                                         double scale)
{
    const auto queryWidth = shape.queryWidth();
    const auto kvWidth = shape.kvWidth();
    const auto perKvHead = shape.queriesPerKvHead();

    auto expected = Vector<double>();
    expected.resize(rows * queryWidth);

    auto weights = Vector<double>();

    for (auto row = 0; row < rows; ++row)
    {
        const auto keyCount = positionOffset + row + 1;

        for (auto head = 0; head < shape.heads; ++head)
        {
            const auto queryColumn = head * shape.headDim;
            const auto kvColumn = head / perKvHead * shape.headDim;

            weights.clear();
            auto largest = -std::numeric_limits<double>::infinity();

            for (auto key = 0; key < keyCount; ++key)
            {
                auto total = 0.0;

                for (auto channel = 0; channel < shape.headDim; ++channel)
                    total +=
                        (double) queries[row * queryWidth + queryColumn + channel]
                        * keys[key * kvWidth + kvColumn + channel];

                weights.add(scale * total);
                largest = std::max(largest, scale * total);
            }

            auto denominator = 0.0;

            for (auto key = 0; key < keyCount; ++key)
            {
                weights[key] = std::exp(weights[key] - largest);
                denominator += weights[key];
            }

            for (auto channel = 0; channel < shape.headDim; ++channel)
            {
                auto total = 0.0;

                for (auto key = 0; key < keyCount; ++key)
                    total +=
                        weights[key] * values[key * kvWidth + kvColumn + channel];

                expected[row * queryWidth + queryColumn + channel] =
                    total / denominator;
            }
        }
    }

    return expected;
}
} // namespace HF::Reference

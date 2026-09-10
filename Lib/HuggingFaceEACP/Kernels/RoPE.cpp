#include "RoPE.h"

#include <cmath>

namespace HF
{
RotaryTable makeRotaryTable(int maxPositions, int headDim, double theta)
{
    const auto half = headDim / 2;

    auto table = RotaryTable {};
    table.cosines.resize(maxPositions * half);
    table.sines.resize(maxPositions * half);

    for (auto channel = 0; channel < half; ++channel)
    {
        // theta^(-2i/headDim) as a division of the exponent rather than a
        // negative power, so the two ends of the table are computed the same
        // way and the i = 0 term is exactly 1.
        const auto exponent = 2.0 * channel / (double) headDim;
        const auto inverseFrequency = 1.0 / std::pow(theta, exponent);

        for (auto position = 0; position < maxPositions; ++position)
        {
            const auto angle = (double) position * inverseFrequency;
            const auto at = position * half + channel;

            table.cosines[at] = (float) std::cos(angle);
            table.sines[at] = (float) std::sin(angle);
        }
    }

    return table;
}
} // namespace HF

#pragma once

#include "KernelTypes.h"

namespace HF
{
// The exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))). erf is eacp's intrinsic —
// a float32 error function rather than the libm-grade one the name has on the
// CPU, since neither shading language has erf and the EDSL emits the
// approximation as a helper, but its error lands under what a float can
// resolve near one.
inline Float exactGelu(const Float& x)
{
    constexpr auto inverseRootTwo = 0.70710678118654752f;

    return 0.5f * x * (1.f + erf(x * inverseRootTwo));
}

// The tanh approximation, which is what Gemma is trained and shipped with:
// config.json names hidden_act "gelu_pytorch_tanh", so this — and not the
// exact form next door — is the function the released weights were fitted
// against. The two differ by about 1e-3 at the peak of their disagreement,
// which is far more than a tolerance and far more than the whole float32
// error of either, so which one runs is a correctness question rather than an
// accuracy one.
//
// tanh is eacp's intrinsic, spelled the same in both shading languages, and
// **its argument is clamped before it reaches one.** The cube grows the
// argument to about 990 by x = 30, and Metal's tanh under the default compile
// options — which is what eacp builds a library with — returns a NaN there
// rather than saturating: an activation of 30 came back as not-a-number, and a
// single NaN in the residual stream is the whole rest of the sequence. Ten is
// past where tanh is 1.0f exactly (float32 resolves nothing between tanh(8.4)
// and one), so the clamp changes no representable value of the function and
// only keeps the intrinsic inside the range it answers on.
inline Float tanhGelu(const Float& x)
{
    constexpr auto rootTwoOverPi = 0.7978845608028654f;
    constexpr auto cubicTerm = 0.044715f;
    constexpr auto saturated = 10.f;

    auto inner = rootTwoOverPi * (x + cubicTerm * x * x * x);

    return 0.5f * x * (1.f + tanh(clamp(inner, -saturated, saturated)));
}

// Which of the two a program applies. The model's own is Tanh; Exact is kept
// because it costs one function and is what a Hugging Face model configured
// with plain "gelu" would need.
enum class GeluForm
{
    Tanh,
    Exact
};

// Elementwise and in place, one thread per element: dispatch(kernel,
// elementCount). The activation replaces what it was computed from, since
// nothing in a transformer reads a pre-activation twice.
//
// Gemma's feed-forward does not dispatch this — it is gated, so the activation
// and the multiply by the up projection happen together in GeGLU.h. What is
// left for this is any ungated activation, and being the one place the two
// forms are checked against a double-precision reference.
template <GeluForm form>
struct GeluProgram final : ComputeProgram
{
    GeluProgram() { compile(); }

    void define() override
    {
        auto at = threadId();
        write(values, at, activation(values[at]));
    }

    static Float activation(const Float& x)
    {
        if constexpr (form == GeluForm::Tanh)
            return tanhGelu(x);
        else
            return exactGelu(x);
    }

    Uniform<OutputBuffer> values;

    EACP_SHADER(values)
};

using Gelu = GeluProgram<GeluForm::Tanh>;
using ExactGelu = GeluProgram<GeluForm::Exact>;
} // namespace HF

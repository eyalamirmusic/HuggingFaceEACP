# NO_MAIN: DecoderWeights uploads every tensor through makeBuffer and a step is
# recorded into a real compute pass, so this module needs a device and
# therefore eacp's event loop, through the shared entry point in Tests/Support.
#
# Common.h reaches the Model and Kernels suites' own headers by relative path —
# the safetensors byte assembly, ScratchDirectory, the spread of inputs and the
# double-precision tanh GELU all already live there. Those are test headers
# rather than library ones, so nothing but this include path depends on them
# and hf-decoder's own dependencies are unchanged.
nano_add_executable(DecoderTests
        NO_MAIN
        SOURCES
        DecoderShapeTests.cpp
        DecoderTests.cpp
        DecoderWeightsTests.cpp
        GemmaDecoderTests.cpp
        ${HF_GPU_TEST_MAIN}
        TARGETS hf-decoder)

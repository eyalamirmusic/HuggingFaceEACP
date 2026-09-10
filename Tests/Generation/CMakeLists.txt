# NO_MAIN: a run uploads every weight through makeBuffer and records a command
# buffer per step, so this module needs a device and therefore eacp's event
# loop, through the shared entry point in Tests/Support.
#
# Common.h reaches Tests/Decoder's headers by relative path — the synthetic
# checkpoint, the scalar reference decoder and the DecoderRun that drives one
# step at a time all already live there, and through them Tests/Model's scratch
# directory and Tests/Kernels' readback plumbing. Those are test headers rather
# than library ones, so nothing but this include path depends on them and
# hf-generation's own dependencies are unchanged.
nano_add_executable(GenerationTests
        NO_MAIN
        SOURCES
        GenerationTests.cpp
        GemmaGenerationTests.cpp
        ${HF_GPU_TEST_MAIN}
        TARGETS hf-generation)

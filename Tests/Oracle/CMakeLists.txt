# llama.cpp as an in-process oracle — the tier above the scalar CPU references
# in Tests/Kernels. Behind HF_EACP_ENABLE_LLAMA_CPP in the root CMakeLists,
# which is also where llama.cpp itself is fetched, with every backend off so
# the reference is exactly ggml's CPU arithmetic.
#
# What it reads is gemma-2b.gguf, the F32 conversion of the same safetensors
# the build fetches — the same weights at the same width, which is what
# makes a logit comparison mean something rather than measuring a quantisation.
if (NOT HF_EACP_ENABLE_LLAMA_CPP)
    return()
endif ()

# NO_MAIN: DecoderOracleTests runs our decoder on the device, so this
# executable needs eacp's event loop through the shared entry point in
# Tests/Support the way Tests/Decoder does.
#
# hf-decoder is the model side of the logit comparison and hf-sampling the
# greedy draw over a row read back from it; hf-tokenizer and hf-model are the
# two the tokenizer and file comparisons need. llama is llama.cpp's own
# library, and it carries ggml on its interface.
#
# DecoderOracleTests.cpp reaches Tests/Kernels/Common.h by relative path, the
# way Tests/Decoder/Common.h does: the upload and readback plumbing is a test
# header rather than a library one, so nothing but that include depends on it
# and this target's own dependencies are unchanged.
nano_add_executable(OracleTests
        NO_MAIN
        SOURCES
        DecoderOracleTests.cpp
        LlamaOracle.cpp
        LlamaOracleTests.cpp
        TokenizerOracleTests.cpp
        ${HF_GPU_TEST_MAIN}
        TARGETS hf-tokenizer hf-model hf-decoder hf-sampling llama)

# Which llama.cpp this is, for the line the oracle prints once per run:
# llama_version() answers the 0.4.0 the source tree carries, which says far
# less than the build number the tag names.
target_compile_definitions(OracleTests PRIVATE
        HF_EACP_LLAMA_CPP_TAG="${HF_EACP_LLAMA_CPP_TAG}")

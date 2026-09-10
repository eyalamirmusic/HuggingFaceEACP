nano_add_executable(TokenizerTests
        SOURCES
        TokenizerTests.cpp
        TARGETS hf-tokenizer)

# gemma-2b is gated and its tokenizer.json is 17.5 MB, so it is a manual
# download and never a commit: the test that wants one returns early when it is
# absent, the way the GPU tests do without a device. Point it at a downloaded
# model with the GEMMA_MODEL_DIR environment variable at run time, or with this
# at configure time. plan.md's item 4 is the same directory.
#
# The committed fixture beside it is reached by absolute path rather than by
# guessing a working directory.
set(HF_EACP_GEMMA_MODEL_DIR "" CACHE PATH
        "A downloaded google/gemma-2b directory to run the tokenizer tests against")

target_compile_definitions(TokenizerTests PRIVATE
        HF_TOKENIZER_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/Fixtures"
        HF_EACP_GEMMA_MODEL_DIR="${HF_EACP_GEMMA_MODEL_DIR}")

# hf-model beside hf-tokenizer, for Support/GemmaModel.h alone: the tokenizer
# knows nothing about a checkpoint's layout, and ModelFiles is what spells the
# file names and reads the GEMMA_MODEL_DIR override. A test dependency only —
# hf-tokenizer's own is unchanged.
nano_add_executable(TokenizerTests
        SOURCES
        TokenizerTests.cpp
        TARGETS hf-tokenizer hf-model)

# The hand-written mini vocabulary this suite is mostly written against, reached
# by absolute path rather than by guessing a working directory. gemma-2b's own
# 17.5 MB tokenizer.json is a download and never a commit, and it arrives
# through HF_EACP_GEMMA_MODEL_DIR, which Tests/CMakeLists.txt defines for every
# target here.
target_compile_definitions(TokenizerTests PRIVATE
        HF_TOKENIZER_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/Fixtures")

add_library(hf-model STATIC)

# Miro rides along on eacp-core, which hf-core links through eacp-gpu, so
# <Miro/Json.h> needs no dependency of its own here.
target_link_libraries(hf-model PUBLIC hf-core)

target_sources(hf-model PRIVATE
        GemmaConfig.cpp
        GemmaTensors.cpp
        ModelFiles.cpp
        ModelIO.cpp
        SafeTensors.cpp
        ShardIndex.cpp
        ShardedTensors.cpp
        TensorLoader.cpp
        TensorType.cpp)

hf_set_default_target_setting(hf-model)
hf_enable_unity_build(hf-model)
hf_add_ide_sources(hf-model)

# Nothing is downloaded here, unlike WhisperEACP's model fetch: google/gemma-2b
# is gated, so a build cannot get it unauthenticated. The checkpoint is a
# manual download and GEMMA_MODEL_DIR is how it announces itself — see
# ModelFiles.h, and plan.md's fourth gap.

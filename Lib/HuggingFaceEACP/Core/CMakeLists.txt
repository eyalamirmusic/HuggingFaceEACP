add_library(hf-core STATIC)

# Lib/ is the include root, so a header of ours is spelled
# <HuggingFaceEACP/Core/Core.h>, beside eacp's own <eacp/...>.
target_include_directories(hf-core PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../)

# eacp-gpu rather than eacp-core, and PUBLIC: every layer above this one is a
# ComputeProgram holding GPU buffers, so eacp's GPU headers and its Metal /
# D3D12 link line are in our own signatures. The /Zc:preprocessor that
# EACP_SHADER needs on MSVC rides along the same way, being PUBLIC on eacp-gpu.
target_link_libraries(hf-core PUBLIC eacp-gpu)

target_sources(hf-core PRIVATE
        Utils/Version.cpp)

# The one place the library's own version lives is project(), so read it from
# there rather than keeping a second copy in the source that can drift.
target_compile_definitions(hf-core PRIVATE
        HF_EACP_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        HF_EACP_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        HF_EACP_VERSION_PATCH=${PROJECT_VERSION_PATCH})

hf_set_default_target_setting(hf-core)
hf_enable_unity_build(hf-core)
hf_add_ide_sources(hf-core)

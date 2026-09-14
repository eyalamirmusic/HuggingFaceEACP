add_library(hf-kernels STATIC)

# PUBLIC: a kernel's uniforms and its shape structs are in the signatures every
# module above holds one through, and eacp-gpu rides along the same way.
target_link_libraries(hf-kernels PUBLIC hf-core)

target_sources(hf-kernels PRIVATE
        Argmax.cpp
        Int8Blocks.cpp
        Kernels.cpp
        MultiQueryAttention.cpp
        RoPE.cpp)

hf_set_default_target_setting(hf-kernels)
hf_enable_unity_build(hf-kernels)
hf_add_ide_sources(hf-kernels)

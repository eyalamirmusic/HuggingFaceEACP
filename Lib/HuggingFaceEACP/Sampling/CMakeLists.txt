add_library(hf-sampling STATIC)

# hf-core and nothing else: a draw is scalar C++ over a row that has already
# been read back, so this module holds no GPU buffer and needs no vocabulary to
# name a TokenId — which is why that type lives in Core.
target_link_libraries(hf-sampling PUBLIC hf-core)

target_sources(hf-sampling PRIVATE
        Sampler.cpp)

hf_set_default_target_setting(hf-sampling)
hf_enable_unity_build(hf-sampling)
hf_add_ide_sources(hf-sampling)

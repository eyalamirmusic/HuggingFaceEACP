add_library(hf-generation STATIC)

# The three modules a run is assembled out of, all PUBLIC because all three are
# in this module's own signatures: a Gemma holds a Decoder and its shape, hands
# back a Tokenizer, and takes a SamplingOptions.
target_link_libraries(hf-generation PUBLIC hf-decoder hf-tokenizer hf-sampling)

target_sources(hf-generation PRIVATE
        Gemma.cpp)

hf_set_default_target_setting(hf-generation)
hf_enable_unity_build(hf-generation)
hf_add_ide_sources(hf-generation)

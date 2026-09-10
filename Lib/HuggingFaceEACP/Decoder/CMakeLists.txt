add_library(hf-decoder STATIC)

# Assembled out of Kernels/, fed by Model/: the op set is what a layer is
# recorded from, and the checkpoint is where its weights come from. PUBLIC
# because a DecoderShape holds a MultiQueryShape and a DecoderWeights holds
# TensorBuffers, both of which are in this module's own signatures.
target_link_libraries(hf-decoder PUBLIC hf-kernels hf-model)

target_sources(hf-decoder PRIVATE
        Decoder.cpp
        DecoderShape.cpp
        DecoderWeights.cpp)

hf_set_default_target_setting(hf-decoder)
hf_enable_unity_build(hf-decoder)
hf_add_ide_sources(hf-decoder)

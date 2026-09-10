add_library(hf-generation STATIC)

# The three modules a run is assembled out of, all PUBLIC because all three are
# in this module's own signatures: a Gemma holds a Decoder and its shape, hands
# back a Tokenizer, and takes a SamplingOptions.
target_link_libraries(hf-generation PUBLIC hf-decoder hf-tokenizer hf-sampling)

target_sources(hf-generation PRIVATE
        Gemma.cpp)

# Where the build's post-build copy put what this binary ships — see
# hf_copy_resources in CMake/HFResources.cmake. One answer per platform:
# CoreFoundation's on Apple, which is the bundle's Resources for a bundle and
# the executable's directory for anything else, and the executable's directory
# on Windows.
if (APPLE)
    target_sources(hf-generation PRIVATE
            ResourcesDirectory-Apple.cpp)

    target_link_libraries(hf-generation PRIVATE "-framework CoreFoundation")
elseif (WIN32)
    target_sources(hf-generation PRIVATE
            ResourcesDirectory-Windows.cpp)
endif ()

hf_set_default_target_setting(hf-generation)
hf_enable_unity_build(hf-generation)
hf_add_ide_sources(hf-generation)

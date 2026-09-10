# NO_MAIN, like every module that reaches the device: loading the bundled model
# and generating with it is the whole GPU stack, so these run inside eacp's
# event loop through the shared entry point in Tests/Support.
nano_add_executable(BundledTests
        NO_MAIN
        SOURCES
        BundledTests.cpp
        ${HF_GPU_TEST_MAIN}
        TARGETS hf-generation)

# The copy under test: the fetched files beside this binary after every link. A
# no-op when the build fetched no model, which is what the option's own value
# below is for — the one thing here that cannot be discovered at runtime.
hf_bundle_model(BundledTests)

target_compile_definitions(BundledTests PRIVATE
        HF_EACP_BUNDLES_MODEL=$<BOOL:${HF_EACP_FETCH_MODEL}>)

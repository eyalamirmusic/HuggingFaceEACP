set(kernel_test_sources
        AddTests.cpp
        ArgmaxTests.cpp
        EmbedTests.cpp
        GeGLUTests.cpp
        GeluTests.cpp
        Int8BlocksTests.cpp
        LinearTests.cpp
        MatMulTests.cpp
        MultiQueryAttentionTests.cpp
        RMSNormTests.cpp
        RoPETests.cpp
        SimdTiledMatMulTests.cpp
        SoftmaxTests.cpp
        TiledMatMulTests.cpp)

# NO_MAIN: these reach the GPU, so they run inside eacp's event loop through
# the shared entry point in Tests/Support.
nano_add_executable(KernelTests
        NO_MAIN
        SOURCES
        ${kernel_test_sources}
        ${HF_GPU_TEST_MAIN}
        TARGETS hf-kernels)

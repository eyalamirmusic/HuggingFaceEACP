# NO_MAIN: nearly nothing here reaches the device — every reader below stops at
# the CPU side and asserts against bytes this suite wrote itself. The exception
# is TensorUploadTests.cpp, and it is the exception for a reason: what a tensor
# is uploaded *as* became a claim worth testing once a BF16 weight started going
# to the device packed, and a buffer's storage and its size are only visible
# there. So the suite takes the shared entry point, and those tests alone return
# early when there is no device — the rest still run on a machine with none.
nano_add_executable(ModelTests
        NO_MAIN
        SOURCES
        CatalogueTests.cpp
        CheckpointTests.cpp
        ConfigTests.cpp
        SafeTensorsTests.cpp
        ShardedTensorsTests.cpp
        TensorUploadTests.cpp
        ${HF_GPU_TEST_MAIN}
        TARGETS hf-model)

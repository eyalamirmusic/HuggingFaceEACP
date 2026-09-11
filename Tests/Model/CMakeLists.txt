# NO_MAIN, for one test's sake: almost everything here stops at the CPU side,
# but which storage makeBuffer leaves a tensor in is this module's decision
# rather than eacp's — a packed one has to reach the device as the bytes the
# file holds — and asserting it means uploading one, which needs a device and
# therefore eacp's event loop through the shared entry point in Tests/Support.
# Every test that reaches the device returns early without one, so the suite
# still runs on a machine that has none.
nano_add_executable(ModelTests
        NO_MAIN
        SOURCES
        CatalogueTests.cpp
        CheckpointTests.cpp
        ConfigTests.cpp
        SafeTensorsTests.cpp
        ShardedTensorsTests.cpp
        ${HF_GPU_TEST_MAIN}
        TARGETS hf-model)

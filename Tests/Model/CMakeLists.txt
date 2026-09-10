# NanoTest's own main, unlike Tests/GPU: nothing here reaches the device. The
# loader's upload path is the one part that does, and a test that called it
# would be testing eacp's makeBuffer rather than this module's reading — so
# every assertion below stops at the CPU side, and the suite runs on a machine
# with no GPU at all.
nano_add_executable(ModelTests
        SOURCES
        CatalogueTests.cpp
        CheckpointTests.cpp
        ConfigTests.cpp
        SafeTensorsTests.cpp
        ShardedTensorsTests.cpp
        TARGETS hf-model)

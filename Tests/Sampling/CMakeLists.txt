# NanoTest's own main, unlike Tests/GPU: nothing here reaches the device. The
# sampler is the CPU sibling of the Argmax kernel and works on a row that has
# already been read back, so every assertion below is scalar C++ and the suite
# runs on a machine with no GPU at all.
nano_add_executable(SamplingTests
        SOURCES
        SamplerTests.cpp
        TARGETS hf-sampling)

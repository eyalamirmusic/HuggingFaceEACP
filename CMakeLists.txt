cmake_minimum_required(VERSION 3.31)

# Claimed before project(), not after: CMake's Darwin module reads the
# deployment target while it is probing the compiler, so a set() further down is
# silently ignored and every object ends up built against whatever SDK the build
# machine happens to have. 11.0 is eacp's own floor.
if (APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING
            "Minimum macOS version HuggingFaceEACP targets")
endif ()

project(HuggingFaceEACP VERSION 0.1.0 LANGUAGES C CXX)

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR
            "HuggingFaceEACP requires a 64-bit toolchain (x64 or ARM64), but "
            "the selected compiler targets 32-bit "
            "(CMAKE_SIZEOF_VOID_P=${CMAKE_SIZEOF_VOID_P}, "
            "arch='${CMAKE_SYSTEM_PROCESSOR}'). On Windows, configure from an "
            "x64 or ARM64 Developer prompt, or pass an explicit 64-bit "
            "target, e.g. -DCMAKE_CXX_FLAGS=--target=x86_64-pc-windows-msvc.")
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# eacp links its static libraries into runtime-loaded plugins and webview
# helpers, so every static lib in the tree — ours and the CPM-fetched ones — has
# to be position-independent or the shared-object link fails on Linux. Set here
# as well as there so it reaches the dependencies configured below.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (APPLE)
    # Objective-C++ only. Nothing in this tree or in eacp has a plain .m file —
    # the platform code on both sides is .mm — and enabling OBJC would only
    # change which language CMake hands such a file to. Enabled here rather than
    # left to eacp because a .mm source of ours is compiled in this project's
    # scope, where eacp's own enable_language() does not reach.
    enable_language(OBJCXX)

    # CMake already puts -lobjc on the link line of everything that pulls in
    # Objective-C. Xcode adds -fobjc-link-runtime on top for any target with its
    # own .mm sources, and ld warns about the duplicate.
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_LINK_OBJC_RUNTIME NO)
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

option(HF_EACP_UNITY_BUILD
        "Compile HuggingFaceEACP libraries as CMake unity builds" OFF)
option(HF_EACP_CI_BUILD "Mirror the exact flag set used by CI builds" OFF)

# There is no CI yet, deliberately — see CLAUDE.md. The switch is here anyway
# because it is what makes a configuration reproducible by name: one flag that
# turns on the unity builds of every project in the tree, so a build that
# behaves differently under unity can be reproduced without knowing which
# project exposes which option.
if (HF_EACP_CI_BUILD)
    set(HF_EACP_UNITY_BUILD ON CACHE BOOL
            "Compile HuggingFaceEACP libraries as CMake unity builds" FORCE)
    set(EACP_UNITY_BUILD ON CACHE BOOL
            "Compile eacp libraries as CMake unity builds" FORCE)
    set(MIRO_UNITY_BUILD ON CACHE BOOL
            "Enable CMake unity build for the Miro library" FORCE)
endif ()

include(HFTargetSetup)
include(HFResources)

if (PROJECT_IS_TOP_LEVEL)
    hf_default_setup()
endif ()

set(CMAKE_CXX_STANDARD 20)

find_package(eacp REQUIRED)

add_subdirectory(Lib)

option(HF_EACP_ENABLE_APPS "Build the HuggingFaceEACP apps"
        ${PROJECT_IS_TOP_LEVEL})

if (HF_EACP_ENABLE_APPS)
    add_subdirectory(Apps)
endif ()

option(HF_EACP_ENABLE_TESTS "Enable tests for HuggingFaceEACP"
        ${PROJECT_IS_TOP_LEVEL})

option(HF_EACP_ENABLE_LLAMA_CPP
        "Fetch llama.cpp and build the tests that compare against it" OFF)

# Off by default, and it only means anything with the tests in the tree, since
# Tests/Oracle is the only thing that links it. Both halves are expensive in a
# way the rest of this build is not: llama.cpp and ggml are minutes of compile,
# and the GGUF the tests read is a 10 GB manual download.
#
# Fetched here rather than inside Tests/Oracle because llama.cpp's targets are
# named globally, so a tree holds one build of it however many directories end
# up wanting one.
if (HF_EACP_ENABLE_TESTS AND HF_EACP_ENABLE_LLAMA_CPP)
    include(HFLlamaCpp)

    # Every backend off, so the oracle computes exactly ggml's own CPU path and
    # a disagreement with ours cannot be a backend's. GGML_BLAS is its own
    # switch and is not implied by GGML_ACCELERATE OFF: without it the
    # configure still finds Accelerate and links BLAS into the reference.
    #
    # GGML_METAL OFF also keeps Objective-C out of the tree — ggml-metal is
    # .m rather than .mm, and this project enables OBJCXX only.
    hf_add_llama_cpp(OPTIONS
            "GGML_METAL OFF"
            "GGML_BLAS OFF"
            "GGML_ACCELERATE OFF"
            "GGML_OPENMP OFF")
endif ()

if (HF_EACP_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif ()

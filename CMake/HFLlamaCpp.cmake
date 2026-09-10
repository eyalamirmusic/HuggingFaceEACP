# llama.cpp, fetched and added the one way it is safe to add it here: the
# oracle plan.md's step 3 checks our logits and our tokens against, running the
# F32 GGUF of google/gemma-2b through ggml's own CPU path.
#
# Not named LlamaCpp after anything in eacp/CMake: this project's
# CMAKE_MODULE_PATH entry is appended before eacp's, so a module here that
# shared a name with one of theirs would shadow it. Same reason for the `hf_`
# prefix on the function — see HFTargetSetup.cmake.
#
#   hf_add_llama_cpp([OPTIONS <option> ...])
#
# adds the `llama` target (and ggml under it) at b10900, static, with its
# tests, tools, examples, server and common utilities off. OPTIONS are
# forwarded to CPM as llama.cpp's and ggml's own switches, which is where a
# caller names the backends it wants: Tests/Oracle turns every one of them off,
# so the reference is exactly ggml's CPU arithmetic and a disagreement cannot
# be a backend's.
#
# The tag is a llama.cpp release tag — every master commit gets one, and a
# build number is the only version llama.cpp publishes. Pinned rather than
# tracking master because the oracle's whole job is to be the fixed side of a
# comparison, and because llama_context_params has gained and lost fields
# repeatedly (logits_all is gone at this tag; per-token logits are a flag on
# the batch).

include(CPM)

set(HF_EACP_LLAMA_CPP_TAG "b10900" CACHE INTERNAL
        "The llama.cpp release tag the oracle is pinned to")

function(hf_add_llama_cpp)
    cmake_parse_arguments(ARG "" "" "OPTIONS" ${ARGN})

    # llama.cpp forces CMAKE_BUILD_TYPE to Release, in the cache, whenever it
    # finds it empty — and the cache is the one scope an add_subdirectory does
    # not contain, so a configure that named no build type would come back
    # Release for the whole tree. Put back below.
    set(buildTypeBefore "${CMAKE_BUILD_TYPE}")

    # The other leak WhisperEACP's whisper.cpp module has to undo does not
    # apply here, and it was checked rather than assumed: hf_default_setup()
    # puts _LIBCPP_REMOVE_TRANSITIVE_INCLUDES on the root directory, every
    # directory added under it inherits it, and all 216 of llama.cpp's and
    # ggml's translation units compile with it on at this tag. If a later tag
    # stops doing so — libc++ stops handing a translation unit the headers it
    # only reached transitively, so the error names a missing declaration
    # rather than a missing include — clear COMPILE_DEFINITIONS around the
    # CPMAddPackage below and put it back afterwards, the way
    # WhisperEACP/CMake/WhisperCpp.cmake does.

    # BUILD_SHARED_LIBS is named because llama.cpp defaults it ON everywhere
    # but MinGW and Emscripten, and a shared ggml is a dylib the test
    # executable would have to find at run time. It holds as a plain option
    # because CPM sets CMAKE_POLICY_DEFAULT_CMP0077 to NEW, so option() defers
    # to the variable rather than writing a cache entry over it — llama.cpp's
    # own cmake_minimum_required of 3.14 would otherwise leave that policy
    # unset. None of these reaches the cache, which is what keeps them out of
    # the rest of the tree.
    #
    # The six build-artifact switches already default to LLAMA_STANDALONE,
    # which is OFF for a subdirectory; they are named anyway so the build here
    # does not change if llama.cpp changes its mind about that default.
    #
    # LLAMA_CURL is gone at this tag: the HTTP dependency now hangs off
    # LLAMA_OPENSSL and reaches the build only through llama-common, which is
    # off, so this is belt as well as braces.
    #
    # LLAMA_BUILD_IS_DEV is on by default and only appends "-dev" to what
    # llama_version() answers. This is a tagged build, so it says so.
    CPMAddPackage(
            NAME llama-cpp
            GITHUB_REPOSITORY ggml-org/llama.cpp
            GIT_TAG ${HF_EACP_LLAMA_CPP_TAG}
            SYSTEM YES
            EXCLUDE_FROM_ALL YES
            OPTIONS
            "BUILD_SHARED_LIBS OFF"
            "LLAMA_BUILD_IS_DEV OFF"
            "LLAMA_BUILD_COMMON OFF"
            "LLAMA_BUILD_TESTS OFF"
            "LLAMA_BUILD_TOOLS OFF"
            "LLAMA_BUILD_EXAMPLES OFF"
            "LLAMA_BUILD_SERVER OFF"
            "LLAMA_BUILD_APP OFF"
            "LLAMA_OPENSSL OFF"
            ${ARG_OPTIONS})

    if (NOT CMAKE_BUILD_TYPE STREQUAL buildTypeBefore)
        set(CMAKE_BUILD_TYPE "${buildTypeBefore}" CACHE STRING
                "Build type" FORCE)
    endif ()
endfunction()

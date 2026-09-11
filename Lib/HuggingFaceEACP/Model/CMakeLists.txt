add_library(hf-model STATIC)

# Miro rides along on eacp-core, which hf-core links through eacp-gpu, so
# <Miro/Json.h> needs no dependency of its own here.
target_link_libraries(hf-model PUBLIC hf-core)

target_sources(hf-model PRIVATE
        GemmaConfig.cpp
        GemmaTensors.cpp
        ModelFiles.cpp
        ModelIO.cpp
        SafeTensors.cpp
        ShardIndex.cpp
        ShardedTensors.cpp
        TensorLoader.cpp
        TensorType.cpp)

hf_set_default_target_setting(hf-model)
hf_enable_unity_build(hf-model)
hf_add_ide_sources(hf-model)

# The checkpoint is a download, never a commit — 5 GB of weights has no
# business in git. On, CPM fetches the four files this module reads, and the
# directory they are assembled into reaches the tests as a compile definition;
# it is also what hf_bundle_model() below copies beside a binary. On by default
# because Generate's no-argument form and Tests/Bundled both need it, and off
# is for a build that only wants the library or has a checkout of its own.
option(HF_EACP_FETCH_MODEL
        "Download the gemma-2b model files, and copy them beside every binary \
that calls hf_bundle_model()" ON)

# Set even when the fetch is off, so a test always compiles against a real path
# and the tests that need a file skip on its absence rather than on an #ifdef.
# Cached because a plain set() would not survive out of this directory.
set(HF_EACP_GEMMA_MODEL_DIR "${CMAKE_BINARY_DIR}/gemma-2b"
        CACHE INTERNAL "Where the fetched gemma-2b files are assembled")

# unsloth/gemma-2b rather than google/gemma-2b, and pinned to a commit rather
# than to main. Google's repo is gated: an unauthenticated download of any file
# in it answers HTTP 401, so CPM cannot fetch it and a build would need an
# account that accepted the terms plus a token. unsloth's is an ungated mirror
# of the same bf16 weights — one unsharded model.safetensors, the same
# tokenizer, the same config numbers — and pinning the revision is what keeps
# the hashes below valid when the mirror moves.
#
# The one file the mirror cannot serve is gemma-2b.gguf, which Tests/Oracle
# reads. It is not a download at all any more: llama.cpp's own converter writes
# it from the files fetched here, and the GEMMA_MODEL_DIR override that
# ModelFiles.h documents is how a machine that has run that says where it went.
set(GEMMA_2B_URL "https://huggingface.co/unsloth/gemma-2b/resolve/\
7ac9d201a57c8cdc6f939069fc0f044c60197a4a")

# Each file separately, because CPM keys a package on its name and a HF repo is
# not an archive: DOWNLOAD_NO_EXTRACT stops FetchContent trying to unpack the
# download, and DOWNLOAD_ONLY says there is no CMakeLists to add. The hashes
# are the ones HuggingFace serves for that revision, so a corrupted or swapped
# file fails the configure rather than the decode.
function(hf_fetch_model_file package_name file_name file_hash)
    CPMAddPackage(
            NAME ${package_name}
            URL ${GEMMA_2B_URL}/${file_name}
            URL_HASH SHA256=${file_hash}
            DOWNLOAD_NO_EXTRACT YES
            DOWNLOAD_ONLY YES)

    # CPM leaves the downloads in separate _deps directories, so link them into
    # one so a caller names a model directory rather than four unrelated paths.
    # COPY_ON_ERROR covers a Windows account without the privilege to make a
    # symlink.
    file(CREATE_LINK
            "${${package_name}_SOURCE_DIR}/${file_name}"
            "${HF_EACP_GEMMA_MODEL_DIR}/${file_name}"
            COPY_ON_ERROR SYMBOLIC)

    list(APPEND HF_EACP_MODEL_FILES "${HF_EACP_GEMMA_MODEL_DIR}/${file_name}")
    set(HF_EACP_MODEL_FILES "${HF_EACP_MODEL_FILES}" PARENT_SCOPE)
endfunction()

set(HF_EACP_MODEL_FILES "")

if (HF_EACP_FETCH_MODEL)
    file(MAKE_DIRECTORY "${HF_EACP_GEMMA_MODEL_DIR}")

    hf_fetch_model_file(gemma-2b-config config.json
            aa31dbdc535024275edec7b3faa4822759c827147a1d3b3ea5127c1b5527ced3)

    hf_fetch_model_file(gemma-2b-generation-config generation_config.json
            4358ba9459393634465be72b3428013c0e946aef4c18f50f26f94443070d3f8a)

    hf_fetch_model_file(gemma-2b-weights model.safetensors
            d00d0537c1c6175a8cb10bf8263bc3af94e06a7ad92962a6c051f45c4eb4e68a)

    # The vocabulary. Tokenizer/ reads it, and it belongs in the same directory
    # as the three files above rather than behind a second switch.
    hf_fetch_model_file(gemma-2b-tokenizer tokenizer.json
            7da53ca29fb16f6b2489482fc0bc6a394162cdab14d12764a1755ebc583fea79)
endif ()

# The four fetched files, as a list a directory processed after this one can
# read. Empty when the fetch is off, which is what makes hf_bundle_model() a
# no-op in that build.
set(HF_EACP_MODEL_FILES "${HF_EACP_MODEL_FILES}"
        CACHE INTERNAL "The fetched gemma-2b files, or empty")

# The same four files again, copied beside whatever asks for them after every
# link — into Contents/Resources of a bundle on macOS, next to the executable
# otherwise — which is why it lives here rather than beside Gemma::loadBundled
# that reads them: this function is about the fetch above, and nothing but the
# file list connects the two.
#
# The directory name is load-bearing: GemmaModel is what
# Gemma::bundledModelDirectoryName spells, so the runtime finds the copy by the
# same name the build gave it, and Gemma.cpp's error message tells the caller
# to call `hf_bundle_model`.
#
# Defined whether or not the fetch is on, so a consumer calls it
# unconditionally and Gemma::hasBundledModel() is what answers — the same shape
# as a test skipping on a missing file rather than on an #ifdef.
function(hf_bundle_model target)
    if (NOT HF_EACP_MODEL_FILES)
        return()
    endif ()

    hf_copy_resources(${target}
            DESTINATION GemmaModel
            FILES ${HF_EACP_MODEL_FILES})
endfunction()

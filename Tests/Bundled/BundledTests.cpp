#include <HuggingFaceEACP/Generation/Gemma.h>
#include <HuggingFaceEACP/Generation/ResourcesDirectory.h>
#include <HuggingFaceEACP/Model/ModelIO.h>

#include <eacp/GPU/GPU.h>

#include <NanoTest/NanoTest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

// What hf_bundle_model(BundledTests) actually put beside this binary: that the
// directory the runtime resolves is a real one, that the files are in it, that
// they are the ones the fetch fetched, and that a runtime loaded out of them
// generates.
//
// The fetched directory is named here rather than through
// Support/GemmaModel.h, because that one answers the GEMMA_MODEL_DIR override
// first and the override is not what the build copied.

using namespace nano;
using namespace HF;
using namespace eacp::GPU;

namespace
{
constexpr auto bundlesModel = HF_EACP_BUNDLES_MODEL != 0;

// Small enough that a first run is seconds rather than a minute, and long
// enough that a continuation has somewhere to go.
constexpr auto smokeTokens = 8;

constexpr auto capitalPrompt = "The capital of France is";

// The three that are small enough to compare byte for byte. The weights are
// 5 GB, so they are compared by size instead — reading them twice to memcmp
// them would cost more than every other test here together.
constexpr const char* jsonFileNames[] = {ModelFileNames::config,
                                         ModelFileNames::generationConfig,
                                         ModelFileNames::tokenizerJson};

std::filesystem::path fetchedDirectory()
{
    return std::filesystem::path {HF_EACP_GEMMA_MODEL_DIR};
}

bool isFile(const std::filesystem::path& path)
{
    auto error = std::error_code {};
    return std::filesystem::is_regular_file(path, error);
}

bool matchesFetchedFile(const char* name)
{
    const auto bundled =
        ModelIO::readFileBytes(Gemma::bundledModelDirectory() / name);
    const auto fetched = ModelIO::readFileBytes(fetchedDirectory() / name);

    return bundled.size() == fetched.size()
           && std::memcmp(
                  bundled.data(), fetched.data(), (std::size_t) fetched.size())
                  == 0;
}

std::uintmax_t sizeOf(const std::filesystem::path& path)
{
    auto error = std::error_code {};
    return std::filesystem::file_size(path, error);
}
} // namespace

// The platform half, on its own and never skipping: whatever the build copied
// or did not, the directory the runtime would look in has to exist, since it
// is the one the executable itself is in — or its bundle's Resources.
auto tResourcesDirectoryIsReal = test("Bundled/resourcesDirectoryIsReal") = []
{
    auto error = std::error_code {};

    check(!resourcesDirectory().empty());
    check(std::filesystem::is_directory(resourcesDirectory(), error));
    check(Gemma::bundledModelDirectory().filename()
          == Gemma::bundledModelDirectoryName);
};

// The build copied a model or it did not, and when it did the runtime has to
// find it: the copy is a POST_BUILD step of this very target, so a step that
// quietly stopped running would leave a binary that built and shipped nothing.
// Only that direction is asserted — a build reconfigured from on to off leaves
// the earlier copy where it was, and a test that read that as a failure would
// be reporting a stale build directory, not a bug.
auto tBundledModelIsThereWhenTheBuildCopiedOne =
    test("Bundled/modelIsThereWhenTheBuildCopiedOne") = []
{
    if (!bundlesModel)
        return;

    check(Gemma::hasBundledModel());
};

// What the copy put beside the binary is what the fetch fetched: byte for byte
// for the three JSON files, and by size for the weights.
auto tBundledFilesMatchTheFetch = test("Bundled/filesMatchTheFetch") = []
{
    if (!bundlesModel || !isFile(fetchedDirectory() / ModelFileNames::config))
        return;

    for (const auto* name: jsonFileNames)
        check(matchesFetchedFile(name));

    const auto bundledWeights =
        Gemma::bundledModelDirectory() / ModelFileNames::weights;
    const auto fetchedWeights = fetchedDirectory() / ModelFileNames::weights;

    check(isFile(bundledWeights));
    check(sizeOf(bundledWeights) == sizeOf(fetchedWeights));
    check(sizeOf(bundledWeights) > 0);
};

// End to end out of the directory beside the binary: the weights are mapped
// from the copy, and nothing is read from anywhere else.
//
// What is asserted is only what is certain of a base completion model at eight
// tokens — that it produced some, that every id is one the embedding holds,
// and that they decode to something. Which words come back is
// Tests/Generation's assertion, not this one; this is about the copy.
auto tBundledModelGenerates = test("Bundled/modelGenerates") = []
{
    if (!bundlesModel || !Device::shared().isValid() || !Gemma::hasBundledModel())
        return;

    auto gemma = Gemma {};
    gemma.loadBundled();
    gemma.prepare();

    gemma.setMaximumTokens(smokeTokens);

    const auto tokens = gemma.generateFromText(capitalPrompt);
    const auto text = gemma.textForTokens(tokens);

    check(tokens.size() > 0);
    check(tokens.size() <= smokeTokens);

    for (const auto token: tokens)
        check(token >= 0 && token < gemma.config().vocabularySize);

    check(!text.empty());

    std::cout << "  \"" << capitalPrompt << "\" ->\"" << text << "\"\n";
};

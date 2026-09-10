#pragma once

#include <HuggingFaceEACP/Model/Model.h>

#include <NanoTest/NanoTest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>
#include <string_view>

// A safetensors file is small enough to write by hand, so the bulk of this
// suite needs no download at all: every header below is assembled here, byte
// for byte, and parsed straight back. The sharded cases go one step further
// and write a whole model directory — two shards and an index — into a
// temporary directory that removes itself.
namespace HF::Testing
{
inline Vector<std::uint8_t> assembleWithLength(std::uint64_t declaredLength,
                                               std::string_view header,
                                               const Vector<std::uint8_t>& blob)
{
    auto file = Vector<std::uint8_t> {};
    file.resize(8);
    std::memcpy(file.data(), &declaredLength, sizeof(declaredLength));

    for (auto character: header)
        file.add(static_cast<std::uint8_t>(character));

    for (auto byte: blob)
        file.add(byte);

    return file;
}

inline Vector<std::uint8_t> assemble(std::string_view header,
                                     const Vector<std::uint8_t>& blob = {})
{
    return assembleWithLength(header.size(), header, blob);
}

template <typename Element>
Vector<std::uint8_t> toBytes(std::initializer_list<Element> values)
{
    auto bytes = Vector<std::uint8_t> {};
    bytes.resize(static_cast<int>(values.size() * sizeof(Element)));

    auto* out = bytes.data();

    for (auto value: values)
    {
        std::memcpy(out, &value, sizeof(Element));
        out += sizeof(Element);
    }

    return bytes;
}

// A directory of the test's own, removed when the test that made it ends, so
// nothing is left behind whether it passed or threw. Each test names its own,
// and the name is what keeps two of them out of each other's way.
class ScratchDirectory
{
public:
    explicit ScratchDirectory(std::string_view name)
        : directoryPath(std::filesystem::temp_directory_path()
                        / ("hf-eacp-" + std::string {name}))
    {
        auto error = std::error_code {};
        std::filesystem::remove_all(directoryPath, error);
        std::filesystem::create_directories(directoryPath, error);
    }

    ScratchDirectory(const ScratchDirectory&) = delete;
    ScratchDirectory& operator=(const ScratchDirectory&) = delete;

    ~ScratchDirectory()
    {
        auto error = std::error_code {};
        std::filesystem::remove_all(directoryPath, error);
    }

    const std::filesystem::path& path() const { return directoryPath; }

    std::filesystem::path write(std::string_view name,
                                const Vector<std::uint8_t>& bytes) const
    {
        const auto filePath = directoryPath / name;
        auto out = std::ofstream {filePath, std::ios::binary | std::ios::trunc};
        out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());

        return filePath;
    }

    std::filesystem::path writeText(std::string_view name,
                                    std::string_view text) const
    {
        const auto filePath = directoryPath / name;
        auto out = std::ofstream {filePath, std::ios::binary | std::ios::trunc};
        out.write(text.data(), static_cast<std::streamsize>(text.size()));

        return filePath;
    }

private:
    std::filesystem::path directoryPath;
};

template <typename Body>
bool throwsModelError(Body&& body)
{
    try
    {
        body();
    }
    catch (const ModelError&)
    {
        return true;
    }
    catch (...)
    {
        return false;
    }

    return false;
}

inline bool nearlyEqual(float actual, float expected, float tolerance = 1.0e-6f)
{
    const auto difference = actual - expected;
    return (difference < 0 ? -difference : difference) <= tolerance;
}
} // namespace HF::Testing

#include "ModelIO.h"

#include <fstream>
#include <limits>

namespace HF::ModelIO
{
namespace
{
std::string describe(std::string_view what, std::string_view key)
{
    return std::string {what} + " field '" + std::string {key} + "'";
}
} // namespace

Vector<std::uint8_t> readFileBytes(const std::filesystem::path& path)
{
    auto stream = std::ifstream {path, std::ios::binary | std::ios::ate};

    if (!stream)
        throw ModelError {"cannot open '" + path.string() + "'"};

    const auto byteCount = static_cast<std::streamoff>(stream.tellg());

    if (byteCount < 0)
        throw ModelError {"cannot size '" + path.string() + "'"};

    // EA::Vector indexes with int. Everything read whole through here is a
    // JSON file — a config or a shard index — and the weights are mapped
    // rather than read, so this limit never meets a tensor.
    if (byteCount > std::numeric_limits<int>::max())
        throw ModelError {"'" + path.string() + "' is larger than 2 GB"};

    auto bytes = Vector<std::uint8_t> {};
    bytes.resize(static_cast<int>(byteCount));
    stream.seekg(0);

    if (byteCount > 0
        && !stream.read(reinterpret_cast<char*>(bytes.data()), byteCount))
        throw ModelError {"cannot read '" + path.string() + "'"};

    return bytes;
}

std::string_view textOf(Span<const std::uint8_t> bytes)
{
    return {reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::size_t>(bytes.size())};
}

Miro::Json::Value parseObject(std::string_view text, std::string_view what)
{
    auto parsed = Miro::Json::Value {};

    try
    {
        parsed = Miro::Json::parse(text);
    }
    catch (const Miro::Json::ParseError& error)
    {
        throw ModelError {std::string {what}
                          + " is not valid JSON: " + error.what()};
    }

    if (!parsed.isObject())
        throw ModelError {std::string {what} + " is not a JSON object"};

    return parsed;
}

Miro::Json::Value parseObjectFile(const std::filesystem::path& path,
                                  std::string_view what)
{
    const auto bytes = readFileBytes(path);
    return parseObject(textOf(bytes), what);
}

const Miro::Json::Value& field(const Miro::Json::Object& object,
                               std::string_view key,
                               std::string_view what)
{
    if (const auto* found = Miro::Json::find(object, key))
        return *found;

    throw ModelError {describe(what, key) + " is missing"};
}

std::int64_t asInteger(const Miro::Json::Value& value, std::string_view what)
{
    if (!value.isNumber())
        throw ModelError {std::string {what} + " is not a number"};

    try
    {
        return value.asInteger();
    }
    catch (const Miro::Json::AccessError&)
    {
        throw ModelError {std::string {what} + " is not a whole 64-bit integer"};
    }
}

double asDouble(const Miro::Json::Value& value, std::string_view what)
{
    if (!value.isNumber())
        throw ModelError {std::string {what} + " is not a number"};

    return value.asNumber();
}

std::int64_t integerField(const Miro::Json::Object& object,
                          std::string_view key,
                          std::string_view what)
{
    return asInteger(field(object, key, what), describe(what, key));
}

int intField(const Miro::Json::Object& object,
             std::string_view key,
             std::string_view what)
{
    const auto value = integerField(object, key, what);

    if (value < std::numeric_limits<int>::min()
        || value > std::numeric_limits<int>::max())
        throw ModelError {describe(what, key) + " does not fit in an int"};

    return static_cast<int>(value);
}

int intFieldOr(const Miro::Json::Object& object,
               std::string_view key,
               int fallback,
               std::string_view what)
{
    if (Miro::Json::find(object, key) == nullptr)
        return fallback;

    return intField(object, key, what);
}

double doubleFieldOr(const Miro::Json::Object& object,
                     std::string_view key,
                     double fallback,
                     std::string_view what)
{
    const auto* found = Miro::Json::find(object, key);

    if (found == nullptr)
        return fallback;

    return asDouble(*found, describe(what, key));
}

bool boolFieldOr(const Miro::Json::Object& object,
                 std::string_view key,
                 bool fallback,
                 std::string_view what)
{
    const auto* found = Miro::Json::find(object, key);

    if (found == nullptr)
        return fallback;

    if (!found->isBool())
        throw ModelError {describe(what, key) + " is not a boolean"};

    return found->asBool();
}

std::string stringFieldOr(const Miro::Json::Object& object,
                          std::string_view key,
                          std::string_view fallback,
                          std::string_view what)
{
    const auto* found = Miro::Json::find(object, key);

    if (found == nullptr)
        return std::string {fallback};

    if (!found->isString())
        throw ModelError {describe(what, key) + " is not a string"};

    return found->asString();
}
} // namespace HF::ModelIO

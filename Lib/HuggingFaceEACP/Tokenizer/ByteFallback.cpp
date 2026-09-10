#include "ByteFallback.h"

#include <Miro/Unicode.h>

namespace HF::ByteFallback
{
namespace
{
constexpr auto nameLength = std::size_t {6};
constexpr auto namePrefix = std::string_view {"<0x"};
constexpr auto hexDigits = std::string_view {"0123456789ABCDEF"};

// U+FFFD REPLACEMENT CHARACTER, spelled as bytes so the source encoding
// cannot change what it means.
constexpr auto replacementCharacter = std::string_view {"\xef\xbf\xbd"};

int valueOfHexDigit(char character)
{
    if (character >= '0' && character <= '9')
        return character - '0';

    if (character >= 'A' && character <= 'F')
        return character - 'A' + 10;

    if (character >= 'a' && character <= 'f')
        return character - 'a' + 10;

    return -1;
}

bool isValidUtf8(std::string_view text)
{
    for (auto position = std::size_t {}; position < text.size();)
    {
        const auto decoded = Miro::Unicode::decodeUtf8(text, position);

        if (!decoded.valid)
            return false;

        position += (std::size_t) decoded.byteLength;
    }

    return true;
}
} // namespace

std::string nameForByte(std::uint8_t byte)
{
    auto name = std::string {namePrefix};

    name += hexDigits[byte >> 4];
    name += hexDigits[byte & 0xF];
    name += '>';

    return name;
}

int byteForName(std::string_view tokenText)
{
    if (tokenText.size() != nameLength || !tokenText.starts_with(namePrefix)
        || tokenText.back() != '>')
        return -1;

    const auto high = valueOfHexDigit(tokenText[3]);
    const auto low = valueOfHexDigit(tokenText[4]);

    if (high < 0 || low < 0)
        return -1;

    return high * 16 + low;
}

void appendRunAsText(std::string& text, std::string_view byteRun)
{
    if (isValidUtf8(byteRun))
    {
        text += byteRun;
        return;
    }

    for (auto index = std::size_t {}; index < byteRun.size(); ++index)
        text += replacementCharacter;
}
} // namespace HF::ByteFallback

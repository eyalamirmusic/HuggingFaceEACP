#include "Metaspace.h"

namespace HF
{
std::string Metaspace::normalize(std::string_view text) const
{
    auto normalized = std::string {};
    normalized.reserve(text.size() + replacement.size());

    if (prependsDummyPrefix)
        normalized += replacement;

    for (auto character: text)
    {
        if (character == ' ')
            normalized += replacement;
        else
            normalized += character;
    }

    return normalized;
}

Vector<std::string_view> Metaspace::split(std::string_view normalizedText) const
{
    auto words = Vector<std::string_view> {};

    if (normalizedText.empty())
        return words;

    if (!splitsBeforeReplacement || replacement.empty())
    {
        words.add(normalizedText);
        return words;
    }

    auto wordStart = std::size_t {};

    // From one rather than from zero: a replacement at the very start opens
    // the first word instead of closing an empty one.
    auto position = normalizedText.find(replacement, 1);

    while (position != std::string_view::npos)
    {
        words.add(normalizedText.substr(wordStart, position - wordStart));
        wordStart = position;
        position = normalizedText.find(replacement, position + replacement.size());
    }

    words.add(normalizedText.substr(wordStart));
    return words;
}

std::string Metaspace::restoreSpaces(std::string_view tokenText) const
{
    if (replacement.empty())
        return std::string {tokenText};

    auto text = std::string {};
    text.reserve(tokenText.size());

    auto position = std::size_t {};

    while (position < tokenText.size())
    {
        const auto found = tokenText.find(replacement, position);

        if (found == std::string_view::npos)
            break;

        text += tokenText.substr(position, found - position);
        text += ' ';
        position = found + replacement.size();
    }

    text += tokenText.substr(position);
    return text;
}
} // namespace HF

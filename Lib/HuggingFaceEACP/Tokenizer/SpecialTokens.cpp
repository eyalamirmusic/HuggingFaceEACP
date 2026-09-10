#include "SpecialTokens.h"

namespace HF
{
void assignSpecialToken(SpecialTokens& tokens,
                        std::string_view content,
                        TokenId token)
{
    if (content == "<pad>")
        tokens.padding = token;
    else if (content == "<eos>")
        tokens.endOfSequence = token;
    else if (content == "<bos>")
        tokens.beginningOfSequence = token;
    else if (content == "<unk>")
        tokens.unknown = token;
    else if (content == "<start_of_turn>")
        tokens.startOfTurn = token;
    else if (content == "<end_of_turn>")
        tokens.endOfTurn = token;
}
} // namespace HF

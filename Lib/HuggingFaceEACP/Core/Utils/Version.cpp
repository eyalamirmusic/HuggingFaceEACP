#include "Version.h"

#include <eacp/Core/Utils/Strings.h>

namespace HF
{
std::string toString(const Version& version)
{
    return eacp::Strings::concat(
        version.major, '.', version.minor, '.', version.patch);
}

Version getLibraryVersion()
{
    return {HF_EACP_VERSION_MAJOR, HF_EACP_VERSION_MINOR, HF_EACP_VERSION_PATCH};
}
} // namespace HF

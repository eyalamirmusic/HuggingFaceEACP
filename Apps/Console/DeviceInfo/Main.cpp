#include <HuggingFaceEACP/HuggingFaceEACP.h>

#include <eacp/Core/App/App.h>
#include <eacp/GPU/GPU.h>

#include <cstdio>

// Bring-up check: the half this project stands on — a device to dispatch
// compute on — reporting what it gives us to plan against. Worth a real binary
// rather than knowledge, because every number below is a property of the eacp
// that was fetched and the machine it came up on, not of this project.

using namespace eacp;
using namespace eacp::GPU;

namespace
{
void printGpuInfo()
{
    auto& device = Device::shared();

    std::printf("GPU\n");

    if (!device.isValid())
    {
        std::printf("  no device available - nothing here can run\n");
        return;
    }

    std::printf("  device                    %s\n", device.name().c_str());
    std::printf("  1D thread group width     %d\n", ComputePass::threadGroupWidth);
    std::printf("  2D thread group size      %d x %d\n",
                ComputePass::threadGroupSize2D,
                ComputePass::threadGroupSize2D);
    std::printf("  3D thread group size      %d x %d x %d\n",
                ComputePass::threadGroupSize3D,
                ComputePass::threadGroupSize3D,
                ComputePass::threadGroupSize3D);
    std::printf("  storage buffer slots      %d\n", ComputePass::maxBufferSlots);
    std::printf("  first uniform slot        %d\n", ComputePass::uniformBase);

    // What a per-tensor view into one shard-wide buffer has to be rounded to,
    // which is the number the zero-copy weight loading in plan.md turns on.
    std::printf("  ranged bind alignment     %d bytes\n",
                device.storageBufferOffsetAlignment());
    std::printf("  block-compressed textures %s\n",
                device.supportsBlockCompression() ? "yes" : "no");
}

void printDeviceInfo()
{
    std::printf("HuggingFaceEACP %s\n\n",
                HF::toString(HF::getLibraryVersion()).c_str());

    printGpuInfo();
}
} // namespace

int main()
{
    return Apps::run(printDeviceInfo);
}

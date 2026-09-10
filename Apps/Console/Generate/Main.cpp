#include <HuggingFaceEACP/HuggingFaceEACP.h>

#include <eacp/Core/App/App.h>
#include <eacp/GPU/GPU.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <string_view>

// The whole runtime as a binary: a prompt in, a continuation streamed out as
// each token arrives, and what the two halves of the run cost on stderr.
//
// The model ships with the binary: the build fetches gemma-2b and copies it
// beside this executable, and a run with no arguments but a prompt loads that.
// GEMMA_MODEL_DIR is the explicit override, and it wins when set — which is
// how a run is pointed at Google's own gated download, the one that carries
// gemma-2b.gguf beside the safetensors for the oracle tests.
//
// Inside eacp::Apps::run for the reason every GPU-touching thing here is: the
// Metal backend is written against the run loop and autorelease pool that owns,
// and DeviceInfo next door is the same shape.

using namespace eacp;

namespace
{
constexpr auto usage =
    "usage: Generate \"<prompt>\" [options]\n"
    "\n"
    "  --max-tokens N   how many tokens to generate (default: to the window)\n"
    "  --temperature T  0 is greedy on the device, above it draws on the CPU\n"
    "  --top-k K        keep only the K largest logits (default: all)\n"
    "  --top-p P        keep the smallest set reaching mass P (default: 1)\n"
    "  --seed S         the draw's seed, so a run is reproducible\n"
    "\n"
    "The model is the gemma-2b the build copied beside this binary. Set\n"
    "GEMMA_MODEL_DIR to run a checkpoint of your own instead: a directory\n"
    "with config.json, the weights as one safetensors or as shards behind an\n"
    "index, and tokenizer.json. It wins over the copy whenever it is set.\n";

std::string missingModel()
{
    return "this binary ships no model, so there is nothing to run.\n"
           "Reconfigure with -DHF_EACP_FETCH_MODEL=ON — the default — so the\n"
           "build fetches gemma-2b and copies it beside the binary, or point\n"
           "GEMMA_MODEL_DIR at a checkpoint of your own:\n\n"
           "  export GEMMA_MODEL_DIR=/path/to/gemma-2b\n\n";
}

double secondsSince(std::chrono::steady_clock::time_point start)
{
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
}

// What the command line asked for, resolved before the device is opened so a
// run with no model says so rather than reporting a missing GPU first.
struct Request
{
    std::string prompt;
    int maximumTokens = 0;
    HF::SamplingOptions sampling = HF::SamplingOptions::greedy();
};

bool readsValue(std::string_view argument,
                std::string_view name,
                const HF::Vector<std::string>& arguments,
                int& index,
                std::string& value)
{
    if (argument != name)
        return false;

    if (index + 1 >= arguments.size())
        return false;

    value = arguments[++index];

    return true;
}

// Each argument is one of the five flags and its value, or the prompt. Two
// prompts is a quoting mistake rather than a second run, so it is refused.
bool parse(const HF::Vector<std::string>& arguments, Request& request)
{
    auto prompts = 0;

    for (auto index = 1; index < arguments.size(); ++index)
    {
        const auto argument = std::string_view {arguments[index]};
        auto value = std::string {};

        if (readsValue(argument, "--max-tokens", arguments, index, value))
            request.maximumTokens = std::atoi(value.c_str());
        else if (readsValue(argument, "--temperature", arguments, index, value))
            request.sampling.temperature = (float) std::atof(value.c_str());
        else if (readsValue(argument, "--top-k", arguments, index, value))
            request.sampling.topK = std::atoi(value.c_str());
        else if (readsValue(argument, "--top-p", arguments, index, value))
            request.sampling.topP = (float) std::atof(value.c_str());
        else if (readsValue(argument, "--seed", arguments, index, value))
            request.sampling.seed = std::strtoull(value.c_str(), nullptr, 10);
        else if (argument.starts_with("--"))
            return false;
        else if (++prompts == 1)
            request.prompt = arguments[index];
        else
            return false;
    }

    return prompts == 1 && !request.prompt.empty();
}

// stderr rather than stdout, so a run whose continuation is piped into
// something else hands that something the text alone.
void report(const HF::Gemma& gemma, int promptTokens, int tokenCount, double loading)
{
    const auto prefill = gemma.lastPrefillSeconds();
    const auto decode = gemma.lastDecodeSeconds();

    const auto prefillRate = prefill > 0.0 ? promptTokens / prefill : 0.0;
    const auto decodeRate = decode > 0.0 ? tokenCount / decode : 0.0;

    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "  load and upload   %8.3f s\n", loading);
    std::fprintf(stderr,
                 "  prefill, %4d in   %8.3f s   %7.1f tokens/s\n",
                 promptTokens,
                 prefill,
                 prefillRate);
    std::fprintf(stderr,
                 "  decode,  %4d out  %8.3f s   %7.1f tokens/s\n",
                 tokenCount,
                 decode,
                 decodeRate);
    std::fprintf(stderr, "  steps             %8d\n", gemma.lastStepCount());
}

void run(const Request& request)
{
    const auto start = std::chrono::steady_clock::now();

    auto gemma = HF::Gemma {};

    if (const auto named = HF::ModelFiles::directoryFromEnvironment();
        !named.empty())
        gemma.load(named);
    else
        gemma.loadBundled();

    gemma.prepare();

    if (request.maximumTokens > 0)
        gemma.setMaximumTokens(request.maximumTokens);

    gemma.setSampling(request.sampling);

    const auto promptTokens = gemma.tokenizer().encodeWithBos(request.prompt);

    // One token's text at a time, flushed, so a long continuation reads as it
    // is produced rather than arriving at the end. Decoding a token on its own
    // is what the streaming costs: a piece that is half a character comes out
    // as the replacement character where decoding the run at once would have
    // joined it to the next.
    gemma.onToken = [&gemma](HF::TokenId token)
    {
        const auto one = HF::Vector<HF::TokenId> {token};

        std::fputs(gemma.textForTokens(one).c_str(), stdout);
        std::fflush(stdout);
    };

    const auto loading = secondsSince(start);
    const auto generated = gemma.generate(promptTokens);

    std::fputs("\n", stdout);

    report(gemma, promptTokens.size(), generated.size(), loading);
}

void generate()
{
    const auto& arguments = Apps::getAppEnvironment().commandLineArgs;
    auto request = Request {};

    if (!parse(arguments, request))
    {
        std::fputs(usage, stderr);
        Apps::setReturnValue(2);
        return;
    }

    if (HF::ModelFiles::directoryFromEnvironment().empty()
        && !HF::Gemma::hasBundledModel())
    {
        std::fputs(missingModel().c_str(), stderr);
        std::fputs(usage, stderr);
        Apps::setReturnValue(2);
        return;
    }

    if (!GPU::Device::shared().isValid())
    {
        std::fputs("no GPU device available - nothing here can run\n", stderr);
        Apps::setReturnValue(1);
        return;
    }

    try
    {
        run(request);
    }
    catch (const std::exception& failure)
    {
        std::fprintf(stderr, "%s\n", failure.what());
        Apps::setReturnValue(1);
    }
}
} // namespace

int main(int argc, char* argv[])
{
    Apps::setCommandLineArgs(argc, argv);
    return Apps::run(generate);
}

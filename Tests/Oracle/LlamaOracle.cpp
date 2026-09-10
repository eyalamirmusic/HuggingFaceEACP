#include "LlamaOracle.h"

#include <HuggingFaceEACP/Model/ModelFiles.h>

#include <ggml-backend.h>
#include <llama.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <thread>
#include <type_traits>
#include <utility>

namespace HF::Testing
{
namespace
{
// Our ids go straight into llama.cpp's arrays rather than through a copy, so
// the two spellings of a token id have to be the same type rather than merely
// the same width.
static_assert(std::is_same_v<TokenId, llama_token>,
              "TokenId and llama_token have parted company");

// llama_backend_init registers ggml's backends for the process and
// llama_backend_free unregisters them, so they belong to the process rather
// than to an oracle: freeing them from ~LlamaOracle would pull them out from
// under a second oracle that was still alive. The guard runs its destructor at
// exit instead.
void useBackendForTheProcess()
{
    struct BackendGuard
    {
        BackendGuard() { llama_backend_init(); }
        ~BackendGuard() { llama_backend_free(); }
    };

    static const auto guard = BackendGuard {};
    (void) guard;
}

// llama.cpp writes a few hundred lines about the model to stderr on every
// load. Silenced so a failing test's output is the failure. llama_log_set
// forwards to ggml_log_set, so this covers the loader as well as the graph.
void silenceLogging()
{
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
}

// Which llama.cpp this is, printed once. The backend list is the load-bearing
// half: the root CMakeLists turns every one of them off, and a number below
// should be read knowing whether that took.
void announceBuild()
{
    static const auto announced = []
    {
        std::cout << "  llama.cpp " << HF_EACP_LLAMA_CPP_TAG << " ("
                  << llama_version() << "), backends";

        for (auto index = std::size_t {}; index < ggml_backend_dev_count(); ++index)
        {
            const auto device = ggml_backend_dev_get(index);
            std::cout << (index == 0 ? " " : ", ")
                      << ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
        }

        std::cout << "\n";
        return true;
    }();

    (void) announced;
}

int threadCount()
{
    const auto reported = (int) std::thread::hardware_concurrency();
    return std::max(reported, 1);
}

// llama_batch_init hands back every member uninitialised, n_tokens included,
// so a batch is only ever filled through this.
class Batch
{
public:
    explicit Batch(int capacity)
        : batch(llama_batch_init(capacity, 0, 1))
    {
        batch.n_tokens = 0;
    }

    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;

    ~Batch() { llama_batch_free(batch); }

    void add(TokenId token, int position, bool wantsLogits)
    {
        const auto index = batch.n_tokens;

        batch.token[index] = token;
        batch.pos[index] = position;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = wantsLogits ? 1 : 0;

        batch.n_tokens = index + 1;
    }

    const llama_batch& get() const { return batch; }

private:
    llama_batch batch;
};

// The two string getters llama.cpp shares a convention for: the length comes
// back from a call that writes nothing, and the second call fills a buffer of
// that size. A key the file does not carry answers -1 rather than zero, which
// is what tells an absent value from an empty one.
template <typename Write>
std::string readString(const Write& write)
{
    const auto length = write(nullptr, std::size_t {});

    if (length < 0)
        return {};

    auto text = std::string((std::size_t) length + 1, '\0');
    const auto written = write(text.data(), text.size());

    if (written < 0)
        return {};

    text.resize((std::size_t) written);
    return text;
}
} // namespace

std::filesystem::path LlamaOracle::pathFromEnvironment()
{
    const auto directory = ModelFiles::directoryFromEnvironment();

    if (directory.empty())
        return {};

    return directory / ModelFileNames::ggufModel;
}

bool LlamaOracle::isAvailable()
{
    const auto path = pathFromEnvironment();

    if (path.empty())
        return false;

    auto error = std::error_code {};
    return std::filesystem::is_regular_file(path, error);
}

LlamaOracle LlamaOracle::load(int contextSize)
{
    return LlamaOracle {pathFromEnvironment(), contextSize};
}

LlamaOracle::LlamaOracle(const std::filesystem::path& ggufFile, int contextSize)
    : contextLength(std::max(contextSize, 1))
{
    silenceLogging();
    useBackendForTheProcess();
    announceBuild();

    auto modelParameters = llama_model_default_params();
    modelParameters.n_gpu_layers = 0;

    model = llama_model_load_from_file(ggufFile.string().c_str(), modelParameters);

    if (model == nullptr)
        return;

    vocabulary = llama_model_get_vocab(model);

    auto contextParameters = llama_context_default_params();

    // One batch has to hold the whole sequence, because logits() asks for
    // every row of it at once and llama caps the outputs of a batch at
    // n_batch. n_ubatch follows so the reference is one graph rather than a
    // sequence of them.
    contextParameters.n_ctx = (std::uint32_t) contextLength;
    contextParameters.n_batch = contextParameters.n_ctx;
    contextParameters.n_ubatch = contextParameters.n_ctx;
    contextParameters.n_threads = threadCount();
    contextParameters.n_threads_batch = contextParameters.n_threads;
    contextParameters.no_perf = true;

    context = llama_init_from_model(model, contextParameters);
}

LlamaOracle::LlamaOracle(LlamaOracle&& other) noexcept
    : model(std::exchange(other.model, nullptr))
    , context(std::exchange(other.context, nullptr))
    , vocabulary(std::exchange(other.vocabulary, nullptr))
    , contextLength(other.contextLength)
{
}

LlamaOracle& LlamaOracle::operator=(LlamaOracle&& other) noexcept
{
    if (this != &other)
    {
        release();

        model = std::exchange(other.model, nullptr);
        context = std::exchange(other.context, nullptr);
        vocabulary = std::exchange(other.vocabulary, nullptr);
        contextLength = other.contextLength;
    }

    return *this;
}

LlamaOracle::~LlamaOracle()
{
    release();
}

void LlamaOracle::release()
{
    if (context != nullptr)
        llama_free(context);

    if (model != nullptr)
        llama_model_free(model);

    context = nullptr;
    model = nullptr;
    vocabulary = nullptr;
}

bool LlamaOracle::isValid() const
{
    return model != nullptr && context != nullptr;
}

int LlamaOracle::vocabularySize() const
{
    return vocabulary == nullptr ? 0 : llama_vocab_n_tokens(vocabulary);
}

TokenId LlamaOracle::bos() const
{
    return vocabulary == nullptr ? invalidTokenId : llama_vocab_bos(vocabulary);
}

TokenId LlamaOracle::eos() const
{
    return vocabulary == nullptr ? invalidTokenId : llama_vocab_eos(vocabulary);
}

int LlamaOracle::embeddingWidth() const
{
    return model == nullptr ? 0 : llama_model_n_embd(model);
}

int LlamaOracle::layerCount() const
{
    return model == nullptr ? 0 : llama_model_n_layer(model);
}

int LlamaOracle::attentionHeads() const
{
    return model == nullptr ? 0 : llama_model_n_head(model);
}

int LlamaOracle::keyValueHeads() const
{
    return model == nullptr ? 0 : llama_model_n_head_kv(model);
}

int LlamaOracle::trainingContext() const
{
    return model == nullptr ? 0 : llama_model_n_ctx_train(model);
}

std::string LlamaOracle::metadata(const char* key) const
{
    if (model == nullptr)
        return {};

    return readString(
        [&](char* buffer, std::size_t size)
        { return llama_model_meta_val_str(model, key, buffer, size); });
}

std::string LlamaOracle::description() const
{
    if (model == nullptr)
        return {};

    return readString([&](char* buffer, std::size_t size)
                      { return llama_model_desc(model, buffer, size); });
}

Vector<TokenId> LlamaOracle::tokenize(std::string_view text, bool addBos) const
{
    auto tokens = Vector<TokenId> {};

    if (vocabulary == nullptr)
        return tokens;

    // An empty string_view's data() may be null, and llama.cpp builds a
    // std::string out of the pair before it looks at the length.
    const auto* raw = text.empty() ? "" : text.data();
    const auto length = (int) text.size();

    // Room for the added token as well as for one token per byte, which is the
    // worst case byte fallback can reach.
    tokens.resize(length + 8);

    const auto write = [&]
    {
        return llama_tokenize(
            vocabulary, raw, length, tokens.data(), tokens.size(), addBos, false);
    };

    auto written = write();

    // A negative return is minus the count that would have fitted, which is
    // llama.cpp's way of asking for a bigger buffer rather than an error.
    if (written < 0)
    {
        tokens.resize(-written);
        written = write();
    }

    tokens.resize(written < 0 ? 0 : written);
    return tokens;
}

std::string LlamaOracle::detokenize(Span<const TokenId> tokens) const
{
    if (vocabulary == nullptr || tokens.empty())
        return {};

    auto text = std::string(((std::size_t) tokens.size() + 1) * 8, '\0');

    const auto write = [&]
    {
        return llama_detokenize(vocabulary,
                                tokens.data(),
                                tokens.size(),
                                text.data(),
                                (int) text.size(),
                                false,
                                false);
    };

    auto written = write();

    if (written < 0)
    {
        text.resize((std::size_t) -written);
        written = write();
    }

    if (written < 0)
        return {};

    text.resize((std::size_t) written);
    return text;
}

void LlamaOracle::clearMemory()
{
    llama_memory_clear(llama_get_memory(context), true);
}

bool LlamaOracle::decode(Span<const TokenId> tokens,
                         int firstPosition,
                         bool everyRow)
{
    if (tokens.empty() || tokens.size() > contextLength)
        return false;

    auto batch = Batch {tokens.size()};

    for (auto index = 0; index < tokens.size(); ++index)
        batch.add(tokens[index],
                  firstPosition + index,
                  everyRow || index == tokens.size() - 1);

    return llama_decode(context, batch.get()) == 0;
}

TokenId LlamaOracle::argmaxOfLastRow() const
{
    const auto* row = llama_get_logits_ith(context, -1);

    if (row == nullptr)
        return invalidTokenId;

    const auto width = vocabularySize();
    return (TokenId) (std::max_element(row, row + width) - row);
}

Vector<float> LlamaOracle::logits(Span<const TokenId> tokens)
{
    auto rows = Vector<float> {};

    if (!isValid() || tokens.empty())
        return rows;

    clearMemory();

    if (!decode(tokens, 0, true))
        return rows;

    const auto width = vocabularySize();
    rows.reserve(tokens.size() * width);

    for (auto index = 0; index < tokens.size(); ++index)
    {
        const auto* row = llama_get_logits_ith(context, index);

        if (row == nullptr)
            return {};

        for (auto column = 0; column < width; ++column)
            rows.add(row[column]);
    }

    return rows;
}

Vector<TokenId> LlamaOracle::greedy(Span<const TokenId> prompt, int maxTokens)
{
    auto generated = Vector<TokenId> {};

    if (!isValid() || prompt.empty() || maxTokens <= 0)
        return generated;

    clearMemory();

    // Only the prompt's last row, because nothing here reads the others and
    // asking for all of them would hold a 256000-wide row per prompt token.
    if (!decode(prompt, 0, false))
        return generated;

    auto position = prompt.size();

    for (auto step = 0; step < maxTokens; ++step)
    {
        const auto next = argmaxOfLastRow();

        if (next == invalidTokenId || llama_vocab_is_eog(vocabulary, next))
            break;

        generated.add(next);

        if (!decode(Span<const TokenId> {&next, 1}, position, false))
            break;

        ++position;
    }

    return generated;
}
} // namespace HF::Testing

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Rules

Claude must never commit or push without explicit permission from the user in
the current conversation.

## Project Overview

HuggingFaceEACP runs Hugging Face models — `google/gemma-2b` first — on `eacp`'s
GPU compute stack. Every layer is authored as an `eacp::GPU::ComputeProgram`
subclass, a C++ struct whose body is written in eacp's shader EDSL, so a kernel
written once emits MSL on Apple and HLSL on Windows and Metal and D3D12 both
come free. The second goal is upstream: writing a real neural net against eacp's
compute layer is what surfaces what that layer is still missing, and a gap
belongs in eacp rather than in a workaround here. `plan.md` is the plan — the
model's shape, what carries over from WhisperEACP, what is new, and the ranked
list of eacp gaps — and is the file to read before adding anything.

Right now this runs end to end against the real checkpoint: `Core`, the `Model`
loader with its shard index and Gemma config, the `Tokenizer`, `Kernels` — the
op set each kernel is checked against a scalar CPU reference — the `Decoder`
that is Gemma's forward pass over a KV cache, the CPU `Sampling`, and
`Generation`, the KV-cached loop that takes a string and returns one. The
weights are a build-time download rather than something to arrange by hand:
`HF_EACP_FETCH_MODEL` fetches gemma-2b at configure time and
`hf_bundle_model(<target>)` copies it beside a binary, which `Generate` and
`Tests/Bundled` both call. `GEMMA_MODEL_DIR` is the explicit override, for a
checkpoint of your own — Google's gated download among them, which is the one
that carries the `gemma-2b.gguf` `Tests/Oracle` reads.

## Build Commands

```bash
# Configure
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DHF_EACP_UNITY_BUILD=OFF

# Build all targets
cmake --build build

# Build a specific target
cmake --build build --target GPUTests
cmake --build build --target DeviceInfo

# Run the tests
ctest --test-dir build --output-on-failure
```

### Build Options

- `HF_EACP_UNITY_BUILD` (default `OFF`): compiles the libraries as CMake unity
  builds. Claude must always configure with `-DHF_EACP_UNITY_BUILD=OFF` so
  per-file compile commands land in `compile_commands.json` and LSP tooling
  returns accurate results. Passing it explicitly matters because a cached `ON`
  in an existing build directory otherwise survives a reconfigure.

- `HF_EACP_ENABLE_TESTS` / `HF_EACP_ENABLE_APPS` (default: on when top-level):
  the `Tests/` and `Apps/` trees.

- `HF_EACP_FETCH_MODEL` (default `ON`): CPM downloads gemma-2b's four files at
  configure time and links them into `<build>/gemma-2b`, which
  `hf_bundle_model(<target>)` then copies beside the binaries that ask —
  `Generate` and `BundledTests` — and which every test target sees as the
  `HF_EACP_GEMMA_MODEL_DIR` compile definition.

  The source is **`unsloth/gemma-2b`**, pinned to a commit, not
  `google/gemma-2b`: Google's repo is gated, so an unauthenticated download of
  any file in it answers HTTP 401 and CPM cannot fetch it. unsloth's is an
  ungated mirror of the same bf16 weights — one unsharded `model.safetensors`,
  the same tokenizer, the same config numbers. What it does not carry is
  `gemma-2b.gguf`, so `Tests/Oracle` still wants Google's own download through
  `GEMMA_MODEL_DIR`.

  It costs disk twice over. The download itself is 5.0 GB of
  `model.safetensors` plus 17.5 MB of `tokenizer.json` and under a kilobyte of
  the two JSON configs, and it lands in `build/_deps` — set `CPM_SOURCE_CACHE`
  to a directory outside the build tree and every build directory on the
  machine shares one copy instead of re-downloading 5 GB each. On top of that
  each `hf_bundle_model` target gets a **copy**, not a link, so `Generate` and
  `BundledTests` are 5 GB apiece beside their executables — about 15 GB for a
  full build tree. Configure with `-DHF_EACP_FETCH_MODEL=OFF` for a build that
  only wants the library; the tests that need a checkpoint then skip the way a
  GPU test skips without a device.

- `HF_EACP_ENABLE_LLAMA_CPP` (default `OFF`): fetches llama.cpp at a pinned
  release tag and builds `Tests/Oracle`, which compares our tokens and our
  logits against it. Off by default because both halves are expensive in a way
  the rest of this build is not — llama.cpp and ggml are minutes of compile,
  and the `gemma-2b.gguf` the tests read is a 10 GB manual download that the
  fetched mirror does not carry, so it takes `GEMMA_MODEL_DIR` pointing at
  Google's own gated download and every test there skips without one. It means
  nothing without
  `HF_EACP_ENABLE_TESTS`. The fetch names every ggml backend off, so the
  reference is exactly ggml's CPU arithmetic and a disagreement cannot be a
  backend's.

  ```bash
  cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DHF_EACP_UNITY_BUILD=OFF \
        -DHF_EACP_ENABLE_LLAMA_CPP=ON
  ```

- `HF_EACP_CI_BUILD` (default `OFF`): turns on the unity builds here and in eacp
  and Miro. There is no CI (see below); the switch is what makes that
  configuration reproducible by name rather than by remembering which project
  exposes which option. Because it turns unity on it is not for LSP-backed
  development.

### Dependencies are fetched, not taken from the machine

eacp comes from CPM at **`develop`**, NanoTest at `main`. That is the default
and the only configuration Claude should use: the plain configure line above is
the whole story, and no build here points at a checkout on this machine.

Claude must **not** pass `-DCPM_eacp_SOURCE` unless the user asks for it in the
current conversation. A local tree drags whatever is uncommitted in it into this
build, so an unrelated refactor in progress over there breaks every target here,
with the error surfacing inside the dependency where it reads as ours.

The override exists for the case it is actually for — changing eacp itself
alongside a change here that needs it — and the build returns to the fetch as
soon as that eacp change is pushed:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DHF_EACP_UNITY_BUILD=OFF \
      -DCPM_eacp_SOURCE=$HOME/Code/eacp
```

Use `$HOME` (not `~`). CMake does not expand `~`, and shell tilde expansion is
suppressed inside quotes — `-DCPM_eacp_SOURCE="~/Code/eacp"` will silently
configure against a non-existent path and fail later with an error about a
missing `eacp-gpu` target.

`develop` rather than `main`: main trails it, and the compute layer this project
is written against moves on develop first. eacp's SIMD-group matrix — the fast
many-row product — is already there, so a plain fetch has it.

### There is deliberately no CI

This is a private repository, and GitHub Actions minutes on a private repository
are billed. **Claude must not add a `.github/` directory, a workflow, or any
other CI configuration**, and must not suggest one as part of a change. The
`cmake` / `ctest` lines above are the whole check, run locally.

## Architecture

New source files are added directly to the module's CMakeLists.txt under the
appropriate `target_sources(...)` call. Platform-specific sources go inside the
matching `APPLE`/`WIN32` branch.

`Lib/` is the include root, so a header of ours is spelled
`<HuggingFaceEACP/Core/Core.h>`, beside eacp's `<eacp/...>`. Everything lives in
namespace `HF`. Library targets are prefixed `hf-`; `HuggingFaceEACP` is the
INTERFACE umbrella that links them all, and `Lib/HuggingFaceEACP/HuggingFaceEACP.h`
the umbrella header.

### CMake modules must not share a name with a dependency's

`CMAKE_MODULE_PATH` is inherited by every subdirectory, and this project's entry
is appended before eacp's. A module here named the same as one of eacp's
therefore **shadows it**, and eacp's own `include(...)` silently picks up our
file. Hence `CMake/HFTargetSetup.cmake` rather than `TargetSetup.cmake`, and the
`hf_` prefix on every function it defines: CMake functions are global once
defined, so a name declared in two projects resolves to whichever directory was
processed last, which is not something a call site can see.

Check `eacp/CMake/` before adding a module here.

### Library (`Lib/HuggingFaceEACP`)

**Core/** — shared types and the library version. Links `eacp-gpu` publicly
rather than `eacp-core`, since every layer above it will hold GPU buffers. The
version comes from `project()` through compile definitions, so there is no
second copy to drift.

The modules `plan.md` calls for sit beside it in that order, each one linking
only what is in its own signatures: the safetensors loader with its shard index,
the tokenizer, the kernels, the decoder, the sampler, and `Generation` on top of
the three of them a run needs.

## Tests (`Tests`)

NanoTest, one executable per module.

`Tests/GPU` links an entry point of its own, `Tests/Support/GpuTestMain.cpp`,
named by the `HF_GPU_TEST_MAIN` variable set in `Tests/CMakeLists.txt`: anything
touching the GPU runs inside `eacp::Apps::run`, which owns the run loop and the
autorelease pool the Metal backend is written against. A test that needs a
device returns early when `Device::shared().isValid()` is false, so the suite
still passes on a machine with no GPU.

Every kernel gets a test that asserts against a scalar CPU reference computed in
the test itself. That is what catches a backend divergence — the same assertion
runs against MSL on Apple and HLSL on Windows.

## Layout

| | |
| --- | --- |
| `plan.md` | The plan: Gemma 2B's shape, what carries over from WhisperEACP, and the ranked eacp gaps |
| `CMake` | `CPM.cmake`, the two `Find` modules that fetch eacp and NanoTest, `HFTargetSetup.cmake` and `HFResources.cmake` |
| `Lib/HuggingFaceEACP/Core` | Shared types and the library version. Links `eacp-gpu` |
| `Lib/HuggingFaceEACP/Model` | Safetensors with the shard index, `config.json`, the tensor catalogue, the `GEMMA_MODEL_DIR` override, and the model fetch with `hf_bundle_model` |
| `Lib/HuggingFaceEACP/Tokenizer` | Gemma's SentencePiece-style BPE from `tokenizer.json`, with byte fallback |
| `Lib/HuggingFaceEACP/Kernels` | The op set: the products, the reductions, RMSNorm, RoPE, GeGLU and the multi-query attention |
| `Lib/HuggingFaceEACP/Decoder` | Gemma's forward pass over a KV cache: the shape, the weights, and one compute pass per step |
| `Lib/HuggingFaceEACP/Sampling` | The CPU sampler over a read-back logits row: greedy, temperature, top-k, top-p |
| `Lib/HuggingFaceEACP/Generation` | `Gemma`: the whole runtime, a string in and a string out, with the greedy loop's token feedback on the device; and `resourcesDirectory()`, which finds the model the build copied |
| `Apps/Console/DeviceInfo` | What this machine's GPU offers, printed from eacp's `Device` |
| `Apps/Console/Generate` | A prompt in and a continuation streamed out, over the model the build copied beside it, or the one `GEMMA_MODEL_DIR` names |
| `Tests/Core` | The version, without a device |
| `Tests/Bundled` | What `hf_bundle_model` put beside the binary: the copy matches the fetch, and a run out of it generates |
| `Tests/GPU` | The compute smoke test: eacp's toolchain end to end |
| `Tests/Model` | Shards and config over safetensors files the tests write themselves |
| `Tests/Tokenizer` | A hand-written mini `tokenizer.json` fixture |
| `Tests/Kernels` | A double-precision scalar reference per kernel |
| `Tests/Decoder` | A double-precision reference decoder over a synthetic checkpoint the tests write |
| `Tests/Sampling` | Greedy, temperature, top-k and top-p on the CPU, seeded and reproducible |
| `Tests/Generation` | The loop against one the test runs itself, over the synthetic checkpoint and a `tokenizer.json` of the same width |
| `Tests/Oracle` | llama.cpp over the F32 GGUF, behind `HF_EACP_ENABLE_LLAMA_CPP` |
| `Tests/Support` | `GpuTestMain.cpp`, the shared entry point for GPU-touching suites, and `GemmaModel.h`, which resolves the checkpoint every suite runs against |

## Code Style

The style is eacp's, and `.clang-format` and `.clang-tidy` are eacp's own files
copied verbatim. Every source here must be clang-format clean:

```bash
clang-format --dry-run --Werror $(git ls-files '*.h' '*.cpp' '*.mm')
```

Always use the most modern C++ and RAII practices.
Use auto for variables and whenever possible.
Don't use auto for functions and member functions

Don't use comments unless absolutely needed. Use named functions to make code
self documenting. Where a comment is warranted it explains why, in full
sentences, not what.

Give std::function members a non-null default — a no-op lambda, or one
returning an empty value — so call sites invoke them directly without null
checks.

Enforced via `.clang-format`:
- Allman brace style
- 85 column limit
- 4-space indentation (no tabs)
- Pointer alignment: left (`int* ptr`)
- Break constructor initializers before comma

Always run clang-format for edited code files

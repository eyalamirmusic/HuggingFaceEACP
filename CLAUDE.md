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

Right now this is plan.md's first two steps: `Core`, the `Model` loader with
its shard index and Gemma config, the `Tokenizer`, and `Kernels` — the op set
the decoder will be assembled out of, each kernel checked against a scalar CPU
reference. The decoder and the generation loop are still to come, and nothing
has been run against the real checkpoint: the repo is gated and there is no
download on this machine, so every test against it returns early until
`GEMMA_MODEL_DIR` points at one.

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

The modules `plan.md` calls for — the safetensors loader with its shard index,
the tokenizer, the kernels, the decoder and the generation loop — land beside it
in that order.

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
| `CMake` | `CPM.cmake`, the two `Find` modules that fetch eacp and NanoTest, and `HFTargetSetup.cmake` |
| `Lib/HuggingFaceEACP/Core` | Shared types and the library version. Links `eacp-gpu` |
| `Lib/HuggingFaceEACP/Model` | Safetensors with the shard index, `config.json`, the tensor catalogue, and the `GEMMA_MODEL_DIR` locator |
| `Lib/HuggingFaceEACP/Tokenizer` | Gemma's SentencePiece-style BPE from `tokenizer.json`, with byte fallback |
| `Lib/HuggingFaceEACP/Kernels` | The op set: the products, the reductions, RMSNorm, RoPE, GeGLU and the multi-query attention |
| `Apps/Console/DeviceInfo` | What this machine's GPU offers, printed from eacp's `Device` |
| `Tests/Core` | The version, without a device |
| `Tests/GPU` | The compute smoke test: eacp's toolchain end to end |
| `Tests/Model` | Shards and config over safetensors files the tests write themselves |
| `Tests/Tokenizer` | A hand-written mini `tokenizer.json` fixture |
| `Tests/Kernels` | A double-precision scalar reference per kernel |
| `Tests/Support` | `GpuTestMain.cpp`, the shared entry point for GPU-touching suites |

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

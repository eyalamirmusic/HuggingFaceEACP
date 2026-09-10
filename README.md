# HuggingFaceEACP

Hugging Face models on [eacp](https://github.com/eyalamirmusic/eacp)'s GPU
compute stack, starting with
[google/gemma-2b](https://huggingface.co/google/gemma-2b).

Every layer of a model is authored as an `eacp::GPU::ComputeProgram` — a C++
struct whose body is written in eacp's shader EDSL rather than in a shading
language. The EDSL emits MSL on Apple and HLSL on Windows from that one source,
so a kernel written once runs on Metal and D3D12 and cannot drift between them.

The other half of the point is upstream: writing a real neural net against
eacp's compute layer is what surfaces what that layer is still missing, and
those gaps get closed in eacp rather than worked around here.

**`plan.md` is the plan** — Gemma 2B's shape in
[WhisperEACP](https://github.com/eyalamirmusic/WhisperEACP)'s terms, what
carries over from it, the four new kernels, and the ranked list of what eacp is
missing. This tree is the skeleton it starts from: `Core`, one console app and
the tests that prove eacp's compute path is reachable. None of the model is
here yet.

## Layout

| | |
| --- | --- |
| `plan.md` | The plan, and the eacp gaps it depends on |
| `Lib/HuggingFaceEACP/Core` | Shared types and the library version. Links `eacp-gpu` |
| `Apps/Console/DeviceInfo` | What this machine's GPU offers, printed from eacp's `Device` |
| `Tests/Core` | The version, without a device |
| `Tests/GPU` | The compute smoke test: a kernel compiled, dispatched and read back |
| `CMake` | CPM, the fetches for eacp and NanoTest, and the target defaults |

Headers are spelled `<HuggingFaceEACP/...>`, beside eacp's own `<eacp/...>`, and
everything lives in namespace `HF`.

## Building

Requires CMake 3.31+, a C++20 compiler and a 64-bit toolchain. eacp and NanoTest
are fetched through CPM — eacp at `develop`, NanoTest at `main` — so the
configure line is the whole story and no build here points at a checkout on the
machine.

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DHF_EACP_UNITY_BUILD=OFF
cmake --build build

ctest --test-dir build --output-on-failure
./build/Apps/Console/DeviceInfo/DeviceInfo
```

macOS and Windows. eacp gates its whole GPU stack behind platforms with a Metal
or D3D12 backend, so there is no Linux target.

| option | default | |
| --- | --- | --- |
| `HF_EACP_UNITY_BUILD` | `OFF` | Unity builds of the libraries. Off, per-file compile commands land in `compile_commands.json` |
| `HF_EACP_ENABLE_TESTS` | on when top-level | The `Tests/` tree |
| `HF_EACP_ENABLE_APPS` | on when top-level | The `Apps/` tree |
| `HF_EACP_CI_BUILD` | `OFF` | The unity builds here, in eacp and in Miro, as one reproducible switch |

There is no CI configuration, deliberately: this is a private repository and the
`cmake` and `ctest` lines above are the whole check.

To build against a local eacp checkout — for changing eacp itself alongside a
change here that needs it, and nothing else:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DHF_EACP_UNITY_BUILD=OFF \
      -DCPM_eacp_SOURCE=$HOME/Code/eacp
```

Use `$HOME`, not `~`: CMake does not expand a tilde and the shell will not
expand one inside quotes, so the path silently resolves to nothing and the
failure surfaces much later as a missing `eacp-gpu` target.

## License

MIT — see `LICENSE`.

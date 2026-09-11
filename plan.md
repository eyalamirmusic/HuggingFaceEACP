# Running google/gemma-2b on eacp's GPU module

A high-level plan, written before any code, for running
[google/gemma-2b](https://huggingface.co/google/gemma-2b) locally on
[eacp](https://github.com/eyalamirmusic/eacp)'s compute stack, with
[WhisperEACP](https://github.com/eyalamirmusic/WhisperEACP) as the reference for
how a model is assembled out of `eacp::GPU::ComputeProgram`s.

The second goal is WhisperEACP's second goal: a real net surfaces what eacp's
compute layer is missing, and a gap belongs in eacp rather than in a workaround
here. The gaps this plan already sees are in the last section.

The Gemma numbers below were from the released config as remembered rather than
read. **They are now read**: `HF_EACP_FETCH_MODEL` fetches the checkpoint at
configure time, and `Model/Checkpoint/config` and `matchesTheCatalogue` assert
every shape against the file. The table holds — `hidden_size` 2048,
`intermediate_size` 16384, 18 layers, 8 query heads of `head_dim` 256 over one
KV head, `vocab_size` 256000, `max_position_embeddings` 8192, `rms_norm_eps`
1e-6, `rope_theta` 10000, tied embeddings, BOS 2 / EOS 1 / PAD 0. What the
table has wrong is the sharding: the mirror the build fetches ships **one
unsharded 5.0 GB `model.safetensors`**, not two shards, which is why
`ModelFiles` requires either an index or a single file rather than the two.

## What Gemma 2B is, in WhisperEACP's terms

A decoder-only transformer: WhisperEACP's decoder without the encoder, the
cross-attention or the mel front-end. The decode loop, the KV cache, the on-GPU
argmax feeding the next embed, and the whole test discipline carry over.

| | Whisper tiny.en decoder | Gemma 2B |
| --- | --- | --- |
| layers | 4 | 18 |
| width | 384 | 2048 |
| feed-forward width | 1536 | 16384, gated (GeGLU) |
| heads | 6 of 64 | 8 query heads of 256, **1 shared KV head** |
| vocabulary | 51,864 | 256,000 |
| context | 448 | 8192 |
| norm | LayerNorm with bias | RMSNorm, scale is `1 + w`, eps 1e-6 |
| positions | learned table | RoPE on q and k, theta 10000 |
| activation | exact GELU | tanh GELU |
| biases | yes | none anywhere |
| logits | tied embedding | tied embedding; input embedding scaled by sqrt(2048) |
| parameters | 37 M | ~2.5 B |
| weights on disk | 151 MB F32 | **5.0 GB BF16**, two safetensors shards |

Per layer, the tensors are `q_proj [2048, 2048]`, `k_proj [256, 2048]`,
`v_proj [256, 2048]`, `o_proj [2048, 2048]`, `gate_proj [16384, 2048]`,
`up_proj [16384, 2048]`, `down_proj [2048, 16384]`, and the two RMSNorm scales.
The embedding is `[256000, 2048]`, the only tensor above a gigabyte.

### The repo

| file | size | |
| --- | --- | --- |
| `model-00001-of-00002.safetensors` | 4.95 GB | BF16 |
| `model-00002-of-00002.safetensors` | 67 MB | BF16 |
| `model.safetensors.index.json` | 13.5 kB | tensor name to shard |
| `config.json`, `generation_config.json` | | BOS 2, EOS 1, PAD 0 |
| `tokenizer.json` | 17.5 MB | the fast-tokenizer form |
| `tokenizer.model` | 4.2 MB | the SentencePiece original |
| `gemma-2b.gguf` | 10 GB | F32, usable as an oracle through llama.cpp |

`google/gemma-2b` is **gated**: a download needs a Hugging Face account that
accepted Google's terms and a token, and an unauthenticated fetch answers 401,
so WhisperEACP's pattern of fetching through CPM cannot point at it. It points
at `unsloth/gemma-2b` instead — an ungated mirror of the same bf16 weights,
pinned to a commit — which is gap 4 below and is how the build now gets a
model. The mirror is **unsharded**: one 5.0 GB `model.safetensors` in place of
the two shards and the index above. It carries `tokenizer.model` too, though
the fetch does not take it — nothing here reads the SentencePiece original —
and what it has no copy of at all is `gemma-2b.gguf`.

The GGUF is what `Tests/Oracle` reads, and it **no longer takes Google's
download either**: `convert_hf_to_gguf.py`, out of the llama.cpp tree this
build already pins at `b10900`, writes one from the fetched safetensors.

```bash
build/_deps/llama-cpp-src/convert_hf_to_gguf.py <staged> --outtype f32 \
      --outfile $HOME/Code/models/gemma-2b/gemma-2b.gguf
```

`<staged>` is a directory holding the four fetched files plus the three the
converter reads and the fetch does not take — `tokenizer.model`,
`tokenizer_config.json` and `special_tokens_map.json` — which come from the
same `unsloth/gemma-2b` commit CMake pins, so nothing in it is from anywhere
else. The result is 10 GB of F32 and about a minute of writing, and it differs
from Google's own file in one key: the converter leaves out
`gemma.rope.freq_base` when it took its default, so
`Oracle/Decoder/configMatchesGguf` reports `rope_theta` rather than asserting
it. `GEMMA_MODEL_DIR` is how a machine that has run the conversion says where
it went. A CMake target that runs it is a possible later step, and is not
written: it wants a Python environment and 10 GB, neither of which the rest of
this build needs.

`gemma-2b` is the base completion model. `gemma-2b-it` is the same architecture
with a chat prompt format, so supporting both is a prompt change only.

## What carries over from WhisperEACP

Unchanged or nearly so:

- Safetensors mapping and validation, `TensorLoader`'s shape checks,
  `config.json` reading. One addition: the shard index, so a tensor name
  resolves to one of two files.
- The many-row product (`SimdTiledMatMul`) for prefill, `SplitLinear` for the
  one-row decode step, softmax, the two-dispatch argmax, the embed gather. The
  argmax already splits a row across groups, so 256k columns is a parameter
  change. The embed drops the positional table and gains the sqrt(width) scale.
- The decoder's structure: one compute pass per step, shapes as uniforms, KV
  cache written through ranged binds, argmax output read as the next step's
  token with no host round trip, a prompt step of many rows and then a token
  step of one.
- The test tiers: a double-precision CPU reference per kernel and per layer, an
  in-process oracle, small committed fixtures. The oracle analogue of
  whisper.cpp is llama.cpp over the GGUF, and ggml is already fetched by the
  whisper oracle build.

## What is new on the model side

- **RMSNorm**, a sibling of the LayerNorm kernel: a group-per-row fp32
  reduction, then `x * rsqrt(mean(x²) + eps) * (1 + w)`.
- **RoPE** on q and k after their projections, rotate-half form over 256-wide
  heads. A small kernel or a fold on the projection's store, with a cos/sin
  table of `[8192, 128]` uploaded once.
- **The gated MLP**: `down(gelu_tanh(gate(x)) * up(x))`. Cheapest form is one
  product over the concatenated gate and up weights writing `[rows, 32768]`,
  then the multiply folded into the down product's operand read, or a fold on
  the store that reads the partner half. eacp has `tanh` natively.
- **Multi-query attention.** Every query head reads the one KV head, so the
  attention kernels need a head-to-KV-head mapping, which for Whisper is the
  identity. The KV cache per layer is tiny: `[8192, 256]` per K or V, 8 MB at
  fp32 over the full context.
- **Head width 256 breaks `SingleQueryAttention` as written.** Its 64 lanes
  each keep a row of the head width in threadgroup memory, and 64 × 256 floats
  is 64 KB against Metal's 32 KB. The value fold has to be laid out by column
  rather than by lane, or the head split four ways. Kernel design, not an eacp
  gap.
- **The tokenizer** is SentencePiece BPE, not GPT-2 byte-level BPE: spaces
  become `▁`, unknown characters fall back to byte tokens, there is no
  byte-level alphabet. WhisperEACP's merge engine is reusable; the normalizer,
  pre-tokenizer and decoder are not. Parsing 17.5 MB of `tokenizer.json`
  through Miro's JSON in Debug will take seconds; measure it, and consider
  reading `tokenizer.model` instead.
- **Generation**: prepend BOS (2), stop at EOS (1), greedy first. Temperature,
  top-k and top-p can start as a 1 MB logits readback per token on the CPU.
- **Logits** are a `[256000, 2048]` product against the tied embedding, 1 GB of
  BF16 read per token. `SplitLinear` at a logits split count, as the Whisper
  decoder does.

## Performance shape

Decode is bandwidth bound: every token reads every weight once, so the storage
format sets the ceiling. Prefill is compute bound and lands on the SIMD-group
matrix product, which is already the fast path.

| weight storage | bytes per token | rough ceiling | measured |
| --- | --- | --- | --- |
| fp32, widened on the CPU — step 5 replaced it | 10 GB | ~50 tok/s | 46 to 49 |
| **bf16 kept packed — what runs today** | 5 GB | ~100 tok/s | **83 to 91** |
| int8 or int4 blocks, later | 1.3 to 2.5 GB | 200 to 400 tok/s | |

The measured column is one M-series device, Release, a 6-token prompt: 91
tokens/s over 16 decode steps and 83 over 64, against 49 and 46 on the widened
path. The ceilings were guesses from the bandwidth alone and the ratio came out
at 1.8, which is what a bandwidth-bound step predicts.

A step is roughly 18 layers × a dozen dispatches, a few hundred per token,
which is the same order as the Whisper decoder's step.

## Dependencies

eacp comes from CPM at `develop`, and that is the only configuration this tree
is built in — `CLAUDE.md` says why a local checkout drags whatever is
uncommitted in it into this build.

**Everything this tree asks of eacp is on `develop`.** Step 5's two rounds
consume six additions — the BF16 reads, `readHalf4` / `readBFloat16x4`,
`saturatingTanh`, `simdSum` / `simdMax` / `simdMin`,
`Device::maxThreadgroupMemory` with `ComputeProgram::threadgroupMemoryBytes` and
`fitsThreadgroupMemory`, and `ComputePipeline::threadExecutionWidth` — and all
of them are there. For a fortnight they were not: they were written on a branch
and this tree built only against a checkout of it, which is what the
`-DCPM_eacp_SOURCE` override exists for and the one case it is for. That is
over; the plain configure line is the whole story again.

Which is the project's second goal working as intended, and worth recording as
a shape rather than as an anecdote. Every one of those six was found by writing
a real net against the compute layer and hitting the missing thing, went into
eacp rather than into a workaround here, and came back as an API this tree then
deleted code to use.

## What eacp is missing, ranked

1. **A BF16 read in the EDSL. Done, in eacp `develop`, and used here.** The one
   that mattered. Gemma ships BF16 and the loader widened it to fp32 on the CPU,
   which doubled the weights on the GPU and halved decode speed. Converting to
   fp16 instead would have been lossy: bf16 has an 8-bit exponent and fp16 a
   5-bit one, so small weights flush and nothing is bit-exact against the
   reference. What landed is exactly the shape this asked for, mirroring
   `readHalf`: `InputBuffer::readBFloat16(i)` counting bfloat16s,
   `readBFloat16x2(i)` counting the words that hold two, `unpackBFloat16x2` /
   `packBFloat16x2`, `writeBFloat16x2`, and `bfloat16FromFloat` /
   `bfloat16ToFloat` on the host in `PackedVertex.h`. Widening is a shift and a
   bitcast, so it is exact and bit-identical on every backend; the narrowing is
   round-to-nearest-even written in integer arithmetic, so it is bit-identical
   too, which `packHalf2` is not.

   Downstream it is step 5's first round, below: `WeightStorage` has its third
   case, every product kernel and the embedding gather have a bf16 variant, and
   the loader uploads a BF16 tensor's bytes as they lie. Decode went from 49 to
   91 tokens/s and the process from 14.4 GB resident to 10.3 GB.
2. **`GPU::Buffer` sizes are `int`.** WhisperEACP's plan recorded this as the
   maintainer's decision, and for 2B in BF16 it holds: the embedding is
   1.05 GB. Widened to fp32 it was 2.10 GB, 50 MB under the limit, and a 7B
   embedding in fp32 is past it. Item 1 has landed, so the largest buffer here
   is now half what it was and this is not blocking for 2B; it blocks anything
   larger, and it still blocks the logits buffer at a large prompt capacity —
   see the step 3 and 4 notes below. Widen `Buffer`, `BufferRange`, `read` and
   `update` to 64-bit before 7B rather than after. Shader-side indexing is
   32-bit and fine: the biggest tensor is 524 M elements.
3. **Zero-copy weight buffers on Metal.** `Buffer` always copies through
   `newBufferWithBytes`, so loading is a 5 GB memcpy and, while the mapping and
   the buffer both exist, 10 GB resident. Fine on 128 GB, painful on a 16 GB
   laptop. Metal can wrap a page-aligned mapping with
   `newBufferWithBytesNoCopy`, and every tensor would then be a `BufferRange`
   into one buffer over the whole shard (offsets are 4-byte aligned, which the
   ranged bind needs; check per tensor). D3D12 has no equivalent and keeps
   copying behind the same API. Worth doing; not needed for a first token.
4. **Loading a gated model. Done, through a mirror.** Not a GPU gap.
   `google/gemma-2b` answers 401 to an unauthenticated download, so CPM cannot
   fetch it and a token in the environment was the only way through it.
   `unsloth/gemma-2b` is an ungated mirror of the same bf16 weights — one
   unsharded `model.safetensors`, the same tokenizer, the same config numbers —
   so `HF_EACP_FETCH_MODEL` fetches the four files from it at a pinned commit,
   `hf_bundle_model(<target>)` copies them beside a binary, and
   `Gemma::loadBundled()` finds the copy. Nothing has to be arranged by hand.
   `GEMMA_MODEL_DIR` stays as the explicit override, and is also what points
   `Tests/Oracle` at `gemma-2b.gguf`. That is the one file the mirror does not
   carry, and it is converted out of these same safetensors by the llama.cpp
   tree this build already pins — see "The repo" above — so the gate is now
   behind us for every purpose and nothing here needs a Hugging Face token.
5. **Threading.** eacp's GPU layer is main-thread only. Two hundred tokens at
   10 ms each is two seconds of message thread if driven synchronously.
   `commitAsync` and the scoped wait exist, so record several steps per command
   buffer, with the on-GPU token feedback and an EOS check on readback, and
   drive the loop from the run loop the way `LiveTranscriber` does. Whether
   eacp wants a compute queue off the main thread is a bigger question than
   this project; not opened here.
6. **Quantization, later.** Int8 and int4 block formats need `readByte` and
   nibble helpers as the analogue of `readHalf`, plus per-block scales. eacp
   has the uint arithmetic already, so this is convenience rather than
   capability. It is the step from 100 to a few hundred tokens per second.

Found while writing steps 1 and 2, unranked. **All four are filled**, on eacp
`develop`, and all four are used here:

- **Metal's `tanh` returns NaN for a large argument. Done.** eacp compiles its
  Metal library with fast math on and no `MTLCompileOptions`, and under that
  mode `tanh(990)` is NaN rather than 1. tanh GELU's argument grows as the cube
  of its input, so an activation of 30 hits it, and `Gelu.h` clamped the
  argument to ±10 by hand. eacp's `saturatingTanh` is that clamp, and the tails
  are answered with the constant rather than computed, ±10 itself included —
  float32 resolves nothing between `tanh(9.011)` and one, so it is a repair of
  the tails and not a different function. `Gelu.h` calls it, and the ±30 and
  ±1000 cases in `Tests/Kernels` are what says so, through the standalone
  activation and through the GeGLU the model actually dispatches.
- **No SIMD-group-scoped reduction. Done, and used in one place.** `groupSum`
  and `groupMax` went through threadgroup scratch and two barriers whatever the
  group was. eacp now has `simdSum`/`simdMax`/`simdMin`, collective over exactly
  one SIMD group of 32 and, on Metal, one instruction with neither scratch nor
  barrier. Where it pays here is `SplitLinear` at `splits == simdWidth`: the
  group is [32, 2], each SIMD group is exactly one output's lanes, and the fold
  replaces 64 published partials and a serial walk of 32 of them. That count is
  `logitsSplitCount`, so it is the widest product a decode step runs.
  `ReducingProgram` picks the narrow fold for any kernel whose lane count is at
  or under 32, which is a path the tests take and the decoder does not.

  **It did not pay in attention, which is where this item was written from.**
  Two whole-group folds per tile over 64 lanes is what the entry called four
  barriers per 64 keys, and making the group one SIMD group removes them — and
  measures *slower*: 76.6 tokens/s against 83.4 over 64 decode steps, since
  halving the lanes doubles the tiles and halves the threads walking the head
  width. Reverted, and recorded in `MultiQueryAttention.h` so it is not tried
  again. A step already at memory bandwidth has no barrier cost left to save.
- **No threadgroup-memory budget in the EDSL. Done.** eacp has
  `Device::maxThreadgroupMemory()`, `ComputeProgram::threadgroupMemoryBytes()`
  and `fitsThreadgroupMemory(device)`, so what a kernel declares and what a
  device allows are both numbers rather than comments. `Decoder::prepare` runs
  every one of its programs through a check that throws a `ModelError` naming
  the kernel and both numbers, before a pipeline is built — eacp logs an
  overspend and then lets the backend refuse, which on Metal happens after the
  library compiled clean and points at the wrong thing.
  `Kernels/multiQueryAttentionFitsTheThreadgroupBudget` asserts the attention
  kernel's declared 1536 bytes against the device's 32768, and asserts that the
  lane-major layout it was written instead of — 64 × 256 floats, 65536 bytes —
  is past it. That last assertion is the one the comment used to make by hand.
- **`readHalf`-style vector reads emit scalar subscripts. Done for the packed
  pair.** `readHalf4` and `readBFloat16x4` take a record of four elements as the
  two words holding it, which on Metal is one eight-byte load;
  `storedWeight4` in `Kernels/WeightStorage.h` now reads through them, so all
  three storages index by records of four and the call site never spells how
  wide an element is. `read4` itself also lowers to a single `packed_float4`
  load on Metal now. Neither moved a decode step, which is at memory bandwidth
  — see step 5's second round.

Found while writing steps 3 and 4, unranked:

- **Gap 2 is reachable without the embedding.** The logits buffer is
  `promptCapacity × 256000 × 4` bytes: 524 MB at the default 512 rows, and
  2048 rows is past what an `int` describes. `Gemma::prepare` counts it in
  64 bits and refuses with a `ModelError` naming this gap rather than
  truncating the allocation. The embedding is bound twice — as the gather's
  table and as the tied logits weight — and at 2.10 GB widened sat 50 MB under
  the limit; packed it is 1.05 GB, so since step 5 the logits buffer is the
  only thing here anywhere near the ceiling.
- **`commitAsync` cannot drive a generation loop.** Its `Async` resolves on
  the main thread, and a loop that never returns to the run loop never sees
  it — eacp's own header says so. `submit()` and the scoped
  `CommandBuffer::read`, which waits for that one buffer rather than the
  newest submission, are what the loop uses, with two steps in flight. Gap 5
  above stands as written for the demo app's sake, not the loop's.
- **A host write into a buffer the GPU is still writing is ordered by
  nothing.** The steps computed past the end of a run are still filling the
  sequence buffer when the next run's `Buffer::update` copies its prompt in,
  and nothing in the API says the two are unordered. The loop waits on the
  ring before it returns; WhisperEACP's `Whisper` carries the same latent
  race. A `Buffer::update` that waited for in-flight writers, or a documented
  rule that it does not, belongs in eacp.
- **No raw-bytes concatenation in the loader. Done, in step 5's first round.**
  Fusing gate and up went through `readFloats`, so a packed repo paid a widened
  copy of the largest weight per layer. It is now
  `TensorLoader::loadStackedProjectionWeights`, which stacks the two tensors'
  raw bytes and reports the storage both halves share — so F32, fp16 and bf16
  all go up at the shape the file has them in, and only a pair whose types
  disagree, whose type no kernel reads, or whose first half has an odd element
  count falls back to widening. That last one is the whole of what the byte
  path has to be careful about: a packed read fetches the word an index lands
  in, so the second half has to begin on one. Gemma's halves are
  [16384, 2048].

Found while writing step 5's bf16 round, unranked. Both are about the same
thing — a packed weight is widened one element at a time and the rest of the
machine only takes fp32 — and the first is the cheap one:

- **No four-wide packed read.** The packed family stops at two: `readHalf2` and
  `readBFloat16x2` each fetch one word and give a `Float2`. The float family
  goes to four, and `read4` is what `SplitLinear`'s hot loop uses — a run of
  four weights against a `read4` of the input, which is the shape a decode step
  spends nearly all its bandwidth in. So the packed variants call
  `readBFloat16x2(i / 2)` and `readBFloat16x2(i / 2 + 1)` and assemble a
  `Float4` by hand, which is two indexed loads where the float path has one.
  A `readHalf4(i)` / `readBFloat16x4(i)` taking a four-aligned element index,
  fetching the two words as one eight-byte load and unpacking four, would be
  the same call for the same shape. The unranked `read4` item above compounds
  it: even the float path's one load lowers to four scalar subscripts today, so
  neither is the load it reads as. A `UIntInputBuffer::read4` plus four
  `unpackBFloat16x2` calls is the workaround, and it is not taken here because
  it makes the *uniform's declared type* depend on the weight's storage, which
  would reach every product kernel's member list rather than its read.

- **The SIMD-group matrix cannot take a packed operand.** `simdMatrix` loads a
  fragment from a `Shared<Float>` tile, an `InputBuffer` or an `OutputBuffer` —
  all fp32 — so `SimdTiledMatMulProgram` widens each bf16 weight on the way
  into threadgroup memory and the fragment load reads fp32 from there. The
  global traffic is halved, which is the win this round measured, but the
  32 x 64 staging block of B is still 8 KB of the 32 KB budget rather than
  4 KB, and there is a widen per element per tile that the hardware would do
  for free. Two things would fix it, and the second is the real one:
  `shared<T>` over a packed sixteen-bit type with a `simdMatrix` overload that
  loads a fragment from it, and — better — a `simdMatrix(const InputBuffer&,
  offset, rowStride)` sibling that loads an 8 x 8 fragment straight out of a
  packed device buffer, which for a ContiguousK weight would remove the B
  staging and its two barriers entirely. Metal has `simdgroup_bfloat8x8` and
  `simdgroup_half8x8` to lower both onto; a backend without them widens, which
  is what this file does by hand today.

  **Left open, and it is an eacp policy question rather than a kernel one.**
  Taken to eacp with the rest of this list, it came back as the one item not
  filled: `bfloat` and `simdgroup_bfloat8x8` are Metal 3.1, which is macOS 14,
  and eacp's deployment target is macOS 11. The fallback on an older OS is the
  staging this file already does, so the same kernel would carry a barrier on
  one OS and not on the other — a collective point that exists or does not
  depending on the SDK, which is the worst shape a barrier can have. Three ways
  out, none of them free:

  - raise eacp's GPU floor to macOS 14, and the overload is unconditional;
  - gate it on `__METAL_VERSION__` with the staging fallback behind it, and
    document that the barrier is there on one branch and not the other — the
    kernel author then has to write for the branch that has it;
  - expose it as a feature level that withholds the overload where it cannot
    be had, so a kernel asks and takes the staging path itself when the answer
    is no.

  Nothing here needs it before quantization, which changes the operand format
  again.

Already filled since the Whisper rounds: 3D dispatch exists, so the head no
longer has to be folded into the dispatch row; `CommandTimer` times
off-screen command buffers.

## Suggested order

Mirrors how WhisperEACP went, one module and one test tier at a time:

1. **Done.** Loader with shards, config, and the tokenizer. The one-layer CPU
   reference moved to step 3, where the decoder's shape it checks exists. What
   changed against the plan above: WhisperEACP's 2 GB whole-file limit had to
   go, since the first shard is 4.95 GB, and is now per tensor; and byte
   fallback runs *before* the merges, at character granularity, which is
   Hugging Face's order. Whether Gemma prepends a dummy `▁` is read from the
   JSON and defaults to off, and **the real file settles it**: gemma-2b's
   normalizer is a bare `Replace(" " -> "▁")` with a null `pre_tokenizer` and
   no `Prepend`, so the dummy prefix and the pre-merge split are both off. It
   also settles the byte pieces, which were assumed to be 256 of them: there
   are **255**, at ids 217..472, and id 226 — where `<0x09>` would sit — is a
   literal tab instead. `Tokenizer/realTokenizerParseTime` asserted the wrong
   thing about that and now asserts what matters, which is that every byte
   reaches a piece standing for exactly it.
2. **Done.** The four new kernels, with scalar references in `Tests/Kernels`.
   The attention's value fold is laid out by column, so a group holds one
   256-wide accumulator (1.5 KB) rather than one per lane. The SIMD-matrix
   switch WhisperEACP carries was dropped: eacp `develop` has it
   unconditionally. The kernels' own references never needed a checkpoint, and
   the suites that do now get one from the build's fetch — a test skips only
   in a build configured with `-DHF_EACP_FETCH_MODEL=OFF`.
3. **Done, and run against the real checkpoint and the oracle.** `Decoder`
   is the forward pass over a KV cache, one compute pass per step, checked
   against a double-precision reference over a synthetic checkpoint the tests
   write (2e-7 relative, asserted at 5e-6),
   through both the many-row tiled product and the one-row split product,
   and through a two-shard checkpoint. The llama.cpp comparison exists behind
   `HF_EACP_ENABLE_LLAMA_CPP` — llama.cpp pinned at `b10900`, every ggml
   backend off — as `Tests/Oracle`: the tokenizer against llama.cpp's, our
   prefill and token-by-token logits against its rows, an eight-token greedy
   continuation, and the config's numbers against the GGUF's header, which is
   what retires the "confirm every number" caveat above. Every one of them
   skips until `GEMMA_MODEL_DIR` names a directory holding `gemma-2b.gguf`,
   which the mirror does not carry but which this tree's own llama.cpp converts
   out of the fetched safetensors — see "The repo".

   **They have run.** Every row's argmax and every row's five largest tokens
   agree, on all three prompts and one token at a time, which is the assertion
   the tier exists for. The elementwise differences measure 8.4e-3, 1.02e-2 and
   3.6e-3 relative over the three prompts — 0.025, 0.043 and 0.013 absolute,
   on logits reaching 88.6, 113.4 and 74.9 — and 8.4e-3 one token at a time.

   Two rounds of step 5 have moved them in the fourth significant digit and no
   further, which is worth a table since it is the one number that says the
   arithmetic is still the same arithmetic:

   | | widened F32 | packed BF16 | + the eacp round |
   | --- | --- | --- | --- |
   | `A:` | 8.3794e-3 | 8.3794e-3 | 8.35314e-3 |
   | `In 1969 …` | 1.02258e-2 | 1.02258e-2 | 1.02578e-2 |
   | `def add(a, b):` | 3.60951e-3 | 3.60951e-3 | 3.59328e-3 |
   | one token at a time | 8.36572e-3 | 8.36572e-3 | 8.36879e-3 |

   The middle column is exact against the first because the shader widens the
   same bytes in the same order. The third moved because the order changed: the
   logits fold at 32 splits is a SIMD tree rather than a serial walk of 32
   partials, and eacp's `read4` became one `packed_float4` load on Metal, which
   is enough to change how the compiler contracts a `dot`. Holding the fold out
   and leaving the load in reproduced the third column's prompt figures exactly,
   so the load is nearly all of it. Every argmax and every top-5 still agrees,
   and the largest of them is a fifth of `logitTolerance`. The third column was
   measured against eacp's branch and then again against `develop` once the
   merge landed, and the two are identical digit for digit.
   That is what a 2048-wide fp32 dot product summed in two different orders
   through eighteen layers costs, and it tracks the magnitude a row reaches
   rather than anything else, so the provisional 2e-3 is now
   `logitTolerance = 5e-2`, five times the worst of them. What changed
   against the plan:
   gate and up are concatenated at load into one `[32768, 2048]` weight, so
   the MLP is one product, one GeGLU and the down product with the residual
   folded into its store; and the per-step intermediates are sized by a step
   capacity of their own (`maxStepRows`) rather than by the window, since at
   8192 rows the gated pair alone was 1.6 GB for buffers a decode step uses one
   row of.
4. **Done, and run against the real checkpoint.** `Generation`'s `Gemma` is
   the runtime: a string in, the prefill in blocks of the prompt capacity,
   then one command
   buffer per token with two in the air, Argmax writing the token into the
   sequence buffer on the device and the next step's Embed reading it there.
   The host reads a slot back for the stop condition and a callback only.
   Checked against a loop the test drives itself over `Decoder::step` and a
   CPU argmax, over a synthetic `tokenizer.json` of the model's width.

   **What the first run against the real checkpoint showed.** Three things,
   all three now settled:

   - `config.json` carries `"hidden_activation": null` beside a
     `"hidden_act": "gelu"`, and the loader threw on it — `stringFieldOr`
     refuses a field that is present and not a string. transformers writes
     that null when the newer key was never set and reads it as Gemma's own
     `gelu_pytorch_tanh` with `hidden_act` ignored, so `activationOf` now
     treats an explicit null as the default and only an absent key defers to
     the older spelling. Nothing dispatches on the value; it is recorded.
   - The 256 byte pieces are 255. See item 1 above: id 226 is a literal tab.
   - **Greedy from "The capital of France is" does not reach Paris — and is
     right not to.** It produces `" a city of contrasts. It is a city of
     history, of art, of"`, and so does everything else. Three independently
     written implementations agree on the ids
     `[476, 3413, 576, 82777, 235265, 1165, 603, 476]` token for token: our
     GPU decoder, llama.cpp over the F32 GGUF on ggml's CPU path, and Hugging
     Face transformers in fp32 on the CPU. At the sampled row ' a' is -16.5291
     and ' Paris' -16.8554, then ' the' -17.0254, ' one' -17.273 and
     ' also' -17.8572 — Paris is the model's second choice and loses by 0.326
     logits, which is eight times the largest elementwise disagreement the
     oracle has ever measured between our logits and llama.cpp's. The one
     thing that might plausibly have moved it — the embedding normalizer,
     `sqrt(2048)` rounded to bf16's 45.25 against fp32's 45.254833 — was run
     both ways and the top five come back bit-identical. So the base model is
     not answering a question here; it is continuing a sentence, which is what
     a base model does. `Generation/Gemma/completesAPrompt` and the two oracle
     greedy tests now probe with `"Q: What is the capital of France?\nA:"`,
     whose answer is ' Paris' at -1.0616 against ' paris' at -5.9548: the same
     claim about the same stack, with a 4.89-logit margin behind it instead of
     a 0.326-logit one.

   Timings, Debug, one M-series device, 16 tokens from a 6-token prompt: 38 s
   to load and upload (the BF16 widening on the CPU, which is gap 1), 0.36 s
   of prefill, and 0.29 s of decode — about 56 tokens/s. **Step 5's first round
   took the widening out**, and the same run is now 15 s to load and upload,
   0.12 s of prefill and 0.18 s of decode, about 90 tokens/s. What is left of
   that 15 s is mostly the 17.5 MB `tokenizer.json` through Miro's JSON in a
   Debug build, not the weights: the same run in Release loads in 0.9 s.
5. **First round done: the weights stay bf16.** The BF16 read went into eacp
   first, since it decides the whole memory budget, and this round is what
   consumes it. `WeightStorage` has a `PackedBFloat16` case; `MatMul`, `Linear`,
   `SplitLinear`, `TiledMatMul` and `SimdTiledMatMul` each read their weight
   through one `storedWeight` / `storedWeight4` written once in `MatMul.h`, so
   a fourth storage would be one edit rather than five; and `Embed` became a
   template over the same enum. The loader uploads a BF16 or an F16 tensor's
   bytes as they lie, padding an odd element count to a whole word, and the
   fused gate-and-up weight is stacked as bytes — see the concatenation item
   above. Everything a product reads is packed. The two RMSNorm scales per
   layer are widened on the way up, deliberately: they are [2048] each, and the
   alternative is a packed read in every small kernel a tensor can reach.

   **The embedding is the case that decided the gather's shape.** It is bound
   twice, as the gather's table and as the tied logits weight, so a gather with
   only a float form would have forced the largest tensor in the model to stay
   widened whatever the product could read. `EmbedProgram` is parameterised by
   `WeightStorage` for that one reason, and the embedding is 1.05 GB rather
   than 2.10 GB because of it.

   **Three storages is twelve pipelines, and a run dispatches four.**
   `Decoder::prepare` therefore takes the weights: it reads
   `DecoderWeights::storages()` and compiles the gather, the tiled product and
   the two split products that storage needs, leaving the other eight empty.
   Compiling all of them built two whole SIMD-group matrix pipelines nothing
   would ever bind. A step against weights in a storage the decoder was not
   prepared for is a `ModelError` raised beside the shape check, before
   anything is recorded.

   **What it cost and what it bought.** Nothing in the numbers moved: the
   oracle's elementwise differences are 8.4e-3, 1.02e-2 and 3.6e-3 relative on
   the three prompts and 8.4e-3 one token at a time, which are the widened
   path's figures to every digit recorded. They have to be — the shader widens
   the same bytes the CPU used to, in the same order, so the arithmetic is
   identical rather than merely close, and every argmax and top-5 still agrees
   with llama.cpp. Release, 6-token prompt, one M-series device:

   | | widened F32 | packed BF16 |
   | --- | --- | --- |
   | load and upload | 1.57 s | 0.94 s |
   | prefill, 6 rows | 0.44 s | 0.12 s |
   | decode, 16 tokens | 0.325 s, 49 tokens/s | 0.175 s, 91 tokens/s |
   | decode, 64 tokens | 1.39 s, 46 tokens/s | 0.77 s, 83 tokens/s |
   | peak resident | 14.42 GB | 10.34 GB |
   | peak footprint | 11.25 GB | 6.38 GB |

   Decode is 1.8x, which is what halving a bandwidth-bound step's traffic
   predicts. Prefill is 3.6x rather than 2x, and the extra is first-touch of
   half as many pages on a run this short rather than anything in the
   arithmetic. The 4.9 GB of footprint is the weights: 2.5 B parameters at four
   bytes against two, to within the rounding. Resident is 4.1 GB rather than
   4.9 because the 5 GB safetensors mapping is counted in both runs and how
   much of it stays resident is the page cache's business, not ours.

   **Second round: the rest of the eacp gaps, and what they were worth.** The
   four unranked items from steps 1 and 2 plus the four-wide packed read all
   landed in eacp together, and this tree now uses every one of them —
   `saturatingTanh` in `Gelu.h`, `readBFloat16x4`/`readHalf4` in
   `storedWeight4`, `simdSum` in `SplitLinear` at the logits' split count and
   in `ReducingProgram` below 32 lanes, and the threadgroup budget as a
   prepare-time refusal in `Decoder`. See the gap list for what each one
   replaced.

   **None of them moved the clock, and the reason is the same one the table at
   the top gives.** Release, 6-token prompt, against the first round's numbers:

   | | round one | round two |
   | --- | --- | --- |
   | load and upload | 0.94 s | 0.84 s |
   | prefill, 6 rows | 0.12 s | 0.11 s |
   | decode, 16 tokens | 91 tokens/s | 92.6 tokens/s |
   | decode, 64 tokens | 83 tokens/s | 83.4 tokens/s |

   A decode step reads 5.0 GB of weights and runs at 92.6 tokens/s, which is
   463 GB/s — at what the memory system gives. Every one of these changes
   removes arithmetic, loads or barriers, and none of them removes bytes, so
   there was nothing left for them to buy. That is worth writing down rather
   than treating as a disappointment: it says the next round is int8 or int4
   and not another pass over the kernels, and it is the same conclusion the
   attention lane-count experiment reached from the other direction.

   The oracle moved in the fourth significant digit — see the table under step
   3 — and holding the `simdSum` fold out showed that almost all of it is
   eacp's own `read4` lowering rather than anything here.

   Still to do in this step: the fused GeGLU, the split counts (the decoder's
   64 and 32 are WhisperEACP's numbers, and 32 now also buys the SIMD-scoped
   fold, which is a reason to keep it that measurement did not supply), and the
   prompt-capacity guess. The two gaps the bf16 round turned up are at the end
   of the gap list, one filled and one left open.
6. **Half done.** `Sampling` is the CPU sampler — temperature, top-k, top-p in
   Hugging Face's processor order, a seeded draw the standard specifies so a
   seed means the same token sequence on every machine — and a temperature
   above zero routes the loop through a 1 MB logits readback per token, one
   step in flight. `Apps/Console/Generate` streams a continuation for a prompt
   and prints the two halves' timings. On-device sampling is a later round.

## Open decisions

- **Should the conversion into `gemma-2b.gguf` be a build target?** It is a
  command a person runs once, written down in "The repo" above, and everything
  it needs is already in the tree — the pinned llama.cpp checkout, the fetched
  safetensors — except a Python environment with `torch` and the three
  tokenizer files the fetch does not take. A target would make the oracle tier
  reachable from a configure line rather than from a paragraph; it would also
  put 10 GB and a `pip install` behind a switch that currently only costs
  compile time. Left as a step, not taken.
- ~~Does the BF16 read go into eacp now, so this project never carries a
  widened fp32 path at all?~~ **Settled: it went in.** The widened path is not
  gone, though, and should not be — `WeightStorage::Float` is what an F32 repo
  and every synthetic test checkpoint take, and `makeFloatBuffer` is what the
  norm scales take. What went is the widening of the *weights*, which is the
  only place it cost anything.
- Widen `Buffer` to 64-bit in the same eacp change, or defer until 7B? The
  logits buffer still hits the limit at a prompt capacity of 2048 rows, so it
  is a 2B question too, though one a smaller prefill block sidesteps. The
  packed weights removed the other half of the pressure: the embedding is
  1.05 GB now rather than 2.10 GB.
- **What does eacp do about `simdgroup_bfloat8x8`?** The one gap taken to eacp
  and not filled, because it is a deployment-target question rather than a
  codegen one: the packed fragment load needs Metal 3.1, which is macOS 14,
  against eacp's floor of 11. Raise the floor, gate it and document a barrier
  that exists on one OS and not the other, or expose a feature level a kernel
  asks — the three are spelled out under the gap itself. Nothing here needs it
  before quantization, which changes the operand format again anyway.

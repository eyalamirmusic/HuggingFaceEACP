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
| fp32, widened on the CPU — step 5 replaced it | 10.0 GB | ~50 tok/s | 46 to 49 |
| bf16 kept packed — the default | 5.0 GB | ~100 tok/s | 85 to 96 |
| **int8 blocks of 32 — what `--int8` runs** | 2.7 GB | ~190 tok/s | **141 to 162** |
| int4 blocks, later | 1.4 GB | ~360 tok/s | |

The measured column is one M4 Max, Release, a 6-token prompt, the median of
three: 95.8 tokens/s over 16 decode steps and 86.5 over 64 on the bf16 path,
against 49 and 46 on the widened one. The ceilings were guesses from the
bandwidth alone and the first ratio came out at 1.9, which is what a
bandwidth-bound step predicts. 95.8 tokens/s against 5.0 GB of weights a step is
479 GB/s, so the remaining headroom to the guessed 100 was the memory system's
and not a kernel's.

**The int8 row reached 1.32x where its bytes predict 1.88x, and the fifth round
took it to 1.63x by making the load as wide as the record.** 2.506 billion
parameters at 1.0625 bytes is 2.66 GB a step. A quantized record of four weights
was one word, four bytes, fetched by one `readInt8x4`, where a bf16 record of
four is two words, eight bytes, fetched by one `readBFloat16x4` — the same load
count for half the data, so a step that should have been bound by bandwidth was
bound by how many loads it could issue. eacp's `readInt8x16` fetches sixteen
weights in one sixteen-byte load, and with the lane count moved to match it the
int8 path went from 298–335 GB/s to 375–430 GB/s against bf16's 433–479. See
gap 6 for the calls and step 5's fifth round for the table.

**What is left of the 1.88x is a fixed cost per step rather than the weights.**
Take the 16-token runs: a bf16 step is 10.44 ms and an int8 step 6.19 ms, so
2.34 GB fewer bytes buys 4.25 ms — 551 GB/s marginal, and subtracting each
path's own weight traffic at that rate leaves 1.37 ms on the bf16 step and
1.36 ms on the int8 one. One line fits both, which is what "bandwidth bound"
means; the 1.36 ms is the couple of hundred dispatches a step is made of and the
work that is not weights. Before the wide read the same arithmetic gave 893 GB/s
marginal and a 4.96 ms fixed cost, which is half a step and not credible — the
two paths did not lie on one line, which is exactly what a load-bound path looks
like from the outside.

A step is roughly 18 layers × a dozen dispatches, a few hundred per token,
which is the same order as the Whisper decoder's step.

## Dependencies

eacp comes from CPM at `develop`, and that is the only configuration this tree
is built in — `CLAUDE.md` says why a local checkout drags whatever is
uncommitted in it into this build.

**That was not always true; the last exception closed with eacp PR #52.** Step
5's fourth and fifth rounds — the int8 weights below and the wide reads that
made them pay — consume eacp's quantized reads, and for a while those were on a
`quantized-reads` branch rather than on `develop`, so this tree was built with
`-DCPM_eacp_SOURCE` pointed at a checkout of it. That was the second time that
override has been used and is the one case it exists for. PR #52 merged the
branch into `develop`, so the plain configure line is the whole story again,
exactly as it became after step 5's first two rounds. What it supplied is
listed under gap 6.

**Everything else this tree asks of eacp is on `develop`.** Step 5's first two
rounds consume six additions — the BF16 reads, `readHalf4` / `readBFloat16x4`,
`saturatingTanh`, `simdSum` / `simdMax` / `simdMin`,
`Device::maxThreadgroupMemory` with `ComputeProgram::threadgroupMemoryBytes` and
`fitsThreadgroupMemory`, and `ComputePipeline::threadExecutionWidth` — and all
of them are there. For a fortnight they were not: they were written on a branch
and this tree built only against a checkout of it, which is what the
`-DCPM_eacp_SOURCE` override exists for and the one case it is for. That is
over for those six, and with PR #52 it is over for the quantized reads too.

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
   too, which `packHalf2` is not. Nothing here *writes* bf16 — the activations,
   the KV cache and the logits are all fp32 — so the pack half of that family
   landed unused.

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
6. **Quantization. Int8 done, in eacp `develop`, and used here; int4 next.**
   The step from 100 to a few hundred tokens per second, and it is the one gap
   whose answer this round can report a number for rather than a prediction.

   **What eacp PR #52 supplied, and this tree uses.** `InputBuffer::readInt8`
   and `readUInt8` counting bytes, `readInt8x4` / `readUInt8x4` counting the
   words that hold four, `readInt4x8` / `readUInt4x8` handing back a
   `Float4Pair` of eight nibbles, the free `unpackInt8x4` / `packInt8x4` family
   beside them, and `int8x4FromBytes` / `int8x4ToByte` and the nibble pair on
   the host in `PackedVertex.h`. Sign extension is written as `(b ^ 0x80) - 128`
   in every dialect and on the host, so a buffer packed on the CPU reads back as
   what was put in it rather than as what a cast happened to mean. The fp16
   scale needed nothing new: `readHalf` was already exact and
   `halfFromFloat` on the host already rounds to nearest even, which is what
   lets the CPU reference be bit-identical to the shader rather than close to
   it.

   Then, after this tree measured the fourth round and found the shape below,
   the same PR supplied **the wide quantized reads**: `readInt8x8` /
   `readUInt8x8` over records of eight bytes, taken through one `read2`;
   `readInt8x16` / `readUInt8x16` over records of sixteen, through one `read4`;
   and `readInt4x16` / `readUInt4x16` over records of sixteen nibbles, through
   one `read2`. Sixteen values do not fit a vector any of the three languages
   has, so they come back as **`Float4Quad {Float4 a, b, c, d;}`** in address
   order — `q.a.x` is the record's first element, `q.d.w` its sixteenth — beside
   the `Float4Pair {low; high;}` the nibble reads already used. Element *k* of a
   buffer is component *k* % 16 of `readInt8x16(k / 16)`, which is the index
   convention `readBFloat16x4` set, one width up.

   Downstream it is step 5's fourth and fifth rounds, below: `WeightStorage` has
   its fourth case, the loader quantizes on the way up, `--int8` is the flag,
   and `storedWeight8` / `storedWeight16` are what the split product and the
   tiled products' staging read through. Decode went from 85 to 112 tokens/s
   over 64 tokens on the fourth round and from 112 to 141 on the fifth, and the
   process from 6.37 GB of footprint to 4.62 GB.

   **What this tree hit while consuming it, in the order it costs.**

   - **No quantized read wider than one word, which held int8 to 1.32x instead
     of 1.88x. Filled, and it was worth 26%.** `readInt8x4(i)` fetched one word
     — four bytes, four weights — where `readBFloat16x4(i)` fetches two words as
     a single eight-byte load for the same four weights. So the two kernels
     issued the same number of loads and the quantized one carried half the
     bytes in each, and a decode step that should have been bound by bandwidth
     was bound by load issue: 427 to 478 GB/s on the bf16 path against 298 to
     335 GB/s on the int8 one, on the same device and the same kernel.

     What landed is exactly the shape this asked for: `readInt8x8` out of the
     two words `read2` fetches and `readInt8x16` out of the four `read4`
     fetches, with the nibble pair at both widths and `Float4Quad` as the answer
     to "no vector wider than four". On Metal a `readInt8x16` emits one
     `packed_float4` load and four register bitcasts, with no second buffer
     access, which an eacp codegen test asserts. Sixteen weights to a load also
     makes the per-block scale a read per sixteen rather than per four.

     **Measured, it is two thirds of the way to the bytes' prediction and the
     rest is not loads.** 112.1 to 141.0 decode tokens/s over 64, 125.9 to 161.6
     over 16, and 299 to 375 GB/s and 335 to 430 GB/s with them — against bf16's
     433 and 479 on the same runs. What the wide read did not buy is explained
     under "Performance shape" above: past this point both paths lie on one line
     of 551 GB/s marginal bandwidth plus 1.36 ms of fixed per-step cost, so the
     remaining 1.63x-against-1.88x is the dispatches and the work that is not
     weights, and no read width reaches it.

   - **No common-subexpression elimination for buffer reads. Still open, and now
     worked around rather than paid.** The eacp side has already recorded it; it
     showed here in the tiled products' staging loop, where a staging thread
     reads eight consecutive weights that are all in one block and each of the
     eight emitted its own `readHalf` of the same scale. That cost prefill
     rather than decode — 994 tokens at 1400 tokens/s against bf16's 1426, 2%
     — and one `readInt8x8` per run is what removes it: the run is now one
     eight-byte load and one scale read, and the quantized prefill measures 1490
     tokens/s, 4% *faster* than the packed path rather than 2% slower. The gap
     itself is unchanged: two reads at the same index are still two loads, and
     it is only that there is now a call that does in one read what eight would
     have done.

   - **No vector wider than four. Answered by an aggregate rather than by a
     vector.** It is why `Float4Pair` exists for the nibble reads, and
     `Float4Quad` is the same answer one width up — four `Float4`s in address
     order, which is what a sixteen-byte load unpacks into and what every
     backend can spell.

   **What is still awkward, after the merge.** Three things, none of them
   blocking:

   - **`asUInt` is scalar only**, with no `Float4 -> UInt4` overload. It is why
     a kernel cannot fetch four words through `read4` and bitcast them itself,
     which is the workaround the wide reads made unnecessary rather than
     possible — the alternative was binding the weight as a
     `Uniform<UIntInputBuffer>`, which makes the *uniform's declared type* depend
     on the weight's storage and so reaches every product kernel's member list
     rather than its read.
   - **HLSL and GLSL still expand a record read componentwise**, `read4`
     included and so `readInt8x16` with it. The win above is Metal's alone
     today; the same kernel is correct on D3D12 and no faster there.
   - **There is no wide *store* to match.** The quantizer runs on the host, so
     nothing here needs one yet; an on-device quantizer would.

   Int4 is the next round and is deliberately not written here. It is a fifth
   `WeightStorage` case reading `readInt4x16` at the same block and the same
   scale, and nothing in the layout, the loader or the decoder's plumbing
   changes shape for it — which is what the fourth case was designed to leave
   true.

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

   The rounds of step 5 have moved them in the fourth significant digit and no
   further, which is worth a table since it is the one number that says the
   arithmetic is still the same arithmetic. The five columns are the first four
   rounds; the fifth round's figures are in the paragraph after it:

   | | widened F32 | packed BF16 | + the eacp round | + 256 split lanes | int8 blocks |
   | --- | --- | --- | --- | --- | --- |
   | `A:` | 8.3794e-3 | 8.3794e-3 | 8.35314e-3 | 8.35314e-3 | **0.364406** |
   | `In 1969 …` | 1.02258e-2 | 1.02258e-2 | 1.02578e-2 | 1.02578e-2 | **0.285007** |
   | `def add(a, b):` | 3.60951e-3 | 3.60951e-3 | 3.59328e-3 | 3.59328e-3 | **0.242264** |
   | one token at a time | 8.36572e-3 | 8.36572e-3 | 8.36879e-3 | 8.36665e-3 | **0.364372** |

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
   `logitTolerance = 5e-2`, five times the worst of them.

   **The last column is a different kind of number, and belongs in the same
   table for exactly that reason.** The first four move in the fourth
   significant digit because they are the accumulation order's; the fifth is a
   hundred times larger because it is the format's — 1.65, 2.53 and 1.39
   absolute on logits reaching 88.6, 113.4 and 74.9. `quantizedLogitTolerance`
   is 2.0, five times the worst of them at one digit, by the same discipline.

   **And the argmax does not move.** Every row's largest token still agrees with
   llama.cpp on all three prompts and one token at a time, and the eight-token
   greedy continuation is the same eight tokens — ` Paris` and the four after it
   — on the quantized path as on the exact one. What does move is the fifth
   place: four rows of the fifty this tier compares swap their fifth token, and
   each of those four is a pair llama.cpp's own row puts 0.196, 0.098, 0.0003
   and 0.075 logits apart. So the top-five check is now "the same five, or a
   swap the reference itself calls a tie", with `quantizedTieGap = 1.0` five
   times the worst of those — and the exact storages answer to a tie gap of
   zero, which is the strict set equality they have always passed.

   **The fourth column moves one row and only one, and which row it is is the
   point.** The split count that went from 64 lanes to 256 is the one-row
   product's, so the three prompt rows — which go through the many-row tiled
   product and never touch it — are identical digit for digit, and the
   token-at-a-time row, whose every projection is that product, moves from
   8.36879e-3 to 8.36665e-3. A fold over 256 lanes is a different tree than one
   over 64, which is all that is, and it moves the answer by less than the
   change from fp32 to bf16 storage did.

   **The fifth round moves the same one row, for the same reason, and says so
   twice over.** The wide read changed which sixteen weights a lane of the
   one-row product owns and the lane count went 256 to 128 with it, so the
   token-at-a-time rows move — 8.36665e-3 to **8.37199e-3** on the packed path
   and 0.364372 to **0.364376** on the quantized one — while all six prompt
   rows, packed and quantized, come back identical to the digit: 8.35314e-3,
   1.02578e-2, 3.59328e-3 and 0.364406, 0.285007, 0.242264. The prompt rows go
   through the tiled product, and the tiled product's staging changed too — a
   run of eight weights out of one `readInt8x8` in place of eight reads — so
   their being unmoved is the assertion that the wide read is the same
   arithmetic in a different number of loads, and not merely that nothing
   touched them. Every argmax still agrees, the greedy continuation is the same
   eight tokens, and the four rows of fifty that swap a fifth token are the same
   four.

   What changed against the plan:
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
5. **Five rounds done; the weights are bf16 by default and int8 on request.**
   The first round is below, then the eacp gaps, then the three numbers that
   were guesses, then the storage format, then the load that format needed.

   **First round: the weights stay bf16.** The BF16 read went into eacp
   first, since it decides the whole memory budget, and this round is what
   consumes it. `WeightStorage` has a `PackedBFloat16` case; `MatMul`, `Linear`,
   `SplitLinear`, `TiledMatMul` and `SimdTiledMatMul` each read their weight
   through one `storedWeight` / `storedWeight4` written once in `MatMul.h`, so
   a fourth storage would be one edit rather than five — see the fourth round
   for how that turned out; and `Embed` became a template over the same enum. The loader uploads a BF16 or an F16 tensor's
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

   **Third round: the three numbers that were still guesses.** The fused
   GeGLU, the two split counts and the prompt capacity were the last things in
   this step carried over from WhisperEACP or from arithmetic rather than
   measured here. All three are settled now, by measurement. One number moved
   — the in-layer split count, from 64 lanes to 256 — and everything else is
   kept for a reason a measurement supplies rather than assumes, which is the
   outcome the second round's conclusion predicts and the reason to write a
   null result down with its figure rather than leave it as an open item.

   | | was | is | what it is worth |
   | --- | --- | --- | --- |
   | in-layer split lanes | 64 | **256** | +2.0% at 16 tokens, +1.9% at 64 |
   | logits split lanes | 32 | 32 | a curve flat to 0.3% either side |
   | fused GeGLU | no | no | 0.07% of a step's bytes, 0.5% measured |
   | prompt capacity | 512 | 512 | 0.69 s prefill against 0.79 s at 128 |

   **The in-layer split count is the one that moved, and 64 was low.** It is
   how many lanes share one output's inner sum in every projection a decode
   step runs, and the 64 it has been is neither a measurement nor WhisperEACP's
   own number — that one is 96, over a 384-wide row, which is not Gemma's
   curve. Release, 6-token prompt, one M-series device, decode tokens/s, the
   median of three interleaved rounds:

   | lanes | 16 tokens | 64 tokens |
   | --- | --- | --- |
   | 32 | 92.8 | 83.6 |
   | 64 — what stood here | 92.7 | 83.7 |
   | 96 | 93.5 | 84.1 |
   | 128 | 93.3 | 84.6 |
   | **256** | **94.6** | **85.3** |
   | 384 | 85.7 | 77.4 |
   | 512 | 67.6 | 61.6 |
   | 1024 | 29.2 | 27.3 |

   Two percent, which is small and is also the only thing this round found that
   is outside the noise. The shape of the curve is the interesting half: it is
   nearly flat from 32 to 256 and then falls off a cliff, losing a third of the
   rate by 512 and two thirds by 1024. 256 is eight SIMD groups, and past it a
   group is asking the scheduler for more threads than one output's 2048-wide
   inner sum has work for. Nothing about this contradicts the second round —
   256 lanes move no bytes either; what they move is how much of the machine a
   one-row product has running at once, which is the one thing a decode step
   was short of.

   **The logits count stays 32, and now for a measured reason.** At 256
   in-layer lanes the logits projection measures 95.0 / 85.5 tokens/s at 16
   lanes, 94.7 / 85.3 at 32, 94.0 / 85.0 at 64, 93.0 / 84.1 at 128 and
   93.2 / 84.4 at 256 — the top of that curve is flat to within the
   run-to-run noise, and 32 is the one point on it where a group is exactly one
   SIMD group per output and the fold is `simdSum` with neither scratch nor a
   barrier. So the second round's remark that 32 "now also buys the SIMD-scoped
   fold, which is a reason to keep it that measurement did not supply" is
   answered: measurement supplies it, by finding nothing better.

   **The fused GeGLU is not worth writing, and the arithmetic said so before
   the measurement agreed.** Today the MLP is one product over the concatenated
   gate-and-up weight writing `[rows, 32768]`, a GeGLU folding that to
   `[rows, 16384]`, and the down product. Per layer a decode step's one row
   therefore writes 128 KB, reads it back, writes 64 KB and reads that back;
   over 18 layers the whole intermediate traffic of the feed-forward is 6.9 MB,
   against the 5.0 GB of weights the same step reads. Fusing the activation
   into the gate-and-up product's store removes 4.6 MB of it, 0.09%; fusing it
   into the down product's operand read removes 3.5 MB, 0.07%. Removing the
   stage outright — the ceiling neither fusion can beat — measures 95.4
   tokens/s against 95.1 over 16 decode tokens and 86.0 against 85.5 over 64.
   Half a percent is what the whole dispatch, both its buffers and its barrier
   are worth together. The two fusions are also not equally priced: the one
   into the down product's read is the cheap one to write and would evaluate
   the activation 2048 times per element, once per output walking that
   16384-wide row; the one into the gate-and-up store is free at run time and
   needs a product kernel that computes two dot products per output and folds
   them, in the split form and the tiled form and in each of the three weight
   storages. Recorded in `GeGLU.h` so it is not tried again, the way the
   attention lane-count experiment is recorded in `MultiQueryAttention.h`.

   **The prompt capacity stays 512 rows, and the two curves it sits between
   are now drawn.** It sizes the logits buffer at `capacity × 256000 × 4` bytes
   and every per-step intermediate beside it, and it is how many rows one
   prefill block carries. Release, a 986-token prompt — long enough to be
   several blocks at every capacity below it:

   | rows | blocks | prefill | prefill rate | peak footprint |
   | --- | --- | --- | --- | --- |
   | 128 | 8 | 0.79 s | 1251 tokens/s | 5.48 GB |
   | 256 | 4 | 0.76 s | 1303 tokens/s | 5.63 GB |
   | **512** | **2** | **0.69 s** | **1434 tokens/s** | **5.94 GB** |
   | 1024 | 1 | 0.69 s | 1433 tokens/s | 6.54 GB |

   512 is where both curves stop paying. Below it the prefill costs 10% at 256
   rows and 15% at 128, since a block is what fills the many-row tiled product
   and a short one leaves it idle at the edges; above it the prefill does not
   improve at all and the footprint grows by 600 MB, because 512 rows already
   fill that product's machine. Half a gigabyte for the last 15% of a prefill
   is the trade this takes, and a caller who wants the memory back has
   `setPromptCapacity` and now knows what it costs. Gap 2 is unchanged by this:
   the logits buffer at 512 rows is 524 MB and at 2048 rows would be past what
   an `int` describes, which is why `Gemma::prepare` counts it in 64 bits.

   With those three settled, what was left in this step was the storage format,
   which is gap 6 and the round below.

   **Fourth round: int8 blocks, which is the storage format.** Thirty-two
   consecutive weights to a block along the contiguous dimension, one scale per
   block, symmetric with no zero point, `q = round-to-nearest-even(x / scale)`
   clamped to [-127, 127] — llama.cpp's Q8_0 shape, taken because it is the
   well-characterised one and because a later comparison against a Q8_0 GGUF
   stays possible. 1.0625 bytes an element against bf16's two.

   **The scale is fp16 and the quantizer divides by it after it has been
   narrowed.** fp16 because an fp32 scale is four bytes per thirty-two elements
   rather than two, 12.5% more bytes on a step whose whole cost is bytes, and
   because bf16's eight-bit mantissa would put visible error on the scale
   itself. Dividing by the narrowed value rather than by `amax / 127` is what
   makes a CPU reference *bit-exact* against the shader rather than close to it:
   both sides multiply the same fp16 number by the same small integer, and a
   float multiply is the same answer on every backend. llama.cpp's own Q8_0
   divides by the unnarrowed one and stores the narrowed, so its dequantized
   values are not the ones its quantizer aimed at; this way round they are, and
   `Tests/Decoder`'s quantized run measures 2.5e-7 against a double-precision
   reference over the dequantized weights — the same figure the exact paths give.

   **One `GPU::Buffer` per weight, in two regions**, because a kernel that
   needed two would need a second `InputBuffer` member and every product's
   member list would change. The elements first, one signed byte each and four
   to a word, in the order the tensor already has them; then one fp16 scale per
   block, two to a word, in the same block order. Element *i* is byte *i*, its
   block is *i* / 32, and the scale of block *b* is the half at
   `elementCount / 2 + b`. A tensor whose contiguous dimension is not a whole
   number of blocks, or whose element count is not a whole number of scale
   words, is a `ModelError` naming it — every Gemma tensor passes, since the
   contiguous dimension of a projection weight is 2048 or 16384.

   **A fourth storage was one edit and a two-liner, not one edit.** `storedWeight`
   and `storedWeight4` in `Kernels/WeightStorage.h` gained one `else if
   constexpr` each, which is what the bf16 round's "one edit rather than five"
   promised. What that round did not foresee is that a block format needs a
   *second* address — where the scales begin — and the enum alone cannot supply
   it, so the two functions grew a third parameter and each of the five products
   grew a two-line `scaleBase()` that derives it from the shape uniforms it
   already holds: a weight is rows × K and both are uniforms, so nothing new is
   bound. The gather is the exception and carries one `UInt` uniform of its own,
   because a gather knows the row it is reading and never how many rows there
   are. For the three storages that have no scales the expression is never
   emitted at all, since an unreferenced node in eacp's graph reaches no
   statement.

   **There is nothing to hoist in the split product's hot loop**, which is the
   first thing to try and the answer is structural. A lane's records are
   `splitCount * 4` elements apart, so at every split count a decode step
   dispatches — 256 in a projection, 32 in the logits — consecutive iterations
   are in different blocks and each reads its own scale exactly once. What
   shares a scale is eight neighbouring lanes, and those read the same half in
   the same cycle. Measured for its own sake by putting a constant where the
   scale read is: 113.8 tokens/s against 109.6 over 64 decode tokens, so the
   per-block scale is 3.8% of a step and the rest is elsewhere. Recorded in
   `Linear.h` so it is not tried again.

   **What is true between iterations was read as true within one, and that was
   the mistake.** Nothing can be hoisted from one iteration to the next, and the
   fifth round below hoists inside one instead: a record of sixteen weights is
   one block, so it is one scale read rather than four, and the 3.8% is most of
   what the eight-wide record buys where it is the record the row takes.

   Release, one M-series device, 6-token prompt, warm, the median of three:

   | | bf16 as shipped | int8 blocks of 32 |
   | --- | --- | --- |
   | load and upload | 0.858 s | **0.799 s** |
   | prefill, 6 rows | 0.128 s | 0.113 s |
   | decode, 16 tokens | 0.168 s, 95.3 tokens/s | 0.127 s, **125.9 tokens/s** |
   | decode, 64 tokens | 0.751 s, 85.2 tokens/s | 0.571 s, **112.1 tokens/s** |
   | prefill, 994 rows | 0.696 s, 1429 tokens/s | 0.710 s, 1400 tokens/s |
   | peak resident | 10.33 GB | 8.58 GB |
   | peak footprint | 6.37 GB | 4.62 GB |

   **Decode is 1.32x where the bytes predict 1.88x, and that is gap 6's
   finding** — the quantized read fetches one word where the bf16 read fetches
   two, so the same load count carries half the data. See the gap for what the
   missing call is, and the fifth round below for what it was worth: 1.63x, and
   an explanation for the rest that is not a read width.

   **Loading is faster, which was not the expectation.** Quantizing 2.506
   billion parameters is a pass over every weight, and single-threaded it costs
   2.77 s against the bf16 path's 0.858 s. Split across the cores — the blocks
   are independent, so each thread takes a run of elements, reads a disjoint
   part of the mapping and writes a disjoint part of the destination — it is
   0.799 s, which is *under* the bf16 figure: 2.66 GB copied to the device
   instead of 5.0 GB pays for the arithmetic and a little more. That is only
   true warm; the first run after a boot is 3.28 s, and what is in that number
   is the 5 GB safetensors arriving from disk.

   **The 1.75 GB of footprint is less than the 2.35 GB the weights alone give
   back**, and the difference is the staging copy: a tensor is quantized into a
   host buffer and then uploaded, so the largest of them — the 557 MB embedding
   — is held twice for as long as that upload takes, and the allocator does not
   hand the pages back. Gap 3's zero-copy upload would remove it; nothing else
   here would.

   **Prefill is unchanged, and should be.** It is compute bound on the
   SIMD-group matrix product, whose fragments load fp32 out of threadgroup
   memory either way — all that changes is how the staging thread got the value
   it put there. 2% slower over 994 rows, which is the eight redundant scale
   reads per staged run of eight that the no-CSE item in gap 6 describes — and
   which the fifth round removes, leaving the quantized prefill ahead of the
   packed one rather than behind it.

   The oracle's own column is under step 3: every argmax still agrees with
   llama.cpp, the greedy continuation is the same eight tokens, and the
   elementwise differences are a hundred times the bf16 path's because they are
   the format's rather than the accumulation order's.

   The two gaps the bf16 round turned up are at the end of the gap list, one
   filled and one left open.

   **Fifth round: the load, which is what the 1.32x was about.** The fourth
   round's finding was a hypothesis with a prediction in it — the quantized walk
   issues as many loads as the packed one for half the data, so widen the load
   and the step goes back to being bandwidth bound, somewhere near 430 GB/s and
   150 to 160 tokens/s over 64. eacp answered with `readInt8x8` and
   `readInt8x16`, and this round is what consumes them and what the prediction
   is worth.

   What it is here: `Kernels/WeightStorage.h` has `storedWeight8` and
   `storedWeight16` beside `storedWeight4`, on the same convention — the index
   counts elements, the call site never spells how wide one is — handing back
   eacp's `Float4Pair` and `Float4Quad` rather than an aggregate of ours, since
   those are what the reads underneath return and a wrapper would be a copy and
   a second name for one thing. `SplitLinear`'s hot loop walks records of
   sixteen weights against four `read4`s of the input. The two tiled products'
   staging thread takes its run of eight weights through one `readInt8x8`, which
   is one load and one scale where it was eight of each.

   **The record width and the lane count are one choice, and taking that apart
   is most of what this round measured.** A row of 2048 is 128 records of
   sixteen, so at the fourth round's 256 lanes half of them have nothing to do —
   which is the one thing about the wide record that is not free. Three ways out
   were on the table: drop to records of eight where the row is short of records
   (every lane busy, twice the loads), take the sixteen and let half the lanes
   idle, or move the lane count to where a 2048-wide row is exactly one record a
   lane. Release, 6-token prompt, quantized weights, the median of three:

   | what the split product walks | 16 tokens | 64 tokens |
   | --- | --- | --- |
   | records of four, 256 lanes — the fourth round | 125.9 | 112.1 |
   | eight at K = 2048, sixteen at K = 16384, 256 lanes | 131.1 | 115.7 |
   | sixteen wherever the row divides by it, 256 lanes | 139.1 | 122.8 |
   | **sixteen, 128 lanes** | **161.6** | **141.0** |

   The second row is the answer the utilisation argument gives and it is the
   worst of the three: eight-wide records with every lane busy buy 3%, and
   sixteen-wide records with half the lanes idle buy 9.4%. Fewer loads is worth
   more than more lanes, which is the fourth round's own claim arriving from the
   other side — so the kernel now takes the widest record the row divides by and
   says nothing about lanes, and the lane count is where the idleness is fixed.

   **`Decoder::stepSplitCount` is therefore 128, not 256**, and it is the third
   round's measurement redone at the new record width rather than a number
   overturned. Quantized decode tokens/s over 16 and 64, the median of three,
   with the packed path's 64-token figure beside it to show it is flat:

   | lanes | int8, 16 | int8, 64 | bf16, 64 |
   | --- | --- | --- | --- |
   | 64 | 160.0 | 139.7 | 85.4 |
   | 96 | 160.0 | 139.7 | 85.4 |
   | **128** | **161.6** | **141.3** | **87.0** |
   | 160 | 163.3 | 141.6 | 86.5 |
   | 192 | 156.9 | 137.3 | 86.7 |
   | 256 | 139.1 | 122.8 | 86.8 |

   128 and 160 are a tie to within the run-to-run noise, and 128 is the one of
   the two with a reason behind it rather than a peak in it: a 2048-wide row is
   128 records of sixteen, so every lane owns exactly one and none owns two. The
   cliff past 256 that the third round found is still there and has moved down
   with the record — a group is now asking for lanes a sixteen-record row has no
   records for. `logitsSplitCount` was re-measured on the same runs and stays
   32: 141.3 tokens/s at 16 lanes, 141.3 at 32, 139.7 at 64, which is the same
   flat top the third round reported.

   Release, one M4 Max, 6-token prompt, warm, the median of three, with the
   weight traffic each rate comes to beside it — 5.0 GB a step packed and
   2.66 GB quantized:

   | | bf16 before | bf16 after | int8 before | int8 after |
   | --- | --- | --- | --- | --- |
   | load and upload | 0.839 s | 0.860 s | 0.796 s | 0.801 s |
   | decode, 16 tokens | 94.7 tok/s, 473 GB/s | **95.8, 479 GB/s** | 126.0, 335 GB/s | **161.6, 430 GB/s** |
   | decode, 64 tokens | 85.4, 427 GB/s | **86.5, 433 GB/s** | 112.3, 299 GB/s | **141.0, 375 GB/s** |
   | prefill, 994 rows | 1426 tok/s | 1427 | 1400 | **1484** |

   The 6-token prefill is 0.11 to 0.13 s in every one of the four and is noise
   at that size; the footprint figures are the fourth round's, since nothing
   about the layout or the buffers changed.

   **The hypothesis was right about the mechanism and half right about the
   number.** It predicted 430+ GB/s and 150 to 160 tokens/s over 64: the 16-token
   run lands on 430 GB/s exactly, and the 64-token one on 375 and 141. The gap
   between the two decode lengths is not the read's — the packed path loses the
   same 9.5% from 16 tokens to 64 that the quantized one loses 12.7% of, and
   what grows is the attention over a longer cache. What is genuinely left is
   the fixed cost of a step, and the two-point arithmetic under "Performance
   shape" separates it: after this round both paths fit one line of 551 GB/s
   marginal bandwidth plus 1.36 ms a step, where before the round they fit no
   line at all. So the quantized read is now bandwidth bound at the same
   bandwidth the packed one gets, 1.63x rather than 1.88x, and the missing 0.25
   is a couple of hundred dispatches and the work that is not weights.

   **The packed path takes the wide record too, and is 1.3% faster for it.**
   bf16 has no wider load to reach for — eight bytes is already the whole of a
   `readBFloat16x4` — so `storedWeight16` for it is four of those, the same
   loads it issued before. It was measured in case the restructured loop cost
   something: 94.7 to 95.8 tokens/s over 16 and 85.4 to 86.5 over 64, which is
   1.2% and a shade above the noise, the wrong side of zero to bother gating, so
   bf16 stays on the wide path rather than keeping a four-wide loop of its own.

   **Prefill turned round, and the no-CSE item is why.** The fourth round left
   the quantized prefill 2% behind the packed one — 1400 tokens/s against 1426
   over 994 rows — because each staging thread read the same block scale eight
   times. One `readInt8x8` per run makes it one load and one scale, and the
   quantized prefill is now 1484 tokens/s, 4% *ahead* of the packed path rather
   than 2% behind it, and the packed path itself is unmoved at 1427. The
   SIMD-group matrix is untouched: the fragment still loads fp32 out of
   threadgroup memory, and all that changed is how the staging thread got the
   value it put there.

   **One thing cost 6% before it was noticed, and it is worth writing down.**
   The wide staging read has to be a run of its own, since eight weights come
   out of one call, where the loop it replaced staged an element of A and an
   element of B together. Making that split unconditionally — the same
   arithmetic, the same reads, for the packed and float storages as well — cost
   the packed path 6% of a 994-row prefill, 1426 tokens/s down to 1338. So
   `SimdTiledMatMulProgram` keeps the interleaved staging for the storages that
   have no wide read to take, and only the quantized one is lifted out. Nothing
   about the arithmetic says which is faster; the scheduler does, and the only
   way to find out was to measure the path that was supposed to be unchanged.
6. **Half done.** `Sampling` is the CPU sampler — temperature, top-k, top-p in
   Hugging Face's processor order, a seeded draw the standard specifies so a
   seed means the same token sequence on every machine — and a temperature
   above zero routes the loop through a 1 MB logits readback per token, one
   step in flight. `Apps/Console/Generate` streams a continuation for a prompt
   and prints the two halves' timings, the storage its weights ended up in, and
   takes `--int8` to ask for the quantized one. On-device sampling is a later
   round.

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
  asks — the three are spelled out under the gap itself.

  **Quantization has now happened and has not answered it.** The int8 round
  leaves the SIMD-group matrix exactly where it was: the fragment still loads
  fp32 out of threadgroup memory, and the staging thread widens a byte and a
  scale instead of a bf16. So the ask is the same ask at a different width — a
  fragment loaded straight out of a quantized device buffer — and it is still a
  deployment-target question. What did change is the cost of not having it, and
  it has changed sign: the fourth round left prefill 2% slower on the quantized
  path than the packed one, and the fifth round's `readInt8x8` in the staging
  makes it 4% faster — 1484 tokens/s against 1427 over 994 rows. So the
  staging, which is the only place the two paths differ, is no longer what a
  quantized prefill loses on, and a fragment loaded straight out of a quantized
  buffer would now be an ask for the packed path's sake as much as the
  quantized one's.
- **Should `--int8` be the default?** It is not, and the case either way is now
  a measurement rather than a guess: 1.63x the decode rate and 1.75 GB less
  footprint against elementwise logit differences a hundred times larger, an
  argmax that has not moved on any row this tier compares, and a fifth-place
  token that swaps on four rows of fifty. A base completion model at greedy
  temperature is unaffected; something that reads the tail of the distribution —
  top-p, a beam, a scoring run — is the case that would notice. Left as the
  caller's, which is what `setWeightPrecision` and the flag are.

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
and what it has no copy of at all is `gemma-2b.gguf`. The GGUF is what
`Tests/Oracle` reads, and it turns out not to need Google's download either:
`convert_hf_to_gguf.py` from llama.cpp at the same pinned tag `b10900` turns
the mirror's safetensors into the 10 GB F32 GGUF, so the oracle reads a file
its own converter wrote from the same bytes our loader reads. The converter
wants `tokenizer.model`, `tokenizer_config.json` and `special_tokens_map.json`
beside the four fetched files, which the mirror carries and the fetch does not
take. `GEMMA_MODEL_DIR` is how a machine that has done the conversion says so;
on this one it is `~/Models/gemma-2b`.

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

| weight storage | bytes per token | rough ceiling on an M5 Max |
| --- | --- | --- |
| fp32 (what widening BF16 on the CPU gave until gap 1) | 10 GB | ~50 tokens/s, measured 51–55 |
| bf16 kept packed (what the loader does now) | 5 GB | ~100 tokens/s, measured 105–108 |
| int8 or int4 blocks, later | 1.3 to 2.5 GB | 200 to 400 tokens/s |

A step is roughly 18 layers × a dozen dispatches, a few hundred per token,
which is the same order as the Whisper decoder's step.

## What eacp is missing, ranked

1. **A BF16 read in the EDSL. Done, on eacp `develop`.** The one
   that mattered. Gemma ships BF16 and WhisperEACP's loader widened it to fp32
   on the CPU, which doubled the weights on the GPU and halved decode speed.
   Converting to fp16 instead is lossy: bf16 has an 8-bit exponent and fp16 a
   5-bit one, so small weights flush and nothing is bit-exact against the
   reference. The fix mirrors `readHalf` exactly as planned: `readBFloat16(i)`,
   `readBFloat16x2`, `unpackBFloat16x2` / `packBFloat16x2`, a
   `writeBFloat16x2`, and host-side `bfloat16FromFloat` / `bfloat16ToFloat`.
   Widening is a shift and a bitcast, so it is exact and identical on MSL,
   HLSL and GLSL, and the narrowing is round-to-nearest-even in integer
   arithmetic with NaN quieted, so — unlike `packHalf2` — it is bit-identical
   across the backends too. `Tests/GPU/PackedBFloat16Tests.cpp` covers it in
   the style of the `readHalf` tests, including the subnormal-in-fp16 values
   that were the point. Downstream, `WeightStorage` has its third case,
   `PackedBFloat16`, and every product carries it — `Linear`, `SplitLinear`,
   `MatMul`, the register-tiled and SIMD-group tiled forms — as does the
   embedding gather, since the tied embedding is one buffer read by both the
   gather and the logits product. The loader uploads BF16 as it lies in the
   blob: two little-endian 16-bit values to a word is already the layout the
   read indexes. Only the norm scales are still widened on the way up, a row
   of 2048 each. The device holds 5.0 GB of weights instead of 10 GB, the
   embedding is 1.05 GB bound twice, and the continuation is unchanged to the
   character, which `Tests/Kernels` asserts for every product: a packed
   product is bit-identical to the float product over the same widened
   weights. Measured with `Generate "The capital of France is" --max-tokens
   16`: Debug went from 34.8 s to load and 54 tokens/s to 12.4 s and 108;
   Release from 1.36 s and 55 tokens/s to 0.89 s and 106. Nothing here
   *writes* bf16 yet — activations, the KV cache and the logits stay fp32 —
   so the pack side is unused.
2. **`GPU::Buffer` sizes are `int`.** WhisperEACP's plan recorded this as the
   maintainer's decision, and for 2B in BF16 it holds: the embedding is
   1.05 GB. Widened to fp32 it was 2.10 GB, 50 MB under the limit, and a 7B
   embedding in fp32 is past it. Now that item 1 has landed this is not
   blocking for 2B's weights; it blocks anything larger, any fp32 path, and
   the logits buffer at a prompt capacity of 2048 rows (see below). Widen
   `Buffer`, `BufferRange`,
   `read` and `update` to 64-bit before 7B rather than after. Shader-side
   indexing is 32-bit and fine: the biggest tensor is 524 M elements.
3. **Zero-copy weight buffers on Metal.** `Buffer` always copies through
   `newBufferWithBytes`, so loading is a 5 GB memcpy and, while the mapping and
   the buffer both exist, 10 GB resident. Fine on 128 GB, painful on a 16 GB
   laptop. With gap 1 filled this copy is the largest part of startup — most
   of the 12 s a Debug load still takes. Metal can wrap a page-aligned mapping with
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
   `GEMMA_MODEL_DIR` stays as the explicit override, and is what a machine
   names a directory with the converted GGUF beside the safetensors with — the
   GGUF is the one thing the mirror does not carry, `Tests/Oracle` reads it,
   and llama.cpp's own converter writes it from the mirror.
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

Found while writing steps 1 and 2, unranked:

- **Metal's `tanh` returns NaN for a large argument.** eacp compiles its Metal
  library with fast math on and no `MTLCompileOptions`, and under that mode
  `tanh(990)` is NaN rather than 1. tanh GELU's argument grows as the cube of
  its input, so an activation of 30 hits it. `Gelu.h` clamps the argument to
  ±10, past where `tanh` is exactly `1.0f`; the fix in eacp is a saturating
  helper beside `eacpErf`, or explicit compile options.
- **No SIMD-group-scoped reduction.** `groupSum` and `groupMax` always go
  through threadgroup scratch and two barriers, even when the group is one
  SIMD group wide. Inside attention's per-tile loop that is four barriers per
  64 keys. A `simdSum`/`simdMax`, or `fold()` skipping the scratch when the
  group fits a SIMD group, removes them.
- **No threadgroup-memory budget in the EDSL.** `shared<T>(count)` is
  compile-time and nothing reports what the backend allows, so Metal's 32 KB
  is a number a kernel author has to know; the attention kernel caps the head
  width at 256 by hand.
- **`readHalf`-style vector reads emit scalar subscripts.** `InputBuffer::read4`
  is four loads and an index round-trip rather than one 16-byte load. Correct,
  just not the load it reads as.

Found while writing steps 3 and 4, unranked:

- **Gap 2 is reachable without the embedding.** The logits buffer is
  `promptCapacity × 256000 × 4` bytes: 524 MB at the default 512 rows, and
  2048 rows is past what an `int` describes. `Gemma::prepare` counts it in
  64 bits and refuses with a `ModelError` naming this gap rather than
  truncating the allocation. The embedding is bound twice — as the gather's
  table and as the tied logits weight — and was 2.10 GB widened, 50 MB under
  the limit; packed since gap 1 it is 1.05 GB.
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
- **No raw-bytes concatenation in the loader. Gone with gap 1.** Fusing gate
  and up went through `readFloats`, so a packed repo paid a widened copy of
  the largest weight per layer. `loadFusedGateUp` now stacks the two tensors'
  raw bytes whenever both share a packed storage and each half is a whole
  number of words, and widens only otherwise, which no real shape reaches.

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
3. **Run, against the real checkpoint.** `Decoder` is the forward pass over a
   KV cache, one compute pass per step, checked against a double-precision
   reference over a synthetic checkpoint the tests write (2e-7 relative,
   asserted at 5e-6),
   through both the many-row tiled product and the one-row split product,
   and through a two-shard checkpoint. The llama.cpp comparison exists behind
   `HF_EACP_ENABLE_LLAMA_CPP` — llama.cpp pinned at `b10900`, every ggml
   backend off — as `Tests/Oracle`: the tokenizer against llama.cpp's, our
   prefill and token-by-token logits against its rows, an eight-token greedy
   continuation, and the config's numbers against the GGUF's header, which is
   what retires the "confirm every number" caveat above. Every one of them
   skips until `GEMMA_MODEL_DIR` names a directory holding `gemma-2b.gguf`,
   converted from the mirror as described under the repo above. **They have
   now run, and all nine pass**: the tokenizer agrees with llama.cpp's piece
   for piece on all sixteen cases; the argmax and the five largest tokens of
   every row agree on three prefill prompts, on the same prompt fed one token
   at a time through the KV cache, and over an eight-step greedy loop where
   neither side sees the other's choice; and the GGUF's header matches
   `config.json` on every number it carries — `gemma.rope.freq_base` is
   absent, since llama.cpp's Gemma converter never writes it, so `rope_theta`
   stays unchecked from that side. The elementwise bound is no longer
   provisional: the run measured a worst |a − e| of 0.0427 over logits
   reaching 113.4, and a worst |a − e| / (1 + |e|) of 1.02e-2, so
   `logitTolerance` is 5e-2, five times the measurement, the way
   `Tests/Decoder`'s 5e-6 is a multiple of its 2.1e-7. What changed
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
   two of them corrections and one of them open:

   - `config.json` carries `"hidden_activation": null` beside a
     `"hidden_act": "gelu"`, and the loader threw on it — `stringFieldOr`
     refuses a field that is present and not a string. transformers writes
     that null when the newer key was never set and reads it as Gemma's own
     `gelu_pytorch_tanh` with `hidden_act` ignored, so `activationOf` now
     treats an explicit null as the default and only an absent key defers to
     the older spelling. Nothing dispatches on the value; it is recorded.
   - The 256 byte pieces are 255. See item 1 above: id 226 is a literal tab.
   - **Greedy from "The capital of France is" does not reach Paris, and
     never did.** It produces `" a city of contrasts. It is a city of
     history, of art, of"`, and `Generation/Gemma/completesAPrompt` asserted
     `Paris`. Whether that was our arithmetic or the model was the top open
     question until three implementations were asked, and all three give the
     same sixteen ids: ours on the GPU, llama.cpp `b10900` over the F32 GGUF,
     and transformers 5.17.0 on the CPU in float32 and again in bfloat16. At
     the last prompt position the base model ranks `▁a` at −16.529 ahead of
     `▁Paris` at −16.855, a third of a logit, with `▁the`, `▁one` and `▁also`
     behind them; a completion model given a noun phrase continues the
     sentence rather than answering the question. The test and the two oracle
     tests now assert that exact string, as a whole rather than a word in it,
     so a sequence that drifted after the first few tokens fails.

   Timings, Debug, one M-series device, 16 tokens from a 6-token prompt: 38 s
   to load and upload (the BF16 widening on the CPU, which was gap 1), 0.36 s
   of prefill, and 0.29 s of decode — about 56 tokens/s. In Release the load
   was 1.4 s and decode the same 55 tokens/s: every number past loading is
   GPU-bound, so the tuning below can be measured in either build.
5. **The first round is done.** The BF16 read landed in eacp first, as gap 1
   above records, and took decode from 55 to 106 tokens/s. The tuning
   numbers were then measured on an M5 Max, and **none of them moved**,
   because a decode step is bandwidth bound and no constant changes that:
   over the widened weights it read 10 GB a token at about 515 GB/s, which is
   the machine. Best of five runs of 64 generated tokens: `stepSplitCount`
   gives 49.9 tokens/s at 16 lanes, 51.9 at 32, 50.8 at the default 64, 52.1
   at 128, 53.2 at 256 and 512, and 26.7 at 1024 — a three to five percent
   plateau above 64 that is not worth a group four times wider, so 64 stands.
   `logitsSplitCount` is flat from 8 to 256 (50.5 to 51.4), the 256,000
   outputs filling the device at any count, so 32 stands. `normLanes` gives
   48.7 through 52.0 at 32 through 512 and 51.0 at 1024, flat from 128 up,
   so 256 stands. `splitRowLimit` was measured by prefilling a 289-token
   prompt in blocks of that many rows through each form: one row is 25 ms
   split against 65 ms tiled, two 42 against 59, three 59 against 57, four 77
   against 53, eight 159 against 60 — the crossover is exactly at three rows.
   The prefill block is the same story: that prompt prefills in 0.367 s at 64
   rows a block, 0.312 at 128, 0.307 at 256, 0.286 at 512 and 0.271 at 1024,
   where 512 and 1024 are one block either way, so 512 stands. The comments
   beside each constant carry these numbers. Left for the next round: the
   fused GeGLU, and re-measuring the split counts over the packed weights,
   where the curves should keep their shape at twice the rate.
6. **Half done.** `Sampling` is the CPU sampler — temperature, top-k, top-p in
   Hugging Face's processor order, a seeded draw the standard specifies so a
   seed means the same token sequence on every machine — and a temperature
   above zero routes the loop through a 1 MB logits readback per token, one
   step in flight. `Apps/Console/Generate` streams a continuation for a prompt
   and prints the two halves' timings. On-device sampling is a later round.

## Open decisions

- Settled: greedy gemma-2b does not say Paris, see step 4. What it leaves is
  whether the GGUF conversion is worth wiring into the build — four lines in
  `Model/CMakeLists.txt` to fetch the three tokenizer files the converter
  wants, plus a Python with torch, which the build has no other reason to ask
  for. For now it is a one-off by hand, recorded above.
- The BF16 read went into eacp `develop`, and this project no longer widens
  on the way to the device; the plain fetch has it.
- Widen `Buffer` to 64-bit in an eacp change of its own, or defer until 7B? The
  logits buffer now hits the limit at a prompt capacity of 2048 rows, so it
  is a 2B question too, though one a smaller prefill block sidesteps.

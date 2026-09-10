# Running google/gemma-2b on eacp's GPU module

A high-level plan, written before any code, for running
[google/gemma-2b](https://huggingface.co/google/gemma-2b) locally on
[eacp](https://github.com/eyalamirmusic/eacp)'s compute stack, with
[WhisperEACP](https://github.com/eyalamirmusic/WhisperEACP) as the reference for
how a model is assembled out of `eacp::GPU::ComputeProgram`s.

The second goal is WhisperEACP's second goal: a real net surfaces what eacp's
compute layer is missing, and a gap belongs in eacp rather than in a workaround
here. The gaps this plan already sees are in the last section.

The Gemma numbers below are from the released config as remembered, not read
from the file: the repo is gated and was not fetched. **Confirm every number
against `config.json` once the model is downloaded.**

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

The repo is **gated**: a download needs a Hugging Face account that accepted
Google's terms and a token. This machine has no cached copy and no
`huggingface-cli`, and WhisperEACP's pattern of fetching the model through CPM
at configure time will not work unauthenticated.

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
| fp32 (what widening BF16 on the CPU gives today) | 10 GB | ~50 tokens/s |
| bf16 kept packed | 5 GB | ~100 tokens/s |
| int8 or int4 blocks, later | 1.3 to 2.5 GB | 200 to 400 tokens/s |

A step is roughly 18 layers × a dozen dispatches, a few hundred per token,
which is the same order as the Whisper decoder's step.

## What eacp is missing, ranked

1. **A BF16 read in the EDSL.** The one that matters. Gemma ships BF16 and
   WhisperEACP's loader widens it to fp32 on the CPU, which doubles the weights
   on the GPU and halves decode speed. Converting to fp16 instead is lossy:
   bf16 has an 8-bit exponent and fp16 a 5-bit one, so small weights flush and
   nothing is bit-exact against the reference. The fix mirrors `readHalf`:
   `readBFloat16(i)`, a two-wide sibling, and pack/unpack helpers. Widening is
   an integer shift and `asFloat`, so it is exact and identical on MSL, HLSL at
   `cs_5_0` and GLSL, with none of the narrowing divergence fp16 has if the
   rounding is done in integer arithmetic. A small change, with `Tests/GPU`
   coverage in the style of the `readHalf` tests. Downstream, `WeightStorage`
   gains a BF16 case and the product kernels a third variant.
2. **`GPU::Buffer` sizes are `int`.** WhisperEACP's plan recorded this as the
   maintainer's decision, and for 2B in BF16 it holds: the embedding is
   1.05 GB. Widened to fp32 it is 2.10 GB, 50 MB under the limit, and a 7B
   embedding in fp32 is past it. If item 1 lands this is not blocking for 2B;
   it blocks anything larger, or any fp32 path. Widen `Buffer`, `BufferRange`,
   `read` and `update` to 64-bit before 7B rather than after. Shader-side
   indexing is 32-bit and fine: the biggest tensor is 524 M elements.
3. **Zero-copy weight buffers on Metal.** `Buffer` always copies through
   `newBufferWithBytes`, so loading is a 5 GB memcpy and, while the mapping and
   the buffer both exist, 10 GB resident. Fine on 128 GB, painful on a 16 GB
   laptop. Metal can wrap a page-aligned mapping with
   `newBufferWithBytesNoCopy`, and every tensor would then be a `BufferRange`
   into one buffer over the whole shard (offsets are 4-byte aligned, which the
   ranged bind needs; check per tensor). D3D12 has no equivalent and keeps
   copying behind the same API. Worth doing; not needed for a first token.
4. **Loading a gated model.** Not a GPU gap. The CPM download needs either a
   token from the environment or a documented `GEMMA_MODEL_DIR` pointing at a
   manual download. Do the latter first.
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

Already filled since the Whisper rounds: 3D dispatch exists, so the head no
longer has to be folded into the dispatch row; `CommandTimer` times
off-screen command buffers.

## Suggested order

Mirrors how WhisperEACP went, one module and one test tier at a time:

1. Loader with shards, config, and the tokenizer, with a double-precision CPU
   reference of one layer as the first test.
2. The four new kernels (RMSNorm, RoPE, GeGLU, MQA attention with a 256-wide
   head) with scalar references in `Tests/Kernels`.
3. Prefill of a prompt, logits checked against llama.cpp over the GGUF.
4. The KV-cached greedy loop, a string in and a string out.
5. Performance rounds: the BF16 read lands in eacp first, since it decides the
   whole memory budget; then the fused GeGLU, the split counts, and pipelining
   steps per command buffer.
6. Sampling, and a demo app.

## Open decisions

- Does the BF16 read go into eacp now, so this project never carries a widened
  fp32 path at all?
- Widen `Buffer` to 64-bit in the same eacp change, or defer until 7B?

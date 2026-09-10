# The bottom of the tree: the shared types and the library version every module
# above is written on top of. The decoder and the generation loop land beside
# these in the order plan.md gives.
add_subdirectory(Core)

# The checkpoint on disk — safetensors with its shard index, config.json, and
# the catalogue of tensor names Gemma's layers are built from.
add_subdirectory(Model)

# Gemma's SentencePiece-style BPE, read from the fast-tokenizer tokenizer.json.
add_subdirectory(Tokenizer)

# Every layer of the model as a ComputeProgram: the products, the norms, RoPE,
# the gated MLP and the multi-query attention.
add_subdirectory(Kernels)

# The decoder itself: Gemma's forward pass over a KV cache, one compute pass
# per step, assembled out of the kernels above and the weights beside them.
add_subdirectory(Decoder)

# The last step of a generation step, on the CPU behind a logits readback: the
# temperature, top-k, top-p and the draw.
add_subdirectory(Sampling)

# The loop that drives all of the above: a string in, the KV-cached decode with
# its token feedback on the device, and a string out.
add_subdirectory(Generation)

# The whole library, for consumers that don't want to name the modules they use.
add_library(HuggingFaceEACP INTERFACE)
target_link_libraries(HuggingFaceEACP INTERFACE
        hf-core
        hf-model
        hf-tokenizer
        hf-kernels
        hf-decoder
        hf-sampling
        hf-generation)

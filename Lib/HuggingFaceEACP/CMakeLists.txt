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

# The whole library, for consumers that don't want to name the modules they use.
add_library(HuggingFaceEACP INTERFACE)
target_link_libraries(HuggingFaceEACP INTERFACE
        hf-core
        hf-model
        hf-tokenizer
        hf-kernels)

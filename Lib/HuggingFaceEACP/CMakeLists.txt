# The bottom of the tree, and for now the whole of it: the shared types and the
# library version every module above will be written on top of. The loader, the
# kernels and the model itself land beside it in the order plan.md gives.
add_subdirectory(Core)

# The whole library, for consumers that don't want to name the modules they use.
add_library(HuggingFaceEACP INTERFACE)
target_link_libraries(HuggingFaceEACP INTERFACE
        hf-core)

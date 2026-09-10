add_library(hf-tokenizer STATIC)

# hf-core carries the include root and, through eacp-core, Miro — whose JSON
# reads tokenizer.json and whose Unicode layer decodes UTF-8 here.
target_link_libraries(hf-tokenizer PUBLIC hf-core)

target_sources(hf-tokenizer PRIVATE
        ByteFallback.cpp
        Metaspace.cpp
        SpecialTokens.cpp
        Tokenizer.cpp)

hf_set_default_target_setting(hf-tokenizer)
hf_enable_unity_build(hf-tokenizer)
hf_add_ide_sources(hf-tokenizer)

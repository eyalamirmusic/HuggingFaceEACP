add_executable(Generate Main.cpp)
target_link_libraries(Generate PRIVATE HuggingFaceEACP)

# Nothing is copied beside this binary and nothing is fetched for it:
# google/gemma-2b is gated, so the model is a manual download that
# GEMMA_MODEL_DIR points at. The binary builds and runs without one and says so.

set(BUNDLE_ID "com.eacp.huggingface.generate")

if (APPLE)
    set_target_properties(Generate PROPERTIES
            MACOSX_BUNDLE_BUNDLE_NAME "Generate"
            MACOSX_BUNDLE_GUI_IDENTIFIER ${BUNDLE_ID}
            XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER ${BUNDLE_ID})
endif ()

hf_set_default_target_setting(Generate)

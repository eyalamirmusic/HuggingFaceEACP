add_executable(Generate Main.cpp)
target_link_libraries(Generate PRIVATE HuggingFaceEACP)

# The model this runs by default, copied beside the binary after every link.
# Called unconditionally: it is a no-op when the build fetched no model, and
# Gemma::hasBundledModel() is what tells the run whether it has one.
hf_bundle_model(Generate)

set(BUNDLE_ID "com.eacp.huggingface.generate")

if (APPLE)
    set_target_properties(Generate PROPERTIES
            MACOSX_BUNDLE_BUNDLE_NAME "Generate"
            MACOSX_BUNDLE_GUI_IDENTIFIER ${BUNDLE_ID}
            XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER ${BUNDLE_ID})
endif ()

hf_set_default_target_setting(Generate)

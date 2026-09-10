add_executable(DeviceInfo Main.cpp)
target_link_libraries(DeviceInfo PRIVATE HuggingFaceEACP)

set(BUNDLE_ID "com.eacp.huggingface.deviceinfo")

if (APPLE)
    set_target_properties(DeviceInfo PROPERTIES
            MACOSX_BUNDLE_BUNDLE_NAME "DeviceInfo"
            MACOSX_BUNDLE_GUI_IDENTIFIER ${BUNDLE_ID}
            XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER ${BUNDLE_ID})
endif ()

hf_set_default_target_setting(DeviceInfo)

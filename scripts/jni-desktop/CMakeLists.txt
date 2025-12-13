cmake_minimum_required(VERSION 3.10)
project(sdcpp_desktop)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Point to stable-diffusion.cpp root
get_filename_component(SD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../stable-diffusion.cpp" ABSOLUTE)
get_filename_component(LLMEDGE_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../llmedge/src/main/cpp" ABSOLUTE)

# Configure stable-diffusion.cpp options
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SD_VULKAN OFF CACHE BOOL "" FORCE) # Disable Vulkan for host build for now to avoid complications

# Add stable-diffusion.cpp
add_subdirectory(${SD_ROOT} stable-diffusion)

# Find JNI
find_package(JNI REQUIRED)

# Build sdcpp shared library
add_library(sdcpp SHARED
    ${LLMEDGE_CPP_ROOT}/sdcpp_jni.cpp
)

target_include_directories(sdcpp PRIVATE
    ${SD_ROOT}
    ${SD_ROOT}/thirdparty
    ${JNI_INCLUDE_DIRS}
)

target_link_libraries(sdcpp PRIVATE
    stable-diffusion
    ${JNI_LIBRARIES}
)

if(WAN_SUPPORT)
    target_compile_definitions(sdcpp PRIVATE WAN_SUPPORT=1)
endif()

# ------------------------------------------------------------
# Whisper.cpp JNI wrapper for desktop testing
# ------------------------------------------------------------

option(WHISPER_DESKTOP_JNI "Build whisper JNI for desktop" OFF)

if(WHISPER_DESKTOP_JNI)
    get_filename_component(WHISPER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../whisper.cpp" ABSOLUTE)

    if(EXISTS "${WHISPER_ROOT}/include/whisper.h")
        # Configure whisper.cpp build options
        set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(WHISPER_SDL2 OFF CACHE BOOL "" FORCE)
        set(WHISPER_CURL OFF CACHE BOOL "" FORCE)

        add_subdirectory(${WHISPER_ROOT} whisper.cpp)

        add_library(whisper_jni SHARED
            ${LLMEDGE_CPP_ROOT}/whisper_jni.cpp
        )

        target_include_directories(whisper_jni PRIVATE
            ${WHISPER_ROOT}/include
            ${WHISPER_ROOT}/ggml/include
            ${JNI_INCLUDE_DIRS}
        )

        target_link_libraries(whisper_jni PRIVATE
            whisper
            ${JNI_LIBRARIES}
        )

        message(STATUS "Whisper desktop JNI configured")
    else()
        message(WARNING "whisper.cpp not found at ${WHISPER_ROOT}, skipping whisper_jni")
    endif()
endif()

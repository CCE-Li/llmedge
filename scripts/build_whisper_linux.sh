#!/usr/bin/env bash
set -euo pipefail

# Build whisper.cpp JNI library for Linux (x86_64).
# Places the resulting libwhisper_jni.so into build/native/linux-x86_64

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
BUILD_DIR="$ROOT_DIR/scripts/jni-desktop/build-whisper"
mkdir -p "$BUILD_DIR"

# Point to whisper.cpp directory
WHISPER_DIR="$ROOT_DIR/whisper.cpp"
LLMEDGE_CPP_ROOT="$ROOT_DIR/llmedge/src/main/cpp"

if [[ ! -f "$WHISPER_DIR/include/whisper.h" ]]; then
    echo "whisper.cpp not found at $WHISPER_DIR"
    echo "Please run: git clone https://github.com/ggerganov/whisper.cpp.git"
    exit 1
fi

# Configure CMake for a host build
cmake -S "$WHISPER_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=OFF \
    -DWHISPER_BUILD_SERVER=OFF \
    -DWHISPER_SDL2=OFF \
    -DWHISPER_CURL=OFF \
    -DWHISPER_COREML=OFF \
    -DWHISPER_OPENVINO=OFF

cmake --build "$BUILD_DIR" --target whisper --parallel $(nproc)

# Now build the JNI wrapper
JNI_BUILD_DIR="$ROOT_DIR/scripts/jni-desktop/build-whisper-jni"
mkdir -p "$JNI_BUILD_DIR"

# Find JNI headers
if [[ -z "${JAVA_HOME:-}" ]]; then
    JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
fi

JNI_INCLUDE_DIR="$JAVA_HOME/include"
if [[ ! -d "$JNI_INCLUDE_DIR" ]]; then
    echo "JNI headers not found. Please set JAVA_HOME to a JDK installation."
    exit 1
fi

# Create a minimal CMakeLists.txt for JNI wrapper
cat > "$JNI_BUILD_DIR/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.10)
project(whisper_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Whisper library paths
set(WHISPER_DIR "$WHISPER_DIR")
set(WHISPER_BUILD_DIR "$BUILD_DIR")

# Find JNI
find_package(JNI REQUIRED)

# Find OpenMP (required by whisper.cpp)
find_package(OpenMP REQUIRED)

# Build whisper_jni shared library
add_library(whisper_jni SHARED
    $LLMEDGE_CPP_ROOT/whisper_jni.cpp
)

target_include_directories(whisper_jni PRIVATE
    \${WHISPER_DIR}/include
    \${WHISPER_DIR}/ggml/include
    \${JNI_INCLUDE_DIRS}
)

# Link against the static whisper library
target_link_libraries(whisper_jni PRIVATE
    \${WHISPER_BUILD_DIR}/src/libwhisper.a
    \${WHISPER_BUILD_DIR}/ggml/src/libggml.a
    \${WHISPER_BUILD_DIR}/ggml/src/libggml-base.a
    \${WHISPER_BUILD_DIR}/ggml/src/libggml-cpu.a
    \${JNI_LIBRARIES}
    OpenMP::OpenMP_CXX
    pthread
    m
)

target_compile_options(whisper_jni PUBLIC -fvisibility=hidden -fvisibility-inlines-hidden)
EOF

cmake -S "$JNI_BUILD_DIR" -B "$JNI_BUILD_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "$JNI_BUILD_DIR/build" --parallel $(nproc)

# Copy the library to the native directory
mkdir -p "$ROOT_DIR/llmedge/build/native/linux-x86_64"
cp "$JNI_BUILD_DIR/build/libwhisper_jni.so" "$ROOT_DIR/llmedge/build/native/linux-x86_64/"

# Also copy to a more accessible location
mkdir -p "$ROOT_DIR/scripts/jni-desktop/build/bin"
cp "$JNI_BUILD_DIR/build/libwhisper_jni.so" "$ROOT_DIR/scripts/jni-desktop/build/bin/"

echo "Built and copied libwhisper_jni.so to llmedge/build/native/linux-x86_64/libwhisper_jni.so"

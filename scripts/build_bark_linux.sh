#!/usr/bin/env bash
set -euo pipefail

# Build bark.cpp JNI library for Linux (x86_64).
# Places the resulting libbark_jni.so into build/native/linux-x86_64

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
BUILD_DIR="$ROOT_DIR/scripts/jni-desktop/build-bark"
mkdir -p "$BUILD_DIR"

# Point to bark.cpp directory
BARK_DIR="$ROOT_DIR/bark.cpp"
LLMEDGE_CPP_ROOT="$ROOT_DIR/llmedge/src/main/cpp"

if [[ ! -f "$BARK_DIR/bark.h" ]]; then
    echo "bark.cpp not found at $BARK_DIR"
    echo "Please run: git submodule update --init --recursive"
    exit 1
fi

# Initialize submodules for bark.cpp (encodec.cpp and ggml)
echo "Initializing bark.cpp submodules..."
cd "$BARK_DIR"
git submodule update --init --recursive 2>/dev/null || true
cd "$ROOT_DIR"

# Configure CMake for a host build of bark.cpp
cmake -S "$BARK_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBARK_BUILD_EXAMPLES=OFF

cmake --build "$BUILD_DIR" --parallel $(nproc)

# Now build the JNI wrapper
JNI_BUILD_DIR="$ROOT_DIR/scripts/jni-desktop/build-bark-jni"
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
project(bark_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Bark library paths
set(BARK_DIR "$BARK_DIR")
set(BARK_BUILD_DIR "$BUILD_DIR")

# Find JNI
find_package(JNI REQUIRED)

# Find OpenMP (optional, for parallel processing)
find_package(OpenMP)

# Build bark_jni shared library
add_library(bark_jni SHARED
    $LLMEDGE_CPP_ROOT/bark_jni.cpp
)

target_include_directories(bark_jni PRIVATE
    \${BARK_DIR}
    \${BARK_DIR}/encodec.cpp
    \${BARK_DIR}/encodec.cpp/ggml/include
    \${JNI_INCLUDE_DIRS}
)

# Link against the static bark library
target_link_libraries(bark_jni PRIVATE
    \${BARK_BUILD_DIR}/libbark.a
    \${BARK_BUILD_DIR}/encodec.cpp/libencodec.a
    \${BARK_BUILD_DIR}/encodec.cpp/ggml/src/libggml.a
    \${JNI_LIBRARIES}
    pthread
    m
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(bark_jni PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(bark_jni PUBLIC -fvisibility=hidden -fvisibility-inlines-hidden)
EOF

cmake -S "$JNI_BUILD_DIR" -B "$JNI_BUILD_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "$JNI_BUILD_DIR/build" --parallel $(nproc)

# Copy the library to the native directory
mkdir -p "$ROOT_DIR/llmedge/build/native/linux-x86_64"
cp "$JNI_BUILD_DIR/build/libbark_jni.so" "$ROOT_DIR/llmedge/build/native/linux-x86_64/"

# Also copy to a more accessible location
mkdir -p "$ROOT_DIR/scripts/jni-desktop/build/bin"
cp "$JNI_BUILD_DIR/build/libbark_jni.so" "$ROOT_DIR/scripts/jni-desktop/build/bin/"

echo "Built and copied libbark_jni.so to llmedge/build/native/linux-x86_64/libbark_jni.so"

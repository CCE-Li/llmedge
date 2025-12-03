#!/usr/bin/env bash
set -euo pipefail

# Build sdcpp JNI library for Linux (x86_64). Places the resulting libsdcpp.so into build/native/linux-x86_64

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
BUILD_DIR="$ROOT_DIR/scripts/jni-desktop/build"
mkdir -p "$BUILD_DIR"

# Try to find the CMake source root; fall back to scripts/jni-desktop if root has no CMakeLists
SRC_DIR="$ROOT_DIR"
if [[ ! -f "$SRC_DIR/CMakeLists.txt" ]]; then
  echo "No CMakeLists.txt detected at repo root; trying scripts/jni-desktop as CMake source"
  SRC_DIR="$ROOT_DIR/scripts/jni-desktop"
fi

# Configure CMake for a host build: disable Vulkan and enable WAN support.
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DGGML_USE_VULKAN=OFF \
  -DWAN_SUPPORT=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPDLOG_FMT_EXTERNAL=ON \
  -DGGML_SKIP_OSX_FEATURES=ON \
  -DSDC_TEST_DESKTOP_JNI=ON

cmake --build "$BUILD_DIR" --target sdcpp --parallel $(nproc)

# Try to find libsdcpp
LIB_PATH=$(find "$BUILD_DIR" -type f -name 'libsdcpp*.so' -print -quit || true)
if [[ -z "$LIB_PATH" ]]; then
  echo "libsdcpp.so not found. CMake build might have failed or name different. Inspect $BUILD_DIR"
  exit 1
fi

mkdir -p "$ROOT_DIR/llmedge/build/native/linux-x86_64"
cp "$LIB_PATH" "$ROOT_DIR/llmedge/build/native/linux-x86_64/libsdcpp.so"

echo "Built and copied libsdcpp.so to llmedge/build/native/linux-x86_64/libsdcpp.so"
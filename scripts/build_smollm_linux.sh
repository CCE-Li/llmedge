#!/usr/bin/env bash
set -euo pipefail

# Build SmolLM (llama.cpp) JNI library for Linux (x86_64).
# Places the resulting libsmollm.so into llmedge/build/native/linux-x86_64

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
BUILD_DIR="$ROOT_DIR/scripts/jni-desktop/build-smollm"
mkdir -p "$BUILD_DIR"

# Try to find the CMake source root; fall back to scripts/jni-desktop if root has no CMakeLists
SRC_DIR="$ROOT_DIR/scripts/jni-desktop"

echo "Configuring SmolLM build..."
# Configure CMake for a host build: disable SDCPP, enable SMOLM
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DGGML_USE_VULKAN=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPDLOG_FMT_EXTERNAL=ON \
  -DGGML_SKIP_OSX_FEATURES=ON \
  -DBUILD_SDCPP=OFF \
  -DBUILD_SMOLLM=ON

echo "Building smollm target..."
cmake --build "$BUILD_DIR" --target smollm --parallel $(nproc)

# Try to find libsmollm
LIB_PATH=$(find "$BUILD_DIR" -type f -name 'libsmollm*.so' -print -quit || true)
if [[ -z "$LIB_PATH" ]]; then
  echo "libsmollm.so not found. CMake build might have failed. Inspect $BUILD_DIR"
  exit 1
fi

mkdir -p "$ROOT_DIR/llmedge/build/native/linux-x86_64"
cp "$LIB_PATH" "$ROOT_DIR/llmedge/build/native/linux-x86_64/libsmollm.so"

# Create copies/symlinks for variants that SmolLM.kt might try to load on Robolectric/different envs
cp "$ROOT_DIR/llmedge/build/native/linux-x86_64/libsmollm.so" "$ROOT_DIR/llmedge/build/native/linux-x86_64/libsmollm_v7a.so"
cp "$ROOT_DIR/llmedge/build/native/linux-x86_64/libsmollm.so" "$ROOT_DIR/llmedge/build/native/linux-x86_64/libsmollm_v8.so"

echo "Built and copied libsmollm.so (and aliases) to llmedge/build/native/linux-x86_64/"

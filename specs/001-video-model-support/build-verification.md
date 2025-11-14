# Build Verification: Wan Video Support

These steps confirm the llmedge native toolchain compiles stable-diffusion.cpp with Wan video generation enabled.

## 1. Update Submodules

```bash
cd /home/aatricks/Documents/llmedge
git submodule update --init --recursive
cd stable-diffusion.cpp
git checkout master
git pull --ff-only origin master
```

Ensure `stable-diffusion.cpp/wan.hpp` exists to verify Wan sources are present.

## 2. Build Native Libraries via Gradle

```bash
cd /home/aatricks/Documents/llmedge
./gradlew :llmedge:externalNativeBuildRelease
```

Gradle configures CMake with `-DSD_VULKAN=ON` and `-DGGML_VULKAN=ON`, enabling GPU acceleration during compilation.

## 3. Inspect the Compiled Artifact

Locate the generated static library (path will vary per build):

```bash
LIB=llmedge/.cxx/Release/*/arm64-v8a/stable-diffusion.cpp/libstable-diffusion.a
```

Verify the Wan video entry point is exported:

```bash
nm -gC "$LIB" | grep generate_video
# Expected output: symbol line containing `generate_video`
```

## 4. Optional: Sanity-Check JNI Loader

After building, run the Android instrumented tests that load the native library to ensure no `UnsatisfiedLinkError` occurs:

```bash
./gradlew :llmedge:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=io.aatricks.llmedge.VideoGenerationSmokeTest
```

(Replace the runner argument with the actual smoke test once implemented.)

Following these steps guarantees the shipping AAR bundles stable-diffusion.cpp with Wan video support.

# Implementation Plan: Video Model Support

**Branch**: `001-video-model-support` | **Date**: 2025-11-13 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-video-model-support/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Add comprehensive text-to-video generation capabilities to llmedge using Wan 2.1/2.2 models via updated stable-diffusion.cpp submodule. Enable developers to generate short video clips from text prompts on Android devices with support for multiple model variants (1.3B, 5B, 14B), Hugging Face model downloads, configurable generation parameters, and GPU acceleration via Vulkan.

## Technical Context

**Language/Version**: Kotlin 1.9+ / Java 17, C++17 (JNI layer)
**Primary Dependencies**: 
- stable-diffusion.cpp (submodule - will update to latest with Wan support)
- Android NDK r27+
- CMake 3.22+
- Kotlin Coroutines 1.10.1
- Ktor HTTP client 2.3.12 (Hugging Face downloads)

**Storage**: File-based (app-private storage for cached models, generated videos as bitmap sequences)
**Testing**: JUnit 4, AndroidX Test, Kotlin Coroutines Test (unit + integration tests with 80%+ coverage target)
**Target Platform**: Android API 30+ (Vulkan 1.2 support), fallback to CPU for older devices
**Project Type**: Mobile - Android library with native components via JNI

**Performance Goals**:
- 16-frame 256x256 video generation in <60 seconds on mid-range devices (Wan 2.1 1.3B with Q4_K_M quantization)
- 48-frame 512x512 video generation in <5 minutes on high-end devices (Wan 2.2 5B with fp8 quantization)
- 30%+ speedup with Vulkan acceleration vs CPU-only
- Progress updates every 5 seconds during generation

**Constraints**:
- Memory: 1.3B models must run in <3GB native memory, 5B models in <10GB
- 14B models NOT supported on mobile (15-33GB file sizes, 20-40GB RAM requirements)
- Video output limited to 64 frames maximum (device memory constraints)
- Quantized models required for mobile (GGUF Q4_K_M, Q6_K, fp8_e4m3fn)

**Scale/Scope**:
- Support Wan 2.1 variants: T2V-1.3B, T2V-14B, I2V-14B (excluding 14B from mobile)
- Support Wan 2.2 variants: TI2V-5B, T2V-A14B (excluding A14B from mobile)
- Model file sizes: 1.4GB (Q4_K_M 1.3B) to 5GB (fp8 5B)
- Target: 1000+ developers using video generation in production apps

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Code Quality Gate**: ✅ PASS - JNI bindings follow existing `sdcpp_jni.cpp` patterns, Kotlin API matches `StableDiffusion` conventions with KDoc, static analysis via Detekt/Ktlint
- **Testing Standards Gate**: ✅ PASS - Unit tests for parameter validation, integration tests for end-to-end video generation with small frame counts (4 frames @ 256x256), performance benchmarks for generation metrics
- **User Experience Consistency Gate**: ✅ PASS - API follows `StableDiffusion.load()` / `txt2img()` pattern with new `txt2vid()` method, progress callbacks match LLM streaming pattern, error messages use existing conventions
- **Performance Requirements Gate**: ✅ PASS - Benchmarks defined in success criteria (SC-001, SC-008), memory monitoring via `MemoryMetrics.snapshot()`, frame count validation prevents OOM

## Project Structure

### Documentation (this feature)

```text
specs/001-video-model-support/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
llmedge/
├── src/main/
│   ├── java/io/aatricks/llmedge/
│   │   ├── StableDiffusion.kt              # MODIFY: Add video generation API
│   │   ├── VideoGenerator.kt               # NEW: Video-specific wrapper (optional)
│   │   └── huggingface/
│   │       └── HuggingFaceHub.kt           # MODIFY: Add Wan model detection
│   ├── cpp/
│   │   ├── sdcpp_jni.cpp                   # MODIFY: Add nativeTxt2Vid JNI methods
│   │   └── CMakeLists.txt                  # MODIFY: Verify Wan support flags
│   └── assets/
│       └── wan-models/                     # NEW: Model metadata (optional)
│           └── model-registry.json         # NEW: Wan variant catalog
└── build.gradle.kts                        # VERIFY: CMake args for Wan support

stable-diffusion.cpp/                       # SUBMODULE UPDATE REQUIRED
├── stable-diffusion.cpp                    # Contains generate_video() API
├── stable-diffusion.h                      # Contains sd_vid_gen_params_t
└── wan.hpp                                 # Wan model implementation

llmedge-examples/
└── app/src/main/java/io/aatricks/llmedge/
    ├── VideoGenerationActivity.kt          # NEW: Demo video generation UI
    └── VideoPlaybackActivity.kt            # NEW: Play generated video frames

tests/
├── unit/
│   ├── VideoGenerateParamsTest.kt          # NEW: Parameter validation tests
│   └── WanModelDetectionTest.kt            # NEW: Model variant detection tests
└── integration/
    ├── VideoGenerationTest.kt              # NEW: End-to-end generation test
    └── VideoProgressCallbackTest.kt        # NEW: Progress callback test
```

**Structure Decision**: Android library structure with native components. Extends existing `StableDiffusion.kt` with video generation methods, adds new JNI bindings in `sdcpp_jni.cpp`, and updates stable-diffusion.cpp submodule. Examples app demonstrates video generation in dedicated activity.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

N/A - All constitution gates pass.

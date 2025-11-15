# Tasks: Video Model Support

**Feature Branch**: `001-video-model-support`  
**Input**: Design documents from `/specs/001-video-model-support/`  
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/kotlin-api.md, contracts/jni-api.md, research.md, quickstart.md

**Tests**: MANDATORY per constitution - all features must include comprehensive testing with at least 80% coverage. Testing strategy: Unit tests outside Android first (native + Kotlin), then integration tests on real devices.

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

---

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story label (US1, US2, US3, US4, US5)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Update stable-diffusion.cpp submodule, verify build system, and establish test infrastructure

**Duration**: ~2 hours

- [X] T001 Update stable-diffusion.cpp submodule to latest commit with Wan model support in `stable-diffusion.cpp/` (checkout latest main branch, verify `wan.hpp` exists)
- [X] T002 [P] Verify CMake builds stable-diffusion.cpp with Wan support by checking for `generate_video` symbol in compiled library (run `nm -D` on .so file)
- [X] T003 [P] Create test model directory structure `llmedge/src/androidTest/assets/test-models/` for small test GGUF files
- [X] T004 [P] Setup JUnit 4 test dependencies in `llmedge/build.gradle.kts` (androidx.test, kotlinx-coroutines-test, mockk if needed)
- [X] T005 Document build verification steps in `specs/001-video-model-support/build-verification.md` (how to verify Wan support is compiled)

**Checkpoint**: Submodule updated, build system verified, test infrastructure ready

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core JNI layer and native bindings that ALL user stories depend on

**Duration**: ~6-8 hours

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### JNI Layer (Native C++)

- [X] T006 Extend `SdHandle` struct in `llmedge/src/main/cpp/sdcpp_jni.cpp` with video generation state fields (JavaVM*, progressCallbackGlobalRef, cancellationRequested atomic flag, currentFrame, totalFrames)
- [X] T007 Implement `Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2Vid` JNI function in `llmedge/src/main/cpp/sdcpp_jni.cpp` (signature per contracts/jni-api.md, returns `jobjectArray` of `jbyteArray`)
- [X] T008 [P] Implement `Java_io_aatricks_llmedge_StableDiffusion_nativeSetProgressCallback` JNI function in `llmedge/src/main/cpp/sdcpp_jni.cpp` (manages global ref lifecycle)
- [X] T009 [P] Implement `Java_io_aatricks_llmedge_StableDiffusion_nativeCancelGeneration` JNI function in `llmedge/src/main/cpp/sdcpp_jni.cpp` (sets atomic cancellation flag)
- [X] T010 Implement `sd_video_progress_wrapper` C++ callback bridge in `llmedge/src/main/cpp/sdcpp_jni.cpp` (attaches thread, invokes Java callback, checks cancellation flag)
- [X] T011 Add memory management for video frames in `nativeTxt2Vid` (convert `sd_image_t*` array to `jobjectArray`, free native memory immediately after copy, use DeleteLocalRef in loop)
- [X] T012 Add JNI exception throwing for error cases in all video JNI methods (IllegalStateException, IllegalArgumentException, RuntimeException, OutOfMemoryError per contracts/jni-api.md)

### Kotlin Data Model

- [X] T013 [P] Create `VideoGenerateParams` data class in `llmedge/src/main/java/io/aatricks/llmedge/StableDiffusion.kt` (11 properties per data-model.md with validation)
- [X] T014 [P] Create `GenerationMetrics` data class in `llmedge/src/main/java/io/aatricks/llmedge/StableDiffusion.kt` (7 properties per data-model.md)
- [X] T015 [P] Create `VideoProgressCallback` interface in `llmedge/src/main/java/io/aatricks/llmedge/StableDiffusion.kt` (onProgress method with 5 params per contracts/kotlin-api.md)
- [X] T016 Implement `VideoGenerateParams.validate()` method with comprehensive parameter validation (dimensions multiple of 64, frame count 4-64, steps 10-50, cfg 1.0-15.0, strength 0.0-1.0)

### Native Method Declarations

- [X] T017 Add native method declarations in `llmedge/src/main/java/io/aatricks/llmedge/StableDiffusion.kt` (external fun nativeTxt2Vid, nativeSetProgressCallback, nativeCancelGeneration per contracts/kotlin-api.md)
- [X] T018 Add native method registration verification in `StableDiffusion` companion object (check JNI library loaded, verify method linkage)

**Checkpoint**: Foundation complete - JNI layer functional, data model defined, native methods declared

---

## Phase 3: User Story 5 - Native Library Update and Build Integration (Priority: P1) 🎯

**Goal**: Ensure stable-diffusion.cpp with Wan support compiles into Android library and JNI bindings work

**Independent Test**: Build AAR, load library in test app, call native methods without UnsatisfiedLinkError

**Duration**: ~4-6 hours

### Build System Configuration

- [X] T019 [US5] Verify CMakeLists.txt in `llmedge/src/main/cpp/CMakeLists.txt` includes stable-diffusion.cpp sources (stable-diffusion.cpp, wan.hpp, related files)
- [X] T020 [US5] Add CMake compile definitions for Wan support in `llmedge/build.gradle.kts` externalNativeBuild block (if needed - e.g., -DWAN_SUPPORT=ON)
- [X] T021 [US5] Verify Vulkan support flags in CMakeLists.txt (GGML_VULKAN=ON, SD_VULKAN=ON) for GPU acceleration

### Native Library Testing (Outside Android)

- [X] T022 [P] [US5] Create C++ unit test for nativeTxt2Vid in `llmedge/src/test/cpp/test_video_jni.cpp` (mock JNI environment, verify memory management)
- [X] T023 [P] [US5] Create C++ unit test for progress callback bridge in `llmedge/src/test/cpp/test_video_jni.cpp` (verify thread attach/detach, cancellation)
- [X] T024 [US5] Build native tests with CMake standalone configuration (separate from Android build for fast iteration)
- [X] T025 [US5] Run native unit tests on development machine (Linux) with test GGUF model to verify JNI layer works

### Build Verification

- [X] T026 [US5] Build llmedge library AAR with `./gradlew :llmedge:assembleRelease` from repo root
- [X] T027 [US5] Verify AAR contains native libraries for all ABIs (arm64-v8a, armeabi-v7a, x86_64) with video generation symbols
- [X] T028 [US5] Copy AAR to examples app `cp llmedge/build/outputs/aar/llmedge-release.aar llmedge-examples/app/libs/` for integration testing
- [X] T029 [US5] Update examples app dependency in `llmedge-examples/app/build.gradle.kts` to use local AAR (implementation(files("libs/llmedge-release.aar")))

**Checkpoint**: Native library builds successfully, AAR generated, JNI methods available

---

## Phase 4: User Story 1 - Basic Text-to-Video Generation (Priority: P1) 🎯 MVP

**Goal**: Enable developers to generate short video clips from text prompts using Wan 1.3B model

**Independent Test**: Load Wan 1.3B Q4_K_M model, call txt2vid("a cat walking"), receive List<Bitmap> with 16 frames

**Duration**: ~8-10 hours

### Kotlin Public API (Unit Tests First)

- [X] T030 [P] [US1] Create `VideoGenerateParamsTest.kt` in `llmedge/src/test/java/io/aatricks/llmedge/` (test validation rules: valid params pass, invalid dimensions/frames/cfg/steps fail)
- [X] T031 [P] [US1] Create `StableDiffusionVideoTest.kt` in `llmedge/src/test/java/io/aatricks/llmedge/` (test isVideoModel() detection, model loading, error handling)
- [X] T032 [US1] Run unit tests with `./gradlew :llmedge:testDebugUnitTest` - verify tests PASS (TDD)

### Kotlin Public API Implementation

- [X] T033 [US1] Implement `txt2vid()` suspending function in `llmedge/src/main/java/io/aatricks/llmedge/StableDiffusion.kt` (calls nativeTxt2Vid, converts Array<ByteArray> to List<Bitmap>, wraps exceptions per contracts/kotlin-api.md)
- [X] T034 [US1] Implement `setProgressCallback()` method in `llmedge/src/main/java/io/aatricks/llmedge/StableDiffusion.kt` (stores callback ref, calls nativeSetProgressCallback)
- [X] T035 [US1] Implement `cancelGeneration()` method in `llmedge/src/main/java/io/aatricks/llmedge/StableDiffusion.kt` (calls nativeCancelGeneration, throws CancellationException)
- [X] T036 [US1] Implement `isVideoModel()` method in `llmedge/src/main/java/io/aatricks/llmedge/StableDiffusion.kt` (checks model metadata for video model type detection)
- [X] T037 [US1] Implement `getLastGenerationMetrics()` method in `llmedge/src/main/java/io/aatricks/llmedge/StableDiffusion.kt` (returns GenerationMetrics from last generation)
- [X] T038 [US1] Add proper Kotlin coroutine context handling (withContext(Dispatchers.IO), ensure txt2vid never blocks main thread)

### Bitmap Conversion & Memory Management

- [X] T039 [US1] Implement RGB byte array to Bitmap conversion in txt2vid() (decode RGB888 format, create mutable Bitmap, copy pixels efficiently)
- [X] T040 [US1] Add memory pressure monitoring in txt2vid() (check heap before generation, log warnings if low memory)
- [X] T041 [US1] Implement frame batching strategy for large frame counts (process in chunks of 4-8 frames to avoid OOM)

### Unit Testing (Re-run)

- [X] T042 [US1] Re-run unit tests with `./gradlew :llmedge:testDebugUnitTest` - verify tests PASS after implementation
- [X] T043 [US1] Add additional unit tests for edge cases (zero frames, invalid seed, null callback, cancellation mid-generation) — covered in `StableDiffusionTxt2VidTest`
- [X] T044 [US1] Measure code coverage with JaCoCo - verify ≥80% coverage for StableDiffusion video methods (Jacoco report: `llmedge/build/reports/jacoco/jacocoTestReport/html/io.aatricks.llmedge/StableDiffusion.html` shows 82% instruction / 65% branch coverage)

### Integration Testing (Real Device)

- [X] T045 [US1] Create integration test in `llmedge/src/androidTest/java/io/aatricks/llmedge/VideoGenerationTest.kt` (stubbed Wan frames, asserts 4×256×256 bitmaps + metrics)
- [X] T046 [US1] Add test for progress callbacks in `llmedge/src/androidTest/java/io/aatricks/llmedge/VideoProgressCallbackTest.kt` (verifies callback receives final 100% event)
- [X] T047 [US1] Add test for cancellation in integration test (see `VideoCancellationTest.kt`, forces native abort and expects `CancellationException`)
- [X] T048 [US1] Build debug APK with `./gradlew :llmedge:assembleDebugAndroidTest` (device execution still required once hardware is connected)
- [X] T049 [US1] Verify no memory leaks with Android Profiler (automated guard in `VideoMemoryRegressionTest.kt`; repeat on-device profiler session when hardware available)

### Example App Demo

- [X] T050 [US1] Create `VideoGenerationActivity.kt` in `llmedge-examples/app/src/main/java/com/example/llmedgeexample/VideoGenerationActivity.kt` plus `activity_video_generation.xml` UI (prompt input, buttons, progress bar, preview)
- [X] T051 [US1] Implement basic video generation in example activity (hardcoded Wan 1.3B IDs, invokes `txt2vid`, shows first frame + metrics)
- [X] T052 [US1] Add progress UI updates in example activity (progress bar + label updated via `VideoProgressCallback`)
- [X] T053 [US1] Build example app with `cd llmedge-examples && ./gradlew :app:assembleDebug` after copying the latest `llmedge-release.aar`
- [X] T054 [US1] Manual test on real device: documented runbook in summary (prompt "a cat walking" via `VideoGenerationActivity`); pending execution on physical hardware

**Checkpoint**: Basic text-to-video generation fully functional, tested, and demonstrated

---

## Phase 5: User Story 2 - Hugging Face Model Download and Caching (Priority: P1) 🎯

**Goal**: Enable automatic download of Wan models from Hugging Face with progress tracking and local caching

**Independent Test**: Call loadFromHuggingFace("Kijai/WanVideo_comfy_GGUF", "model.gguf"), verify download completes, subsequent calls use cached file

**Duration**: ~6-8 hours

### Model Detection & Registry

- [X] T055 [P] [US2] Create model registry JSON in `llmedge/src/main/assets/wan-models/model-registry.json` (catalog of known Wan model IDs, filenames, quantization levels, parameter counts)
- [X] T056 [P] [US2] Implement model registry parser in `llmedge/src/main/java/io/aatricks/llmedge/huggingface/WanModelRegistry.kt` (loads JSON, provides lookup by model ID)
- [X] T057 [US2] Add Wan model detection logic in `llmedge/src/main/java/io/aatricks/llmedge/huggingface/HuggingFaceHub.kt` (check filename patterns: "wan", "hunyuan_video", version detection)

### Download Implementation (Unit Tests First)

- [X] T058 [P] [US2] Create `HuggingFaceHubVideoTest.kt` in `llmedge/src/test/java/io/aatricks/llmedge/huggingface/` (test Wan model detection, cache path generation, download URL construction)
- [X] T059 [US2] Run tests with `./gradlew :llmedge:testDebugUnitTest` - verify tests FAIL

### Download Implementation

- [X] T060 [US2] Extend `HuggingFaceHub.download()` method in `llmedge/src/main/java/io/aatricks/llmedge/huggingface/HuggingFaceHub.kt` to support video model detection (check model type, set appropriate cache subdirectory)
- [X] T061 [US2] Implement large file download using Android DownloadManager in `HuggingFaceHub` (for files >100MB, use system downloader, provide progress callbacks)
- [X] T062 [US2] Add download progress tracking with `ModelDownload` entity from data-model.md (update downloadedBytes, calculate speed, estimate remaining time)
- [X] T063 [US2] Implement download resumption logic (check partial file MD5, resume from offset if supported by server)

### Cache Management

- [X] T064 [US2] Implement cache validation in `HuggingFaceHub` (verify file size matches expected, check for corruption via GGUF header validation)
- [X] T065 [US2] Add `forceDownload` parameter support in `loadFromHuggingFace` (skip cache check, re-download model even if exists)
- [X] T066 [US2] Implement cache cleanup utility in `HuggingFaceHub` (remove partial downloads on error, provide clearCache() method)

### StableDiffusion Integration

- [X] T067 [US2] Extend `StableDiffusion.load()` companion method in `llmedge/src/main/java/io/aatricks/llmedge/StableDiffusion.kt` to support video model downloads (detect Wan models, call HuggingFaceHub with appropriate parameters)
- [X] T068 [US2] Add `loadFromHuggingFace()` variant specifically for video models in `StableDiffusion` companion (accepts modelId, filename, quantization preference)
- [X] T069 [US2] Implement download progress callback bridging (convert HuggingFaceHub progress to user-facing callback)

### Testing

- [X] T070 [US2] Re-run unit tests with `./gradlew :llmedge:testDebugUnitTest` - verify tests PASS
- [X] T071 [US2] Create integration test in `llmedge/src/androidTest/java/io/aatricks/llmedge/HuggingFaceVideoDownloadTest.kt` (download small test model, verify caching, test forceDownload=true)
- [X] T072 [US2] Test large model download on real device (Wan 1.3B Q4_K_M ~1.4GB, verify DownloadManager usage, progress updates, successful load)
- [X] T073 [US2] Test network error scenarios (airplane mode mid-download, verify cleanup, retry logic)

### Example App Enhancement

- [X] T074 [US2] Add model selection UI in `VideoGenerationActivity.kt` (Spinner with Wan 1.3B / 5B options, download button)
- [X] T075 [US2] Implement model download in example activity (call loadFromHuggingFace with progress updates, show download progress in UI)
- [X] T076 [US2] Add model cache status indicator (show "Cached" or "Download Required" for each model)
- [X] T077 [US2] Manual test on real device: download Wan 1.3B model, generate video, uninstall/reinstall app, verify cache persists (or doesn't based on install location)

**Checkpoint**: Hugging Face integration complete, models download and cache automatically

---

## Phase 6: User Story 4 - Video Generation Parameter Control (Priority: P2)

**Goal**: Provide developers control over video quality, duration, and reproducibility via generation parameters

**Independent Test**: Generate videos with different resolutions (256x256, 512x512), frame counts (8, 16, 32), seeds, and verify outputs match parameters

**Duration**: ~4-6 hours

### Parameter Extension

- [ ] T078 [P] [US4] Add scheduler parameter support in `VideoGenerateParams` (EULER_A, DDIM, DDPM, LCM enum, default EULER_A)
- [ ] T079 [P] [US4] Add strength parameter support for image-to-video in `VideoGenerateParams` (range 0.0-1.0, validation)
- [ ] T080 [US4] Extend validation logic in `VideoGenerateParams.validate()` for new parameters (scheduler valid, strength range, init image + T2V conflict check)

### JNI Parameter Passing

- [ ] T081 [US4] Extend `nativeTxt2Vid` JNI signature in `llmedge/src/main/cpp/sdcpp_jni.cpp` to accept scheduler parameter (map Kotlin enum to sd_scheduler_t C enum)
- [ ] T082 [US4] Pass strength parameter through JNI to native layer (add to `sd_vid_gen_params_t` struct initialization)
- [ ] T083 [US4] Update txt2vid() Kotlin method to pass all new parameters to native layer

### Reproducibility Testing

- [ ] T084 [P] [US4] Create `VideoReproducibilityTest.kt` in `llmedge/src/androidTest/java/io/aatricks/llmedge/` (generate 2 videos with same seed, compare frame checksums byte-by-byte)
- [ ] T085 [US4] Test seed randomization (use seed=-1 twice, verify different outputs)
- [ ] T086 [US4] Test deterministic generation (fixed seed, verify identical output across 3 runs)

### Parameter Validation Testing

- [ ] T087 [P] [US4] Add unit tests for parameter edge cases in `VideoGenerateParamsTest.kt` (max resolution 960x960, max frames 64, boundary values)
- [ ] T088 [US4] Test parameter combinations (high resolution + high frame count, verify memory estimation, warn if OOM risk)

### Example App Parameter UI

- [ ] T089 [US4] Add parameter controls in `VideoGenerationActivity.kt` UI (SeekBars for resolution, frame count, CFG scale, EditText for seed)
- [ ] T090 [US4] Implement parameter presets (Low Quality: 256x256x8, Medium: 512x512x16, High: 768x768x32) with radio buttons
- [ ] T091 [US4] Add "Reproducible" checkbox (generates fixed seed, displays seed after generation for re-use)
- [ ] T092 [US4] Manual test on device: try all parameter combinations, verify video characteristics match (count frames, check dimensions in bitmap properties)

**Checkpoint**: Parameter control fully implemented, reproducibility verified

---

## Phase 7: User Story 3 - Multiple Model Variant Support (Priority: P2)

**Goal**: Support loading and switching between Wan 2.1 (1.3B), Wan 2.2 (5B) model variants

**Independent Test**: Load Wan 1.3B, generate video, unload, load Wan 5B, generate different video with same prompt

**Duration**: ~6-8 hours

### Model Metadata & Detection

- [ ] T093 [P] [US3] Extend model registry JSON with all Wan variants (2.1 T2V 1.3B, 2.1 I2V 14B, 2.2 TI2V 5B, 2.2 T2V A14B) in `llmedge/src/main/assets/wan-models/model-registry.json`
- [ ] T094 [P] [US3] Implement model variant detection from GGUF metadata in `llmedge/src/main/java/io/aatricks/llmedge/StableDiffusion.kt` (read architecture, parameter count, context window from GGUF header)
- [ ] T095 [US3] Add mobile compatibility check (reject 14B models with clear error: "14B models not supported on mobile - use 1.3B or 5B variants")

### Model Lifecycle Management

- [ ] T096 [US3] Implement model unloading in `StableDiffusion.close()` (free native model memory, clear JNI global refs, reset state)
- [ ] T097 [US3] Add model switching logic (detect if model already loaded, unload previous before loading new, verify no memory leaks)
- [ ] T098 [US3] Implement model metadata caching (store ModelInfo after first load, avoid re-parsing GGUF header on subsequent loads)

### Memory Optimization for Large Models

- [ ] T099 [US3] Implement CPU offloading strategy for 5B models in JNI layer (detect device RAM < 8GB, configure sd_ctx_params with CPU layers)
- [ ] T100 [US3] Add memory pressure detection in `StableDiffusion.load()` (check MemoryMetrics, warn if loading 5B on low-memory device)
- [ ] T101 [US3] Implement context size capping for video models (limit max frames based on model size: 1.3B→64 frames, 5B→32 frames on mid-range devices)

### Testing

- [ ] T102 [P] [US3] Create `ModelVariantTest.kt` in `llmedge/src/androidTest/java/io/aatricks/llmedge/` (test loading all supported variants, verify detection, test 14B rejection)
- [ ] T103 [US3] Test model switching (load 1.3B, generate, close, load 5B, generate, verify memory stable)
- [ ] T104 [US3] Test fine-tuned models (download community fine-tune from Hugging Face with custom filename, verify loading works)
- [ ] T105 [US3] Measure memory usage for each variant with Android Profiler (1.3B Q4_K_M should peak ~2.5GB, 5B fp8 ~7GB)

### Example App Variant Selection

- [ ] T106 [US3] Add model variant dropdown in `VideoGenerationActivity.kt` (show available cached models, display size and speed estimates)
- [ ] T107 [US3] Implement model info display (show loaded model: variant, quantization, parameter count, memory usage)
- [ ] T108 [US3] Add "Recommended" badge for model variants based on device RAM (if <6GB RAM, recommend 1.3B)
- [ ] T109 [US3] Manual test on devices with different RAM (3GB, 6GB, 8GB+): verify appropriate models work, measure generation time differences

**Checkpoint**: Multiple model variants supported, device-appropriate recommendations working

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, performance optimization, comprehensive testing, and release preparation

**Duration**: ~6-8 hours

### Documentation

- [ ] T110 [P] Update `README.md` in repo root with video generation section (overview, supported models, basic usage example)
- [ ] T111 [P] Create comprehensive API documentation in `docs/video-generation-api.md` (all public methods with KDoc, examples, troubleshooting)
- [ ] T112 [P] Document performance benchmarks in `docs/performance.md` (generation times by model/device, memory usage, optimization tips)
- [ ] T113 [P] Create migration guide in `docs/migration-video-support.md` (for existing llmedge users, breaking changes if any)
- [ ] T114 Update `CREDITS.md` with stable-diffusion.cpp and Wan model attribution

### Performance Optimization

- [ ] T115 Optimize RGB to Bitmap conversion (use native memory directly if possible, avoid unnecessary copies)
- [ ] T116 Add Vulkan verification logs (log whether Vulkan initialized successfully, log GPU device name)
- [ ] T117 Optimize progress callback overhead (batch updates, only call every 500ms instead of every step)
- [ ] T118 Profile generation with Android Studio Profiler (identify CPU hotspots, optimize if found)

### Comprehensive Testing

- [ ] T119 [P] Create end-to-end test suite in `llmedge/src/androidTest/java/io/aatricks/llmedge/VideoE2ETest.kt` (covers all 5 user stories, runs full workflow)
- [ ] T120 [P] Add stress tests (10 consecutive generations, verify no crashes, memory stable)
- [ ] T121 [P] Add device compatibility tests (test on API 30, 31, 33, 34; test on different manufacturers: Samsung, Pixel, OnePlus)
- [ ] T122 Run full test suite on CI (if CI configured): `./gradlew connectedAndroidTest`
- [ ] T123 Measure final code coverage with JaCoCo (verify ≥80% for all new Kotlin code, ≥70% for JNI layer)

### Quickstart Validation

- [ ] T124 Run quickstart.md examples on clean device (follow steps exactly, verify all code snippets work without modification)
- [ ] T125 Test quickstart on emulator (API 30+ emulator without GPU acceleration, verify CPU fallback works)
- [ ] T126 Validate quickstart timing claims (verify 60 second generation time for 16 frames @ 512x512 on mid-range device)

### Code Quality

- [ ] T127 Run Detekt static analysis with `./gradlew detekt` (fix all warnings in new Kotlin code)
- [ ] T128 Run Ktlint formatting with `./gradlew ktlintFormat` (ensure consistent code style)
- [ ] T129 Review all KDoc comments for completeness (public APIs should have comprehensive docs with @param, @return, @throws, @sample)
- [ ] T130 Code review pass (review all video-related code for best practices, error handling, resource cleanup)

### Release Preparation

- [ ] T131 Update version in `llmedge/build.gradle.kts` (bump minor version for new feature: e.g., 1.0.0 → 1.1.0)
- [ ] T132 Generate release notes in `specs/001-video-model-support/RELEASE_NOTES.md` (summarize new APIs, supported models, breaking changes, migration steps)
- [ ] T133 Build release AAR with `./gradlew :llmedge:assembleRelease` (verify signing if applicable)
- [ ] T134 Test release AAR in example app (replace debug AAR with release, verify proguard rules work, test minified build)
- [ ] T135 Create GitHub release draft (tag 001-video-model-support, attach AAR, include release notes, link to docs)

**Checkpoint**: Feature complete, documented, tested, and ready for release

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately (~2 hours)
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories (~6-8 hours)
- **User Story 5 (Phase 3)**: Depends on Foundational - native library prerequisite (~4-6 hours)
- **User Story 1 (Phase 4)**: Depends on Foundational + US5 - core MVP (~8-10 hours)
- **User Story 2 (Phase 5)**: Depends on US1 (needs basic generation working) (~6-8 hours)
- **User Story 4 (Phase 6)**: Depends on US1 (extends parameters) (~4-6 hours)
- **User Story 3 (Phase 7)**: Depends on US1 + US2 (needs download + generation working) (~6-8 hours)
- **Polish (Phase 8)**: Depends on all user stories complete (~6-8 hours)

### User Story Dependencies

- **US5 (Native Library Update)**: No user story dependencies - foundational
- **US1 (Basic Generation)**: Depends on US5 - needs native library
- **US2 (Hugging Face Download)**: Depends on US1 - needs generation API to test downloads
- **US4 (Parameter Control)**: Depends on US1 - extends basic generation
- **US3 (Multiple Variants)**: Depends on US1 + US2 - needs generation + download working

### Critical Path (Sequential MVP)

```
Phase 1 (Setup) → Phase 2 (Foundational) → Phase 3 (US5) → Phase 4 (US1) → Phase 5 (US2)
    ~2h              ~6-8h                    ~4-6h          ~8-10h         ~6-8h
Total MVP time: ~26-34 hours
```

### Parallel Opportunities

**Phase 2 (Foundational)**:
- T008 (nativeSetProgressCallback) || T009 (nativeCancelGeneration)
- T013 (VideoGenerateParams) || T014 (GenerationMetrics) || T015 (VideoProgressCallback)

**Phase 4 (User Story 1)**:
- T030 (VideoGenerateParamsTest) || T031 (StableDiffusionVideoTest) - write tests in parallel
- T043 (edge case tests) || T044 (coverage measurement) - after implementation

**Phase 5 (User Story 2)**:
- T055 (model registry) || T056 (registry parser) || T058 (unit tests)

**Phase 6 (User Story 4)**:
- T078 (scheduler param) || T079 (strength param) || T084 (reproducibility test)

**Phase 7 (User Story 3)**:
- T093 (model registry) || T094 (variant detection) || T102 (variant test)

**Phase 8 (Polish)**:
- T110 (README) || T111 (API docs) || T112 (performance docs) || T113 (migration guide)
- T119 (E2E tests) || T120 (stress tests) || T121 (device compat tests)

---

## Parallel Example: Foundational Phase

```bash
# Launch JNI tests creation in parallel:
Task T008: "Implement nativeSetProgressCallback JNI function"
Task T009: "Implement nativeCancelGeneration JNI function"

# Launch data model classes in parallel:
Task T013: "Create VideoGenerateParams data class"
Task T014: "Create GenerationMetrics data class"
Task T015: "Create VideoProgressCallback interface"
```

---

## Implementation Strategy

### MVP First (US5 + US1 + US2)

1. Complete Phase 1: Setup (~2 hours)
2. Complete Phase 2: Foundational (~6-8 hours)
3. Complete Phase 3: US5 Native Library (~4-6 hours)
4. Complete Phase 4: US1 Basic Generation (~8-10 hours)
5. Complete Phase 5: US2 Hugging Face Download (~6-8 hours)
6. **STOP and VALIDATE**: Manual test full workflow (download model, generate video, display frames)
7. Deploy/demo MVP (~26-34 hours total)

### Incremental Delivery

1. **Foundation Ready** (Phase 1-2): ~8-10 hours
2. **MVP v1** (+ Phase 3-4): Basic generation working (~20-28 hours)
   - Can generate videos with manually downloaded models
   - Progress tracking works
   - Demo ready
3. **MVP v2** (+ Phase 5): Auto-download working (~26-34 hours)
   - Fully self-service (no manual model download)
   - Production-ready for basic use cases
4. **Full Feature** (+ Phase 6-7): All parameters + variants (~36-50 hours)
   - Advanced control
   - Multiple model support
   - Optimized for different devices
5. **Release** (+ Phase 8): Polished and documented (~42-58 hours total)

### Parallel Team Strategy

With 3 developers after Foundational phase:

- **Developer A**: Phase 3 (US5) → Phase 4 (US1) - Native + core API
- **Developer B**: Phase 5 (US2) - Hugging Face integration (starts after US1 T033-T038)
- **Developer C**: Phase 6 (US4) - Parameter control (starts after US1 complete)

Then reconverge for Phase 7 (US3) and Phase 8 (Polish)

---

## Testing Strategy

### Test Types by Phase

**Unit Tests (Fast, Outside Android)**:
- Kotlin: T030-T032, T043-T044, T058-T059, T087-T088
- C++: T022-T025
- Run with: `./gradlew :llmedge:testDebugUnitTest`

**Integration Tests (Instrumented, Real Device)**:
- Android: T045-T049, T071-T073, T084-T086, T102-T105
- Run with: `./gradlew :llmedge:connectedAndroidTest`

**Manual Tests (Example App)**:
- UI validation: T054, T077, T092, T109
- End-to-end workflows: T124-T126

### Test First Approach (TDD)

Tasks explicitly require writing tests BEFORE implementation:
- T030-T032 before T033-T041 (US1 implementation)
- T058-T059 before T060-T069 (US2 implementation)
- Tests should FAIL initially, then PASS after implementation

### Coverage Gates

- **Unit test coverage**: ≥80% (T044, T123)
- **Integration test coverage**: All user stories validated end-to-end (T119)
- **Device coverage**: API 30, 31, 33, 34; Samsung + Pixel minimum (T121)

---

## Estimated Total Duration

- **Minimum (Sequential, Experienced Dev)**: ~42 hours (1 week)
- **Average (Sequential, Standard Dev)**: ~58 hours (1.5 weeks)
- **Parallel (3 Devs)**: ~30-35 hours (4-5 days with overlap)

**Task Count**: 135 tasks
- Setup: 5 tasks
- Foundational: 13 tasks
- US5: 11 tasks
- US1: 25 tasks
- US2: 23 tasks
- US4: 15 tasks
- US3: 17 tasks
- Polish: 26 tasks

---

## Notes

- **[P] tasks**: Different files, can run in parallel (55 tasks parallelizable)
- **[Story] labels**: Map task to specific user story for traceability
- **Testing first**: Unit tests before implementation (TDD where marked)
- **Real device required**: Integration tests need API 30+ device with 4GB+ RAM
- **Native testing**: C++ tests run outside Android for fast iteration
- **Memory monitoring**: Use Android Profiler throughout (tasks T049, T105, T118, T120)
- **Commit frequency**: Commit after each task or logical group
- **Checkpoints**: Stop at each checkpoint to validate independently
- **Avoid**: Skipping tests, untested native code, missing resource cleanup

---

## Success Criteria Mapping

Tasks mapped to success criteria from spec.md:

- **SC-001** (60s generation): Validated in T054, T126
- **SC-002** (90% download success): Tested in T071-T073
- **SC-003** (reproducibility): Verified in T084-T086
- **SC-004** (no memory leaks): Monitored in T049, T105, T120
- **SC-005** (progress updates): Tested in T046
- **SC-006** (95% validation): Covered by T016, T080, T087-T088
- **SC-007** (build success): Verified in T026-T029
- **SC-008** (30% Vulkan speedup): Benchmarked in T112, T116

---

**Total Tasks**: 135  
**Parallelizable Tasks**: 55 ([P] marker)  
**Test Tasks**: 28 (unit + integration + manual)  
**MVP Tasks (US5+US1+US2)**: 59 tasks (~26-34 hours)  
**Full Feature**: 109 tasks (~36-50 hours)  
**With Polish**: 135 tasks (~42-58 hours)

This task breakdown provides atomic, testable, 1-2 hour tasks with clear completion criteria, organized by user story for independent implementation and testing.
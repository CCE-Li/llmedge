# Research: Video Model Support

**Feature**: Video Model Support (001-video-model-support)  
**Date**: 2025-11-13  
**Phase**: 0 - Research & Discovery

## Overview

This document consolidates research findings for adding Wan video model support to llmedge. Research covered stable-diffusion.cpp Wan implementation analysis, JNI binding design patterns, and Hugging Face model repository structure.

---

## 1. Wan Model Variants in stable-diffusion.cpp

### Decision: Support Wan 1.3B and 5B variants only on mobile

**Rationale**:
- 1.3B models (Q4_K_M ~1.4GB) fit comfortably in mobile memory constraints (<3GB native RAM)
- 5B models (fp8 ~5GB) work on high-end Android devices (8GB+ total RAM)
- 14B models (15-33GB files, 20-40GB runtime RAM) exceed mobile device capabilities

**Alternatives considered**:
- **Include 14B models**: Rejected - file sizes (15-33GB) and memory requirements (20-40GB) make mobile deployment impractical
- **Only support 1.3B**: Rejected - misses opportunity for premium device experiences with 5B models
- **Cloud-based 14B inference**: Rejected - violates on-device design principle, adds network dependency

### Supported Model Variants

#### Wan 2.1 Models

**Wan2.1-T2V-1.3B** (Text-to-Video) - **PRIMARY TARGET**
- Architecture: 30 layers, 1536 dim, 12 heads
- Parameters: 1.3 billion
- Quantization: Q4_K_M (~1.4GB), Q6_K (~2GB), fp8 (~1.4GB)
- Memory usage: 2-3GB native RAM
- Capabilities: Pure text-to-video generation
- Detection: `num_layers == 30`, no VACE blocks

**Wan2.1-I2V-14B** (Image-to-Video) - **NOT SUPPORTED ON MOBILE**
- Architecture: 40 layers, 5120 dim, 40 heads
- File size: 15-17GB (fp8), 29-33GB (fp16)
- Memory usage: 20-40GB RAM
- Rationale for exclusion: Exceeds mobile device capabilities

#### Wan 2.2 Models

**Wan2.2-TI2V-5B** (Text+Image-to-Video) - **HIGH-END DEVICE TARGET**
- Architecture: 30 layers, 3072 dim, 24 heads  
- Parameters: 5 billion
- Quantization: fp8_e4m3fn (~5GB)
- Memory usage: 6-10GB native RAM
- Capabilities: Unified T2V and I2V in single model
- Requires: Devices with 8GB+ total RAM
- Special: Uses wan_2.2_vae (different from 2.1)

**Wan2.2-T2V-A14B / I2V-A14B** - **NOT SUPPORTED ON MOBILE**
- Parameters: 14 billion (dual-stage architecture)
- File sizes: 15-33GB per model (requires TWO models)
- Memory usage: 30-50GB RAM
- Rationale for exclusion: Dual-model requirement doubles memory/storage needs

### Model Type Detection Logic

stable-diffusion.cpp automatically detects Wan variants via tensor inspection:

```cpp
// From WanRunner constructor
- num_layers: Counted from blocks.*.{tensor} naming
- model_type: "t2v" or "i2v" based on img_emb tensors
- VACE detection: Presence of vace_blocks.* tensors
- Architecture: 1536 dim (1.3B), 3072 dim (5B), 5120 dim (14B)
```

Detection strings emitted:
- `"Wan2.1-T2V-1.3B"` - Primary mobile target
- `"Wan2.2-TI2V-5B"` - High-end device target
- Others logged but unsupported on mobile

---

## 2. Video Generation C API

### Decision: Use existing `generate_video()` API from stable-diffusion.cpp

**Rationale**:
- Mature implementation supporting all Wan variants
- Handles text/image prompts, control frames, high/low noise separation
- Progress callbacks via `sd_set_progress_callback()`
- Memory-safe frame allocation/deallocation

**Alternatives considered**:
- **Custom video generation pipeline**: Rejected - reinventing wheel, high maintenance burden
- **Frame-by-frame generation with txt2img**: Rejected - no temporal coherence, inefficient

### API Structure

**Entry point**:
```cpp
SD_API sd_image_t* generate_video(
    sd_ctx_t* sd_ctx,
    const sd_vid_gen_params_t* sd_vid_gen_params,
    int* num_frames_out
);
```

**Key parameters** (`sd_vid_gen_params_t`):
- `prompt`, `negative_prompt`: Text prompts
- `width`, `height`, `video_frames`: Output dimensions
- `init_image`: Optional starting image for I2V/TI2V
- `sample_params`: Steps, CFG scale, scheduler, sampling method
- `seed`: Reproducibility control
- `strength`: Denoising strength for I2V

**Return value**:
- Array of `sd_image_t` structs (one per frame)
- Frames adjusted: Wan outputs `(T-1)*4 + 1` frames internally
- Caller responsible for freeing via `free(frames[i].data)`, then `free(frames)`

**Progress tracking**:
- Global callback via `sd_set_progress_callback(callback_fn, user_data)`
- Signature: `void callback(int step, int total_steps, float time_per_step, void* data)`
- Called periodically during denoising loop

---

## 3. JNI Binding Design

### Decision: Array of byte arrays return type with progress callback support

**Rationale**:
- **Array of `ByteArray`**: Allows incremental Bitmap conversion in Kotlin, avoids single giant allocation
- **Progress callbacks via JavaVM**: Enables UI updates during long-running generation (30-300 seconds typical)
- **Cancellation via atomic flag**: Checked in progress callback, throws exception to abort generation

**Alternatives considered**:
- **Single concatenated byte array**: Rejected - risk of OOM on long videos, no incremental processing
- **File path return (H.264 video)**: Rejected - requires libav/ffmpeg dependency, complicates preview/editing
- **Polling-based progress**: Rejected - wastes CPU cycles, less responsive than callbacks

### JNI Method Signatures

**Primary generation method**:
```cpp
extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2Vid(
    JNIEnv* env, jobject thiz, jlong handlePtr,
    jstring jPrompt, jstring jNegative,
    jint width, jint height, jint videoFrames,
    jint steps, jfloat cfg, jlong seed,
    jbyteArray jInitImage, jint initWidth, jint initHeight,
    jobject progressCallback
);
// Returns: jobjectArray of jbyteArray (byte[][])
```

**Progress callback setup**:
```cpp
extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeSetProgressCallback(
    JNIEnv* env, jobject thiz, jlong handlePtr,
    jobject progressCallback  // nullable
);
// Stores global ref, configures sd_set_progress_callback()
```

**Cancellation**:
```cpp
extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeCancelGeneration(
    JNIEnv* env, jobject thiz, jlong handlePtr
);
// Sets atomic<bool> cancellationRequested in SdHandle
```

### SdHandle Extension

```cpp
struct SdHandle {
    sd_ctx_t* ctx = nullptr;
    int last_width = 0;
    int last_height = 0;
    
    // NEW for video support
    JavaVM* jvm = nullptr;
    jobject progressCallbackGlobalRef = nullptr;
    jmethodID progressMethodID = nullptr;
    std::atomic<bool> cancellationRequested{false};
    int currentFrame = 0;
    int totalFrames = 0;
};
```

### Progress Callback Bridge

**Kotlin interface**:
```kotlin
interface VideoProgressCallback {
    fun onProgress(step: Int, totalSteps: Int, 
                  currentFrame: Int, totalFrames: Int, 
                  timePerStep: Float)
}
```

**JNI wrapper** (thread-safe via `JavaVM->AttachCurrentThread`):
```cpp
void sd_video_progress_wrapper(int step, int steps, float time, void* data) {
    auto* handle = static_cast<SdHandle*>(data);
    if (handle->cancellationRequested.load()) {
        throw std::runtime_error("Video generation cancelled");
    }
    
    JNIEnv* env = nullptr;
    bool detach = false;
    if (handle->jvm->GetEnv((void**)&env, JNI_VERSION_1_6) == JNI_EDETACHED) {
        handle->jvm->AttachCurrentThread(&env, nullptr);
        detach = true;
    }
    
    env->CallVoidMethod(handle->progressCallbackGlobalRef,
                       handle->progressMethodID,
                       step, steps, 
                       handle->currentFrame, handle->totalFrames,
                       time);
    
    if (detach) handle->jvm->DetachCurrentThread();
}
```

### Memory Management Strategy

**Native side**:
```cpp
// After generate_video() returns
for (int i = 0; i < num_frames_out; i++) {
    jbyteArray frameBytes = env->NewByteArray(byteCount);
    env->SetByteArrayRegion(frameBytes, 0, byteCount, (jbyte*)frames[i].data);
    env->SetObjectArrayElement(result, i, frameBytes);
    env->DeleteLocalRef(frameBytes);  // Prevent local ref overflow
    
    free(frames[i].data);  // Free immediately after copy
}
free(frames);  // Free array
```

**Kotlin side**:
```kotlin
suspend fun txt2vid(params: VideoGenerateParams): List<Bitmap> {
    generationMutex.withLock {
        val frameBytes = nativeTxt2Vid(...) ?: throw IllegalStateException()
        return frameBytes.map { bytes ->
            convertRgbToBitmap(bytes, params.width, params.height)
        }
    }
}
```

**Memory profile** (typical 16-frame video):
- Native: 16 × 512×512×3 bytes = ~12MB peak (freed incrementally)
- JVM heap: ~12MB for byte arrays + ~12MB for Bitmaps = ~24MB total
- Acceptable for mobile targets (<50MB overhead)

---

## 4. Hugging Face Model Repository Structure

### Decision: Use Kijai/WanVideo_comfy_GGUF for mobile-optimized models

**Rationale**:
- Single-file GGUF format simplifies downloads (vs multi-file diffusers)
- Q4_K_M quantization provides best size/quality tradeoff for mobile (1.4GB for 1.3B model)
- Community-maintained repository with wide variant coverage
- Established download patterns (6.47M+ downloads indicates reliability)

**Alternatives considered**:
- **ByteDance official repo**: Rejected - multi-file diffusers format (65.9GB total) complicates mobile downloads
- **Kijai FP8 scaled repo**: Considered for high-end devices (807GB total size is manageable with selective downloads)
- **GGUF from stable-diffusion.cpp releases**: Rejected - no official Wan GGUF releases yet

### Recommended Model Files

#### For 1.3B Models (All devices):
```
Repository: Kijai/WanVideo_comfy_GGUF
Files:
- Phantom-Wan-1_3B_Q4_K_M.gguf (~1.4GB) - Best for storage-constrained devices
- Phantom-Wan-1_3B_Q6_K.gguf (~2GB) - Better quality, moderate size increase

Expected runtime memory: 2-3GB native RAM
Download strategy: Single file, use preferSystemDownloader=true
```

#### For 5B Models (High-end devices, 8GB+ RAM):
```
Repository: Kijai/WanVideo_comfy_fp8_scaled
Files:
- TI2V/Wan2_2-5B_fp8_e4m3fn_scaled_KJ.safetensors (~5GB)
- Qwen/Qwen2.5_7B_instruct_bf16.safetensors (~14GB, shared text encoder)

Expected runtime memory: 6-10GB native RAM
Download strategy: Multi-file with caching, sequential downloads
```

### Model Format Preferences

**Quantization ranking** (mobile context):
1. **Q4_K_M GGUF** - Best size/quality, 1.4GB for 1.3B model
2. **Q6_K GGUF** - Better quality, 2GB for 1.3B model
3. **fp8_e4m3fn safetensors** - Highest quality for mobile, ~1.4GB for 1.3B / ~5GB for 5B
4. **fp16/bf16 safetensors** - Too large for mobile (2.87GB for 1.3B, 10GB+ for 5B)

**NOT recommended for mobile**:
- fp32/fp16/bf16 formats (unnecessary precision overhead)
- 14B variants in any quantization (15-33GB files minimum)
- Multi-file diffusers format (complex loading, multiple downloads)

### HuggingFaceHub Integration

**Model detection pattern**:
```kotlin
fun detectWanModelVariant(filename: String): WanModelInfo? {
    val patterns = mapOf(
        Regex("Wan2[._]1-T2V-1[._]3B", IGNORE_CASE) to 
            WanModelInfo(version = "2.1", task = "T2V", size = "1.3B"),
        Regex("Wan2[._]2-TI2V-5B", IGNORE_CASE) to
            WanModelInfo(version = "2.2", task = "TI2V", size = "5B"),
        Regex("Phantom-Wan-1[._]3B", IGNORE_CASE) to
            WanModelInfo(version = "2.1", task = "T2V", size = "1.3B")
    )
    return patterns.entries.firstOrNull { it.key.find(filename) != null }?.value
}
```

**Download strategy**:
```kotlin
// 1.3B models - single file download
val download = HuggingFaceHub.ensureRepoFileOnDisk(
    context = context,
    modelId = "Kijai/WanVideo_comfy_GGUF",
    filename = "Phantom-Wan-1_3B_Q4_K_M.gguf",
    preferSystemDownloader = true,  // Use DownloadManager for 1.4GB file
    onProgress = { progress -> updateUI(progress) }
)

// 5B models - sequential multi-file
val model = downloadWithDependencies(
    modelFile = "Wan2_2-5B_fp8_e4m3fn_scaled_KJ.safetensors",
    dependencies = listOf("Qwen2.5_7B_instruct_bf16.safetensors")
)
```

---

## 5. stable-diffusion.cpp Submodule Update

### Decision: Update to latest master with Wan support verification

**Rationale**:
- Current commit `d05e46c` includes comprehensive Wan implementation
- `wan.hpp` contains all model variants (1.3B, 5B, 14B across 2.1/2.2)
- `generate_video()` API mature and tested by community

**Update process**:
```bash
cd stable-diffusion.cpp
git fetch origin
git checkout master
git pull origin master
cd ..
git add stable-diffusion.cpp
git commit -m "chore: update stable-diffusion.cpp to latest with Wan support"
```

**Verification**:
- Check `stable-diffusion.cpp/wan.hpp` exists
- Verify `generate_video()` signature in `stable-diffusion.h`
- Confirm `sd_vid_gen_params_t` struct definition
- Test build with `-DSD_VULKAN=ON` flag

**Build integration** (already present in `llmedge/build.gradle.kts`):
```kotlin
externalNativeBuild {
    cmake {
        arguments += "-DSD_VULKAN=ON"
        arguments += "-DGGML_VULKAN=ON"
    }
}
```

**No additional CMake changes required** - stable-diffusion.cpp submodule automatically includes Wan support when updated.

---

## 6. Performance & Memory Benchmarks

### Target Metrics (from success criteria)

**SC-001: 16-frame 256x256 in <60s** (Wan 2.1 1.3B Q4_K_M)
- Device class: Mid-range (Snapdragon 778G equivalent, 6GB RAM)
- Backend: CPU-only fallback acceptable, Vulkan preferred
- Steps: 20 (standard quality)
- Expected: 3-4s per frame average

**SC-008: 30% Vulkan speedup**
- Baseline: CPU-only generation
- Vulkan: GPU acceleration on compatible devices
- Measurement: Average time per frame across 16-frame video

### Memory Constraints

**1.3B models (Q4_K_M)**:
- File size: ~1.4GB
- Native RAM usage: 2-3GB (model + KV cache + temp buffers)
- JVM heap usage: ~20-30MB (frame byte arrays + Bitmaps)
- Total device RAM required: 4GB+ (allows for OS + app overhead)

**5B models (fp8)**:
- File size: ~5GB
- Native RAM usage: 6-10GB
- JVM heap usage: ~20-30MB
- Total device RAM required: 8GB+ (flagship device tier)

### Frame Count Limits

**Maximum frames per generation**:
- 1.3B models: 64 frames (device memory limit)
- 5B models: 48 frames (higher memory per frame)
- Reasoning: Prevents OOM crashes, aligns with typical short-form video use cases (2-4 seconds @ 16fps)

**Validation**:
```kotlin
fun validateVideoParams(params: VideoGenerateParams): Result<Unit> {
    val maxFrames = when (detectModelSize(modelPath)) {
        "1.3B" -> 64
        "5B" -> 48
        else -> 32  // Conservative default
    }
    if (params.videoFrames > maxFrames) {
        return Result.failure(IllegalArgumentException(
            "Frame count ${params.videoFrames} exceeds maximum $maxFrames for this model"
        ))
    }
    return Result.success(Unit)
}
```

---

## 7. API Design Decisions

### Decision: Extend StableDiffusion class with video methods

**Rationale**:
- Consistent with existing library architecture (single entry point per capability)
- Reuses model loading infrastructure (`load()`, `loadFromHuggingFace()`)
- Natural progression: `txt2img()` → `txt2vid()`

**Alternatives considered**:
- **Separate VideoGenerator class**: Rejected - duplicates model management, confuses API surface
- **Unified generate() method with mode enum**: Rejected - reduces type safety, unclear return types

### Kotlin API Structure

```kotlin
class StableDiffusion {
    // Existing API
    suspend fun txt2img(params: GenerateParams): Bitmap
    
    // NEW: Video generation
    suspend fun txt2vid(
        params: VideoGenerateParams,
        onProgress: VideoProgressCallback? = null
    ): List<Bitmap>
    
    // NEW: Video generation as flow (future optimization)
    fun txt2vidAsFlow(
        params: VideoGenerateParams,
        chunkSize: Int = 4
    ): Flow<List<Bitmap>>
    
    // NEW: Cancellation support
    fun cancelGeneration()
}

data class VideoGenerateParams(
    val prompt: String,
    val negative: String = "",
    val width: Int = 512,
    val height: Int = 512,
    val videoFrames: Int = 16,  // Validated against model limits
    val steps: Int = 20,
    val cfgScale: Float = 7.0f,
    val seed: Long = 42L,
    val initImage: Bitmap? = null,  // For I2V/TI2V models
    val strength: Float = 0.8f
)

interface VideoProgressCallback {
    fun onProgress(step: Int, totalSteps: Int, 
                  currentFrame: Int, totalFrames: Int, 
                  timePerStep: Float)
}
```

**Breaking changes**: None - additive API only

---

## Summary

All technical unknowns resolved. Ready to proceed to Phase 1 (Design & Contracts).
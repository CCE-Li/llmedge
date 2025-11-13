# Data Model: Video Model Support

**Feature**: Video Model Support (001-video-model-support)  
**Date**: 2025-11-13  
**Phase**: 1 - Design & Contracts

## Overview

This document defines the data entities and their relationships for video model support in llmedge. Entities model video generation parameters, model metadata, generation results, and progress tracking.

---

## Core Entities

### 1. VideoModel

Represents a loaded Wan video model variant with metadata.

**Attributes**:
- `modelType: ModelType` - Variant identifier (T2V, I2V, TI2V)
- `version: String` - Wan version ("2.1", "2.2")
- `parameterCount: ParameterSize` - Model size (1.3B, 5B, 14B)
- `filePath: String` - Absolute path to model file on device
- `quantization: Quantization` - Quantization level (Q4_K_M, Q6_K, fp8, fp16)
- `fileSize: Long` - Model file size in bytes
- `contextWindow: Int` - Maximum video frames supported (16, 32, 64)
- `requiresVulkan: Boolean` - Whether model benefits from GPU acceleration
- `memoryRequirement: Long` - Estimated native RAM usage in bytes

**Enums**:
```kotlin
enum class ModelType { T2V, I2V, TI2V }
enum class ParameterSize { 
    SIZE_1_3B,   // 1.3 billion parameters
    SIZE_5B,     // 5 billion parameters
    SIZE_14B     // 14 billion parameters (not supported on mobile)
}
enum class Quantization { Q4_K_M, Q6_K, Q8_0, FP8_E4M3FN, FP16, BF16, FP32 }
```

**Relationships**:
- One `VideoModel` has many `VideoGenerationResult` (history)

**Validation Rules**:
- `parameterCount == SIZE_14B` → ERROR on mobile platforms
- `quantization == FP16/BF16/FP32` + `parameterCount > SIZE_1_3B` → WARNING (large file size)
- `contextWindow` must be power of 2 in range [16, 64]
- `memoryRequirement < DeviceInfo.availableMemory * 0.7` → WARNING (OOM risk)

**State Transitions**:
```
[UNINITIALIZED] → [LOADING] → [READY] → [GENERATING] → [READY]
                      ↓            ↓
                  [ERROR]      [ERROR]
```

---

### 2. VideoGenerationParams

Configuration for a single video generation request.

**Attributes**:
- `prompt: String` - Text description of desired video content (required)
- `negativePrompt: String` - Undesired content description (optional, default: "")
- `width: Int` - Output frame width in pixels (multiple of 64, range: 256-960)
- `height: Int` - Output frame height in pixels (multiple of 64, range: 256-960)
- `videoFrames: Int` - Number of frames to generate (range: 4-64, validated per model)
- `steps: Int` - Denoising steps (range: 10-50, typical: 20)
- `cfgScale: Float` - Classifier-free guidance scale (range: 1.0-15.0, typical: 7.0)
- `seed: Long` - Random seed for reproducibility (-1 for random)
- `initImage: Bitmap?` - Starting image for I2V/TI2V (optional, null for T2V)
- `strength: Float` - Denoising strength for I2V (range: 0.0-1.0, typical: 0.8)
- `scheduler: Scheduler` - Noise scheduler algorithm (default: EULER_A)

**Enums**:
```kotlin
enum class Scheduler { EULER_A, DDIM, DDPM, LCM }
```

**Relationships**:
- One `VideoGenerationParams` produces one `VideoGenerationResult`
- Immutable value object (Kotlin data class)

**Validation Rules**:
```kotlin
fun validate(): Result<Unit> {
    require(prompt.isNotBlank()) { "Prompt cannot be empty" }
    require(width % 64 == 0 && width in 256..960) { 
        "Width must be multiple of 64 in range 256-960, got $width" 
    }
    require(height % 64 == 0 && height in 256..960) { 
        "Height must be multiple of 64 in range 256-960, got $height" 
    }
    require(videoFrames in 4..64) { 
        "Frame count must be 4-64, got $videoFrames" 
    }
    require(steps in 10..50) { 
        "Steps must be 10-50, got $steps" 
    }
    require(cfgScale in 1.0f..15.0f) { 
        "CFG scale must be 1.0-15.0, got $cfgScale" 
    }
    require(strength in 0.0f..1.0f) { 
        "Strength must be 0.0-1.0, got $strength" 
    }
    
    // Model-specific validation
    if (initImage != null && modelType == ModelType.T2V) {
        return Result.failure(IllegalArgumentException(
            "T2V models do not accept init images"
        ))
    }
    if (initImage == null && modelType == ModelType.I2V) {
        return Result.failure(IllegalArgumentException(
            "I2V models require init image"
        ))
    }
    
    return Result.success(Unit)
}
```

**Defaults**:
```kotlin
companion object {
    fun default() = VideoGenerationParams(
        prompt = "",
        negativePrompt = "",
        width = 512,
        height = 512,
        videoFrames = 16,
        steps = 20,
        cfgScale = 7.0f,
        seed = -1L,  // Random
        initImage = null,
        strength = 0.8f,
        scheduler = Scheduler.EULER_A
    )
}
```

---

### 3. VideoOutput

Generated video data as a sequence of frames with metadata.

**Attributes**:
- `frames: List<Bitmap>` - Output frames in generation order
- `width: Int` - Frame width in pixels
- `height: Int` - Frame height in pixels
- `frameCount: Int` - Number of frames (derived: `frames.size`)
- `duration: Float` - Video duration in seconds (calculated from frame count and FPS)
- `fps: Int` - Frames per second for playback (default: 16)
- `seed: Long` - Seed used for generation (for reproducibility)
- `createdAt: Instant` - Generation timestamp

**Relationships**:
- Part of `VideoGenerationResult` (composition)

**Derived Properties**:
```kotlin
val duration: Float
    get() = frameCount.toFloat() / fps

val sizeInBytes: Long
    get() = frames.sumOf { it.byteCount.toLong() }

val averageFrameSizeKB: Float
    get() = (sizeInBytes / 1024f) / frameCount
```

**Methods**:
```kotlin
fun saveToFile(directory: File, format: VideoFormat): File
fun getFrame(index: Int): Bitmap
fun toByteArraySequence(): Sequence<ByteArray>
```

---

### 4. VideoGenerationResult

Complete result of a video generation operation including metrics.

**Attributes**:
- `output: VideoOutput` - Generated video frames and metadata
- `params: VideoGenerationParams` - Parameters used for generation
- `metrics: GenerationMetrics` - Performance and resource usage metrics
- `status: GenerationStatus` - Final status of operation
- `errorMessage: String?` - Error description if status == FAILED

**Enums**:
```kotlin
enum class GenerationStatus { SUCCESS, FAILED, CANCELLED }
```

**Relationships**:
- Contains one `VideoOutput` (composition)
- Contains one `VideoGenerationParams` (composition)
- Contains one `GenerationMetrics` (composition)

**Validation Rules**:
- `status == SUCCESS` → `output != null && errorMessage == null`
- `status == FAILED` → `output == null && errorMessage != null`
- `status == CANCELLED` → `output == null`

---

### 5. GenerationMetrics

Performance and resource usage metrics for video generation.

**Attributes**:
- `totalTimeSeconds: Float` - Total generation time from start to finish
- `framesPerSecond: Float` - Average generation throughput
- `timePerStep: Float` - Average time per denoising step in seconds
- `peakMemoryUsageMB: Long` - Peak native memory usage during generation
- `vulkanEnabled: Boolean` - Whether GPU acceleration was used
- `modelLoadTimeSeconds: Float` - Time to load model (if applicable)
- `frameConversionTimeSeconds: Float` - Time to convert RGB to Bitmap

**Derived Properties**:
```kotlin
val averageFrameTime: Float
    get() = totalTimeSeconds / frameCount

val stepsPerSecond: Float
    get() = 1.0f / timePerStep

val throughput: String
    get() = "${framesPerSecond.format(2)} fps"
```

**Methods**:
```kotlin
fun toPrettyString(): String {
    return """
        Total time: ${totalTimeSeconds.format(2)}s
        Throughput: ${framesPerSecond.format(2)} fps
        Average time/step: ${timePerStep.format(3)}s
        Peak memory: ${peakMemoryUsageMB}MB
        Vulkan: ${if (vulkanEnabled) "enabled" else "disabled"}
    """.trimIndent()
}
```

---

### 6. ModelDownload

Represents a Hugging Face model download operation with progress tracking.

**Attributes**:
- `modelId: String` - Hugging Face repository ID (e.g., "Kijai/WanVideo_comfy_GGUF")
- `filename: String` - Target file in repository
- `file: File` - Local file path after download
- `fileSize: Long` - Total file size in bytes
- `downloadedBytes: Long` - Bytes downloaded so far
- `status: DownloadStatus` - Current download state
- `startTime: Instant` - Download start timestamp
- `endTime: Instant?` - Download completion timestamp (null if in progress)
- `errorMessage: String?` - Error description if status == FAILED

**Enums**:
```kotlin
enum class DownloadStatus { PENDING, IN_PROGRESS, COMPLETED, FAILED, CANCELLED }
```

**Derived Properties**:
```kotlin
val progress: Float
    get() = if (fileSize > 0) downloadedBytes.toFloat() / fileSize else 0f

val downloadSpeedMBps: Float
    get() {
        val elapsed = Duration.between(startTime, Instant.now()).seconds
        return if (elapsed > 0) (downloadedBytes / 1024f / 1024f) / elapsed else 0f
    }

val estimatedTimeRemainingSeconds: Long
    get() {
        val speed = downloadSpeedMBps
        return if (speed > 0) {
            ((fileSize - downloadedBytes) / 1024f / 1024f / speed).toLong()
        } else 0L
    }
```

**State Transitions**:
```
[PENDING] → [IN_PROGRESS] → [COMPLETED]
                ↓
            [FAILED]
            [CANCELLED]
```

---

### 7. VideoProgressState

Real-time progress tracking during video generation.

**Attributes**:
- `currentStep: Int` - Current denoising step (0-based)
- `totalSteps: Int` - Total denoising steps for all frames
- `currentFrame: Int` - Frame being generated (0-based)
- `totalFrames: Int` - Total frames in video
- `timePerStep: Float` - Average seconds per step
- `elapsedSeconds: Float` - Time since generation started
- `estimatedRemainingSeconds: Float` - Estimated time to completion

**Derived Properties**:
```kotlin
val overallProgress: Float
    get() = if (totalSteps > 0) currentStep.toFloat() / totalSteps else 0f

val frameProgress: Float
    get() = if (totalFrames > 0) currentFrame.toFloat() / totalFrames else 0f

val progressPercent: Int
    get() = (overallProgress * 100).toInt()

val statusText: String
    get() = "Frame $currentFrame/$totalFrames (Step $currentStep/$totalSteps)"
```

**Methods**:
```kotlin
fun update(step: Int, frame: Int, timePerStep: Float) {
    currentStep = step
    currentFrame = frame
    this.timePerStep = timePerStep
    elapsedSeconds += timePerStep
    estimatedRemainingSeconds = (totalSteps - currentStep) * timePerStep
}
```

**Relationships**:
- Created fresh for each generation operation
- Updated via `VideoProgressCallback` interface

---

## Entity Relationships Diagram

```
┌─────────────────┐
│   VideoModel    │
│  - modelType    │
│  - version      │──┐
│  - paramCount   │  │ has many
│  - filePath     │  │
│  - quantization │  │
│  - fileSize     │  │
│  - contextWin   │  │
│  - memReq       │  │
└─────────────────┘  │
                     │
                     ↓
┌───────────────────────┐
│ VideoGenerationResult │
│  - output             │
│  - params             │
│  - metrics            │◄────┐
│  - status             │     │
│  - errorMessage       │     │
└───────────────────────┘     │
         │ contains           │ contains
         ├────────────┐       │
         ↓            ↓       │
┌─────────────────┐ ┌──────────────────┐
│  VideoOutput    │ │ GenerationMetrics│
│  - frames       │ │  - totalTime     │
│  - width        │ │  - fps           │
│  - height       │ │  - timePerStep   │
│  - frameCount   │ │  - peakMemory    │
│  - duration     │ │  - vulkanEnabled │
│  - fps          │ │  - loadTime      │
│  - seed         │ │  - convTime      │
│  - createdAt    │ └──────────────────┘
└─────────────────┘
         ↑
         │ created from
         │
┌───────────────────────┐
│VideoGenerationParams  │
│  - prompt             │
│  - negativePrompt     │
│  - width              │
│  - height             │
│  - videoFrames        │
│  - steps              │
│  - cfgScale           │
│  - seed               │
│  - initImage          │
│  - strength           │
│  - scheduler          │
└───────────────────────┘

┌─────────────────┐         ┌────────────────────┐
│  ModelDownload  │         │VideoProgressState  │
│  - modelId      │         │  - currentStep     │
│  - filename     │         │  - totalSteps      │
│  - file         │         │  - currentFrame    │
│  - fileSize     │         │  - totalFrames     │
│  - downloaded   │         │  - timePerStep     │
│  - status       │         │  - elapsed         │
│  - startTime    │         │  - estimatedRemain │
│  - endTime      │         └────────────────────┘
│  - errorMsg     │
└─────────────────┘
```

---

## Data Flow

### 1. Model Loading Flow
```
User → load() → ModelDownload → File → VideoModel(READY)
```

### 2. Video Generation Flow
```
User → VideoGenerationParams → validate() → 
    nativeTxt2Vid() → Array<ByteArray> → 
    List<Bitmap> → VideoOutput → 
    VideoGenerationResult(SUCCESS)
```

### 3. Progress Update Flow
```
Native C++ → sd_progress_callback → JNI → JavaVM attach → 
    VideoProgressCallback.onProgress() → VideoProgressState update → 
    UI refresh
```

### 4. Error Handling Flow
```
Native error → JNI exception → Kotlin catch → 
    VideoGenerationResult(FAILED, errorMessage)
```

---

## Persistence Strategy

### Not Persisted:
- `VideoModel` - Transient, loaded per session
- `VideoProgressState` - Real-time only, discarded after generation
- `VideoOutput.frames` - Large Bitmap arrays, not stored (user exports if needed)

### Optionally Persisted (User Choice):
- `VideoGenerationResult` - Can be serialized to JSON for history tracking
- `ModelDownload` - Cache metadata in SharedPreferences for resume support

### Always Cached:
- Model files - Cached in `filesDir/hf-models/` directory via `HuggingFaceHub`

**Serialization Example** (JSON):
```json
{
  "status": "SUCCESS",
  "params": {
    "prompt": "a lovely cat walking",
    "width": 512,
    "height": 512,
    "videoFrames": 16,
    "seed": 42
  },
  "output": {
    "frameCount": 16,
    "duration": 1.0,
    "fps": 16,
    "createdAt": "2025-11-13T10:30:00Z"
  },
  "metrics": {
    "totalTimeSeconds": 45.3,
    "framesPerSecond": 0.35,
    "peakMemoryUsageMB": 2847,
    "vulkanEnabled": true
  }
}
```

---

## Summary

Data model defined with 7 core entities covering model metadata, generation parameters, output representation, performance metrics, download tracking, and progress state. All entities include validation rules, state transitions, and relationships necessary for type-safe video generation implementation.
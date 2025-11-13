# Kotlin API Contract: Video Generation

**Feature**: Video Model Support  
**Version**: 1.0.0  
**Date**: 2025-11-13

## Overview

This document defines the Kotlin API contract for video generation functionality in llmedge. The API extends the existing `StableDiffusion` class with video generation methods while maintaining backward compatibility.

---

## Public API

### StableDiffusion Class Extensions

#### 1. txt2vid() - Synchronous Video Generation

```kotlin
/**
 * Generates a video from a text prompt using the loaded Wan model.
 *
 * This is a suspending function that blocks until video generation completes.
 * Use from a coroutine scope with Dispatchers.IO or Default.
 *
 * @param params Video generation parameters (prompt, dimensions, frame count, etc.)
 * @param onProgress Optional callback for progress updates during generation
 * @return List of Bitmap frames in generation order
 * @throws IllegalStateException if model is not loaded or is not a video model
 * @throws IllegalArgumentException if params validation fails
 * @throws CancellationException if generation is cancelled via cancelGeneration()
 * @throws OutOfMemoryError if device runs out of memory during generation
 *
 * @sample
 * ```kotlin
 * val sd = StableDiffusion.load(context, modelId = "Kijai/WanVideo_comfy_GGUF", 
 *                              filename = "Phantom-Wan-1_3B_Q4_K_M.gguf")
 * val params = VideoGenerateParams(
 *     prompt = "a lovely cat walking",
 *     width = 512,
 *     height = 512,
 *     videoFrames = 16
 * )
 * val frames = sd.txt2vid(params) { step, totalSteps, frame, totalFrames, timePerStep ->
 *     println("Progress: Frame $frame/$totalFrames, Step $step/$totalSteps")
 * }
 * println("Generated ${frames.size} frames")
 * ```
 */
suspend fun txt2vid(
    params: VideoGenerateParams,
    onProgress: VideoProgressCallback? = null
): List<Bitmap>
```

#### 2. txt2vidAsFlow() - Streaming Video Generation

```kotlin
/**
 * Generates a video from a text prompt as a Flow, emitting frames in chunks.
 *
 * Useful for processing or displaying frames incrementally without waiting for
 * full video generation. Reduces peak memory usage for long videos.
 *
 * @param params Video generation parameters
 * @param chunkSize Number of frames to generate before emitting (default: 4)
 * @return Flow emitting List<Bitmap> chunks as they're generated
 * @throws IllegalStateException if model is not loaded
 * @throws IllegalArgumentException if params validation fails
 *
 * @sample
 * ```kotlin
 * sd.txt2vidAsFlow(params, chunkSize = 4)
 *     .collect { frameChunk ->
 *         // Process chunk of 4 frames
 *         frameChunk.forEach { bitmap ->
 *             videoEncoder.addFrame(bitmap)
 *         }
 *     }
 * ```
 */
fun txt2vidAsFlow(
    params: VideoGenerateParams,
    chunkSize: Int = 4
): Flow<List<Bitmap>>
```

#### 3. cancelGeneration() - Cancel In-Progress Generation

```kotlin
/**
 * Requests cancellation of the current video generation operation.
 *
 * Cancellation is cooperative and may not take effect immediately. The txt2vid()
 * function will throw CancellationException when cancellation completes.
 *
 * Safe to call when no generation is in progress (no-op).
 *
 * @sample
 * ```kotlin
 * val job = launch {
 *     try {
 *         val frames = sd.txt2vid(params)
 *     } catch (e: CancellationException) {
 *         println("Generation cancelled")
 *     }
 * }
 * delay(10000)  // Wait 10 seconds
 * sd.cancelGeneration()
 * job.join()
 * ```
 */
fun cancelGeneration()
```

#### 4. getLastGenerationMetrics() - Retrieve Performance Metrics

```kotlin
/**
 * Returns performance metrics from the most recent video generation.
 *
 * Useful for benchmarking, debugging, and user-facing performance stats.
 *
 * @return GenerationMetrics from last txt2vid() call, or null if no generation yet
 *
 * @sample
 * ```kotlin
 * val frames = sd.txt2vid(params)
 * val metrics = sd.getLastGenerationMetrics()
 * println("Generation took ${metrics.totalTimeSeconds}s")
 * println("Average: ${metrics.framesPerSecond} fps")
 * ```
 */
fun getLastGenerationMetrics(): GenerationMetrics?
```

#### 5. isVideoModel() - Check Model Type

```kotlin
/**
 * Checks if the currently loaded model supports video generation.
 *
 * @return true if model is a Wan variant (T2V, I2V, TI2V), false otherwise
 *
 * @sample
 * ```kotlin
 * val sd = StableDiffusion.load(context, modelPath = "/path/to/model.gguf")
 * if (sd.isVideoModel()) {
 *     val frames = sd.txt2vid(params)
 * } else {
 *     val image = sd.txt2img(params)
 * }
 * ```
 */
fun isVideoModel(): Boolean
```

---

## Data Classes

### VideoGenerateParams

```kotlin
/**
 * Parameters for video generation.
 *
 * @property prompt Text description of desired video content (required, non-blank)
 * @property negative Text description of undesired content (optional, default: "")
 * @property width Output frame width in pixels (must be multiple of 64, range: 256-960)
 * @property height Output frame height in pixels (must be multiple of 64, range: 256-960)
 * @property videoFrames Number of frames to generate (range: 4-64, validated per model)
 * @property steps Denoising steps (range: 10-50, typical: 20)
 * @property cfgScale Classifier-free guidance scale (range: 1.0-15.0, typical: 7.0)
 * @property seed Random seed for reproducibility (-1 for random, non-negative for fixed)
 * @property initImage Starting image for I2V/TI2V models (null for T2V)
 * @property strength Denoising strength for I2V (range: 0.0-1.0, typical: 0.8)
 * @property scheduler Noise scheduler algorithm (default: EULER_A)
 */
data class VideoGenerateParams(
    val prompt: String,
    val negative: String = "",
    val width: Int = 512,
    val height: Int = 512,
    val videoFrames: Int = 16,
    val steps: Int = 20,
    val cfgScale: Float = 7.0f,
    val seed: Long = -1L,
    val initImage: Bitmap? = null,
    val strength: Float = 0.8f,
    val scheduler: Scheduler = Scheduler.EULER_A
) {
    /**
     * Validates parameters against constraints.
     *
     * @return Result.success if valid, Result.failure with exception if invalid
     */
    fun validate(): Result<Unit>
    
    /**
     * Creates a copy with modified prompt.
     */
    fun withPrompt(prompt: String): VideoGenerateParams
    
    companion object {
        /**
         * Returns default parameters suitable for most use cases.
         */
        fun default(): VideoGenerateParams
    }
}

enum class Scheduler {
    EULER_A,  // Euler Ancestral (default, good quality/speed balance)
    DDIM,     // DDIM (Denoising Diffusion Implicit Models)
    DDPM,     // DDPM (Denoising Diffusion Probabilistic Models)
    LCM       // Latent Consistency Models (fast, fewer steps)
}
```

### GenerationMetrics

```kotlin
/**
 * Performance and resource usage metrics for video generation.
 *
 * @property totalTimeSeconds Total generation time from start to finish
 * @property framesPerSecond Average generation throughput (frames/sec)
 * @property timePerStep Average time per denoising step in seconds
 * @property peakMemoryUsageMB Peak native memory usage during generation
 * @property vulkanEnabled Whether GPU acceleration was used
 * @property modelLoadTimeSeconds Time to load model (0 if already loaded)
 * @property frameConversionTimeSeconds Time to convert native RGB to Bitmaps
 */
data class GenerationMetrics(
    val totalTimeSeconds: Float,
    val framesPerSecond: Float,
    val timePerStep: Float,
    val peakMemoryUsageMB: Long,
    val vulkanEnabled: Boolean,
    val modelLoadTimeSeconds: Float = 0f,
    val frameConversionTimeSeconds: Float
) {
    /**
     * Average time to generate one frame.
     */
    val averageFrameTime: Float
        get() = 1.0f / framesPerSecond
    
    /**
     * Denoising steps per second.
     */
    val stepsPerSecond: Float
        get() = 1.0f / timePerStep
    
    /**
     * Human-readable throughput string.
     */
    val throughput: String
        get() = "%.2f fps".format(framesPerSecond)
    
    /**
     * Formats metrics as multi-line string for logging.
     */
    fun toPrettyString(): String
}
```

---

## Callback Interfaces

### VideoProgressCallback

```kotlin
/**
 * Callback interface for receiving progress updates during video generation.
 *
 * All methods are called on a background thread (not UI thread). Use appropriate
 * dispatchers (e.g., withContext(Dispatchers.Main)) for UI updates.
 *
 * @sample
 * ```kotlin
 * val callback = object : VideoProgressCallback {
 *     override fun onProgress(step: Int, totalSteps: Int, 
 *                            currentFrame: Int, totalFrames: Int, 
 *                            timePerStep: Float) {
 *         val progress = (step.toFloat() / totalSteps * 100).toInt()
 *         println("Progress: $progress% (Frame $currentFrame/$totalFrames)")
 *     }
 * }
 * val frames = sd.txt2vid(params, callback)
 * ```
 */
fun interface VideoProgressCallback {
    /**
     * Called periodically during video generation to report progress.
     *
     * @param step Current denoising step (0-based)
     * @param totalSteps Total denoising steps across all frames
     * @param currentFrame Frame being generated (0-based)
     * @param totalFrames Total frames in video
     * @param timePerStep Average seconds per step (updated incrementally)
     */
    fun onProgress(
        step: Int,
        totalSteps: Int,
        currentFrame: Int,
        totalFrames: Int,
        timePerStep: Float
    )
}
```

---

## JNI Native Methods

### Internal JNI Interface (Not Public API)

```kotlin
/**
 * Native method for video generation. Do not call directly.
 *
 * @return Array of byte arrays (RGB format), one per frame, or null on failure
 */
private external fun nativeTxt2Vid(
    handle: Long,
    prompt: String,
    negative: String,
    width: Int,
    height: Int,
    videoFrames: Int,
    steps: Int,
    cfg: Float,
    seed: Long,
    initImageBytes: ByteArray?,
    initWidth: Int,
    initHeight: Int
): Array<ByteArray>?

/**
 * Sets progress callback for video generation.
 *
 * @param callback VideoProgressCallback instance or null to clear
 */
private external fun nativeSetProgressCallback(
    handle: Long,
    callback: VideoProgressCallback?
)

/**
 * Requests cancellation of in-progress generation.
 */
private external fun nativeCancelGeneration(handle: Long)
```

---

## Error Handling

### Exception Types

```kotlin
// IllegalStateException: Model not loaded or wrong type
throw IllegalStateException("Model is not loaded")
throw IllegalStateException("Loaded model is not a video model (use txt2img instead)")
throw IllegalStateException("Video generation failed - check logcat for native errors")

// IllegalArgumentException: Invalid parameters
throw IllegalArgumentException("Prompt cannot be empty")
throw IllegalArgumentException("Width must be multiple of 64, got: $width")
throw IllegalArgumentException("Frame count 128 exceeds maximum 64 for 1.3B models")
throw IllegalArgumentException("I2V models require initImage parameter")

// CancellationException: User cancelled
throw CancellationException("Video generation cancelled by user")

// OutOfMemoryError: Device ran out of memory
throw OutOfMemoryError("Insufficient memory for video generation")
```

### Error Handling Patterns

```kotlin
// Pattern 1: Try-catch with result type
suspend fun generateVideoSafe(params: VideoGenerateParams): Result<List<Bitmap>> {
    return try {
        val frames = sd.txt2vid(params)
        Result.success(frames)
    } catch (e: IllegalStateException) {
        Result.failure(e)
    } catch (e: IllegalArgumentException) {
        Result.failure(e)
    } catch (e: CancellationException) {
        Result.failure(e)
    } catch (e: OutOfMemoryError) {
        Result.failure(e)
    }
}

// Pattern 2: Specific error handling
try {
    val frames = sd.txt2vid(params)
    // Success
} catch (e: IllegalArgumentException) {
    // Show validation error to user
    Toast.makeText(context, "Invalid parameters: ${e.message}", LENGTH_LONG).show()
} catch (e: CancellationException) {
    // User cancelled - no action needed
} catch (e: OutOfMemoryError) {
    // Suggest lower resolution/frame count
    Toast.makeText(context, "Not enough memory - try fewer frames", LENGTH_LONG).show()
}
```

---

## Threading Model

### Coroutine Dispatchers

```kotlin
// txt2vid() must be called from IO or Default dispatcher
CoroutineScope(Dispatchers.IO).launch {
    val frames = sd.txt2vid(params)
    // frames available here
}

// Or use withContext
val frames = withContext(Dispatchers.Default) {
    sd.txt2vid(params)
}
```

### Progress Callback Threading

```kotlin
// Callbacks invoked on background thread - use Dispatchers.Main for UI
val callback = object : VideoProgressCallback {
    override fun onProgress(step: Int, totalSteps: Int, ...) {
        // Background thread - safe for logging
        Log.d("Progress", "Step $step/$totalSteps")
        
        // UI updates require Main dispatcher
        launch(Dispatchers.Main) {
            progressBar.progress = (step * 100 / totalSteps)
        }
    }
}
```

### Concurrency Guarantees

- **Single generation at a time**: Internal `generationMutex` prevents concurrent txt2vid() calls
- **Reentrant-safe callbacks**: Progress callbacks can call other StableDiffusion methods (e.g., getLastGenerationMetrics())
- **Thread-safe cancellation**: cancelGeneration() can be called from any thread

---

## Backward Compatibility

### No Breaking Changes

- All existing `StableDiffusion` APIs remain unchanged
- `txt2img()` continues to work for image models
- Existing model loading patterns compatible with video models

### Version Detection

```kotlin
// Check library version (in BuildConfig or companion object)
if (StableDiffusion.VERSION >= "2.0.0") {
    // Video generation available
    if (sd.isVideoModel()) {
        val frames = sd.txt2vid(params)
    }
} else {
    // Fallback to image generation only
    val image = sd.txt2img(imageParams)
}
```

---

## Performance Expectations

### Typical Generation Times (1.3B Model, Q4_K_M)

| Resolution | Frames | Steps | Time (CPU) | Time (Vulkan) |
|-----------|--------|-------|-----------|---------------|
| 256×256 | 16 | 20 | ~60s | ~40s |
| 512×512 | 16 | 20 | ~120s | ~80s |
| 512×512 | 32 | 20 | ~240s | ~160s |
| 768×768 | 16 | 20 | ~300s | ~200s |

*Note: Times measured on mid-range Android device (Snapdragon 778G, 6GB RAM)*

### Memory Usage

| Model Size | Quantization | Native RAM | JVM Heap |
|-----------|--------------|-----------|----------|
| 1.3B | Q4_K_M | 2-3GB | 20-30MB |
| 1.3B | Q6_K | 3-4GB | 20-30MB |
| 5B | fp8 | 6-10GB | 20-30MB |

---

## Complete Usage Example

```kotlin
import io.aatricks.llmedge.StableDiffusion
import io.aatricks.llmedge.VideoGenerateParams
import io.aatricks.llmedge.VideoProgressCallback
import kotlinx.coroutines.*
import android.graphics.Bitmap

class VideoGenerationActivity : AppCompatActivity() {
    private lateinit var sd: StableDiffusion
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Load Wan model
        lifecycleScope.launch(Dispatchers.IO) {
            sd = StableDiffusion.load(
                context = this@VideoGenerationActivity,
                modelId = "Kijai/WanVideo_comfy_GGUF",
                filename = "Phantom-Wan-1_3B_Q4_K_M.gguf",
                preferSystemDownloader = true
            )
            
            // Verify model type
            if (!sd.isVideoModel()) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@VideoGenerationActivity, 
                                 "Model does not support video generation", 
                                 Toast.LENGTH_LONG).show()
                }
                return@launch
            }
            
            // Generate video
            generateVideo()
        }
    }
    
    private suspend fun generateVideo() {
        val params = VideoGenerateParams(
            prompt = "a lovely cat walking in a garden",
            negative = "blurry, low quality, distorted",
            width = 512,
            height = 512,
            videoFrames = 16,
            steps = 20,
            cfgScale = 7.0f,
            seed = 42L
        )
        
        // Validate parameters
        params.validate().getOrElse { error ->
            withContext(Dispatchers.Main) {
                Toast.makeText(this@VideoGenerationActivity, 
                             "Invalid parameters: ${error.message}", 
                             Toast.LENGTH_LONG).show()
            }
            return
        }
        
        // Progress callback
        val progressCallback = object : VideoProgressCallback {
            override fun onProgress(step: Int, totalSteps: Int, 
                                  currentFrame: Int, totalFrames: Int, 
                                  timePerStep: Float) {
                val progress = (step * 100 / totalSteps)
                val eta = (totalSteps - step) * timePerStep
                
                launch(Dispatchers.Main) {
                    progressBar.progress = progress
                    statusText.text = "Frame $currentFrame/$totalFrames (${eta.toInt()}s remaining)"
                }
            }
        }
        
        try {
            val frames = sd.txt2vid(params, progressCallback)
            
            // Get metrics
            val metrics = sd.getLastGenerationMetrics()
            Log.d("VideoGen", "Generated ${frames.size} frames in ${metrics?.totalTimeSeconds}s")
            
            // Display or save frames
            withContext(Dispatchers.Main) {
                displayVideo(frames)
            }
            
        } catch (e: CancellationException) {
            Log.d("VideoGen", "Generation cancelled")
        } catch (e: IllegalArgumentException) {
            withContext(Dispatchers.Main) {
                Toast.makeText(this@VideoGenerationActivity, 
                             "Parameter error: ${e.message}", 
                             Toast.LENGTH_LONG).show()
            }
        } catch (e: OutOfMemoryError) {
            withContext(Dispatchers.Main) {
                Toast.makeText(this@VideoGenerationActivity, 
                             "Out of memory - try fewer frames or lower resolution", 
                             Toast.LENGTH_LONG).show()
            }
        }
    }
    
    private fun displayVideo(frames: List<Bitmap>) {
        // Play frames at 16 fps
        var currentFrame = 0
        val handler = Handler(Looper.getMainLooper())
        val runnable = object : Runnable {
            override fun run() {
                if (currentFrame < frames.size) {
                    imageView.setImageBitmap(frames[currentFrame])
                    currentFrame++
                    handler.postDelayed(this, 62)  // ~16 fps
                }
            }
        }
        handler.post(runnable)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        sd.close()  // Free native resources
    }
}
```

---

## Summary

API contract defines 5 new public methods, 2 data classes, and 1 callback interface for video generation. All methods follow existing llmedge patterns with comprehensive error handling, progress tracking, and performance metrics. No breaking changes to existing APIs.
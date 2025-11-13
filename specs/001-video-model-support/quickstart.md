# Video Generation Quickstart

Get started with on-device text-to-video generation in 5 minutes.

---

## Prerequisites

- **Android Studio**: Latest stable version
- **minSdk 30**: Android 11+ (Vulkan acceleration support)
- **Device RAM**: 4GB+ recommended for 1.3B models, 8GB+ for 5B models
- **Storage**: 2-6GB free space for model downloads

---

## Installation

### Step 1: Add llmedge to your project

**build.gradle.kts (app level):**
```kotlin
dependencies {
    implementation("io.aatricks:llmedge:1.0.0")
}
```

### Step 2: Add required permissions

**AndroidManifest.xml:**
```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

---

## Basic Usage

### Generate Your First Video

**MainActivity.kt:**
```kotlin
import android.os.Bundle
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import io.aatricks.llmedge.StableDiffusion
import io.aatricks.llmedge.StableDiffusion.VideoGenerateParams
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        val imageView = findViewById<ImageView>(R.id.preview_image)
        
        lifecycleScope.launch {
            try {
                // 1. Load Wan 1.3B model from Hugging Face
                val sd = StableDiffusion.load(
                    context = applicationContext,
                    modelId = "Kijai/WanVideo_comfy_GGUF",
                    filename = "hunyuan_video_720_cfgdistill_fp8_e4m3fn-Q4_K_M.gguf"
                )
                
                // 2. Generate 16-frame video at 512x512
                val frames = sd.txt2vid(VideoGenerateParams(
                    prompt = "A cute cat playing with yarn, high quality, detailed",
                    negativePrompt = "blurry, low quality, distorted",
                    width = 512,
                    height = 512,
                    videoFrames = 16,
                    steps = 20,
                    cfgScale = 7.0f,
                    seed = -1 // Random seed
                ))
                
                // 3. Display first frame (for demo)
                withContext(Dispatchers.Main) {
                    imageView.setImageBitmap(frames[0])
                }
                
                // 4. Cleanup
                sd.close()
                
            } catch (e: Exception) {
                e.printStackTrace()
                // Handle error (show toast, etc.)
            }
        }
    }
}
```

**Expected output:**
- **Generation time**: ~60 seconds on modern devices (Vulkan)
- **Memory usage**: ~2.5GB peak (1.3B Q4_K_M model)
- **Frame count**: 16 frames at 512×512 RGB
- **Quality**: Good temporal coherence for cat movement

---

## Advanced: Progress Tracking

Display generation progress to users:

```kotlin
lifecycleScope.launch {
    val sd = StableDiffusion.load(applicationContext, modelId, filename)
    
    val progressBar = findViewById<ProgressBar>(R.id.progress_bar)
    val statusText = findViewById<TextView>(R.id.status_text)
    
    // Setup progress callback
    val callback = object : StableDiffusion.VideoProgressCallback {
        override fun onProgress(
            currentStep: Int,
            totalSteps: Int,
            currentFrame: Int,
            totalFrames: Int,
            timeElapsed: Float
        ) {
            runOnUiThread {
                val progress = ((currentFrame * totalSteps + currentStep) * 100) / 
                              (totalFrames * totalSteps)
                progressBar.progress = progress
                statusText.text = "Frame $currentFrame/$totalFrames - Step $currentStep/$totalSteps"
            }
        }
    }
    sd.setProgressCallback(callback)
    
    // Generate with progress updates
    val frames = sd.txt2vid(VideoGenerateParams(
        prompt = "Sunset over mountains, cinematic",
        width = 512,
        height = 512,
        videoFrames = 16
    ))
    
    // Clear callback
    sd.setProgressCallback(null)
    
    // Display frames...
    sd.close()
}
```

---

## Display Video Frames

### Option 1: ImageView Slideshow

Simple frame-by-frame display:

```kotlin
val imageView = findViewById<ImageView>(R.id.video_view)
val frames: List<Bitmap> = ... // from txt2vid()

lifecycleScope.launch {
    frames.forEach { frame ->
        withContext(Dispatchers.Main) {
            imageView.setImageBitmap(frame)
        }
        delay(100) // 10 FPS
    }
}
```

### Option 2: VideoView Export

Save as MP4 video file:

```kotlin
import android.media.MediaCodec
import android.media.MediaFormat
import android.media.MediaMuxer

fun exportToVideo(frames: List<Bitmap>, outputPath: String) {
    val width = frames[0].width
    val height = frames[0].height
    val fps = 10
    
    val format = MediaFormat.createVideoFormat(
        MediaFormat.MIMETYPE_VIDEO_AVC,
        width,
        height
    ).apply {
        setInteger(MediaFormat.KEY_BIT_RATE, 2_000_000)
        setInteger(MediaFormat.KEY_FRAME_RATE, fps)
        setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        setInteger(MediaFormat.KEY_COLOR_FORMAT, 
                  MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
    }
    
    val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
    codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
    codec.start()
    
    val muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    var trackIndex = -1
    
    frames.forEachIndexed { index, frame ->
        // Convert Bitmap to YUV and encode
        // (Implementation details omitted for brevity)
    }
    
    codec.stop()
    codec.release()
    muxer.stop()
    muxer.release()
}
```

---

## Model Selection Guide

Choose model based on device capabilities:

| Model Variant | Size | RAM Required | Generation Time (16 frames) | Quality |
|--------------|------|--------------|------------------------------|---------|
| **Wan 1.3B Q4_K_M** | 1.4GB | 2-3GB | ~60s (Vulkan), ~90s (CPU) | Good |
| **Wan 5B FP8** | 5GB | 6-8GB | ~150s (Vulkan), ~300s (CPU) | Excellent |

**Download paths**:
- 1.3B: `Kijai/WanVideo_comfy_GGUF` → `hunyuan_video_720_cfgdistill_fp8_e4m3fn-Q4_K_M.gguf`
- 5B: `Kijai/WanVideo_comfy_GGUF` → `hunyuan_video_720_cfgdistill_fp8_e4m3fn.gguf`

---

## Common Pitfalls

### ❌ OutOfMemoryError during generation

**Problem**: Device runs out of RAM.

**Solutions**:
- Use 1.3B model instead of 5B
- Reduce `videoFrames` (try 8 instead of 16)
- Lower resolution (try 256×256 instead of 512×512)
- Close other apps before generation

### ❌ Generation takes too long (>5 minutes)

**Problem**: CPU-only inference or low-end device.

**Solutions**:
- Verify Vulkan enabled: `StableDiffusion.isVulkanEnabled()`
- Reduce `steps` (try 15 instead of 20)
- Use smaller model (1.3B instead of 5B)
- Upgrade to device with Vulkan support

### ❌ IllegalStateException: "Model is not a video model"

**Problem**: Loaded image generation model instead of video model.

**Solutions**:
- Verify model filename ends with `.gguf` and is Wan model
- Check model type: `sd.isVideoModel()` should return `true`
- Re-download model if corrupted

### ❌ Poor video quality (flickering, artifacts)

**Problem**: Insufficient denoising steps or bad seed.

**Solutions**:
- Increase `steps` to 25-30
- Try different seeds: `seed = 42L` (fixed) instead of `-1` (random)
- Use negative prompt: `"blurry, low quality, distorted, artifacts"`
- Increase `cfgScale` to 8.0-9.0 for more prompt adherence

### ❌ ANR (Application Not Responding)

**Problem**: Generation running on main thread.

**Solutions**:
- Always use coroutines: `lifecycleScope.launch { ... }`
- Verify `Dispatchers.IO` context: `withContext(Dispatchers.IO) { sd.txt2vid(...) }`
- Use streaming API for long generations: `sd.txt2vidAsFlow(...)`

---

## Performance Tips

### 1. Enable Vulkan Acceleration

**Check at runtime**:
```kotlin
if (StableDiffusion.isVulkanEnabled()) {
    println("Vulkan GPU acceleration active - 2-3x faster generation")
} else {
    println("CPU fallback - consider using smaller model")
}
```

### 2. Batch Multiple Generations

Reuse model instance:
```kotlin
val sd = StableDiffusion.load(context, modelId, filename)

val prompts = listOf(
    "A cat playing",
    "A dog running",
    "A bird flying"
)

prompts.forEach { prompt ->
    val frames = sd.txt2vid(VideoGenerateParams(prompt = prompt))
    // Process frames...
}

sd.close() // Cleanup once at end
```

### 3. Monitor Memory Usage

Track native memory:
```kotlin
val metrics = sd.getLastGenerationMetrics()
println("Native memory used: ${metrics.nativeMemoryUsedMb} MB")
println("Tokens/sec: ${metrics.tokensPerSecond}")
```

---

## Next Steps

1. **Explore image-to-video**: Use `initImage` parameter for guided generation
2. **Experiment with CFG scale**: Try values 5.0-12.0 for different styles
3. **Tune resolution vs speed**: Balance quality and generation time
4. **Implement caching**: Save generated frames to avoid re-generation
5. **Read full API docs**: See `contracts/kotlin-api.md` for complete reference

---

## Full Working Example

**Complete project structure**:

```
app/
  src/main/
    java/com/example/videoapp/
      MainActivity.kt  ← Code above
    res/layout/
      activity_main.xml  ← UI layout
    AndroidManifest.xml  ← Permissions
  build.gradle.kts  ← Dependencies
```

**activity_main.xml**:
```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">
    
    <ImageView
        android:id="@+id/preview_image"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1"
        android:scaleType="centerCrop" />
    
    <ProgressBar
        android:id="@+id/progress_bar"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        style="@android:style/Widget.ProgressBar.Horizontal" />
    
    <TextView
        android:id="@+id/status_text"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_gravity="center"
        android:text="Ready" />
</LinearLayout>
```

---

## Need Help?

- **Documentation**: Check `/docs` folder for detailed guides
- **Examples**: See `llmedge-examples` app for more complex use cases
- **Issues**: Report bugs at GitHub repository
- **Performance**: Read `contracts/kotlin-api.md` for optimization techniques
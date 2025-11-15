# llmedge

**llmedge** is a lightweight Android library for running GGUF language models fully on-device, powered by [llama.cpp](https://github.com/ggerganov/llama.cpp).

See the [examples repository](https://github.com/Aatricks/llmedge-examples) for sample usage.

Acknowledgments to Shubham Panchal and upstream projects are listed in [`CREDITS.md`](./CREDITS.md).

> [!NOTE]
> This library is in early development and may change significantly.

---

## Features

- Run GGUF models directly on Android using llama.cpp (JNI)
- Download and cache models from Hugging Face
- **Video Generation**: Generate short video clips (4-64 frames) from text using Wan models
- **Image Generation**: Stable Diffusion integration for txt2img
- Minimal on-device RAG (retrieval-augmented generation) pipeline
- **OCR Support**: Extract text from images using Google ML Kit
- **Vision Model Ready**: Architecture prepared for vision-capable LLMs (LLaVA)
- Built-in memory usage metrics
- Optional Vulkan acceleration

---

## Table of Contents

1. [Installation](#installation)  
2. [Usage](#usage)  
   - [Downloading Models](#downloading-models)  
   - [Reasoning Controls](#reasoning-controls)  
   - [Image Text Extraction (OCR)](#image-text-extraction-ocr)
   - [Vision Models](#vision-models)
   - [Stable Diffusion (image generation)](#stable-diffusion-image-generation)
   - [Video Generation](#video-generation)
   - [On-device RAG](#on-device-rag)  
3. [Building](#building)  
4. [Architecture](#architecture)  
5. [Technologies](#technologies)  
6. [Memory Metrics](#memory-metrics)  
7. [Notes](#notes)

---

## Installation

> [!WARNING]
> For development, it is strongly recommended to work on linux due to Vulkan backend build for Stable Diffusion not working on Windows.

Clone the repository along with the `llama.cpp` and `stable-diffusion.cpp` submodule:

```bash
git clone --depth=1 https://github.com/Aatricks/llmedge
cd llmedge
git submodule update --init --recursive
```

Open the project in Android Studio. If it does not build automatically, use ***Build > Rebuild Project.***

## Usage

### Quick Start

Load a local GGUF file and run a blocking prompt from a background coroutine:

```kotlin
val smol = SmolLM()

CoroutineScope(Dispatchers.IO).launch {
    val modelFile = File(context.filesDir, "models/tinyllama.gguf")
    smol.load(modelFile.absolutePath)

    val reply = smol.getResponse("Summarize on-device LLMs in one sentence.")
    withContext(Dispatchers.Main) {
        outputView.text = reply
    }
}
```

Call `smol.close()` when the instance is no longer needed to free native memory.

### Downloading Models

llmedge can download and cache GGUF model weights directly from Hugging Face:

```kotlin
val smol = SmolLM()

val download = smol.loadFromHuggingFace(
    context = context,
    modelId = "unsloth/Qwen3-0.6B-GGUF",
    filename = "Qwen3-0.6B-Q4_K_M.gguf", // optional
    forceDownload = false,
    preferSystemDownloader = true
)

Log.d("llmedge", "Loaded ${download.file.name} from ${download.file.parent}")
```

#### Key points:

- loadFromHuggingFace downloads (if needed) and loads the model immediately after.

- Supports onProgress callbacks and private repositories via token.

- Requests to old mirrors automatically resolve to up-to-date Hugging Face repos.

- Automatically uses the model's declared context window (minimum 1K tokens) and caps it to a heap-aware limit (2K–8K). Override with `InferenceParams(contextSize = …)` if needed.

- Large downloads use Android's DownloadManager when `preferSystemDownloader = true` to keep transfers out of the Dalvik heap.

- Advanced users can call `HuggingFaceHub.ensureModelOnDisk()` to manage caching and quantization manually.

### Reasoning Controls

`SmolLM` lets you disable or re-enable "thinking" traces produced by reasoning-aware models through the `ThinkingMode` enum and the optional `reasoningBudget` parameter. The default configuration keeps thinking enabled (`ThinkingMode.DEFAULT`, reasoning budget `-1`). To start a session with thinking disabled (equivalent to passing `--no-think` or `--reasoning-budget 0`), specify it when loading the model:

```kotlin
val smol = SmolLM()

val params = SmolLM.InferenceParams(
    thinkingMode = SmolLM.ThinkingMode.DISABLED,
    reasoningBudget = 0, // explicit override, optional when the mode is DISABLED
)
smol.load(modelPath, params)
```

At runtime you can flip the behaviour without reloading the model:

```kotlin
smol.setThinkingEnabled(true)              // restore the default
smol.setReasoningBudget(0)                 // force-disable thoughts again
val budget = smol.getReasoningBudget()     // inspect the current budget
val mode = smol.getThinkingMode()          // inspect the current mode
```

Setting the budget to `0` always disables thinking, while `-1` leaves it unrestricted. If you omit `reasoningBudget`, the library chooses `0` when the mode is `DISABLED` and `-1` otherwise. The API also injects the `/no_think` tag automatically when thinking is disabled, so you do not need to modify prompts manually.

### Image Text Extraction (OCR)

llmedge uses Google ML Kit Text Recognition for extracting text from images.

#### Quick Start

```kotlin
import io.aatricks.llmedge.vision.*
import io.aatricks.llmedge.vision.ocr.*

// Initialize OCR engine
val mlKitEngine = MlKitOcrEngine(context)

// Create image understanding instance with OCR
val imageUnderstanding = ImageUnderstanding(
    visionAnalyzer = null, // Add vision model when available
    ocrEngines = listOf(mlKitEngine)
)

// Process an image
val imageFile = File("/path/to/image.jpg")
val result = imageUnderstanding.process(
    image = ImageSource.FileSource(imageFile),
    mode = VisionMode.AUTO_PREFER_OCR
)

println("Extracted text: ${result.text}")
println("Engine used: ${result.engine}")
```

#### OCR Engines

**Google ML Kit Text Recognition**
- Fast and lightweight
- No additional data files needed
- Good for Latin scripts
- Add dependency: `implementation("com.google.mlkit:text-recognition:16.0.0")`

#### Processing Modes
#### Processing Modes

```kotlin
enum class VisionMode {
        AUTO_PREFER_VISION,  // Try vision model first, fall back to OCR
        AUTO_PREFER_OCR,     // Try OCR first (ML Kit)
        FORCE_VISION,        // Vision model only (error if unavailable)
        FORCE_MLKIT          // ML Kit OCR only
}
```

### Vision Models

The library architecture supports vision-capable language models (like LLaVA), though native vision support in llama.cpp is still being integrated.

#### Prepared Architecture

```kotlin
// Future usage when vision models are fully supported
val visionAdapter = SmolLMVisionAdapter(context, smolLM)
visionAdapter.loadVisionModel(
    modelPath = "/path/to/llava.gguf",
    mmprojPath = "/path/to/mmproj.bin"
)

val imageUnderstanding = ImageUnderstanding(
    visionAnalyzer = visionAdapter,
    ocrEngines = listOf(mlKitEngine) // Fallback
)

val result = imageUnderstanding.process(
    image = ImageSource.FileSource(imageFile),
    mode = VisionMode.AUTO_PREFER_VISION,
    prompt = "Describe what you see in this image."
)
```

Vision model support will be enabled once llama.cpp's multimodal capabilities are integrated into the Android build.


### Stable Diffusion (image generation)

llmedge now includes a Stable Diffusion integration for on-device image generation via the `StableDiffusion` Kotlin API. The examples app (`llmedge-examples`) contains a working demo at `StableDiffusionActivity.kt` that demonstrates downloading required assets from Hugging Face, loading the model, and generating images safely on memory-constrained devices.

Key points:
- The library can download model weights and auxiliary files (for example a VAE) from Hugging Face and cache them under the app files directory.
- Large files should be downloaded using the system downloader (`preferSystemDownloader = true`) to avoid allocating large buffers on the app heap which can lead to OOMs.
- Stable Diffusion models are memory intensive. Use conservative image resolutions and enable CPU offloading when necessary (see example below).

Quick Kotlin example (adapted from `StableDiffusionActivity`):

```kotlin
// Ensure a VAE safetensors file is present (optional depending on the model repo)
val vaeDownload = io.aatricks.llmedge.huggingface.HuggingFaceHub.ensureRepoFileOnDisk(
    context = this,
    modelId = "Meina/MeinaMix",
    filename = "MeinaPastel - baked VAE.safetensors",
    token = null,
    forceDownload = false,
    preferSystemDownloader = true,
    onProgress = null
)

val sd = StableDiffusion.load(
    context = this,
    modelId = "Meina/MeinaMix",
    filename = "MeinaPastel - baked VAE.safetensors", // let the loader pick the gguf in the repo
    nThreads = Runtime.getRuntime().availableProcessors(),
    offloadToCpu = true,     // reduce native memory pressure by offloading some tensors to CPU
    keepClipOnCpu = true,    // keep CLIP on CPU if device GPU/ram is constrained
    keepVaeOnCpu = false,    // VAE can be kept on GPU when available, choose based on device
    vaePath = vaeDownload.file.absolutePath
)

val bmp = sd.txt2img(
    StableDiffusion.GenerateParams(
        prompt = "a cute pastel anime cat, soft colors, high quality",
        steps = 20,
        cfgScale = 7.0f,
        width = 128,
        height = 128,
        seed = 42L
    )
)

imageView.setImageBitmap(bmp)
sd.close()
```

Memory & troubleshooting:
- If you encounter OutOfMemoryError during generation, reduce width/height and number of steps, or enable more aggressive CPU offloading.
- The example app falls back to a smaller resolution when a generation OOM occurs; you should do the same in production code.

### Video Generation

llmedge now supports **on-device video generation** using Wan models via the `StableDiffusion` class. Generate short video clips (4-64 frames) from text prompts entirely on Android devices.

**⚠️ Hardware Requirements**:
- **Minimum RAM**: 12GB recommended for Wan 2.1 T2V-1.3B with mixed precision (fp16 main + Q3_K_S T5XXL)
- **Supported Devices**: Galaxy S23 Ultra (12GB), Pixel 8 Pro (12GB), OnePlus 12 (16GB+)
- **Not Supported**: 8GB RAM devices (Galaxy S22, Pixel 7) - insufficient memory for 3-model pipeline

**Why 12GB?**
Wan models require loading **three separate components**:
1. Main diffusion model (fp16): ~2.7GB RAM
2. T5XXL text encoder (Q3_K_S GGUF): ~5.9GB RAM (decompresses from 2.86GB file)
3. VAE decoder (fp16): ~0.14GB RAM
4. Working memory + overhead: ~1GB RAM

**Total**: ~9.7GB minimum, 12GB recommended for stable operation.

**Supported Models**:
- **Wan 2.1**: T2V-1.3B (text-to-video) - **12GB+ RAM required**
- **Wan 2.2**: TI2V-5B (text+image-to-video) - **16GB+ RAM required**
- **14B models**: Not supported on mobile (rejected at load time)

**Note**: GGUF quantization of the main model is currently blocked by metadata issues in community conversions. Only safetensors format works reliably.

**Basic Usage**:

```kotlin
// Download all three required model files explicitly
val modelFile = io.aatricks.llmedge.huggingface.HuggingFaceHub.ensureRepoFileOnDisk(
    context = this,
    modelId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
    revision = "main",
    filename = "wan2.1_t2v_1.3B_fp16.safetensors",
    allowedExtensions = listOf(".safetensors"),
    preferSystemDownloader = true
)

val vaeFile = io.aatricks.llmedge.huggingface.HuggingFaceHub.ensureRepoFileOnDisk(
    context = this,
    modelId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
    revision = "main",
    filename = "wan_2.1_vae.safetensors",
    allowedExtensions = listOf(".safetensors"),
    preferSystemDownloader = true
)

val t5xxlFile = io.aatricks.llmedge.huggingface.HuggingFaceHub.ensureRepoFileOnDisk(
    context = this,
    modelId = "city96/umt5-xxl-encoder-gguf",
    revision = "main",
    filename = "umt5-xxl-encoder-Q3_K_S.gguf",
    allowedExtensions = listOf(".gguf"),
    preferSystemDownloader = true
)

// Load all three models together
val sd = StableDiffusion.load(
    context = this,
    modelPath = modelFile.file.absolutePath,
    vaePath = vaeFile.file.absolutePath,
    t5xxlPath = t5xxlFile.file.absolutePath,
    nThreads = Runtime.getRuntime().availableProcessors(),
    offloadToCpu = true,
    keepClipOnCpu = true,
    keepVaeOnCpu = true
)

val params = StableDiffusion.VideoGenerateParams(
    prompt = "a cat walking in a garden, high quality",
    videoFrames = 8,  // Start small: 4-8 frames
    width = 256,      // Conservative resolution
    height = 256,
    steps = 20,
    cfgScale = 7.0,
    seed = 42
)

val frames: List<Bitmap> = sd.txt2vid(params)

// Save frames to video file or display
frames.forEachIndexed { index, bitmap ->
    // Process each frame
}

sd.close()
```

**Model Selection**:

⚠️ **Important**: Due to memory constraints, video generation requires explicit multi-file loading. The simplified `modelId` + `filename` approach does not work for Wan models.

Check device RAM before attempting to load:

```kotlin
val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
val memInfo = ActivityManager.MemoryInfo()
activityManager.getMemoryInfo(memInfo)
val totalRamGB = memInfo.totalMem / (1024.0 * 1024.0 * 1024.0)

if (totalRamGB < 12.0) {
    // Show error: device does not have enough RAM for video generation
    // Recommend cloud inference API or alternative approaches
    throw UnsupportedOperationException(
        "Video generation requires 12GB+ RAM. " +
        "This device has ${String.format("%.1f", totalRamGB)}GB. " +
        "Consider using cloud inference APIs instead."
    )
}

// Proceed with model loading (see Basic Usage above)
```

**Advanced Parameters**:

```kotlin
val params = StableDiffusion.VideoGenerateParams(
    prompt = "a serene ocean sunset, waves gently rolling",
    videoFrames = 32,
    width = 512,
    height = 512,
    steps = 25,
    cfgScale = 7.5,
    seed = -1,  // -1 for random seed
    scheduler = StableDiffusion.Scheduler.DDIM,
    strength = 0.8,  // For image-to-video (0.0-1.0)
    initImage = null  // Optional: initial image for I2V models
)
```

**Progress Monitoring**:

```kotlin
sd.setProgressCallback { step, totalSteps ->
    val progress = (step.toFloat() / totalSteps * 100).toInt()
    runOnUiThread {
        progressBar.progress = progress
        statusText.text = "Generating: $progress%"
    }
}

val frames = sd.txt2vid(params)
```

**Cancellation**:

```kotlin
// Cancel long-running generation
sd.cancelGeneration()
```

**Performance Tips**:
- Start with 256x256 resolution and 4-8 frames for testing
- Use 512x512 only on 16GB+ devices
- Use 10-20 steps for faster generation (20-30 for quality)
- Enable Vulkan acceleration on Android 11+ devices for 2-5x speedup
- 1.3B models generate ~3-8 seconds/frame on mid-range devices (Vulkan enabled)
- Expect 30-120 seconds for an 8-frame 256x256 video

**Memory Management**:
- Wan 2.1 T2V-1.3B peaks at ~9.7GB RAM (fp16 main + Q3_K_S T5XXL + fp16 VAE)
- Wan 2.2 TI2V-5B peaks at ~16GB+ RAM
- 14B models are rejected at load time (require 20-40GB RAM)
- All memory optimizations are already enabled:
  - `free_params_immediately = true` (frees buffers after loading)
  - `offload_params_to_cpu = true` (uses CPU backend)
  - Quantized T5XXL encoder (Q3_K_S reduces 9.5GB → 5.9GB at runtime)

**Known Limitations**:
- GGUF quantization of main model blocked by metadata issues
- Sequential loading not supported - all three models load simultaneously
- No disk streaming - models must fit in RAM
- 8GB RAM devices cannot run Wan models (architectural constraint)

See `llmedge-examples/app/src/main/java/io/aatricks/llmedge/VideoGenerationActivity.kt` for a complete working example.


Running the example app:
1. Build the library AAR and copy it into the example app (from the repo root):

```bash
./gradlew :llmedge:assembleRelease
cp llmedge/build/outputs/aar/llmedge-release.aar llmedge-examples/app/libs/llmedge-release.aar
```

2. Build and install the example app:

```bash
cd llmedge-examples
../gradlew :app:assembleDebug
../gradlew :app:installDebug
```

3. Open the app on device and pick the "Stable Diffusion" demo from the launcher. The demo downloads any missing files from Hugging Face and runs a quick txt2img generation.

Notes:
- The example explicitly downloads a VAE safetensors file for the `Meina/MeinaMix` demo; many repos include VAE files, but some GGUF model repos bundle everything you need. If the repo lacks a GGUF model file you'll get an obvious IllegalArgumentException — provide a `filename` or choose a different repo in that case.
- Use the system downloader for large safetensors/gguf files to avoid heap pressure on Android.
### On-device RAG

The library includes a minimal on-device RAG pipeline, similar to Android-Doc-QA, built with:
- Sentence embeddings (ONNX)
- Whitespace `TextSplitter`
- In-memory cosine `VectorStore` with JSON persistence
- `SmolLM` for context-aware responses

### Setup

1. Download embeddings

   From the Hugging Face repository `sentence-transformers/all-MiniLM-L6-v2`, place:

```
llmedge/src/main/assets/embeddings/all-minilm-l6-v2/model.onnx
llmedge/src/main/assets/embeddings/all-minilm-l6-v2/tokenizer.json
```

2. Build the library

```
./gradlew :llmedge:assembleRelease
```

3. Use in your application

```kotlin
    val smol = SmolLM()
    val rag = RAGEngine(context = this, smolLM = smol)

    CoroutineScope(Dispatchers.IO).launch {
        rag.init()
        val count = rag.indexPdf(pdfUri)
        val answer = rag.ask("What are the key points?")
        withContext(Dispatchers.Main) {
            // render answer
        }
    }
```

## Building

Building with Vulkan enabled
---------------------------

If you want to enable Vulkan acceleration for the native inference backend, follow these additional notes and requirements. The project supports building the Vulkan backend for Android but the runtime device must also support Vulkan 1.2.

Prerequisites
- Android NDK r27 or newer (NDK r27 used in development; NDK provides the Vulkan C headers). Ensure your NDK matches the version used by your build environment.
- CMake 3.22+ and Ninja (the Android Gradle plugin will pick up CMake when configured).
- Gradle (use the wrapper: `./gradlew`).
- Android API (minSdk) 30 or higher when enabling the Vulkan backend — ggml-vulkan requires Vulkan 1.2 which is guaranteed on Android 11+ devices.
- (Optional) VULKAN_SDK set in environment if you build shaders or use Vulkan SDK tools on the host. The build will fetch a matching `vulkan.hpp` header if needed.

Build flags
- Enable Vulkan at CMake configure time using Gradle external native build arguments. For example (bash/fish):

```bash
./gradlew :llmedge:assembleRelease -Pandroid.injected.build.api=30 -Pandroid.jniCmakeArgs="-DSD_VULKAN=ON -DGGML_VULKAN=ON"
```

Alternatively, set these flags in `llmedge/src/main/cpp/CMakeLists.txt` or in your Android Studio CMake configuration. The important flags are `-DSD_VULKAN=ON` and `-DGGML_VULKAN=ON` so ggml's Vulkan backend and the Stable Diffusion integration compile with Vulkan support.

Notes about headers and toolchain
- The build fetches `Vulkan-Hpp` (`vulkan.hpp`) and pins it to the NDK's Vulkan headers to avoid API mismatch. If you have a local `VULKAN_SDK` you can point to it, otherwise the project will use the fetched headers.
- The repository also builds a small host toolchain to generate SPIR-V shaders at build time; ensure your build host has a working C++ toolchain (clang/gcc) and CMake configured.

Runtime verification
- The AAR will include native libraries that link against `libvulkan.so` when Vulkan is enabled. To verify Vulkan is actually being used at runtime:
    - Run the app on a device with Android 11+ and a Vulkan 1.2-capable GPU.
    - Use the Kotlin API `SmolLM.isVulkanEnabled()` to check whether the library thinks Vulkan is available.
    - Inspect runtime logs: filter logcat for the `SmolSD` tag (the native logger) and look for backend initialization messages. Example:

```bash
adb logcat -s SmolSD:* | sed -n '1,200p'
```

    Look for messages indicating successful Vulkan initialization. Note: some builds logged "Using Vulkan backend" before initialization completed — make sure you see no subsequent "Failed to initialize Vulkan backend" or "Using CPU backend" messages.

Troubleshooting
- If you see "Vulkan 1.2 required" or linker errors for Vulkan symbols, confirm `minSdk` is set to 30 or higher in `llmedge/build.gradle.kts` and that your NDK provides the expected Vulkan headers.
- If your device lacks Vulkan 1.2 support, the native code will fall back to the CPU backend. Use a modern device (Android 11+) or an emulator/image with Vulkan 1.2 support.

#### Notes:

- Uses `com.tom-roush:pdfbox-android` for PDF parsing.
- Embeddings library: `io.gitlab.shubham0204:sentence-embeddings:v6`.
- Scanned PDFs require OCR (e.g., ML Kit or Tesseract) before indexing.
- ONNX `token_type_ids` errors are automatically handled; override via `EmbeddingConfig` if required.

## Architecture

1. llama.cpp (C/C++) provides the core inference engine, built via the Android NDK.
2. Stable Diffusion (C/C++) provides the image generation backend, built via the Android NDK.
3. `LLMInference.cpp` wraps the llama.cpp C API.
4. `smollm.cpp` exposes JNI bindings for Kotlin.
5. The `SmolLM` Kotlin class provides a high-level API for model loading and inference.

## Technologies

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Core LLM backend
- GGUF — Model format
- Android NDK / JNI — Native bindings
- ONNX Runtime — Sentence embeddings
- Android DownloadManager — Large file downloads

## Memory Metrics

You can measure RAM usage at runtime:

```kotlin
val snapshot = MemoryMetrics.snapshot(context)
Log.d("Memory", snapshot.toPretty(context))
```

Typical measurement points:

- Before model load
- After model load
- After blocking prompt
- After streaming prompt

#### Key fields:

- `totalPssKb`: Total proportional RAM usage. Best for overall tracking.
- `dalvikPssKb`: JVM-managed heap and runtime.
- `nativePssKb`: Native heap (llama.cpp, ONNX, tensors, KV cache).
- `otherPssKb`: Miscellaneous memory.

Monitor `nativePssKb` closely during model loading and inference to understand LLM memory footprint.

## Notes

- Vulkan SDK may be required; set the `VULKAN_SDK` environment variable when building with Vulkan.
- Vulkan acceleration can be checked via `SmolLM.isVulkanEnabled()`.

### ProGuard/R8 Configuration

The library includes consumer ProGuard rules. If you need to add custom rules:

```proguard
# Keep OCR engines
-keep class io.aatricks.llmedge.vision.** { *; }
-keep class org.bytedeco.** { *; }
-keep class com.google.mlkit.** { *; }

# Suppress warnings for optional dependencies
-dontwarn org.bytedeco.**
-dontwarn com.google.mlkit.**
```

### Licenses

- **llmedge**: Apache 2.0
- **llama.cpp**: MIT
- **stable-diffusion.cpp**: MIT
- **Leptonica**: Custom (BSD-like)
- **Google ML Kit**: Proprietary (see ML Kit terms)
- **JavaCPP**: Apache 2.0

## License and Credits

This project builds upon work by [Shubham Panchal](https://github.com/shubham0204) and [ggerganov](https://github.com/ggerganov).
See [CREDITS.md](CREDITS.md) for full details.
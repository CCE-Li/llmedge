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
- Video Generation: Generate short video clips (4-64 frames) from text using Wan models
- Image Generation: Stable Diffusion integration for txt2img
- Minimal on-device RAG (retrieval-augmented generation) pipeline
- OCR Support: Extract text from images using Google ML Kit
- Vision Model Ready: Architecture prepared for vision-capable LLMs (LLaVA)
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
8. [Testing](#testing)

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

The easiest way to use the library is via the `LLMEdgeManager` singleton, which handles model loading, caching, and threading automatically.

```kotlin
// Generate text using a default lightweight model (SmolLM-135M)
// The model is automatically downloaded if needed.
CoroutineScope(Dispatchers.IO).launch {
    val reply = LLMEdgeManager.generateText(
        context = context,
        params = LLMEdgeManager.TextGenerationParams(
            prompt = "Summarize on-device LLMs in one sentence."
        )
    )
    
    withContext(Dispatchers.Main) {
        outputView.text = reply
    }
}
```

For streaming responses or custom model paths, you can also access the underlying `SmolLM` instance directly or use `LLMEdgeManager`'s streaming callbacks (see Usage).

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
import io.aatricks.llmedge.LLMEdgeManager

// Process an image (Bitmap)
val text = LLMEdgeManager.extractText(context, bitmap)
println("Extracted text: $text")
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

Analyze images using Vision Language Models (like LLaVA or Phi-3 Vision) via `LLMEdgeManager`.

```kotlin
val description = LLMEdgeManager.analyzeImage(
    context = context,
    params = LLMEdgeManager.VisionAnalysisParams(
        image = bitmap,
        prompt = "Describe this image in detail."
    )
) { status ->
    Log.d("Vision", "Status: $status")
}
```

The manager handles the complex pipeline of:
1. Preprocessing the image
2. Loading the vision projector and model
3. Encoding the image to embeddings
4. Generating the textual response

Vision model support is currently experimental and requires specific model architectures (like LLaVA-Phi-3).


### Stable Diffusion (image generation)

Generate images on-device using `LLMEdgeManager`. This automatically handles model downloading (MeinaMix), caching, and memory safety.

```kotlin
val bitmap = LLMEdgeManager.generateImage(
    context = context,
    params = LLMEdgeManager.ImageGenerationParams(
        prompt = "a cute pastel anime cat, soft colors, high quality",
        width = 256,
        height = 256,
        steps = 20
    )
)
imageView.setImageBitmap(bitmap)
```

For advanced usage (custom models, explicit memory control), you can still use the `StableDiffusion` class directly as shown in the `llmedge-examples` repository.

### Video Generation

Generate short video clips using Wan models via `LLMEdgeManager`. This handles the complex requirement of loading three separate models (Diffusion, VAE, T5 Encoder) and offers sequential loading for devices with limited RAM.

**Hardware Requirements**:
- **12GB+ RAM** recommended for standard loading.
- **8GB+ RAM** supported via `forceSequentialLoad = true` (slower but memory-safe).

```kotlin
// 1. Define parameters
val params = LLMEdgeManager.VideoGenerationParams(
    prompt = "a cat walking in a garden, high quality",
    videoFrames = 8,  // Start small: 8-16 frames
    width = 480,      // Balanced resolution
    height = 480,
    steps = 15,
    forceSequentialLoad = true // Recommended for mobile
)

// 2. Generate video
val frames = LLMEdgeManager.generateVideo(
    context = context,
    params = params
) { status, current, total ->
    // Update progress: "Generating frame 1/8"
    Log.d("VideoGen", "$status")
}

// 3. Use the frames (List<Bitmap>)
previewImageView.setImageBitmap(frames.first())
```

`LLMEdgeManager` automatically:
1. Downloads the necessary Wan 2.1 model files (Diffusion, VAE, T5).
2. Sequentially loads components to minimize peak memory usage (if requested).
3. Manages the generation loop and frame conversion.

See `llmedge-examples` for a complete UI implementation.


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

## Testing

Looking to run unit and instrumentation tests locally, including optional native txt2img E2E checks? See the step-by-step guide in [docs/testing.md](docs/testing.md).

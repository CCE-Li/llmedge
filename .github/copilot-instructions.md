# Copilot Instructions for llmedge

## Repository Overview

llmedge is an Android library for running GGUF language models and Stable Diffusion models fully on-device. It provides Kotlin APIs wrapping native C++ implementations from llama.cpp and stable-diffusion.cpp submodules, enabling efficient inference on mobile devices with optional Vulkan acceleration.

**Key Facts:**
- **Primary language**: Kotlin (Android) with C++ JNI bindings
- **Architecture**: Modular Android library with native submodules
- **Core components**: SmolLM (LLM inference), StableDiffusion (image generation), RAG pipeline, vision/OCR support
- **Model formats**: GGUF for LLMs, safetensors/ckpt for diffusion models
- **Hardware acceleration**: Vulkan backend for GPU acceleration (Android 11+)
- **License**: Apache 2.0

## Build System & Dependencies

### Prerequisites
- JDK 17+ (Gradle Kotlin DSL)
- Android SDK & NDK r27+ (for native compilation)
- CMake 3.22+ and Ninja
- Android Studio (latest stable)

### Core Build Commands
**ALWAYS run these commands in sequence for development:**

```bash
# Initialize submodules (required first time)
git submodule update --init --recursive

# Build main library AAR
./gradlew :llmedge:assembleRelease

# Copy AAR to examples for testing
cp llmedge/build/outputs/aar/llmedge-release.aar llmedge-examples/app/libs/llmedge-release.aar

# Build and install example app
cd llmedge-examples
./gradlew :app:assembleDebug
./gradlew :app:installDebug
```

**Build time**: ~5-10 minutes depending on hardware
**Important Notes**:
- Use Gradle wrapper (`./gradlew`) - never install Gradle globally
- Clean builds: `./gradlew clean` before major changes
- Native rebuilds: Delete `llmedge/.cxx/` directory to force CMake regeneration

### Vulkan Acceleration Build
Enable Vulkan for GPU acceleration (requires Android 11+ device):

```bash
./gradlew :llmedge:assembleRelease -Pandroid.jniCmakeArgs="-DGGML_VULKAN=ON -DSD_VULKAN=ON"
```

**Runtime verification**: Call `SmolLM.isVulkanEnabled()` to check if Vulkan initialized successfully.

### Common Build Issues
- **Native compilation fails**: Ensure NDK r27+ and CMake 3.22+
- **Vulkan linker errors**: Confirm `minSdk = 30` in `llmedge/build.gradle.kts`
- **Submodule issues**: Run `git submodule update --init --recursive` after cloning

## Project Architecture

### Core Components
- **`SmolLM`**: Main LLM inference API - loads GGUF models, manages context, handles generation
- **`StableDiffusion`**: Image generation API - txt2img with model/VAE loading from Hugging Face
- **`RAGEngine`**: On-device retrieval-augmented generation with embeddings and vector search
- **`ImageUnderstanding`**: Vision pipeline combining OCR (ML Kit) and vision models
- **`HuggingFaceHub`**: Model/asset download and caching system

### Key Architectural Patterns
- **CPU Feature Detection**: Automatically selects optimal native library variant based on device capabilities (ARMv8.2+, FP16, dotprod, SVE, I8MM)
- **Memory-Aware Context Sizing**: Automatically caps LLM context windows based on device heap (2K-8K tokens)
- **Background I/O**: All model loading and generation uses `Dispatchers.IO` coroutines
- **JNI Resource Management**: Native pointers managed via `AutoCloseable` pattern
- **Hugging Face Integration**: Downloads cached in `filesDir/hf-models/` with automatic fallback to system DownloadManager for large files

### Data Flow Patterns
- **LLM Inference**: Kotlin → JNI → llama.cpp → GGML tensor operations → Vulkan/OpenCL/CPU backends
- **Model Loading**: HuggingFaceHub downloads → GGUFReader metadata parsing → SmolLM native loading
- **RAG Pipeline**: PDF/text → TextSplitter → EmbeddingProvider (ONNX) → VectorStore → SmolLM generation
- **Vision Pipeline**: Image → ML Kit OCR / Vision model → SmolLM multimodal generation

## Developer Workflows

### Adding New Features
1. **Kotlin API first**: Design public API in Kotlin with proper KDoc
2. **JNI bridge**: Add native method declarations and implement C++ side
3. **Update examples**: Add demo in `llmedge-examples` app
4. **Documentation**: Update relevant `docs/` files and README

### Testing Protocol
**ALWAYS test on real devices - emulators miss native issues:**
- Test different Android versions (API 30+ for Vulkan)
- Test memory-constrained devices (<2GB RAM)
- Use `MemoryMetrics.snapshot()` to monitor native heap usage
- Verify `SmolLM.getLastGenerationMetrics()` for performance regression

### Memory Management
- Monitor `nativePssKb` in `MemoryMetrics` during model loading/inference
- Use `InferenceParams(contextSize = ...)` to override automatic context capping
- Enable CPU offloading in StableDiffusion for memory-constrained devices
- Call `close()` on SmolLM/StableDiffusion instances to free native memory

### Performance Optimization
- Profile with `SmolLM.getLastGenerationMetrics()` (tokens/sec, latency)
- Adjust `InferenceParams.numThreads` based on device cores
- Use streaming (`getResponseAsFlow`) for large outputs to avoid UI blocking
- Enable Vulkan on compatible devices for 2-5x speedup

## Code Patterns & Conventions

### Kotlin API Design
**SmolLM loading pattern:**
```kotlin
val smol = SmolLM()
CoroutineScope(Dispatchers.IO).launch {
    smol.loadFromHuggingFace(
        context = context,
        modelId = "unsloth/Qwen3-0.6B-GGUF",
        filename = "Qwen3-0.6B-Q4_K_M.gguf"
    )
    val response = smol.getResponse("Hello world")
    withContext(Dispatchers.Main) { /* update UI */ }
}
smol.close() // Always cleanup
```

**StableDiffusion generation:**
```kotlin
val sd = StableDiffusion.load(context, modelId = "Meina/MeinaMix")
val bitmap = sd.txt2img(StableDiffusion.GenerateParams(
    prompt = "cute cat",
    width = 256, height = 256,
    steps = 20
))
sd.close()
```

### Reasoning Controls
**Thinking mode management:**
```kotlin
val params = SmolLM.InferenceParams(thinkingMode = SmolLM.ThinkingMode.DISABLED)
// or runtime control:
smol.setThinkingEnabled(false)
smol.setReasoningBudget(0) // disable thinking
```

### Error Handling
- Check `FileNotFoundException` for invalid model paths
- Handle `IllegalStateException` for uninitialized models
- Use try/catch around native operations - JNI can throw `UnsatisfiedLinkError`
- Log native errors with `adb logcat -s SmolLM:* SmolSD:*`

### Resource Management
- **Always call `close()`** on SmolLM/StableDiffusion instances
- Use `use` block for automatic cleanup: `SmolLM().use { /* operations */ }`
- Background coroutines prevent ANR during model loading
- Large downloads use `DownloadManager` to avoid Dalvik heap pressure

## Integration Points

### External Dependencies
- **Hugging Face Hub**: Model downloads with token auth and progress callbacks
- **Google ML Kit**: OCR via `MlKitOcrEngine` (no additional setup needed)
- **ONNX Runtime**: Embeddings via `EmbeddingProvider` (model files in assets)
- **PDFBox**: PDF parsing for RAG (`PDFReader`)

### Model Management
- **GGUF models**: Loaded via `SmolLM.load()` or `loadFromHuggingFace()`
- **Diffusion models**: Auto-detected GGUF/safetensors via `StableDiffusion.load()`
- **Embeddings**: ONNX models placed in `llmedge/src/main/assets/embeddings/`
- **Caching**: All downloads cached in app-private storage

### Cross-Component Communication
- **RAG**: `RAGEngine` coordinates embeddings → vector search → SmolLM generation
- **Vision**: `ImageUnderstanding` routes to OCR or vision models based on `VisionMode`
- **HuggingFace**: Shared `ensureModelOnDisk()` for consistent download/caching

## Important Guidelines

### Code Changes
- **Minimal dependencies**: Avoid adding new external libraries
- **Android compatibility**: Test on API 30+ for Vulkan features
- **Memory awareness**: Profile heap usage for all native operations
- **JNI safety**: Never pass null pointers from Kotlin to native code
- **API stability**: Document breaking changes in migration guides

### Git Workflow
- **Submodule updates**: Test thoroughly after `llama.cpp`/`stable-diffusion.cpp` changes
- **Native changes**: Rebuild examples app after modifying C++ code
- **Branch naming**: `feature/`, `fix/`, `refactor/` prefixes
- **Commit messages**: Include component context (e.g., "SmolLM: add reasoning controls")

### Performance Considerations
- Vulkan provides significant speedup but requires modern devices
- Context size directly impacts memory usage (monitor with `getContextLengthUsed()`)
- Thread count tuning affects latency vs throughput tradeoff
- Streaming responses reduce memory pressure for long outputs

### Testing Requirements
- **Real device testing**: Emulators cannot validate native performance
- **Memory profiling**: Use `MemoryMetrics` for regression detection
- **Model compatibility**: Test with multiple GGUF quantization levels
- **Error scenarios**: Invalid models, network failures, memory pressure

## Key Files Reference

**Core APIs:**
- `llmedge/src/main/java/io/aatricks/llmedge/SmolLM.kt` - Main LLM interface
- `llmedge/src/main/java/io/aatricks/llmedge/StableDiffusion.kt` - Image generation
- `llmedge/src/main/java/io/aatricks/llmedge/huggingface/HuggingFaceHub.kt` - Model downloads

**Architecture Components:**
- `llmedge/src/main/java/io/aatricks/llmedge/rag/RAGEngine.kt` - RAG pipeline
- `llmedge/src/main/java/io/aatricks/llmedge/vision/ImageUnderstanding.kt` - Vision processing
- `llmedge/src/main/cpp/LLMInference.cpp` - Native LLM wrapper
- `llmedge/src/main/cpp/sdcpp_jni.cpp` - StableDiffusion JNI

**Build Configuration:**
- `llmedge/build.gradle.kts` - Android library config
- `llmedge/src/main/cpp/CMakeLists.txt` - Native build setup
- `settings.gradle.kts` - Project structure

**Examples & Testing:**
- `llmedge-examples/app/src/main/java/io/aatricks/llmedge/` - Demo implementations
- `docs/architecture.md` - System design details
- `docs/contributing.md` - Development guidelines

Trust these instructions - they contain validated workflows that work reliably across different development environments. Search for additional context only if these instructions prove incomplete or incorrect.</content>
<parameter name="filePath">/home/aatricks/Dev/llmedge/.github/copilot-instructions.md

## Active Technologies
- Kotlin 1.9+ / Java 17, C++17 (JNI layer) (001-video-model-support)
- File-based (app-private storage for cached models, generated videos as bitmap sequences) (001-video-model-support)

## Recent Changes
- 001-video-model-support: Added Kotlin 1.9+ / Java 17, C++17 (JNI layer)

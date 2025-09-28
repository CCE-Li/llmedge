# llmedge

**llmedge** is a lightweight Android library for running GGUF language models fully on-device, powered by [llama.cpp](https://github.com/ggerganov/llama.cpp).

See the [examples repository](https://github.com/Aatricks/llmedge-examples) for sample usage.

Acknowledgments to Shubham Panchal and upstream projects are listed in [`CREDITS.md`](./CREDITS.md).

---

## Features

- Run GGUF models directly on Android using llama.cpp (JNI)
- Download and cache models from Hugging Face
- Minimal on-device RAG (retrieval-augmented generation) pipeline
- Built-in memory usage metrics
- Optional Vulkan acceleration

---

## Table of Contents

1. [Installation](#installation)  
2. [Usage](#usage)  
   - [Downloading Models](#downloading-models)  
   - [Reasoning Controls](#reasoning-controls)  
   - [On-device RAG](#on-device-rag)  
3. [Building](#building)  
4. [Architecture](#architecture)  
5. [Technologies](#technologies)  
6. [Memory Metrics](#memory-metrics)  
7. [Notes](#notes)

---

## Installation

Clone the repository along with the `llama.cpp` submodule:

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
    modelId = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf", // optional
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

#### Notes:

- Uses `com.tom-roush:pdfbox-android` for PDF parsing.
- Embeddings library: `io.gitlab.shubham0204:sentence-embeddings:v6`.
- Scanned PDFs require OCR (e.g., ML Kit or Tesseract) before indexing.
- ONNX `token_type_ids` errors are automatically handled; override via `EmbeddingConfig` if required.

## Architecture

1. llama.cpp (C/C++) provides the core inference engine, built via the Android NDK.
2. `LLMInference.cpp` wraps the llama.cpp C API.
3. `smollm.cpp` exposes JNI bindings for Kotlin.
4. The `SmolLM` Kotlin class provides a high-level API for model loading and inference.

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

## License and Credits

This project builds upon work by [Shubham Panchal](https://github.com/shubham0204) and [ggerganov](https://github.com/ggerganov).
See [CREDITS.md](CREDITS.md) for full details.
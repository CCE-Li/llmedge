package io.aatricks.llmedge

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import java.io.File
import java.lang.ref.WeakReference
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * High-level manager for LLMEdge operations. Simplifies model loading, generation, and memory
 * management.
 */
object LLMEdgeManager {
        private const val TAG = "LLMEdgeManager"
        private const val MIN_AVAILABLE_MEMORY_MB = 2000L

        // Default Video Model (Wan 2.1)
        private const val DEFAULT_VIDEO_MODEL_ID = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
        private const val DEFAULT_VIDEO_MODEL_FILENAME = "wan2.1_t2v_1.3B_fp16.safetensors"
        private const val DEFAULT_VIDEO_VAE_ID = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
        private const val DEFAULT_VIDEO_VAE_FILENAME = "wan_2.1_vae.safetensors"
        private const val DEFAULT_VIDEO_T5XXL_ID = "city96/umt5-xxl-encoder-gguf"
        private const val DEFAULT_VIDEO_T5XXL_FILENAME = "umt5-xxl-encoder-Q3_K_S.gguf"

        // Default Image Model (MeinaMix)
        private const val DEFAULT_IMAGE_MODEL_ID = "Meina/MeinaMix"
        private const val DEFAULT_IMAGE_MODEL_FILENAME = "MeinaPastel - baked VAE.safetensors"

        // Default Text Model (SmolLM 135M)
        private const val DEFAULT_TEXT_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct-GGUF"
        private const val DEFAULT_TEXT_MODEL_FILENAME = "smollm-135m-instruct.q4_k_m.gguf"

        // Default Vision Model (LLaVA Phi-3 Mini)
        private const val DEFAULT_VISION_MODEL_ID = "xtuner/llava-phi-3-mini-gguf"
        private const val DEFAULT_VISION_MODEL_FILENAME = "llava-phi-3-mini-int4.gguf"
        private const val DEFAULT_VISION_PROJ_FILENAME = "llava-phi-3-mini-mmproj-f16.gguf"

        private val textModelMutex = Mutex() // For SmolLM text generation
        private val diffusionModelMutex = Mutex() // For Stable Diffusion image/video generation

        // Phase 3: Model caching with LRU eviction
        private val textModelCache =
                ModelCache<io.aatricks.llmedge.SmolLM>(
                        maxCacheSize = 2,
                        maxMemoryMB = 2048L // 2GB max for text models
                )
        private val diffusionModelCache =
                ModelCache<StableDiffusion>(
                        maxCacheSize = 1,
                        maxMemoryMB = 4096L // 4GB max for diffusion models
                )

        @Volatile private var cachedModel: StableDiffusion? = null
        @Volatile private var cachedSmolLM: io.aatricks.llmedge.SmolLM? = null
        @Volatile private var isLoading = false
        private var contextRef: WeakReference<Context>? = null
        /** When true, prefer higher throughput and lower memory-safety heuristics. */
        @Volatile
        var preferPerformanceMode: Boolean = false

        // `preferPerformanceMode` property provides a runtime setter (no need for explicit wrapper)

        // Phase 3: Core topology for optimal threading
        private val coreInfo by lazy { CpuTopology.detectCoreTopology() }

        // Track currently loaded text model to allow switching
        private data class LoadedTextModelSpec(
                val modelId: String,
                val filename: String,
                val path: String?
        )
        private var currentTextModelSpec: LoadedTextModelSpec? = null

        // Track loaded diffusion model (image/video) to allow switching and cache removal
        private data class LoadedDiffusionModelSpec(
                val modelId: String,
                val filename: String,
                val path: String?,
                val vaePath: String?,
                val t5xxlPath: String?
        )
        private var currentDiffusionModelSpec: LoadedDiffusionModelSpec? = null

        data class ImageGenerationParams(
                val prompt: String,
                val negative: String = "",
                val width: Int = 512,
                val height: Int = 512,
                val steps: Int = 20,
                val cfgScale: Float = 7.0f,
                val seed: Long = -1L,
                val flashAttn: Boolean = true,
                val forceSequentialLoad: Boolean = false
        )

        data class VideoGenerationParams(
                val prompt: String,
                val negative: String = "",
                val width: Int = 512,
                val height: Int = 512,
                val videoFrames: Int = 16,
                val steps: Int = 20,
                val cfgScale: Float = 7.0f,
                val seed: Long = -1L,
                val flashAttn: Boolean = true,
                val forceSequentialLoad: Boolean = false
        )

        data class TextGenerationParams(
                val prompt: String,
                val systemPrompt: String = "You are a helpful assistant.",
                val modelId: String = DEFAULT_TEXT_MODEL_ID,
                val modelFilename: String = DEFAULT_TEXT_MODEL_FILENAME,
                val modelPath: String? = null, // Absolute path override
                val temperature: Float = 0.7f,
                val maxTokens: Int = -1, // -1 for infinite/until EOS
                val revision: String = "main",
                val thinkingMode: io.aatricks.llmedge.SmolLM.ThinkingMode =
                        io.aatricks.llmedge.SmolLM.ThinkingMode.DEFAULT,
                val reasoningBudget: Int? = null
        )

        data class VisionAnalysisParams(
                val image: Bitmap,
                val prompt: String,
                val modelId: String = DEFAULT_VISION_MODEL_ID,
                val modelFilename: String = DEFAULT_VISION_MODEL_FILENAME,
                val projFilename: String = DEFAULT_VISION_PROJ_FILENAME
        )

        /** Information about Vulkan GPU device capabilities */
        data class VulkanDeviceInfo(
                val deviceCount: Int,
                val totalMemoryMB: Long,
                val freeMemoryMB: Long,
                val deviceIndex: Int = 0
        )

        data class PerformanceSnapshot(
                val textMetrics: io.aatricks.llmedge.SmolLM.GenerationMetrics?,
                val diffusionMetrics: io.aatricks.llmedge.StableDiffusion.GenerationMetrics?,
                val timestamp: Long = System.currentTimeMillis()
        )

        /** Check if Vulkan GPU acceleration is available on this device */
        fun isVulkanAvailable(): Boolean {
                return try {
                        io.aatricks.llmedge.StableDiffusion.getVulkanDeviceCount() > 0
                } catch (e: Exception) {
                        false
                }
        }

        /**
         * Get Vulkan device information (memory, device count) Returns null if Vulkan is not
         * available
         */
        fun getVulkanDeviceInfo(): VulkanDeviceInfo? {
                return try {
                        val deviceCount = io.aatricks.llmedge.StableDiffusion.getVulkanDeviceCount()
                        if (deviceCount <= 0) return null

                        val memory = io.aatricks.llmedge.StableDiffusion.getVulkanDeviceMemory(0)
                        if (memory == null || memory.size < 2) return null

                        VulkanDeviceInfo(
                                deviceCount = deviceCount,
                                freeMemoryMB = memory[0] / (1024 * 1024),
                                totalMemoryMB = memory[1] / (1024 * 1024),
                                deviceIndex = 0
                        )
                } catch (e: Exception) {
                        null
                }
        }

        /** Get combined performance metrics from both text and diffusion models */
        fun getPerformanceSnapshot(): PerformanceSnapshot {
                return PerformanceSnapshot(
                        textMetrics = cachedSmolLM?.getLastGenerationMetrics(),
                        diffusionMetrics = cachedModel?.getLastGenerationMetrics()
                )
        }

        /** Log performance snapshot to Android logcat for debugging */
        fun logPerformanceSnapshot() {
                val snapshot = getPerformanceSnapshot()
                val vulkanInfo = getVulkanDeviceInfo()

                Log.i(TAG, "=== Performance Snapshot ===")

                snapshot.textMetrics?.let { metrics ->
                        Log.i(
                                TAG,
                                "Text Generation: ${metrics.tokensPerSecond} tok/s, " +
                                        "${metrics.tokenCount} tokens, ${metrics.elapsedMillis}ms"
                        )
                }

                snapshot.diffusionMetrics?.let { metrics ->
                        Log.i(
                                TAG,
                                "Diffusion: ${metrics.stepsPerSecond} steps/s, " +
                                        "${metrics.totalTimeSeconds}s total"
                        )
                }

                vulkanInfo?.let { info ->
                        Log.i(
                                TAG,
                                "Vulkan: ${info.deviceCount} device(s), " +
                                        "${info.freeMemoryMB}MB free / ${info.totalMemoryMB}MB total"
                        )
                }

                Log.i(TAG, "===========================")
        }

        /** Generates text using a local LLM. */
        suspend fun generateText(
                context: Context,
                params: TextGenerationParams,
                onProgress: ((String) -> Unit)? = null
        ): String =
                textModelMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)

                        // Unload heavy diffusion models if loaded to free up memory
                        unloadDiffusionModel()

                        val smol =
                                getOrLoadSmolLM(
                                        context,
                                        params.modelId,
                                        params.modelFilename,
                                        params.modelPath
                                )

                        // Reset and set system prompt if needed
                        // smol.addSystemPrompt(params.systemPrompt)

                        // Apply runtime params that might change per generation
                        smol.setThinkingMode(params.thinkingMode)
                        params.reasoningBudget?.let { smol.setReasoningBudget(it) }

                        if (onProgress != null) {
                                val sb = StringBuilder()
                                // We need to use flow for streaming
                                // Assuming SmolLM has getResponseAsFlow
                                smol.getResponseAsFlow(params.prompt).collect { token ->
                                        if (token != "[EOG]") {
                                                sb.append(token)
                                                onProgress(token)
                                        }
                                }
                                return@withLock sb.toString()
                        } else {
                                // Use Dispatchers.IO for blocking native JNI operations
                                // Dispatchers.Default has limited parallelism and is meant for CPU-bound work
                                return@withLock kotlinx.coroutines.withContext(
                                        kotlinx.coroutines.Dispatchers.IO
                                ) { smol.getResponse(params.prompt) }
                        }
                }

        fun getLastTextGenerationMetrics(): io.aatricks.llmedge.SmolLM.GenerationMetrics? {
                return cachedSmolLM?.getLastGenerationMetrics()
        }

        fun getLastDiffusionMetrics(): io.aatricks.llmedge.StableDiffusion.GenerationMetrics? {
                return cachedModel?.getLastGenerationMetrics()
        }

        /**
         * Returns the shared SmolLM instance, loading it if necessary. Useful for advanced use
         * cases like RAG.
         */
        suspend fun getSmolLM(context: Context): io.aatricks.llmedge.SmolLM {
                return textModelMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)
                        unloadDiffusionModel()
                        getOrLoadSmolLM(context, DEFAULT_TEXT_MODEL_ID, DEFAULT_TEXT_MODEL_FILENAME)
                }
        }

        /** Returns the shared SmolLM instance WITHOUT loading a model. Useful for activities that want to manage the loading process themselves (e.g. HuggingFaceDemoActivity). */
        suspend fun getSmolLMInstance(context: Context): io.aatricks.llmedge.SmolLM {
                return textModelMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)
                        unloadDiffusionModel()
                        val smol = cachedSmolLM ?: io.aatricks.llmedge.SmolLM().also { cachedSmolLM = it }
                        return@withLock smol
                }
        }

        /** Extracts text from an image using OCR. */
        suspend fun extractText(context: Context, image: Bitmap): String =
                textModelMutex.withLock {
                        // Create a fresh engine instance and close it immediately after use to free resources
                        val engine = io.aatricks.llmedge.vision.ocr.MlKitOcrEngine(context)
                        try {
                                val result =
                                        engine.extractText(
                                                io.aatricks.llmedge.vision.ImageSource.BitmapSource(
                                                        image
                                                )
                                        )
                                return@withLock result.text
                        } finally {
                                engine.close()
                        }
                }

                /** Analyzes an image using a Vision Language Model (VLM). */
                suspend fun analyzeImage(
                        context: Context,
                        params: VisionAnalysisParams,
                        onProgress: ((String) -> Unit)? = null
                ): String =
                        textModelMutex.withLock {
                                contextRef = WeakReference(context.applicationContext)
        
                                // Unload heavy diffusion models
                                unloadDiffusionModel()
                                // Also unload cached SmolLM to free up memory for this heavy operation
                                unloadSmolLM()
        
                                // Ensure files
                                val modelFile = getFile(context, params.modelId, params.modelFilename)
                                val projFile = getFile(context, params.modelId, params.projFilename)
        
                                // Instantiate a fresh SmolLM for this specific vision task
                                // mirroring the configuration in the working backup:
                                // - storeChats = false (saves memory/context)
                                // - temperature = 0.0f (deterministic)
                                // - thinkingMode = DISABLED
                                val optimalThreads = if (preferPerformanceMode) {
                                        CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING)
                                } else {
                                        2
                                }
                                
                                // Use explicit Vulkan setting based on preference
                                val smol = io.aatricks.llmedge.SmolLM(useVulkan = preferPerformanceMode)
                                
                                try {
                                        prepareMemoryForLoading()
                                        
                                        smol.load(
                                                modelPath = modelFile.absolutePath,
                                                params = io.aatricks.llmedge.SmolLM.InferenceParams(
                                                        numThreads = optimalThreads,
                                                        contextSize = null, 
                                                        storeChats = false,
                                                        temperature = 0.0f,
                                                        thinkingMode = io.aatricks.llmedge.SmolLM.ThinkingMode.DISABLED
                                                )
                                        )
        
                                        // Prepare Vision Adapter
                                        val adapter = io.aatricks.llmedge.vision.SmolLMVisionAdapter(context, smol)
        
                                        try {
                                                // 1. Save bitmap to temp file
                                                onProgress?.invoke("Preparing image")
                                                val tempImageFile =
                                                        File.createTempFile(
                                                                "vision_input",
                                                                ".jpg",
                                                                context.cacheDir
                                                        )
                                                val preparedImageFile =
                                                        File.createTempFile(
                                                                "vision_prepared",
                                                                ".bin",
                                                                context.cacheDir
                                                        )
        
                                                try {
                                                        // Preprocess and save
                                                        val scaled =
                                                                io.aatricks.llmedge.vision.ImageUtils
                                                                        .preprocessImage(
                                                                                params.image,
                                                                                correctOrientation = true,
                                                                                maxDimension = 1024,
                                                                                enhance = false
                                                                        )
                                                        tempImageFile.outputStream().use { out ->
                                                                scaled.compress(Bitmap.CompressFormat.JPEG, 90, out)
                                                        }
        
                                                        // 2. Run Projector
                                                        onProgress?.invoke("Encoding image")
                                                        val projector = io.aatricks.llmedge.vision.Projector()
                                                        projector.init(
                                                                projFile.absolutePath,
                                                                smol.getNativeModelPointer()
                                                        )
                                                        val ok =
                                                                projector.encodeImageToFile(
                                                                        tempImageFile.absolutePath,
                                                                        preparedImageFile.absolutePath
                                                                )
                                                        projector.close()
        
                                                        val imageSource =
                                                                if (ok) {
                                                                        io.aatricks.llmedge.vision.ImageSource
                                                                                .FileSource(preparedImageFile)
                                                                } else {
                                                                        io.aatricks.llmedge.vision.ImageSource
                                                                                .FileSource(tempImageFile)
                                                                }
        
                                                        // 3. Run Analysis
                                                        onProgress?.invoke("Running vision analysis")
                                                        adapter.loadVisionModel(modelFile.absolutePath)
                                                        val result =
                                                                adapter.analyze(
                                                                        imageSource,
                                                                        params.prompt,
                                                                        io.aatricks.llmedge.vision.VisionParams()
                                                                )
                                                        return@withLock result.text
                                                } finally {
                                                        tempImageFile.delete()
                                                        preparedImageFile.delete()
                                                        adapter.close() 
                                                }
                                        } catch (e: Exception) {
                                                Log.e(TAG, "Vision analysis failed", e)
                                                throw e
                                        }
                                } finally {
                                        smol.close()
                                }
                        }
        /**
         * Generates an image using the default or configured model. Automatically handles
         * sequential loading for low-memory devices.
         */
        suspend fun generateImage(
                context: Context,
                params: ImageGenerationParams,
                onProgress: ((String, Int, Int) -> Unit)? = null
        ): Bitmap? =
                diffusionModelMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)
                        unloadSmolLM() // Free up memory from LLM

                        val isLowMem = isLowMemoryDevice(context)
                        val useSequential = params.forceSequentialLoad || isLowMem

                        if (useSequential) {
                                // Sequential load not supported for single-file SD models like
                                // MeinaMix yet
                                // Fallback to standard load or implement if needed
                                // For now, just use standard load as MeinaMix is smaller than Wan
                                // 2.1
                                val model =
                                        getOrLoadImageModel(
                                                context,
                                                params.flashAttn,
                                                params.width,
                                                params.height,
                                                onProgress
                                        )
                                // Use txt2img(GenerateParams) which returns Bitmap directly
                                // The raw ByteArray version returns RGB bytes, not encoded image
                                val sdParams = StableDiffusion.GenerateParams(
                                        prompt = params.prompt,
                                        negative = params.negative,
                                        width = params.width,
                                        height = params.height,
                                        steps = params.steps,
                                        cfgScale = params.cfgScale,
                                        seed = params.seed
                                )
                                return model.txt2img(sdParams)
                        } else {
                                val model =
                                        getOrLoadImageModel(
                                                context,
                                                params.flashAttn,
                                                params.width,
                                                params.height,
                                                onProgress
                                        )
                                // Use txt2img(GenerateParams) which returns Bitmap directly
                                // The raw ByteArray version returns RGB bytes, not encoded image
                                val sdParams = StableDiffusion.GenerateParams(
                                        prompt = params.prompt,
                                        negative = params.negative,
                                        width = params.width,
                                        height = params.height,
                                        steps = params.steps,
                                        cfgScale = params.cfgScale,
                                        seed = params.seed
                                )
                                return model.txt2img(sdParams)
                        }
                }

        /** Generates a video using the default or configured model. */
        suspend fun generateVideo(
                context: Context,
                params: VideoGenerationParams,
                onProgress: ((String, Int, Int) -> Unit)? = null
        ): List<Bitmap> =
                diffusionModelMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)
                        unloadSmolLM() // Free up memory from LLM

                        val isLowMem = isLowMemoryDevice(context)
                        val useSequential = params.forceSequentialLoad || isLowMem

                        if (useSequential) {
                                return generateVideoSequentially(context, params, onProgress)
                        } else {
                                val model =
                                        getOrLoadVideoModel(
                                                context,
                                                params.flashAttn,
                                                onProgress,
                                                sequentialLoad = useSequential
                                        )
                                val sdParams =
                                        StableDiffusion.VideoGenerateParams(
                                                prompt = params.prompt,
                                                negative = params.negative,
                                                width = params.width,
                                                height = params.height,
                                                videoFrames = params.videoFrames,
                                                steps = params.steps,
                                                cfgScale = params.cfgScale,
                                                seed = params.seed
                                        )

                                // We need a progress callback wrapper
                                val progressWrapper =
                                        StableDiffusion.VideoProgressCallback {
                                                step,
                                                totalSteps,
                                                currentFrame,
                                                totalFrames,
                                                _ ->
                                                onProgress?.invoke(
                                                        "Generating frame $currentFrame/$totalFrames",
                                                        step,
                                                        totalSteps
                                                )
                                        }
                                model.setProgressCallback(progressWrapper)

                                val framesBytes = model.txt2vid(sdParams)
                                return framesBytes
                        }
                }

        private suspend fun generateImageSequentially(
                context: Context,
                params: ImageGenerationParams,
                onProgress: ((String, Int, Int) -> Unit)?
        ): Bitmap? {
                // ... Implementation similar to video sequential load ...
                // 1. Load T5 -> Encode
                // 2. Unload T5
                // 3. Load Diffusion -> Generate (using new txt2ImgWithPrecomputedCondition)

                // Ensure files
                // Update memory provider for diffusion cache
                if (!preferPerformanceMode) {
                        diffusionModelCache.systemMemoryProvider = {
                                val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                                val mi = ActivityManager.MemoryInfo()
                                am.getMemoryInfo(mi)
                                mi.availMem / (1024L * 1024L)
                        }
                } else {
                        // In performance mode, avoid using system memory to aggressively evict models so
                        // we can keep heavy models resident for throughput
                        diffusionModelCache.systemMemoryProvider = null
                }
                ensureVideoFiles(context, onProgress)

                // Load T5
                prepareMemoryForLoading()
                var t5Model: StableDiffusion? = null
                var cond: StableDiffusion.PrecomputedCondition?
                var uncond: StableDiffusion.PrecomputedCondition?

                try {
                        val t5File =
                                getFile(
                                        context,
                                        DEFAULT_VIDEO_T5XXL_ID,
                                        DEFAULT_VIDEO_T5XXL_FILENAME
                                )
                        t5Model =
                                StableDiffusion.load(
                                        context = context,
                                        modelPath = t5File.absolutePath,
                                        vaePath = null,
                                        t5xxlPath = null,
                                        nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING),
                                        offloadToCpu = true,
                                        keepClipOnCpu = true,
                                        keepVaeOnCpu = true,
                                        flashAttn = params.flashAttn
                                )

                        cond =
                                t5Model.precomputeCondition(
                                        params.prompt,
                                        params.negative,
                                        params.width,
                                        params.height
                                )
                        uncond =
                                if (params.negative.isNotEmpty()) {
                                        t5Model.precomputeCondition(
                                                params.negative,
                                                "",
                                                params.width,
                                                params.height
                                        )
                                } else {
                                        t5Model.precomputeCondition(
                                                "",
                                                "",
                                                params.width,
                                                params.height
                                        )
                                }
                        } finally {
                        t5Model?.close()
                }

                // Load Diffusion
                prepareMemoryForLoading()
                var diffusionModel: StableDiffusion? = null
                try {
                                // Model and VAE files are handled in getOrLoadImageModel

                        diffusionModel =
                                getOrLoadImageModel(
                                        context = context,
                                        flashAttn = params.flashAttn,
                                        width = params.width,
                                        height = params.height,
                                        onProgress = onProgress,
                                        sequentialLoad = true
                                )

                        val bytes =
                                diffusionModel.txt2ImgWithPrecomputedCondition(
                                        prompt = params.prompt,
                                        negative = params.negative,
                                        width = params.width,
                                        height = params.height,
                                        steps = params.steps,
                                        cfg = params.cfgScale,
                                        seed = params.seed,
                                        cond = cond,
                                        uncond = uncond
                                )
                        // Convert raw RGB bytes to Bitmap - BitmapFactory.decodeByteArray expects
                        // encoded PNG/JPEG, not raw RGB pixels
                        return bytes?.let { rgbBytesToBitmap(it, params.width, params.height) }
                } finally {
                        diffusionModel?.close()
                }
        }

        private suspend fun generateVideoSequentially(
                context: Context,
                params: VideoGenerationParams,
                onProgress: ((String, Int, Int) -> Unit)?
        ): List<Bitmap> {
                ensureVideoFiles(context, onProgress)

                prepareMemoryForLoading()
                var t5Model: StableDiffusion? = null
                var cond: StableDiffusion.PrecomputedCondition?
                var uncond: StableDiffusion.PrecomputedCondition?

                try {
                        val t5File =
                                getFile(
                                        context,
                                        DEFAULT_VIDEO_T5XXL_ID,
                                        DEFAULT_VIDEO_T5XXL_FILENAME
                                )
                        t5Model =
                                StableDiffusion.load(
                                        context = context,
                                        modelPath = t5File.absolutePath,
                                        vaePath = null,
                                        t5xxlPath = null,
                                        nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING),
                                        offloadToCpu = true,
                                        keepClipOnCpu = true,
                                        keepVaeOnCpu = true,
                                        flashAttn = params.flashAttn
                                )

                        cond =
                                t5Model.precomputeCondition(
                                        params.prompt,
                                        params.negative,
                                        params.width,
                                        params.height
                                )
                        uncond =
                                if (params.negative.isNotEmpty()) {
                                        t5Model.precomputeCondition(
                                                params.negative,
                                                "",
                                                params.width,
                                                params.height
                                        )
                                } else {
                                        t5Model.precomputeCondition(
                                                "",
                                                "",
                                                params.width,
                                                params.height
                                        )
                                }
                } finally {
                        t5Model?.close()
                }

                prepareMemoryForLoading()
                var diffusionModel: StableDiffusion? = null
                try {
                        // Model and VAE files are handled in getOrLoadVideoModel

                        diffusionModel =
                                getOrLoadVideoModel(
                                        context = context,
                                        flashAttn = params.flashAttn,
                                        onProgress = onProgress,
                                        sequentialLoad = true
                                )

                        val sdParams =
                                StableDiffusion.VideoGenerateParams(
                                        prompt = params.prompt,
                                        negative = params.negative,
                                        width = params.width,
                                        height = params.height,
                                        videoFrames = params.videoFrames,
                                        steps = params.steps,
                                        cfgScale = params.cfgScale,
                                        seed = params.seed
                                )

                        val progressWrapper =
                                StableDiffusion.VideoProgressCallback {
                                        step,
                                        totalSteps,
                                        currentFrame,
                                        totalFrames,
                                        _ ->
                                        onProgress?.invoke(
                                                "Generating frame $currentFrame/$totalFrames",
                                                step,
                                                totalSteps
                                        )
                                }

                        return diffusionModel.txt2VidWithPrecomputedCondition(
                                params = sdParams,
                                cond = cond,
                                uncond = uncond,
                                onProgress = progressWrapper
                        )
                } finally {
                        diffusionModel?.close()
                }
        }

        fun cancelGeneration() {
                cachedModel?.cancelGeneration()
                // SmolLM doesn't expose cancel explicitly in the wrapper yet, but we can try to
                // close it if
                // needed
                // or just rely on it checking interruption.
        }

        private suspend fun getOrLoadImageModel(
                context: Context,
                flashAttn: Boolean,
                width: Int = 512,
                height: Int = 512,
                onProgress: ((String, Int, Int) -> Unit)?,
                sequentialLoad: Boolean? = null
        ): StableDiffusion {
                // Check if we already have the correct IMAGE model loaded
                val spec = currentDiffusionModelSpec
                cachedModel?.let {
                        // Only return cached model if it's specifically an image model (not video)
                        if (spec != null && 
                            spec.filename == DEFAULT_IMAGE_MODEL_FILENAME &&
                            spec.vaePath == null && // Image models don't use separate VAE
                            spec.t5xxlPath == null) { // Image models don't use T5
                                return it
                        }
                        // Wrong model type loaded - unload it first
                        unloadDiffusionModel()
                }

                if (!preferPerformanceMode) {
                        diffusionModelCache.systemMemoryProvider = {
                                val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                                val mi = ActivityManager.MemoryInfo()
                                am.getMemoryInfo(mi)
                                mi.availMem / (1024L * 1024L)
                        }
                } else {
                        diffusionModelCache.systemMemoryProvider = null
                }
                ensureImageFiles(context, onProgress)
                prepareMemoryForLoading()

                // Phase 3: Adaptive flash attention based on image dimensions
                // When flashAttn=true (default), let the helper decide based on dimensions.
                // When flashAttn=false, force disable flash attention.
                // This ensures small images (128x128) don't use flash attention which can be
                // inefficient on mobile GPUs that may lack proper hardware support (coopmat2/subgroup_shuffle).
                val adaptiveFlashAttn =
                        FlashAttentionHelper.shouldUseFlashAttention(
                                width = width,
                                height = height,
                                forceEnable = if (flashAttn) null else false
                        )
                Log.i(
                        TAG,
                        "Loading image model with flash_attn=$adaptiveFlashAttn " +
                                "(requested=$flashAttn, dimensions=${width}x${height}, " +
                                "seqLen=${(width / 8) * (height / 8)}, sequentialLoad=${sequentialLoad})"
                )

                val modelFile = getFile(context, DEFAULT_IMAGE_MODEL_ID, DEFAULT_IMAGE_MODEL_FILENAME)

                // Build cache key
                val cacheKey = makeDiffusionCacheKey(modelFile.absolutePath, null, null)

                // Check cache
                diffusionModelCache.get(cacheKey)?.let { cached ->
                        Log.i(TAG, "Loaded Image model from cache: $cacheKey")
                        cachedModel = cached
                        currentDiffusionModelSpec = LoadedDiffusionModelSpec(
                                modelId = DEFAULT_IMAGE_MODEL_ID,
                                filename = DEFAULT_IMAGE_MODEL_FILENAME,
                                path = modelFile.absolutePath,
                                vaePath = null,
                                t5xxlPath = null
                        )
                        return cached
                }

                val loadStart = System.currentTimeMillis()
                // Let StableDiffusion.load() auto-detect the best backend.
                // On devices with slow Vulkan implementations (e.g., Samsung Xclipse 920),
                // CPU backend with sequential load can be 5x faster than Vulkan.
                // The auto-detection enables CPU backend on low-memory devices which is
                // often the better choice for mobile diffusion workloads.
                val finalSequentialLoad = if (preferPerformanceMode) null else sequentialLoad
                Log.i(TAG, "StableDiffusion.load(image) called with finalSequentialLoad=${finalSequentialLoad}, forceVulkan=${preferPerformanceMode}, offloadToCpu=false, flashAttn=$adaptiveFlashAttn")
                val model = StableDiffusion.load(
                        context = context,
                        modelPath = modelFile.absolutePath,
                        nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                        offloadToCpu = false,
                        sequentialLoad = finalSequentialLoad,
                        forceVulkan = preferPerformanceMode,
                        preferPerformanceMode = preferPerformanceMode,
                        flashAttn = adaptiveFlashAttn
                        // sequentialLoad defaults to null, allowing auto-detection
                )
                val loadTime = System.currentTimeMillis() - loadStart
                // Use file size as cache size estimate to avoid re-parsing the model file.
                val modelSize = modelFile.length()
                Log.i(TAG, "Loaded image model in ${loadTime}ms (size=${modelSize / 1024 / 1024}MB)")
                // StableDiffusion.load() already performs estimation internally if needed for
                // Vulkan VRAM heuristics, so we don't need to call it again here.

                diffusionModelCache.put(cacheKey, model, modelSize, loadTime)
                cachedModel = model
                currentDiffusionModelSpec = LoadedDiffusionModelSpec(
                        modelId = DEFAULT_IMAGE_MODEL_ID,
                        filename = DEFAULT_IMAGE_MODEL_FILENAME,
                        path = modelFile.absolutePath,
                        vaePath = null,
                        t5xxlPath = null
                )
                return model
        }

        private suspend fun getOrLoadVideoModel(
                context: Context,
                flashAttn: Boolean,
                onProgress: ((String, Int, Int) -> Unit)?,
                sequentialLoad: Boolean? = null
        ): StableDiffusion {
                // Check if we already have the correct VIDEO model loaded
                val spec = currentDiffusionModelSpec
                cachedModel?.let {
                        // Only return cached model if it's specifically a video model with VAE and T5
                        if (spec != null && 
                            spec.filename == DEFAULT_VIDEO_MODEL_FILENAME &&
                            spec.vaePath != null && // Video models require VAE
                            spec.t5xxlPath != null) { // Video models require T5
                                return it
                        }
                        // Wrong model type loaded - unload it first
                        unloadDiffusionModel()
                }

                // Set memory provider for cache before loading
                diffusionModelCache.systemMemoryProvider = {
                        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                        val mi = ActivityManager.MemoryInfo()
                        am.getMemoryInfo(mi)
                        mi.availMem / (1024L * 1024L)
                }

                ensureVideoFiles(context, onProgress)
                prepareMemoryForLoading()

                val modelFile =
                        getFile(context, DEFAULT_VIDEO_MODEL_ID, DEFAULT_VIDEO_MODEL_FILENAME)
                val vaeFile = getFile(context, DEFAULT_VIDEO_VAE_ID, DEFAULT_VIDEO_VAE_FILENAME)
                val t5File = getFile(context, DEFAULT_VIDEO_T5XXL_ID, DEFAULT_VIDEO_T5XXL_FILENAME)

                val cacheKey = makeDiffusionCacheKey(modelFile.absolutePath, vaeFile.absolutePath, t5File.absolutePath)
                diffusionModelCache.get(cacheKey)?.let { cached ->
                        Log.i(TAG, "Loaded Video model from cache: $cacheKey")
                        cachedModel = cached
                        currentDiffusionModelSpec = LoadedDiffusionModelSpec(
                                modelId = DEFAULT_VIDEO_MODEL_ID,
                                filename = DEFAULT_VIDEO_MODEL_FILENAME,
                                path = modelFile.absolutePath,
                                vaePath = vaeFile.absolutePath,
                                t5xxlPath = t5File.absolutePath
                        )
                        return cached
                }

                val loadStart = System.currentTimeMillis()
                val finalSequentialLoadV = if (preferPerformanceMode) null else sequentialLoad
                val finalKeepClipOnCpu = if (preferPerformanceMode) false else true
                val finalKeepVaeOnCpu = if (preferPerformanceMode) false else true
                Log.i(TAG, "StableDiffusion.load(video) called with finalSequentialLoad=${finalSequentialLoadV}, forceVulkan=${preferPerformanceMode}, offloadToCpu=false, keepClipOnCpu=${finalKeepClipOnCpu}, keepVaeOnCpu=${finalKeepVaeOnCpu}, flashAttn=$flashAttn")
                val model = StableDiffusion.load(
                        context = context,
                        modelPath = modelFile.absolutePath,
                        vaePath = vaeFile.absolutePath,
                        t5xxlPath = t5File.absolutePath,
                        nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                        offloadToCpu = false,
                        sequentialLoad = finalSequentialLoadV,
                        forceVulkan = preferPerformanceMode,
                        preferPerformanceMode = preferPerformanceMode,
                        keepClipOnCpu = finalKeepClipOnCpu,
                        keepVaeOnCpu = finalKeepVaeOnCpu,
                        flashAttn = flashAttn
                )
                val loadTime = System.currentTimeMillis() - loadStart
                val modelSize = modelFile.length()
                Log.i(TAG, "Loaded video model in ${loadTime}ms (size=${modelSize / 1024 / 1024}MB) sequentialLoad=${sequentialLoad}")
                // Use file size as cache size estimate to avoid re-parsing the model file.

                diffusionModelCache.put(cacheKey, model, modelSize, loadTime)
                cachedModel = model
                currentDiffusionModelSpec = LoadedDiffusionModelSpec(
                        modelId = DEFAULT_VIDEO_MODEL_ID,
                        filename = DEFAULT_VIDEO_MODEL_FILENAME,
                        path = modelFile.absolutePath,
                        vaePath = vaeFile.absolutePath,
                        t5xxlPath = t5File.absolutePath
                )
                Log.i(TAG, "Loaded Video model from cache: $cacheKey, sequentialLoad=${sequentialLoad}")
                return model
        }

        private suspend fun getOrLoadSmolLM(
                context: Context,
                modelId: String,
                filename: String,
                absolutePath: String? = null
        ): io.aatricks.llmedge.SmolLM {
                // Check existing cache first
                cachedSmolLM?.let {
                        // Check if the loaded model matches the request
                        val spec = currentTextModelSpec
                        if (spec != null &&
                                        spec.modelId == modelId &&
                                        spec.filename == filename &&
                                        spec.path == absolutePath
                        ) {
                                return it
                        }
                        // If not match, properly unload and remove from cache
                        unloadSmolLM()
                }

                val finalPath =
                        if (absolutePath != null) {
                                File(absolutePath)
                        } else {
                                getFile(context, modelId, filename)
                        }

                // Update memory provider for the cache based on context
                textModelCache.systemMemoryProvider = {
                        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                        val mi = ActivityManager.MemoryInfo()
                        am.getMemoryInfo(mi)
                        mi.availMem / (1024L * 1024L)
                }

                // Phase 3: Check model cache
                val cacheKey = finalPath.absolutePath
                textModelCache.get(cacheKey)?.let { cachedModel ->
                        Log.i(TAG, "Loaded SmolLM from cache: $cacheKey")
                        cachedSmolLM = cachedModel
                        currentTextModelSpec = LoadedTextModelSpec(modelId, filename, absolutePath)
                        return cachedModel
                }

                // Phase 3: Use core-aware threading
                val optimalThreads = if (preferPerformanceMode) {
                        CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING)
                } else {
                        // Conservative threading for stability/background use
                        2
                }

                Log.i(TAG, "Loading SmolLM with $optimalThreads threads (${coreInfo}), vulkan=$preferPerformanceMode")
                val loadStart = System.currentTimeMillis()

                // Initialize SmolLM with Vulkan setting based on performance mode
                // If preferPerformanceMode is false, useVulkan=false (CPU only) to avoid hangs on some devices
                val smol = io.aatricks.llmedge.SmolLM(useVulkan = preferPerformanceMode)

                // Help clear heap before loading large model
                prepareMemoryForLoading()

                smol.load(
                        modelPath = finalPath.absolutePath,
                        params =
                                io.aatricks.llmedge.SmolLM.InferenceParams(
                                        numThreads = optimalThreads,
                                        // Let SmolLM handle context size automatically based on
                                        // heap
                                        contextSize = null
                                )
                )

                val loadTime = System.currentTimeMillis() - loadStart
                val modelSize = finalPath.length()

                // Cache the loaded model
                textModelCache.put(cacheKey, smol, modelSize, loadTime)

                cachedSmolLM = smol
                currentTextModelSpec = LoadedTextModelSpec(modelId, filename, absolutePath)
                return smol
        }

        /**
         * Downloads a model from Hugging Face with progress updates. Useful for activities that
         * need to show download progress before generation.
         * 
         * Uses Android's system DownloadManager by default to avoid heap memory issues with
         * large model files. The system downloader streams directly to disk without using
         * the app's Java heap.
         */
        suspend fun downloadModel(
                context: Context,
                modelId: String,
                filename: String?,
                revision: String = "main",
                preferSystemDownloader: Boolean = true,
                onProgress: ((Long, Long?) -> Unit)? = null
        ): File {
                return HuggingFaceHub.ensureModelOnDisk(
                                context = context,
                                modelId = modelId,
                                revision = revision,
                                filename = filename,
                                preferSystemDownloader = preferSystemDownloader,
                                onProgress = onProgress
                        )
                        .file
        }

        private fun unloadDiffusionModel() {
                // Prefer removing from cache (which will also close the model) if we have a cache
                // key; otherwise, fall back to closing the cached instance.
                val spec = currentDiffusionModelSpec
                if (spec != null) {
                        val key = makeDiffusionCacheKey(spec.path ?: "", spec.vaePath, spec.t5xxlPath)
                        diffusionModelCache.remove(key)
                        currentDiffusionModelSpec = null
                        cachedModel = null
                } else {
                        cachedModel?.close()
                        cachedModel = null
                }
        }

        private fun unloadSmolLM() {
                // Remove from cache if we have a spec (this will also close the model)
                val spec = currentTextModelSpec
                if (spec != null) {
                        val cacheKey = spec.path ?: getModelCacheKey(spec.modelId, spec.filename)
                        textModelCache.remove(cacheKey)
                }
                // Clear local references
                cachedSmolLM = null
                currentTextModelSpec = null
        }

        private fun getModelCacheKey(modelId: String, filename: String): String {
                return "$modelId/$filename"
        }

        private suspend fun ensureVideoFiles(
                context: Context,
                onProgress: ((String, Int, Int) -> Unit)?
        ) {
                val hfCallback: ((Long, Long?) -> Unit)? = onProgress?.let { genCb ->
                        { downloaded: Long, total: Long? ->
                                genCb("Downloading video asset: $downloaded/${total ?: "?"}", 0, 0)
                        }
                }

                HuggingFaceHub.ensureRepoFileOnDisk(
                        context,
                        DEFAULT_VIDEO_MODEL_ID,
                        "main",
                        DEFAULT_VIDEO_MODEL_FILENAME,
                        emptyList(),
                        null,
                        false,
                        true,
                        hfCallback
                )
                HuggingFaceHub.ensureRepoFileOnDisk(
                        context,
                        DEFAULT_VIDEO_VAE_ID,
                        "main",
                        DEFAULT_VIDEO_VAE_FILENAME,
                        emptyList(),
                        null,
                        false,
                        true,
                        hfCallback
                )
                HuggingFaceHub.ensureRepoFileOnDisk(
                        context,
                        DEFAULT_VIDEO_T5XXL_ID,
                        "main",
                        DEFAULT_VIDEO_T5XXL_FILENAME,
                        emptyList(),
                        null,
                        false,
                        true,
                        hfCallback
                )
        }

        private suspend fun ensureImageFiles(
                context: Context,
                onProgress: ((String, Int, Int) -> Unit)?
        ) {
                val hfCallback: ((Long, Long?) -> Unit)? = onProgress?.let { genCb ->
                        { downloaded: Long, total: Long? -> genCb("Downloading image asset: $downloaded/${total ?: "?"}", 0, 0) }
                }

                HuggingFaceHub.ensureRepoFileOnDisk(
                        context,
                        DEFAULT_IMAGE_MODEL_ID,
                        "main",
                        DEFAULT_IMAGE_MODEL_FILENAME,
                        emptyList(),
                        null,
                        false,
                        true,
                        hfCallback
                )
        }

        private suspend fun getFile(context: Context, repoId: String, filename: String): File {
                return HuggingFaceHub.ensureRepoFileOnDisk(
                                context,
                                repoId,
                                "main",
                                filename,
                                emptyList(),
                                null,
                                false,
                                true,
                                null
                        )
                        .file
        }

        private fun isLowMemoryDevice(context: Context): Boolean {
                val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                val memInfo = ActivityManager.MemoryInfo()
                am.getMemoryInfo(memInfo)
                val totalRamGB = memInfo.totalMem / (1024L * 1024L * 1024L)
                return totalRamGB < 8
        }

        private fun prepareMemoryForLoading() {
                // Native memory is freed immediately on close()
                // Try to nudge the VM to free and compact memory before loading large models in
                // conservative mode. If preferPerformanceMode is enabled, skip the GC hint to
                // avoid unnecessary pauses and let the OS manage memory instead.
                if (!preferPerformanceMode) {
                        try {
                                // Log current memory
                                val rt = Runtime.getRuntime()
                                val used = (rt.totalMemory() - rt.freeMemory()) / (1024L * 1024L)
                                val max = rt.maxMemory() / (1024L * 1024L)
                                Log.d(TAG, "Preparing memory: heap_used=${used}MB heap_max=${max}MB")
                                // Ask for GC; this is a hint to ART and may help on memory-constrained devices
                                System.gc()
                        } catch (e: Exception) {
                                // no-op
                        }
                }
        }

        /**
         * Build a cache key for diffusion models using model path + optional VAE + T5 path.
         */
        private fun makeDiffusionCacheKey(modelPath: String, vaePath: String?, t5Path: String?): String {
                return listOf(modelPath, vaePath ?: "", t5Path ?: "").joinToString("|")
        }

        /**
         * Convert raw RGB byte array to Bitmap.
         * The native txt2img/txt2ImgWithPrecomputedCondition return raw RGB bytes (3 bytes per pixel),
         * not encoded image formats like PNG/JPEG.
         */
        // Reuse pixel buffers across conversions to reduce GC pressure
        private val pixelBufferThreadLocal = ThreadLocal<IntArray>()

        private fun rgbBytesToBitmap(rgb: ByteArray, width: Int, height: Int): Bitmap {
                val total = width * height
                var pixels = pixelBufferThreadLocal.get()
                if (pixels == null || pixels.size < total) {
                        pixels = IntArray(total)
                        pixelBufferThreadLocal.set(pixels)
                }
                return io.aatricks.llmedge.vision.ImageUtils.rgbBytesToBitmap(rgb, width, height, pixels)
        }
}

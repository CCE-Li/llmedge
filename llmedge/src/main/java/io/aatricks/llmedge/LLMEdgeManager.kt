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
        @Volatile private var cachedOcrEngine: io.aatricks.llmedge.vision.ocr.MlKitOcrEngine? = null
        @Volatile private var isLoading = false
        private var contextRef: WeakReference<Context>? = null

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
                                return@withLock kotlinx.coroutines.withContext(
                                        kotlinx.coroutines.Dispatchers.Default
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

        /**
         * Returns the shared SmolLM instance WITHOUT loading a model. Useful for activities that
         * want to manage the loading process themselves (e.g. HuggingFaceDemoActivity).
         */
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
                        val engine = getOrLoadOcrEngine(context)
                        val result =
                                engine.extractText(
                                        io.aatricks.llmedge.vision.ImageSource.BitmapSource(image)
                                )
                        return@withLock result.text
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

                        // Ensure files
                        val modelFile = getFile(context, params.modelId, params.modelFilename)
                        val projFile = getFile(context, params.modelId, params.projFilename)

                        // Load SmolLM
                        val smol = getOrLoadSmolLM(context, params.modelId, params.modelFilename)

                        // Prepare Vision Adapter
                        val adapter = io.aatricks.llmedge.vision.SmolLMVisionAdapter(context, smol)

                        try {
                                // We need to initialize the projector and encode the image
                                // This is complex logic from LlavaVisionActivity, simplified here.

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
                                                        // Fallback
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
                                        adapter.close() // Close adapter but keep SmolLM cached? Or
                                        // close both?
                                        // Adapter doesn't own SmolLM, so it's safe to close
                                        // adapter.
                                }
                        } catch (e: Exception) {
                                Log.e(TAG, "Vision analysis failed", e)
                                throw e
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
                                val bytes =
                                        model.txt2img(
                                                prompt = params.prompt,
                                                negative = params.negative,
                                                width = params.width,
                                                height = params.height,
                                                steps = params.steps,
                                                cfg = params.cfgScale,
                                                seed = params.seed
                                        )
                                return bytes?.let {
                                        android.graphics.BitmapFactory.decodeByteArray(
                                                it,
                                                0,
                                                it.size
                                        )
                                }
                        } else {
                                val model =
                                        getOrLoadImageModel(
                                                context,
                                                params.flashAttn,
                                                params.width,
                                                params.height,
                                                onProgress
                                        )
                                val bytes =
                                        model.txt2img(
                                                prompt = params.prompt,
                                                negative = params.negative,
                                                width = params.width,
                                                height = params.height,
                                                steps = params.steps,
                                                cfg = params.cfgScale,
                                                seed = params.seed
                                        )
                                return bytes?.let {
                                        android.graphics.BitmapFactory.decodeByteArray(
                                                it,
                                                0,
                                                it.size
                                        )
                                }
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
                                        getOrLoadVideoModel(context, params.flashAttn, onProgress)
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
                diffusionModelCache.systemMemoryProvider = {
                        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                        val mi = ActivityManager.MemoryInfo()
                        am.getMemoryInfo(mi)
                        mi.availMem / (1024L * 1024L)
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
                                        onProgress = onProgress
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
                        return bytes?.let {
                                android.graphics.BitmapFactory.decodeByteArray(it, 0, it.size)
                        }
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
                                        onProgress = onProgress
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
                onProgress: ((String, Int, Int) -> Unit)?
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

                diffusionModelCache.systemMemoryProvider = {
                        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                        val mi = ActivityManager.MemoryInfo()
                        am.getMemoryInfo(mi)
                        mi.availMem / (1024L * 1024L)
                }
                ensureImageFiles(context, onProgress)
                prepareMemoryForLoading()

                // Phase 3: Adaptive flash attention based on image dimensions
                val adaptiveFlashAttn =
                        FlashAttentionHelper.shouldUseFlashAttention(
                                width = width,
                                height = height,
                                forceEnable = if (flashAttn) true else null
                        )

                Log.i(
                        TAG,
                        "Loading image model with flash_attn=$adaptiveFlashAttn " +
                                "(requested=$flashAttn, dimensions=${width}x${height})"
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
                val model = StableDiffusion.load(
                        context = context,
                        modelPath = modelFile.absolutePath,
                        nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                        offloadToCpu = false,
                        flashAttn = adaptiveFlashAttn
                )
                val loadTime = System.currentTimeMillis() - loadStart

                // Estimate model footprint using native estimator, fallback to file size
                val chosenDevice = StableDiffusion.getVulkanDeviceCount().let { devices ->
                        var ci = -1
                        if (devices > 0) {
                                var maxTotal = 0L
                                for (i in 0 until devices) {
                                        val mem = StableDiffusion.getVulkanDeviceMemory(i)
                                        if (mem != null && mem.size >= 2 && mem[1] > maxTotal) {
                                                maxTotal = mem[1]
                                                ci = i
                                        }
                                }
                        }
                        ci
                }

                val estimatedBytes = StableDiffusion.estimateModelParamsMemoryBytes(modelFile.absolutePath, if (chosenDevice >= 0) chosenDevice else 0)
                val modelSize = if (estimatedBytes > 0L) estimatedBytes else modelFile.length()

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
                onProgress: ((String, Int, Int) -> Unit)?
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
                val model = StableDiffusion.load(
                        context = context,
                        modelPath = modelFile.absolutePath,
                        vaePath = vaeFile.absolutePath,
                        t5xxlPath = t5File.absolutePath,
                        nThreads = CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.DIFFUSION),
                        offloadToCpu = false,
                        keepClipOnCpu = true,
                        keepVaeOnCpu = true,
                        flashAttn = flashAttn
                )
                val loadTime = System.currentTimeMillis() - loadStart

                // Estimate model footprint using native estimator, fallback to file size
                val chosenDevice = StableDiffusion.getVulkanDeviceCount().let { devices ->
                        var ci = -1
                        if (devices > 0) {
                                var maxTotal = 0L
                                for (i in 0 until devices) {
                                        val mem = StableDiffusion.getVulkanDeviceMemory(i)
                                        if (mem != null && mem.size >= 2 && mem[1] > maxTotal) {
                                                maxTotal = mem[1]
                                                ci = i
                                        }
                                }
                        }
                        ci
                }

                val estimatedBytes = StableDiffusion.estimateModelParamsMemoryBytes(modelFile.absolutePath, if (chosenDevice >= 0) chosenDevice else 0)
                val modelSize = if (estimatedBytes > 0L) estimatedBytes else modelFile.length()

                diffusionModelCache.put(cacheKey, model, modelSize, loadTime)
                cachedModel = model
                currentDiffusionModelSpec = LoadedDiffusionModelSpec(
                        modelId = DEFAULT_VIDEO_MODEL_ID,
                        filename = DEFAULT_VIDEO_MODEL_FILENAME,
                        path = modelFile.absolutePath,
                        vaePath = vaeFile.absolutePath,
                        t5xxlPath = t5File.absolutePath
                )
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

                val smol = io.aatricks.llmedge.SmolLM()

                // Phase 3: Use core-aware threading
                val optimalThreads =
                        CpuTopology.getOptimalThreadCount(CpuTopology.TaskType.PROMPT_PROCESSING)

                Log.i(TAG, "Loading SmolLM with $optimalThreads threads (${coreInfo})")
                val loadStart = System.currentTimeMillis()

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

        private fun getOrLoadOcrEngine(
                context: Context
        ): io.aatricks.llmedge.vision.ocr.MlKitOcrEngine {
                cachedOcrEngine?.let {
                        return it
                }
                val engine = io.aatricks.llmedge.vision.ocr.MlKitOcrEngine(context)
                cachedOcrEngine = engine
                return engine
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
                // Let Android VM handle GC naturally to avoid blocking delays
        }

        /**
         * Build a cache key for diffusion models using model path + optional VAE + T5 path.
         */
        private fun makeDiffusionCacheKey(modelPath: String, vaePath: String?, t5Path: String?): String {
                return listOf(modelPath, vaePath ?: "", t5Path ?: "").joinToString("|")
        }
}

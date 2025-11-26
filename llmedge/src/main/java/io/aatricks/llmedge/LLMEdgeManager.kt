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

        private val loadMutex = Mutex()
        @Volatile private var cachedModel: StableDiffusion? = null
        @Volatile private var cachedSmolLM: io.aatricks.llmedge.SmolLM? = null
        @Volatile private var cachedOcrEngine: io.aatricks.llmedge.vision.ocr.MlKitOcrEngine? = null
        @Volatile private var isLoading = false
        private var contextRef: WeakReference<Context>? = null

        // Track currently loaded text model to allow switching
        private data class LoadedTextModelSpec(
                val modelId: String,
                val filename: String,
                val path: String?
        )
        private var currentTextModelSpec: LoadedTextModelSpec? = null

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

        /** Generates text using a local LLM. */
        suspend fun generateText(
                context: Context,
                params: TextGenerationParams,
                onProgress: ((String) -> Unit)? = null
        ): String =
                loadMutex.withLock {
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
                return loadMutex.withLock {
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
                return loadMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)
                        unloadDiffusionModel()
                        if (cachedSmolLM == null) {
                                cachedSmolLM = io.aatricks.llmedge.SmolLM()
                        }
                        cachedSmolLM!!
                }
        }

        /** Extracts text from an image using OCR. */
        suspend fun extractText(context: Context, image: Bitmap): String =
                loadMutex.withLock {
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
                loadMutex.withLock {
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
                loadMutex.withLock {
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
                                        getOrLoadImageModel(context, params.flashAttn, onProgress)
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
                                        getOrLoadImageModel(context, params.flashAttn, onProgress)
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
                loadMutex.withLock {
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
                                return framesBytes ?: emptyList()
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
                ensureVideoFiles(context, onProgress)

                // Load T5
                prepareMemoryForLoading(context)
                var t5Model: StableDiffusion? = null
                var cond: StableDiffusion.PrecomputedCondition? = null
                var uncond: StableDiffusion.PrecomputedCondition? = null

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
                                        nThreads = Runtime.getRuntime().availableProcessors(),
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
                        t5Model = null
                        System.gc()
                }

                // Load Diffusion
                prepareMemoryForLoading(context)
                var diffusionModel: StableDiffusion? = null
                try {
                        val modelFile =
                                getFile(
                                        context,
                                        DEFAULT_VIDEO_MODEL_ID,
                                        DEFAULT_VIDEO_MODEL_FILENAME
                                )
                        val vaeFile =
                                getFile(context, DEFAULT_VIDEO_VAE_ID, DEFAULT_VIDEO_VAE_FILENAME)

                        diffusionModel =
                                StableDiffusion.load(
                                        context = context,
                                        modelPath = modelFile.absolutePath,
                                        vaePath = vaeFile.absolutePath,
                                        t5xxlPath = null,
                                        nThreads = Runtime.getRuntime().availableProcessors(),
                                        offloadToCpu = false,
                                        keepClipOnCpu = false,
                                        keepVaeOnCpu = false,
                                        flashAttn = params.flashAttn
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
                        System.gc()
                }
        }

        private suspend fun generateVideoSequentially(
                context: Context,
                params: VideoGenerationParams,
                onProgress: ((String, Int, Int) -> Unit)?
        ): List<Bitmap> {
                ensureVideoFiles(context, onProgress)

                prepareMemoryForLoading(context)
                var t5Model: StableDiffusion? = null
                var cond: StableDiffusion.PrecomputedCondition? = null
                var uncond: StableDiffusion.PrecomputedCondition? = null

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
                                        nThreads = Runtime.getRuntime().availableProcessors(),
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
                        t5Model = null
                        System.gc()
                }

                prepareMemoryForLoading(context)
                var diffusionModel: StableDiffusion? = null
                try {
                        val modelFile =
                                getFile(
                                        context,
                                        DEFAULT_VIDEO_MODEL_ID,
                                        DEFAULT_VIDEO_MODEL_FILENAME
                                )
                        val vaeFile =
                                getFile(context, DEFAULT_VIDEO_VAE_ID, DEFAULT_VIDEO_VAE_FILENAME)

                        diffusionModel =
                                StableDiffusion.load(
                                        context = context,
                                        modelPath = modelFile.absolutePath,
                                        vaePath = vaeFile.absolutePath,
                                        t5xxlPath = null,
                                        nThreads = Runtime.getRuntime().availableProcessors(),
                                        offloadToCpu = false,
                                        keepClipOnCpu = false,
                                        keepVaeOnCpu = false,
                                        flashAttn = params.flashAttn
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
                        System.gc()
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
                onProgress: ((String, Int, Int) -> Unit)?
        ): StableDiffusion {
                cachedModel?.let {
                        return it
                }

                ensureImageFiles(context, onProgress)
                prepareMemoryForLoading(context)

                val modelFile =
                        getFile(context, DEFAULT_IMAGE_MODEL_ID, DEFAULT_IMAGE_MODEL_FILENAME)

                val model =
                        StableDiffusion.load(
                                context = context,
                                modelPath = modelFile.absolutePath,
                                vaePath = null, // VAE is baked into MeinaMix
                                t5xxlPath = null, // Not needed for MeinaMix
                                nThreads = Runtime.getRuntime().availableProcessors(),
                                offloadToCpu = false, // Try to use GPU/Vulkan
                                keepClipOnCpu = true,
                                keepVaeOnCpu = true,
                                flashAttn = flashAttn
                        )
                cachedModel = model
                return model
        }

        private suspend fun getOrLoadVideoModel(
                context: Context,
                flashAttn: Boolean,
                onProgress: ((String, Int, Int) -> Unit)?
        ): StableDiffusion {
                cachedModel?.let {
                        return it
                }

                ensureVideoFiles(context, onProgress)
                prepareMemoryForLoading(context)

                val modelFile =
                        getFile(context, DEFAULT_VIDEO_MODEL_ID, DEFAULT_VIDEO_MODEL_FILENAME)
                val vaeFile = getFile(context, DEFAULT_VIDEO_VAE_ID, DEFAULT_VIDEO_VAE_FILENAME)
                val t5File = getFile(context, DEFAULT_VIDEO_T5XXL_ID, DEFAULT_VIDEO_T5XXL_FILENAME)

                val model =
                        StableDiffusion.load(
                                context = context,
                                modelPath = modelFile.absolutePath,
                                vaePath = vaeFile.absolutePath,
                                t5xxlPath = t5File.absolutePath,
                                nThreads = Runtime.getRuntime().availableProcessors(),
                                offloadToCpu = false, // Try to use GPU/Vulkan
                                keepClipOnCpu = true,
                                keepVaeOnCpu = true,
                                flashAttn = flashAttn
                        )
                cachedModel = model
                return model
        }

        private suspend fun getOrLoadSmolLM(
                context: Context,
                modelId: String,
                filename: String,
                absolutePath: String? = null
        ): io.aatricks.llmedge.SmolLM {
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
                        // If not match, close and reload
                        it.close()
                        cachedSmolLM = null
                        currentTextModelSpec = null
                }

                val finalPath =
                        if (absolutePath != null) {
                                File(absolutePath)
                        } else {
                                getFile(context, modelId, filename)
                        }

                val smol = io.aatricks.llmedge.SmolLM()

                smol.load(
                        modelPath = finalPath.absolutePath,
                        params =
                                io.aatricks.llmedge.SmolLM.InferenceParams(
                                        numThreads =
                                                Runtime.getRuntime()
                                                        .availableProcessors()
                                                        .coerceAtMost(4),
                                        // Let SmolLM handle context size automatically based on
                                        // heap
                                        contextSize = null
                                )
                )
                cachedSmolLM = smol
                currentTextModelSpec = LoadedTextModelSpec(modelId, filename, absolutePath)
                return smol
        }

        /**
         * Downloads a model from Hugging Face with progress updates. Useful for activities that
         * need to show download progress before generation.
         */
        suspend fun downloadModel(
                context: Context,
                modelId: String,
                filename: String?,
                revision: String = "main",
                onProgress: ((Long, Long?) -> Unit)? = null
        ): File {
                return HuggingFaceHub.ensureModelOnDisk(
                                context = context,
                                modelId = modelId,
                                revision = revision,
                                filename = filename,
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
                cachedModel?.close()
                cachedModel = null
                System.gc()
        }

        private fun unloadSmolLM() {
                cachedSmolLM?.close()
                cachedSmolLM = null
                System.gc()
        }

        private suspend fun ensureVideoFiles(
                context: Context,
                onProgress: ((String, Int, Int) -> Unit)?
        ) {
                HuggingFaceHub.ensureRepoFileOnDisk(
                        context,
                        DEFAULT_VIDEO_MODEL_ID,
                        "main",
                        DEFAULT_VIDEO_MODEL_FILENAME,
                        emptyList(),
                        null,
                        false,
                        true,
                        null
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
                        null
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
                        null
                )
        }

        private suspend fun ensureImageFiles(
                context: Context,
                onProgress: ((String, Int, Int) -> Unit)?
        ) {
                HuggingFaceHub.ensureRepoFileOnDisk(
                        context,
                        DEFAULT_IMAGE_MODEL_ID,
                        "main",
                        DEFAULT_IMAGE_MODEL_FILENAME,
                        emptyList(),
                        null,
                        false,
                        true,
                        null
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

        private fun prepareMemoryForLoading(context: Context) {
                System.gc()
                Thread.sleep(100)
                System.gc()
        }
}

package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import android.os.Debug
import android.util.Log
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import io.aatricks.llmedge.huggingface.WanModelRegistry
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.Locale
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.min

class StableDiffusion private constructor(
    private val handle: Long
) : AutoCloseable {
    // Serialize concurrent generation calls - native library is not guaranteed to be reentrant.
    private val generationMutex = Mutex()
    private var modelMetadata: VideoModelMetadata? = null
    private val cancellationRequested = AtomicBoolean(false)
    @Volatile private var cachedProgressCallback: VideoProgressCallback? = null
    @Volatile private var lastGenerationMetrics: GenerationMetrics? = null
    private val nativeBridge: NativeBridge = Companion.nativeBridgeProvider(this)

    companion object {
        private const val LOG_TAG = "StableDiffusion"
        private const val BYTES_IN_MB = 1024L * 1024L
        private const val MEMORY_PRESSURE_THRESHOLD = 0.85f
        private const val MIN_FRAME_BATCH = 4
        private const val MAX_FRAME_BATCH = 8
        @Volatile private var isNativeLibraryAvailable: Boolean
        private val defaultNativeBridgeProvider: (StableDiffusion) -> NativeBridge = { instance ->
            object : NativeBridge {
                override fun txt2vid(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    videoFrames: Int,
                    steps: Int,
                    cfg: Float,
                    seed: Long,
                    initImage: ByteArray?,
                    initWidth: Int,
                    initHeight: Int,
                ): Array<ByteArray>? = instance.nativeTxt2Vid(
                    handle,
                    prompt,
                    negative,
                    width,
                    height,
                    videoFrames,
                    steps,
                    cfg,
                    seed,
                    initImage,
                    initWidth,
                    initHeight,
                )

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) {
                    instance.nativeSetProgressCallback(handle, callback)
                }

                override fun cancelGeneration(handle: Long) {
                    instance.nativeCancelGeneration(handle)
                }
            }
        }
        @Volatile private var nativeBridgeProvider: (StableDiffusion) -> NativeBridge = defaultNativeBridgeProvider

        init {
            val disableNativeLoad = java.lang.Boolean.getBoolean("llmedge.disableNativeLoad")
            isNativeLibraryAvailable = !disableNativeLoad
            if (disableNativeLoad) {
                println("[StableDiffusion] Native load disabled via llmedge.disableNativeLoad=true")
            } else {
                // Reuse SmolLM CPU feature detection to choose the best lib variant if needed later.
                try {
                    System.loadLibrary("sdcpp")
                    check(nativeCheckBindings()) { "Failed to link StableDiffusion JNI bindings" }
                } catch (e: UnsatisfiedLinkError) {
                    Log.e(LOG_TAG, "Failed to load sdcpp native library", e)
                    throw e
                }
            }
        }

        internal fun enableNativeBridgeForTests() {
            if (!isNativeLibraryAvailable) {
                isNativeLibraryAvailable = true
            }
        }

        internal fun overrideNativeBridgeForTests(provider: (StableDiffusion) -> NativeBridge) {
            nativeBridgeProvider = provider
        }

        internal fun resetNativeBridgeForTests() {
            nativeBridgeProvider = defaultNativeBridgeProvider
        }

        // JNI factory function implemented in native code. Declared as @JvmStatic
        // so it maps to a static native method (jclass) in JNI.
        @JvmStatic
        private external fun nativeCreate(
            modelPath: String?,
            vaePath: String?,
            t5xxlPath: String?,
            nThreads: Int,
            offloadToCpu: Boolean,
            keepClipOnCpu: Boolean,
            keepVaeOnCpu: Boolean,
        ): Long

        @JvmStatic
        private external fun nativeCheckBindings(): Boolean

        private fun inferVideoModelMetadata(
            resolvedModelPath: String,
            modelId: String?,
            explicitFilename: String?,
        ): VideoModelMetadata {
            val filename = explicitFilename ?: resolvedModelPath.substringAfterLast('/')
            val lowerName = filename.lowercase(Locale.US)
            val tags = mutableSetOf<String>()
            if (lowerName.contains("wan")) tags += "wan"
            if (lowerName.contains("video")) tags += "text-to-video"
            if (lowerName.contains("hunyuan")) tags += "hunyuan"

            val architectureGuess = when {
                !modelId.isNullOrBlank() -> modelId
                lowerName.contains("hunyuan") -> "hunyuan_video"
                lowerName.contains("wan") -> "wan"
                else -> null
            }

            val modelTypeGuess = when {
                lowerName.contains("ti2v") -> "ti2v"
                lowerName.contains("i2v") -> "i2v"
                lowerName.contains("t2v") -> "t2v"
                else -> null
            }

            return VideoModelMetadata(
                architecture = architectureGuess,
                modelType = modelTypeGuess,
                tags = tags,
                filename = filename,
            )
        }

        suspend fun load(
            context: Context,
            modelId: String? = null,
            filename: String? = null,
            modelPath: String? = null,
            vaePath: String? = null,
            t5xxlPath: String? = null,
            nThreads: Int = Runtime.getRuntime().availableProcessors(),
            offloadToCpu: Boolean = false,
            keepClipOnCpu: Boolean = false,
            keepVaeOnCpu: Boolean = false,
            token: String? = null,
            forceDownload: Boolean = false,
        ): StableDiffusion = withContext(Dispatchers.IO) {
            var resolvedModelPath: String
            var resolvedVaePath: String?

            if (modelPath != null) {
                resolvedModelPath = modelPath
                resolvedVaePath = vaePath
            } else if (modelId != null) {
                // If this model ID corresponds to a Wan video model in our registry, use
                // the convenience loader that pulls model + vae + t5 encoder as needed.
                try {
                    val possibleWan = WanModelRegistry.findById(context, modelId) ?: WanModelRegistry.findByModelIdPrefix(context, modelId.removePrefix("wan/"))
                    if (possibleWan != null) {
                        return@withContext loadFromHuggingFace(
                            context = context,
                            modelId = modelId,
                            filename = filename,
                            nThreads = nThreads,
                            offloadToCpu = offloadToCpu,
                            keepClipOnCpu = keepClipOnCpu,
                            keepVaeOnCpu = keepVaeOnCpu,
                            token = token,
                            forceDownload = forceDownload,
                            preferSystemDownloader = true,
                        )
                    }
                } catch (t: Throwable) {
                    // If registry lookup fails, continue with default downloader behavior
                }
                // First, try to find a GGUF model (preferred). If none exists in the repo,
                // fall back to looking for other supported diffusion weight formats
                // such as .safetensors or .ckpt using ensureRepoFileOnDisk.
                try {
                    val res = HuggingFaceHub.ensureModelOnDisk(
                        context = context,
                        modelId = modelId,
                        revision = "main",
                        preferredQuantizations = emptyList(),
                        filename = filename,
                        token = token,
                        forceDownload = forceDownload,
                        preferSystemDownloader = true,
                        onProgress = null,
                    )
                    resolvedModelPath = res.file.absolutePath
                    resolvedVaePath = vaePath
                } catch (iae: IllegalArgumentException) {
                    // No GGUF found — try to fetch a safetensors/ckpt (or other allowed extensions)
                    iae.printStackTrace()
                    val alt = HuggingFaceHub.ensureRepoFileOnDisk(
                        context = context,
                        modelId = modelId,
                        revision = "main",
                        filename = filename,
                        allowedExtensions = listOf(".gguf", ".safetensors", ".ckpt", ".pt", ".bin"),
                        token = token,
                        forceDownload = forceDownload,
                        preferSystemDownloader = true,
                        onProgress = null,
                    )
                    resolvedModelPath = alt.file.absolutePath
                    // If the selected repo-file is itself a VAE or ancillary file, the caller should
                    // explicitly pass a VAe path. We keep vaePath as given.
                    resolvedVaePath = vaePath
                }
            } else {
                throw IllegalArgumentException("Provide either modelPath or modelId")
            }

            val handle = nativeCreate(
                resolvedModelPath,
                resolvedVaePath,
                t5xxlPath,
                nThreads,
                offloadToCpu,
                keepClipOnCpu,
                keepVaeOnCpu,
            )
            if (handle == 0L) throw IllegalStateException("Failed to initialize Stable Diffusion context")
            val instance = StableDiffusion(handle)
            instance.modelMetadata = inferVideoModelMetadata(
                resolvedModelPath = resolvedModelPath,
                modelId = modelId,
                explicitFilename = filename,
            )
            instance
        }

        suspend fun loadFromHuggingFace(
            context: Context,
            modelId: String,
            filename: String? = null,
            nThreads: Int = Runtime.getRuntime().availableProcessors(),
            offloadToCpu: Boolean = false,
            keepClipOnCpu: Boolean = false,
            keepVaeOnCpu: Boolean = false,
            token: String? = null,
            forceDownload: Boolean = false,
            preferSystemDownloader: Boolean = true,
            onProgress: ((name: String, downloaded: Long, total: Long?) -> Unit)? = null,
        ): StableDiffusion = withContext(Dispatchers.IO) {
            val destinationRoot = File(context.filesDir, "hf-models")
            val (modelRes, vaeRes, t5Res) = HuggingFaceHub.ensureWanAssetsOnDisk(
                context = context,
                wanModelId = modelId,
                preferSystemDownloader = preferSystemDownloader,
                token = token,
                onProgress = { downloaded, total -> onProgress?.invoke(modelId, downloaded, total) },
            )

            val handle = nativeCreate(
                modelRes.file.absolutePath,
                vaeRes?.file?.absolutePath,
                t5Res?.file?.absolutePath,
                nThreads,
                offloadToCpu,
                keepClipOnCpu,
                keepVaeOnCpu,
            )
            if (handle == 0L) throw IllegalStateException("Failed to initialize Stable Diffusion context")
            val instance = StableDiffusion(handle)
            instance.modelMetadata = inferVideoModelMetadata(
                resolvedModelPath = modelRes.file.absolutePath,
                modelId = modelId,
                explicitFilename = filename,
            )
            instance
        }
    }

    data class GenerateParams(
        val prompt: String,
        val negative: String = "",
        val width: Int = 512,
        val height: Int = 512,
        val steps: Int = 20,
        val cfgScale: Float = 7.0f,
        val seed: Long = 42L
    )

    enum class Scheduler {
        EULER_A,
        DDIM,
        DDPM,
        LCM,
    }

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
        val scheduler: Scheduler = Scheduler.EULER_A,
    ) {
        fun validate(): Result<Unit> = runCatching {
            require(prompt.isNotBlank()) { "Prompt cannot be blank" }
            require(width % 64 == 0 && width in 256..960) {
                "Width must be a multiple of 64 in range 256..960"
            }
            require(height % 64 == 0 && height in 256..960) {
                "Height must be a multiple of 64 in range 256..960"
            }
            require(videoFrames in 4..64) { "Frame count must be between 4 and 64" }
            require(steps in 10..50) { "Steps must be between 10 and 50" }
            require(cfgScale in 1.0f..15.0f) { "CFG scale must be between 1.0 and 15.0" }
            require(strength in 0.0f..1.0f) { "Strength must be between 0.0 and 1.0" }
            require(seed >= -1L) { "Seed must be -1 or non-negative" }
        }

        fun withPrompt(prompt: String): VideoGenerateParams = copy(prompt = prompt)

        companion object {
            fun default(prompt: String = "") = VideoGenerateParams(prompt = prompt)
        }
    }

    data class GenerationMetrics(
        val totalTimeSeconds: Float,
        val framesPerSecond: Float,
        val timePerStep: Float,
        val peakMemoryUsageMb: Long,
        val vulkanEnabled: Boolean,
        val modelLoadTimeSeconds: Float = 0f,
        val frameConversionTimeSeconds: Float = 0f,
    ) {
        val averageFrameTime: Float
            get() = if (framesPerSecond > 0f) 1f / framesPerSecond else 0f

        val stepsPerSecond: Float
            get() = if (timePerStep > 0f) 1f / timePerStep else 0f

        val throughput: String
            get() = String.format("%.2f fps", framesPerSecond)
    }

    fun interface VideoProgressCallback {
        fun onProgress(
            step: Int,
            totalSteps: Int,
            currentFrame: Int,
            totalFrames: Int,
            timePerStep: Float,
        )
    }

    internal interface NativeBridge {
        fun txt2vid(
            handle: Long,
            prompt: String,
            negative: String,
            width: Int,
            height: Int,
            videoFrames: Int,
            steps: Int,
            cfg: Float,
            seed: Long,
            initImage: ByteArray?,
            initWidth: Int,
            initHeight: Int,
        ): Array<ByteArray>?

        fun setProgressCallback(handle: Long, callback: VideoProgressCallback?)

        fun cancelGeneration(handle: Long)
    }

    internal data class VideoModelMetadata(
        val architecture: String? = null,
        val modelType: String? = null,
        val tags: Set<String> = emptySet(),
        val filename: String? = null,
        val parameterCount: Long? = null,
        val contextWindow: Int? = null,
    )

    internal fun updateModelMetadata(metadata: VideoModelMetadata?) {
        modelMetadata = metadata
    }

    fun isVideoModel(): Boolean {
        val metadata = modelMetadata ?: return false
        return VideoModelDetector.isVideoModel(metadata)
    }

    suspend fun txt2vid(
        params: VideoGenerateParams,
        onProgress: VideoProgressCallback? = null,
    ): List<Bitmap> = withContext(Dispatchers.IO) {
        check(isNativeLibraryAvailable) { "Video generation is unavailable on this platform" }
        params.validate().getOrThrow()
        check(isVideoModel()) { "Loaded model is not a video model (use txt2img instead)" }

        val estimatedBytes = estimateFrameFootprintBytes(
            width = params.width,
            height = params.height,
            frameCount = params.videoFrames,
        )
        warnIfLowMemory(estimatedBytes)

        val (initBytes, initWidth, initHeight) = params.initImage?.let { bitmapToRgbBytes(it) }
            ?: Triple(null, 0, 0)

        val tempCallback = onProgress
        if (tempCallback != null) {
            nativeBridge.setProgressCallback(handle, tempCallback)
        }

        val startNanos = System.nanoTime()
        val memoryBefore = readNativeMemoryMb()
        val frameBytes = try {
            generationMutex.withLock {
                cancellationRequested.set(false)
                nativeBridge.txt2vid(
                    handle,
                    params.prompt,
                    params.negative,
                    params.width,
                    params.height,
                    params.videoFrames,
                    params.steps,
                    params.cfgScale,
                    params.seed,
                    initBytes,
                    initWidth,
                    initHeight,
                ) ?: throw IllegalStateException("Video generation failed")
            }
        } catch (t: Throwable) {
            if (cancellationRequested.get()) {
                cancellationRequested.set(false)
                throw CancellationException("Video generation cancelled", t)
            }
            throw t
        } finally {
            cancellationRequested.set(false)
        }

        if (frameBytes.isEmpty()) {
            throw IllegalStateException("Video generation returned no frames")
        }

        if (frameBytes.size != params.videoFrames) {
            Log.w(
                LOG_TAG,
                "Expected ${params.videoFrames} frames but received ${frameBytes.size}",
            )
        }

        val conversionStart = System.nanoTime()
        val bitmaps = convertFramesToBitmaps(frameBytes, params.width, params.height)
        val conversionSeconds = ((System.nanoTime() - conversionStart) / 1_000_000_000f)
        val totalSeconds = ((System.nanoTime() - startNanos) / 1_000_000_000f)
        val memoryAfter = readNativeMemoryMb()

        lastGenerationMetrics = GenerationMetrics(
            totalTimeSeconds = totalSeconds,
            framesPerSecond = if (totalSeconds > 0f) bitmaps.size / totalSeconds else 0f,
            timePerStep = if (params.steps > 0 && params.videoFrames > 0) {
                totalSeconds / (params.steps * params.videoFrames)
            } else {
                0f
            },
            peakMemoryUsageMb = maxOf(memoryBefore, memoryAfter),
            vulkanEnabled = false,
            frameConversionTimeSeconds = conversionSeconds,
        )

        warnIfLowMemory(estimatedBytes)

        if (tempCallback != null) {
            nativeBridge.setProgressCallback(handle, cachedProgressCallback)
        }

        bitmaps
    }

    fun setProgressCallback(callback: VideoProgressCallback?) {
        cachedProgressCallback = callback
        if (!isNativeLibraryAvailable) return
        nativeBridge.setProgressCallback(handle, callback)
    }

    fun cancelGeneration() {
        cancellationRequested.set(true)
        if (!isNativeLibraryAvailable) return
        nativeBridge.cancelGeneration(handle)
    }

    fun getLastGenerationMetrics(): GenerationMetrics? = lastGenerationMetrics

    suspend fun txt2img(params: GenerateParams): Bitmap = withContext(Dispatchers.Default) {
        val bytes = generationMutex.withLock {
            nativeTxt2Img(
                handle,
                params.prompt,
                params.negative,
                params.width,
                params.height,
                params.steps,
                params.cfgScale,
                params.seed
            ) ?: throw IllegalStateException("Image generation failed")
        }

        // Convert raw RGB bytes to Bitmap
        val bmp = Bitmap.createBitmap(params.width, params.height, Bitmap.Config.ARGB_8888)
        // Convert RGB to ARGB
        val rgb = bytes
        val pixels = IntArray(params.width * params.height)
        var idx = 0
        var p = 0
        while (idx < rgb.size && p < pixels.size) {
            val r = (rgb[idx].toInt() and 0xFF); val g = (rgb[idx + 1].toInt() and 0xFF); val b = (rgb[idx + 2].toInt() and 0xFF)
            pixels[p] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            idx += 3; p += 1
        }
        bmp.setPixels(pixels, 0, params.width, 0, 0, params.width, params.height)
        bmp
    }

    override fun close() {
        nativeDestroy(handle)
    }
    
    private external fun nativeDestroy(handle: Long)

    private external fun nativeTxt2Img(
        handle: Long,
        prompt: String,
        negative: String,
        width: Int,
        height: Int,
        steps: Int,
        cfg: Float,
        seed: Long,
    ): ByteArray?

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
        initImage: ByteArray?,
        initWidth: Int,
        initHeight: Int,
    ): Array<ByteArray>?

    private external fun nativeSetProgressCallback(
        handle: Long,
        callback: VideoProgressCallback?,
    )

    private external fun nativeCancelGeneration(handle: Long)

    private fun bitmapToRgbBytes(bitmap: Bitmap): Triple<ByteArray, Int, Int> {
        val safeBitmap = if (bitmap.config == Bitmap.Config.ARGB_8888) {
            bitmap
        } else {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }
        val width = safeBitmap.width
        val height = safeBitmap.height
        val pixels = IntArray(width * height)
        safeBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val rgb = ByteArray(width * height * 3)
        var rgbIndex = 0
        for (pixel in pixels) {
            rgb[rgbIndex++] = ((pixel shr 16) and 0xFF).toByte()
            rgb[rgbIndex++] = ((pixel shr 8) and 0xFF).toByte()
            rgb[rgbIndex++] = (pixel and 0xFF).toByte()
        }
        return Triple(rgb, width, height)
    }

    private fun convertFramesToBitmaps(
        frameBytes: Array<ByteArray>,
        width: Int,
        height: Int,
    ): List<Bitmap> {
        val batchSize = determineBatchSize(frameBytes.size)
        val bitmaps = ArrayList<Bitmap>(frameBytes.size)
        var index = 0
        while (index < frameBytes.size) {
            val end = min(index + batchSize, frameBytes.size)
            for (i in index until end) {
                bitmaps += rgbBytesToBitmap(frameBytes[i], width, height)
            }
            val remaining = frameBytes.size - end
            if (remaining > 0) {
                warnIfLowMemory(estimateFrameFootprintBytes(width, height, remaining))
            }
            index = end
        }
        return bitmaps
    }

    private fun rgbBytesToBitmap(bytes: ByteArray, width: Int, height: Int): Bitmap {
        val pixels = IntArray(width * height)
        var idx = 0
        var out = 0
        while (idx + 2 < bytes.size && out < pixels.size) {
            val r = bytes[idx].toInt() and 0xFF
            val g = bytes[idx + 1].toInt() and 0xFF
            val b = bytes[idx + 2].toInt() and 0xFF
            pixels[out] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            idx += 3
            out += 1
        }
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            setPixels(pixels, 0, width, 0, 0, width, height)
        }
    }

    private fun determineBatchSize(frameCount: Int): Int = when {
        frameCount >= 48 -> MIN_FRAME_BATCH
        frameCount >= 24 -> 6
        else -> MAX_FRAME_BATCH
    }

    private fun warnIfLowMemory(estimatedAdditionalBytes: Long) {
        val runtime = Runtime.getRuntime()
        val maxMemory = runtime.maxMemory().coerceAtLeast(BYTES_IN_MB)
        val used = runtime.totalMemory() - runtime.freeMemory()
        val projected = used + estimatedAdditionalBytes.coerceAtLeast(0L)
        val ratio = projected.toDouble() / maxMemory.toDouble()
        if (ratio >= MEMORY_PRESSURE_THRESHOLD) {
            Log.w(
                LOG_TAG,
                "Memory pressure warning: projected ${(projected / BYTES_IN_MB)} MB of ${(maxMemory / BYTES_IN_MB)} MB heap",
            )
        }
    }

    private fun estimateFrameFootprintBytes(width: Int, height: Int, frameCount: Int): Long {
        val pixels = width.toLong() * height.toLong()
        return pixels * 4L * frameCount
    }

    private fun readNativeMemoryMb(): Long = try {
        Debug.getNativeHeapAllocatedSize().coerceAtLeast(0L) / BYTES_IN_MB
    } catch (_: Throwable) {
        val runtime = Runtime.getRuntime()
        (runtime.totalMemory() - runtime.freeMemory()) / BYTES_IN_MB
    }

    private object VideoModelDetector {
        private val VIDEO_KEYWORDS = setOf(
            "wan",
            "hunyuan",
            "video",
            "t2v",
            "i2v",
            "ti2v",
        )

        fun isVideoModel(metadata: VideoModelMetadata): Boolean {
            val architecture = metadata.architecture.orEmpty().lowercase(Locale.US)
            if (containsKeyword(architecture)) return true

            val modelType = metadata.modelType.orEmpty().lowercase(Locale.US)
            if (containsKeyword(modelType)) return true

            val filename = metadata.filename.orEmpty().lowercase(Locale.US)
            if (containsKeyword(filename)) return true

            if (metadata.tags.any { containsKeyword(it.lowercase(Locale.US)) }) {
                return true
            }

            return false
        }

        private fun containsKeyword(value: String): Boolean {
            if (value.isEmpty()) return false
            return VIDEO_KEYWORDS.any { keyword -> value.contains(keyword) }
        }
    }
}

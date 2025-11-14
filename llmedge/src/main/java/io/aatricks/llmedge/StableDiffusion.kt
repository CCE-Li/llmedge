package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class StableDiffusion private constructor(
    private val handle: Long
) : AutoCloseable {
    // Serialize concurrent generation calls - native library is not guaranteed to be reentrant.
    private val generationMutex = Mutex()

    companion object {
        private const val LOG_TAG = "StableDiffusion"

        init {
            // Reuse SmolLM CPU feature detection to choose the best lib variant if needed later.
            try {
                System.loadLibrary("sdcpp")
                check(nativeCheckBindings()) { "Failed to link StableDiffusion JNI bindings" }
            } catch (e: UnsatisfiedLinkError) {
                Log.e(LOG_TAG, "Failed to load sdcpp native library", e)
                throw e
            }
        }

        // JNI factory function implemented in native code. Declared as @JvmStatic
        // so it maps to a static native method (jclass) in JNI.
        @JvmStatic
        private external fun nativeCreate(
            modelPath: String?,
            vaePath: String?,
            nThreads: Int,
            offloadToCpu: Boolean,
            keepClipOnCpu: Boolean,
            keepVaeOnCpu: Boolean,
        ): Long

        @JvmStatic
        private external fun nativeCheckBindings(): Boolean

        suspend fun load(
            context: Context,
            modelId: String? = null,
            filename: String? = null,
            modelPath: String? = null,
            vaePath: String? = null,
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
                nThreads,
                offloadToCpu,
                keepClipOnCpu,
                keepVaeOnCpu,
            )
            if (handle == 0L) throw IllegalStateException("Failed to initialize Stable Diffusion context")
            StableDiffusion(handle)
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
}

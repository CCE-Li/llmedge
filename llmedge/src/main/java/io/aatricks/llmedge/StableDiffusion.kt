package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
}

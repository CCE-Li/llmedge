package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import io.aatricks.llmedge.vision.ImageUtils
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import java.io.File
import java.io.FileOutputStream

/**
 * Test that generates 8 frames and exports them as a video file in the project root.
 * This is for manual verification of video quality.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VideoGenerationExportTest {

    private val LIB_PATH_ENV = "LLMEDGE_BUILD_NATIVE_LIB_PATH"
    private val MODEL_PATH_ENV = "LLMEDGE_TEST_MODEL_PATH"

    private fun envInt(name: String, default: Int): Int {
        val raw = System.getenv(name) ?: System.getProperty(name)
        return raw?.toIntOrNull() ?: default
    }

    private fun envBool(name: String, default: Boolean): Boolean {
        val raw = System.getenv(name) ?: System.getProperty(name)
        return when (raw?.trim()?.lowercase()) {
            null -> default
            "1", "true", "yes", "y" -> true
            "0", "false", "no", "n" -> false
            else -> default
        }
    }

    private fun findRepoRoot(startDir: File): File {
        // We want the repo root (the directory that contains settings.gradle.kts).
        // Depending on how Gradle/Robolectric is invoked, user.dir may be either:
        // - <repo>/ (root project)
        // - <repo>/llmedge (module dir)
        // - <repo>/llmedge-examples (another root)
        var cur: File? = startDir
        repeat(10) {
            val d = cur ?: return@repeat
            if (File(d, "settings.gradle.kts").exists()) return d
            cur = d.parentFile
        }
        // Fallback: best-effort
        return startDir
    }

    private fun estimateAvgLuma(bmp: Bitmap, step: Int = 8): Double {
        var sum = 0.0
        var count = 0
        for (y in 0 until bmp.height step step) {
            for (x in 0 until bmp.width step step) {
                val px = bmp.getPixel(x, y)
                val r = (px shr 16) and 0xFF
                val g = (px shr 8) and 0xFF
                val b = px and 0xFF
                // Rec. 709 luma, normalized to [0,1]
                sum += (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
                count++
            }
        }
        return if (count == 0) 0.0 else sum / count
    }

    @Test
    fun `generate 8 frame video and export`() = runBlocking {
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        val modelId = System.getenv("LLMEDGE_TEST_MODEL_ID") ?: System.getProperty("LLMEDGE_TEST_MODEL_ID")
        val modelFilename = System.getenv("LLMEDGE_TEST_MODEL_FILENAME") ?: System.getProperty("LLMEDGE_TEST_MODEL_FILENAME")
        Assume.assumeTrue("No test model specified", !modelPath.isNullOrBlank() || !modelId.isNullOrBlank())

        val libPath = System.getenv(LIB_PATH_ENV)
            ?: System.getProperty(LIB_PATH_ENV)
            ?: "${System.getProperty("user.dir")}/llmedge/build/native/linux-x86_64/libsdcpp.so"
        Assume.assumeTrue("Native library not found", File(libPath).exists())

        val disableNativeLoad = System.getProperty("llmedge.disableNativeLoad")
        Assume.assumeTrue("Native loading disabled", disableNativeLoad != "true")

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val t5PathEnv = System.getenv("LLMEDGE_TEST_T5_PATH") ?: System.getProperty("LLMEDGE_TEST_T5_PATH")
        val vaePathEnv = System.getenv("LLMEDGE_TEST_VAE_PATH") ?: System.getProperty("LLMEDGE_TEST_VAE_PATH")
        // Assume paths are set or will be downloaded
        // Assume.assumeTrue("T5 path not set", !t5Path.isNullOrBlank())
        // Assume.assumeTrue("VAE path not set", !vaePath.isNullOrBlank())

        // Defaults are chosen to be fast on CPU.
        // NOTE: Wan models generate frames in 1 + 4k increments. A request like 8 will be rounded.
        val width = envInt("LLMEDGE_TEST_WIDTH", 512)
        val height = envInt("LLMEDGE_TEST_HEIGHT", 512)
        val videoFrames = envInt("LLMEDGE_TEST_VIDEO_FRAMES", 5)
        val steps = envInt("LLMEDGE_TEST_STEPS", 12)
        val cfgScale = 6.0f
        val seed = 42L
        val prompt = "a beautiful sunset over ocean waves, golden light, cinematic"

        val forceDownload = envBool("LLMEDGE_TEST_FORCE_DOWNLOAD", false)
        val hfToken = System.getenv("HF_TOKEN") ?: System.getProperty("HF_TOKEN")

        // Resolve model component paths.
        var resolvedModelPath: String? = modelPath
        var resolvedVaePath: String? = vaePathEnv
        var resolvedT5Path: String? = t5PathEnv

        if (resolvedModelPath.isNullOrBlank() && !modelId.isNullOrBlank()) {
            // For Robolectric/JVM runs we prefer the pure-JVM downloader.
            // The Android system downloader (DownloadManager) can hang or be unavailable.
            val preferSystemDownloader = false

            // If the user passed the Comfy-Org model repo, download all three required components explicitly.
            // This bypasses the wan registry, which may point at different repos/filenames.
            if (modelId.equals("Comfy-Org/Wan_2.1_ComfyUI_repackaged", ignoreCase = true)) {
                val comfyRepoId = requireNotNull(modelId) { "LLMEDGE_TEST_MODEL_ID must be set" }
                val mainName = modelFilename ?: "wan2.1_t2v_1.3B_fp16.safetensors"
                val vaeName = System.getenv("LLMEDGE_TEST_VAE_FILENAME")
                    ?: System.getProperty("LLMEDGE_TEST_VAE_FILENAME")
                    ?: "wan_2.1_vae.safetensors"
                val t5ModelId = System.getenv("LLMEDGE_TEST_T5_MODEL_ID")
                    ?: System.getProperty("LLMEDGE_TEST_T5_MODEL_ID")
                    ?: "city96/umt5-xxl-encoder-gguf"
                val t5Name = System.getenv("LLMEDGE_TEST_T5_FILENAME")
                    ?: System.getProperty("LLMEDGE_TEST_T5_FILENAME")
                    ?: "umt5-xxl-encoder-Q3_K_S.gguf"

                println("[VideoExport] Downloading WAN components... (this can be several GB)")
                println("[VideoExport]   main: $comfyRepoId / $mainName")
                println("[VideoExport]   vae : $comfyRepoId / $vaeName")
                println("[VideoExport]   t5  : $t5ModelId / $t5Name")

                val mainRes = HuggingFaceHub.ensureRepoFileOnDisk(
                    context = context,
                    modelId = comfyRepoId,
                    revision = "main",
                    filename = mainName,
                    allowedExtensions = listOf(".safetensors"),
                    token = hfToken,
                    forceDownload = forceDownload,
                    preferSystemDownloader = preferSystemDownloader,
                    onProgress = { downloaded, total ->
                        if (total != null && total > 0) {
                            val pct = (downloaded * 100.0 / total)
                            println("[VideoExport] main download: ${String.format("%.1f", pct)}% (${downloaded}/${total})")
                        } else {
                            println("[VideoExport] main download: ${downloaded} bytes")
                        }
                    }
                )

                val vaeRes = HuggingFaceHub.ensureRepoFileOnDisk(
                    context = context,
                    modelId = comfyRepoId,
                    revision = "main",
                    filename = vaeName,
                    allowedExtensions = listOf(".safetensors", ".pt"),
                    token = hfToken,
                    forceDownload = forceDownload,
                    preferSystemDownloader = preferSystemDownloader,
                    onProgress = { downloaded, total ->
                        if (total != null && total > 0) {
                            val pct = (downloaded * 100.0 / total)
                            println("[VideoExport] vae download: ${String.format("%.1f", pct)}% (${downloaded}/${total})")
                        } else {
                            println("[VideoExport] vae download: ${downloaded} bytes")
                        }
                    }
                )

                val t5Res = HuggingFaceHub.ensureRepoFileOnDisk(
                    context = context,
                    modelId = t5ModelId,
                    revision = "main",
                    filename = t5Name,
                    allowedExtensions = listOf(".gguf"),
                    token = hfToken,
                    forceDownload = forceDownload,
                    preferSystemDownloader = preferSystemDownloader,
                    onProgress = { downloaded, total ->
                        if (total != null && total > 0) {
                            val pct = (downloaded * 100.0 / total)
                            println("[VideoExport] t5 download: ${String.format("%.1f", pct)}% (${downloaded}/${total})")
                        } else {
                            println("[VideoExport] t5 download: ${downloaded} bytes")
                        }
                    }
                )

                resolvedModelPath = mainRes.file.absolutePath
                resolvedVaePath = vaeRes.file.absolutePath
                resolvedT5Path = t5Res.file.absolutePath
                println("[VideoExport] Download complete:")
                println("[VideoExport]   main=${resolvedModelPath}")
                println("[VideoExport]   vae =${resolvedVaePath}")
                println("[VideoExport]   t5  =${resolvedT5Path}")
            }
        }

        Assume.assumeTrue(
            "Model path not resolved (set $MODEL_PATH_ENV or use LLMEDGE_TEST_MODEL_ID=Comfy-Org/Wan_2.1_ComfyUI_repackaged)",
            !resolvedModelPath.isNullOrBlank()
        )
        Assume.assumeTrue(
            "VAE path not resolved (set LLMEDGE_TEST_VAE_PATH or allow auto-download)",
            !resolvedVaePath.isNullOrBlank()
        )
        Assume.assumeTrue(
            "T5 path not resolved (set LLMEDGE_TEST_T5_PATH or allow auto-download)",
            !resolvedT5Path.isNullOrBlank()
        )

        println("[VideoExport] Starting video generation: ${width}x${height}, frames=$videoFrames, steps=$steps")
        val startTime = System.currentTimeMillis()

        val sd = StableDiffusion.load(
            context = context,
            modelPath = resolvedModelPath,
            vaePath = resolvedVaePath,
            t5xxlPath = resolvedT5Path,
            nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
            offloadToCpu = true,
            keepClipOnCpu = true,
            keepVaeOnCpu = true,
            flashAttn = true,
            sequentialLoad = false
        )

        println("[VideoExport] Model loaded, generating video...")

        val bitmaps = try {
            val params = StableDiffusion.VideoGenerateParams(
                prompt = prompt,
                negative = "blurry, low quality, distorted",
                width = width,
                height = height,
                videoFrames = videoFrames,
                steps = steps,
                cfgScale = cfgScale,
                seed = seed
            )

            sd.txt2vid(params) { step, totalSteps, currentFrame, totalFrames, _ ->
                println("[VideoExport] Progress: step=$step/$totalSteps, frame=$currentFrame/$totalFrames")
            }
        } finally {
            sd.close()
        }

        val elapsed = System.currentTimeMillis() - startTime
        println("[VideoExport] Generation completed in ${elapsed}ms, got ${bitmaps.size} frames")

        assertTrue("Expected at least 1 frame", bitmaps.isNotEmpty())

        // Export frames + GIF to repo root
        val projectRoot = findRepoRoot(File(System.getProperty("user.dir")))
        val framesDir = File(projectRoot, "generated_frames")
        framesDir.mkdirs()

        // Clear old frames
        framesDir.listFiles()?.forEach { it.delete() }

        // Save each frame as PNG
        bitmaps.forEachIndexed { index, bmp ->
            val frameFile = File(framesDir, "frame_${String.format("%03d", index)}.png")
            FileOutputStream(frameFile).use { fos ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, fos)
            }
            val luma = estimateAvgLuma(bmp)
            println("[VideoExport] Saved ${frameFile.absolutePath} (avgLuma=${String.format("%.4f", luma)})")
        }

        // Create GIF. Prefer ffmpeg (best quality via palettegen+paletteuse over *all* frames),
        // but fall back to an in-process encoder if ffmpeg is unavailable.
        val outputGif = File(projectRoot, "generated_video.gif")
        val fps = 8

        var gifCreated = false
        try {
            val ffmpegCmd = arrayOf(
                "ffmpeg", "-y",
                "-framerate", fps.toString(),
                "-i", "${framesDir.absolutePath}/frame_%03d.png",
                "-filter_complex",
                "[0:v]fps=$fps,split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither=sierra2_4a",
                "-loop", "0",
                outputGif.absolutePath
            )

            println("[VideoExport] Running ffmpeg GIF export: ${ffmpegCmd.joinToString(" ")}")
            val process = ProcessBuilder(*ffmpegCmd)
                .redirectErrorStream(true)
                .start()
            val ffmpegOutput = process.inputStream.bufferedReader().readText()
            val exitCode = process.waitFor()

            println("[VideoExport] ffmpeg output: $ffmpegOutput")
            println("[VideoExport] ffmpeg exit code: $exitCode")

            gifCreated = exitCode == 0 && outputGif.exists() && outputGif.length() > 0
        } catch (t: Throwable) {
            println("[VideoExport] ffmpeg not available or failed to run: ${t.message}")
        }

        if (!gifCreated) {
            try {
                FileOutputStream(outputGif).use { fos ->
                    ImageUtils.createAnimatedGif(
                        frames = bitmaps,
                        delayMs = (1000.0 / fps).toInt(),
                        output = fos,
                        loop = 0
                    )
                }
                gifCreated = outputGif.exists() && outputGif.length() > 0
                println("[VideoExport] Fallback GIF encoder wrote: ${outputGif.absolutePath}")
            } catch (t: Throwable) {
                println("[VideoExport] Fallback GIF encoder failed: ${t.message}")
            }
        }

        if (gifCreated) {
            println("[VideoExport] GIF exported to: ${outputGif.absolutePath}")
            println("[VideoExport] GIF size: ${outputGif.length() / 1024} KB")
        } else {
            println("[VideoExport] GIF export failed; frames saved to: ${framesDir.absolutePath}")
        }

        // Verify frames are not blank
        bitmaps.forEach { bmp ->
            var nonBlankFound = false
            outer@ for (y in 0 until bmp.height step 10) {
                for (x in 0 until bmp.width step 10) {
                    val px = bmp.getPixel(x, y)
                    if ((px ushr 24) != 0 && (px and 0x00FFFFFF) != 0x000000) {
                        nonBlankFound = true
                        break@outer
                    }
                }
            }
            assertTrue("Frame is blank", nonBlankFound)
        }
    }
}

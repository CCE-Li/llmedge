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

    @Test
    fun `generate 8 frame video and export`() = runBlocking {
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        Assume.assumeTrue("No test model specified", !modelPath.isNullOrBlank())

        val libPath = System.getenv(LIB_PATH_ENV)
            ?: System.getProperty(LIB_PATH_ENV)
            ?: "${System.getProperty("user.dir")}/llmedge/build/native/linux-x86_64/libsdcpp.so"
        Assume.assumeTrue("Native library not found", File(libPath).exists())

        val disableNativeLoad = System.getProperty("llmedge.disableNativeLoad")
        Assume.assumeTrue("Native loading disabled", disableNativeLoad != "true")

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context
        val t5Path = System.getenv("LLMEDGE_TEST_T5_PATH") ?: System.getProperty("LLMEDGE_TEST_T5_PATH")
        val vaePath = System.getenv("LLMEDGE_TEST_VAE_PATH") ?: System.getProperty("LLMEDGE_TEST_VAE_PATH")
        Assume.assumeTrue("T5 path not set", !t5Path.isNullOrBlank())
        Assume.assumeTrue("VAE path not set", !vaePath.isNullOrBlank())

        // Parameters for 8 frames
        val width = 512
        val height = 512
        val videoFrames = 8
        val steps = 15
        val cfgScale = 6.0f
        val seed = 42L
        val prompt = "a beautiful sunset over ocean waves, golden light, cinematic"

        println("[VideoExport] Starting 8-frame video generation: ${width}x${height}, $steps steps")
        val startTime = System.currentTimeMillis()

        val sd = StableDiffusion.load(
            context = context,
            modelPath = modelPath,
            vaePath = vaePath,
            t5xxlPath = t5Path,
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

            sd.txt2vid(params) { step, totalSteps, currentFrame, totalFrames, timePerStep ->
                println("[VideoExport] Progress: step=$step/$totalSteps, frame=$currentFrame/$totalFrames")
            }
        } finally {
            sd.close()
        }

        val elapsed = System.currentTimeMillis() - startTime
        println("[VideoExport] Generation completed in ${elapsed}ms, got ${bitmaps.size} frames")

        assertTrue("Expected at least 1 frame", bitmaps.isNotEmpty())

        // Export frames to project root
        val projectRoot = File(System.getProperty("user.dir")).parentFile ?: File(System.getProperty("user.dir"))
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
            println("[VideoExport] Saved ${frameFile.absolutePath}")
        }

        // Create video using ffmpeg
        val outputVideo = File(projectRoot, "generated_video.mp4")
        val ffmpegCmd = arrayOf(
            "ffmpeg", "-y",
            "-framerate", "8",
            "-i", "${framesDir.absolutePath}/frame_%03d.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            outputVideo.absolutePath
        )

        println("[VideoExport] Running ffmpeg: ${ffmpegCmd.joinToString(" ")}")
        val process = ProcessBuilder(*ffmpegCmd)
            .redirectErrorStream(true)
            .start()
        val ffmpegOutput = process.inputStream.bufferedReader().readText()
        val exitCode = process.waitFor()

        println("[VideoExport] ffmpeg output: $ffmpegOutput")
        println("[VideoExport] ffmpeg exit code: $exitCode")

        if (exitCode == 0 && outputVideo.exists()) {
            println("[VideoExport] Video exported to: ${outputVideo.absolutePath}")
            println("[VideoExport] Video size: ${outputVideo.length() / 1024} KB")
        } else {
            println("[VideoExport] ffmpeg failed, frames saved to: ${framesDir.absolutePath}")
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

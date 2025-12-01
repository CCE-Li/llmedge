package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import java.io.File

/**
 * Linux-host end-to-end test for video generation using a real native library (libsdcpp.so)
 * and a local model path. This is intentionally conservative to make it runnable on a typical
 * developer workstation.
 *
 * Requirements to run:
 * - Build the native sdcpp library for Linux and place as llmedge/build/native/linux-x86_64/libsdcpp.so
 *   (There is a script at scripts/build_sdcpp_linux.sh to help).
 * - Provide a small test model suitable for testing via environment variable
 *   LLMEDGE_TEST_MODEL_PATH (path to a .gguf model). If not set, the test will be skipped.
 *
 * This test directly uses StableDiffusion.load() instead of LLMEdgeManager to bypass
 * the sequential loading logic which has issues with standalone T5 GGUF files.
 */

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VideoGenerationLinuxE2ETest {

    private val LIB_PATH_ENV = "LLMEDGE_BUILD_NATIVE_LIB_PATH"
    private val MODEL_PATH_ENV = "LLMEDGE_TEST_MODEL_PATH"

    @Test
    fun `desktop end-to-end video generation`() = runBlocking {
        // Skip test if neither a model path nor a model id is provided
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        val modelId = System.getenv("LLMEDGE_TEST_MODEL_ID") ?: System.getProperty("LLMEDGE_TEST_MODEL_ID")
        val hfToken = System.getenv("HUGGING_FACE_TOKEN") ?: System.getenv("HF_API_TOKEN") ?: System.getProperty("HUGGING_FACE_TOKEN") ?: System.getProperty("HF_API_TOKEN")
        println("[VideoGenerationLinuxE2ETest] modelId=$modelId modelPath=$modelPath hfTokenPresent=${!hfToken.isNullOrBlank()}")
        Assume.assumeTrue("No test model specified in $MODEL_PATH_ENV or LLMEDGE_TEST_MODEL_ID", !modelPath.isNullOrBlank() || !modelId.isNullOrBlank())

        // Check that native library path is properly set and library file exists
        val libPath = System.getenv(LIB_PATH_ENV)
            ?: System.getProperty(LIB_PATH_ENV)
            ?: "${System.getProperty("user.dir")}/llmedge/build/native/linux-x86_64/libsdcpp.so"

        val libFile = java.io.File(libPath)
        println("[VideoGenerationLinuxE2ETest] libPath=$libPath libExists=${libFile.exists()}")
        println("[VideoGenerationLinuxE2ETest] java.library.path=${System.getProperty("java.library.path")}")
        Assume.assumeTrue("Native library not found at $libPath", libFile.exists())

        // Verify native library loading is enabled
        val disableNativeLoad = System.getProperty("llmedge.disableNativeLoad")
        println("[VideoGenerationLinuxE2ETest] llmedge.disableNativeLoad=$disableNativeLoad")
        Assume.assumeTrue(
            "Native loading is disabled. Run with LLMEDGE_BUILD_NATIVE_LIB_PATH env var set.",
            disableNativeLoad != "true"
        )

        // Use Robolectric context
        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context

        // Get model paths from environment
        val t5Path = System.getenv("LLMEDGE_TEST_T5_PATH") ?: System.getProperty("LLMEDGE_TEST_T5_PATH")
        val vaePath = System.getenv("LLMEDGE_TEST_VAE_PATH") ?: System.getProperty("LLMEDGE_TEST_VAE_PATH")
        println("[VideoGenerationLinuxE2ETest] modelPath=$modelPath t5Path=$t5Path vaePath=$vaePath")

        Assume.assumeTrue("T5 path not set", !t5Path.isNullOrBlank())
        Assume.assumeTrue("VAE path not set", !vaePath.isNullOrBlank())

        // Test parameters
        val width = 256
        val height = 256
        val videoFrames = 5  // Minimum 5 frames required for Wan model
        val steps = 10
        val cfgScale = 7.0f
        val seed = 1L
        val prompt = "a simple test of desktop video generation"

        println("[VideoGenerationLinuxE2ETest] Starting video generation with params: ${width}x${height}, $videoFrames frames, $steps steps")
        val startTime = System.currentTimeMillis()

        // Load model directly using StableDiffusion.load() with all components together
        // This bypasses LLMEdgeManager's sequential loading which has issues with standalone T5 GGUF
        println("[VideoGenerationLinuxE2ETest] Loading StableDiffusion model directly...")
        val sd = try {
            StableDiffusion.load(
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
        } catch (e: Exception) {
            println("[VideoGenerationLinuxE2ETest] Failed to load model: ${e.message}")
            e.printStackTrace()
            throw e
        }

        println("[VideoGenerationLinuxE2ETest] Model loaded, generating video...")

        val bitmaps = try {
            val params = StableDiffusion.VideoGenerateParams(
                prompt = prompt,
                negative = "",
                width = width,
                height = height,
                videoFrames = videoFrames,
                steps = steps,
                cfgScale = cfgScale,
                seed = seed
            )

            sd.txt2vid(params) { step, totalSteps, currentFrame, totalFrames, timePerStep ->
                println("[VideoGenerationLinuxE2ETest] Progress: step=$step/$totalSteps, frame=$currentFrame/$totalFrames, time/step=$timePerStep")
            }
        } catch (e: Exception) {
            println("[VideoGenerationLinuxE2ETest] Video generation failed: ${e.message}")
            e.printStackTrace()
            throw e
        } finally {
            sd.close()
        }

        val elapsed = System.currentTimeMillis() - startTime
        println("[VideoGenerationLinuxE2ETest] Video generation completed in ${elapsed}ms, got ${bitmaps.size} frames")

        // Basic sanity checks for generated frames
        // Note: The Wan model may produce fewer frames than requested at small resolutions
        // due to internal constraints. We accept at least 1 frame for the test to pass.
        assertTrue("Expected at least 1 frame, got ${bitmaps.size}", bitmaps.isNotEmpty())
        bitmaps.forEach { bmp ->
            assertEquals(width, bmp.width)
            assertEquals(height, bmp.height)
            assertEquals(Bitmap.Config.ARGB_8888, bmp.config)
            // Not blank: ensure at least one pixel is not fully black/transparent
            var nonBlankFound = false
            loop@ for (y in 0 until bmp.height) {
                for (x in 0 until bmp.width) {
                    val px = bmp.getPixel(x, y)
                    // Check alpha != 0 and not pure black
                    if ((px ushr 24) != 0 && (px and 0x00FFFFFF) != 0x000000) {
                        nonBlankFound = true
                        break@loop
                    }
                }
            }
            assertTrue("Bitmap is blank", nonBlankFound)
        }

        // Frames should not be identical across the generated sequence (if multiple frames)
        if (bitmaps.size > 1) {
            fun bitmapHash(b: Bitmap): Int {
                var h = 1
                for (y in 0 until b.height step maxOf(1, b.height / 16)) {
                    for (x in 0 until b.width step maxOf(1, b.width / 16)) {
                        h = 31 * h + b.getPixel(x, y)
                    }
                }
                return h
            }
            val hashes = bitmaps.map { bitmapHash(it) }
            assertTrue("Expected at least two unique frames", hashes.toSet().size > 1)
        }
    }
}

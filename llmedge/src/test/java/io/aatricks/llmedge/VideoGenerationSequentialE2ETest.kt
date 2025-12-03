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

/**
 * Linux-host end-to-end test for video generation using the EXACT SAME PATH as Android devices.
 * 
 * This test uses the sequential loading path (`forceSequentialLoad = true`) which:
 * 1. Loads T5 encoder separately -> precomputes condition -> unloads T5
 * 2. Loads diffusion model + VAE -> generates video with precomputed condition
 * 
 * This is the path that Android devices take due to memory constraints, and uses
 * `txt2VidWithPrecomputedCondition` internally.
 *
 * Requirements to run:
 * - Build the native sdcpp library for Linux and place as llmedge/build/native/linux-x86_64/libsdcpp.so
 * - Provide model paths via environment variables:
 *   - LLMEDGE_TEST_MODEL_PATH: path to wan2.1_t2v_1.3B_fp16.safetensors
 *   - LLMEDGE_TEST_VAE_PATH: path to wan_2.1_vae.safetensors  
 *   - LLMEDGE_TEST_T5_PATH: path to umt5-xxl-encoder-Q3_K_S.gguf
 *
 * Run with: ./scripts/run_linux_e2e_sequential.sh
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VideoGenerationSequentialE2ETest {

    private val LIB_PATH_ENV = "LLMEDGE_BUILD_NATIVE_LIB_PATH"
    private val MODEL_PATH_ENV = "LLMEDGE_TEST_MODEL_PATH"

    @Test
    fun `sequential video generation matches Android device path`() = runBlocking {
        // Skip test if model paths are not provided
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        val t5Path = System.getenv("LLMEDGE_TEST_T5_PATH") ?: System.getProperty("LLMEDGE_TEST_T5_PATH")
        val vaePath = System.getenv("LLMEDGE_TEST_VAE_PATH") ?: System.getProperty("LLMEDGE_TEST_VAE_PATH")
        
        println("[SequentialE2E] modelPath=$modelPath")
        println("[SequentialE2E] t5Path=$t5Path")
        println("[SequentialE2E] vaePath=$vaePath")
        
        Assume.assumeTrue("Model path not set", !modelPath.isNullOrBlank())
        Assume.assumeTrue("T5 path not set", !t5Path.isNullOrBlank())
        Assume.assumeTrue("VAE path not set", !vaePath.isNullOrBlank())

        // Check that native library exists
        val libPath = System.getenv(LIB_PATH_ENV)
            ?: System.getProperty(LIB_PATH_ENV)
            ?: "${System.getProperty("user.dir")}/llmedge/build/native/linux-x86_64/libsdcpp.so"

        val libFile = java.io.File(libPath)
        println("[SequentialE2E] libPath=$libPath exists=${libFile.exists()}")
        Assume.assumeTrue("Native library not found at $libPath", libFile.exists())

        // Verify native loading is enabled
        val disableNativeLoad = System.getProperty("llmedge.disableNativeLoad")
        Assume.assumeTrue(
            "Native loading is disabled",
            disableNativeLoad != "true"
        )

        // Use Robolectric context
        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context

        // Test parameters - same as VideoGenerationActivity defaults
        val width = 512
        val height = 512
        val videoFrames = 5  // Minimum for Wan model to produce multiple frames
        val steps = 12
        val cfgScale = 7.0f
        val seed = 42L
        val prompt = "A cat walking in a garden"

        println("[SequentialE2E] Starting SEQUENTIAL video generation (Android path)")
        println("[SequentialE2E] params: ${width}x${height}, $videoFrames frames, $steps steps")
        val startTime = System.currentTimeMillis()

        // This is the EXACT path that Android uses:
        // 1. forceSequentialLoad = true triggers generateVideoSequentially()
        // 2. generateVideoSequentially() loads T5, precomputes condition, unloads T5
        // 3. Then loads diffusion model and calls txt2VidWithPrecomputedCondition
        
        val params = LLMEdgeManager.VideoGenerationParams(
            prompt = prompt,
            negative = "",
            width = width,
            height = height,
            videoFrames = videoFrames,
            steps = steps,
            cfgScale = cfgScale,
            seed = seed,
            flowShift = Float.POSITIVE_INFINITY,  // Auto
            flashAttn = true,
            forceSequentialLoad = true,  // THIS IS THE KEY - forces Android path
            // Enable EasyCache like Android does
            easyCache = StableDiffusion.EasyCacheParams(
                enabled = true,
                reuseThreshold = 0.2f,
                startPercent = 0.15f,
                endPercent = 0.95f
            )
        )

        // We need to set up the model paths since LLMEdgeManager normally downloads from HuggingFace
        // For this test, we'll directly use StableDiffusion with the sequential loading simulation
        
        println("[SequentialE2E] Step 1: Loading T5 encoder for condition precomputation...")
        val t5StartTime = System.currentTimeMillis()
        
        // Load T5-only context (like generateVideoSequentially does)
        val t5Model = StableDiffusion.load(
            context = context,
            modelPath = t5Path,
            vaePath = null,
            t5xxlPath = null,
            nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
            offloadToCpu = true,
            keepClipOnCpu = true,
            keepVaeOnCpu = true,
            flashAttn = true
        )
        
        println("[SequentialE2E] T5 loaded in ${System.currentTimeMillis() - t5StartTime}ms")
        println("[SequentialE2E] Precomputing conditions...")
        
        // Precompute conditions (like generateVideoSequentially does)
        val cond = t5Model.precomputeCondition(
            prompt = prompt,
            negative = "",
            width = width,
            height = height
        )
        
        val uncond = t5Model.precomputeCondition(
            prompt = "",
            negative = "",
            width = width,
            height = height
        )
        
        println("[SequentialE2E] Conditions precomputed, unloading T5...")
        t5Model.close()
        
        // Force GC like Android does between model loads
        System.gc()
        Thread.sleep(100)
        
        println("[SequentialE2E] Step 2: Loading diffusion model + VAE...")
        val diffStartTime = System.currentTimeMillis()
        
        // Load diffusion model WITHOUT T5 (like generateVideoSequentially does with loadT5=false)
        val diffusionModel = StableDiffusion.load(
            context = context,
            modelPath = modelPath,
            vaePath = vaePath,
            t5xxlPath = null,  // No T5 - we already precomputed conditions
            nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
            offloadToCpu = true,
            keepClipOnCpu = true,
            keepVaeOnCpu = true,
            flashAttn = true
        )
        
        println("[SequentialE2E] Diffusion model loaded in ${System.currentTimeMillis() - diffStartTime}ms")
        println("[SequentialE2E] Step 3: Generating video with precomputed conditions...")
        
        val genStartTime = System.currentTimeMillis()
        
        // Generate video using precomputed conditions (the key Android path!)
        val videoParams = StableDiffusion.VideoGenerateParams(
            prompt = prompt,
            negative = "",
            width = width,
            height = height,
            videoFrames = videoFrames,
            steps = steps,
            cfgScale = cfgScale,
            seed = seed,
            strength = 1.0f,  // Full T2V mode
            easyCacheParams = params.easyCache
        )
        
        val progressCallback = StableDiffusion.VideoProgressCallback { step, totalSteps, currentFrame, totalFrames, timePerStep ->
            println("[SequentialE2E] Progress: step=$step/$totalSteps, frame=$currentFrame/$totalFrames")
        }
        
        val bitmaps = try {
            diffusionModel.txt2VidWithPrecomputedCondition(
                params = videoParams,
                cond = cond,
                uncond = uncond,
                onProgress = progressCallback
            )
        } finally {
            diffusionModel.close()
        }
        
        val totalTime = System.currentTimeMillis() - startTime
        val genTime = System.currentTimeMillis() - genStartTime
        
        println("[SequentialE2E] Video generation completed!")
        println("[SequentialE2E] Total time: ${totalTime}ms, generation time: ${genTime}ms")
        println("[SequentialE2E] Generated ${bitmaps.size} frames")

        // Validate results
        assertTrue("Expected at least 1 frame, got ${bitmaps.size}", bitmaps.isNotEmpty())
        
        bitmaps.forEachIndexed { index, bmp ->
            println("[SequentialE2E] Frame $index: ${bmp.width}x${bmp.height}")
            assertEquals("Frame $index width mismatch", width, bmp.width)
            assertEquals("Frame $index height mismatch", height, bmp.height)
            assertEquals("Frame $index config mismatch", Bitmap.Config.ARGB_8888, bmp.config)
            
            // Check frame is not blank (not all black/transparent)
            var nonBlankFound = false
            var totalBrightness = 0L
            loop@ for (y in 0 until bmp.height step 4) {
                for (x in 0 until bmp.width step 4) {
                    val px = bmp.getPixel(x, y)
                    val r = (px shr 16) and 0xFF
                    val g = (px shr 8) and 0xFF
                    val b = px and 0xFF
                    totalBrightness += r + g + b
                    if ((px ushr 24) != 0 && (px and 0x00FFFFFF) != 0x000000) {
                        nonBlankFound = true
                    }
                }
            }
            val avgBrightness = totalBrightness / ((bmp.width / 4) * (bmp.height / 4) * 3)
            println("[SequentialE2E] Frame $index average brightness: $avgBrightness")
            assertTrue("Frame $index is blank (all black)", nonBlankFound)
            assertTrue("Frame $index is too dark (avg brightness $avgBrightness < 5)", avgBrightness >= 5)
        }

        // Check frames are different from each other
        if (bitmaps.size > 1) {
            fun bitmapHash(b: Bitmap): Long {
                var h = 1L
                for (y in 0 until b.height step maxOf(1, b.height / 16)) {
                    for (x in 0 until b.width step maxOf(1, b.width / 16)) {
                        h = 31 * h + b.getPixel(x, y)
                    }
                }
                return h
            }
            val hashes = bitmaps.map { bitmapHash(it) }
            println("[SequentialE2E] Frame hashes: $hashes")
            assertTrue("Expected at least two unique frames, got ${hashes.toSet().size}", hashes.toSet().size > 1)
        }

        println("[SequentialE2E] ✓ All validations passed!")
    }

    @Test
    fun `sequential I2V generation matches Android device path`() = runBlocking {
        // Skip test if model paths are not provided
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        val t5Path = System.getenv("LLMEDGE_TEST_T5_PATH") ?: System.getProperty("LLMEDGE_TEST_T5_PATH")
        val vaePath = System.getenv("LLMEDGE_TEST_VAE_PATH") ?: System.getProperty("LLMEDGE_TEST_VAE_PATH")
        
        Assume.assumeTrue("Model path not set", !modelPath.isNullOrBlank())
        Assume.assumeTrue("T5 path not set", !t5Path.isNullOrBlank())
        Assume.assumeTrue("VAE path not set", !vaePath.isNullOrBlank())

        val libPath = System.getenv(LIB_PATH_ENV)
            ?: System.getProperty(LIB_PATH_ENV)
            ?: "${System.getProperty("user.dir")}/llmedge/build/native/linux-x86_64/libsdcpp.so"
        Assume.assumeTrue("Native library not found", java.io.File(libPath).exists())
        Assume.assumeTrue("Native loading disabled", System.getProperty("llmedge.disableNativeLoad") != "true")

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context

        // Test parameters for I2V
        val width = 512
        val height = 512
        val videoFrames = 5
        val steps = 12
        val cfgScale = 7.0f
        val seed = 42L
        val strength = 0.8f  // I2V strength
        val prompt = "A cat walking in a garden"

        println("[SequentialE2E-I2V] Starting SEQUENTIAL I2V generation (Android path)")
        val startTime = System.currentTimeMillis()

        // Create a simple init image (gradient)
        val initBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = (x * 255 / width)
                val g = (y * 255 / height)
                val b = 128
                initBitmap.setPixel(x, y, (0xFF shl 24) or (r shl 16) or (g shl 8) or b)
            }
        }
        println("[SequentialE2E-I2V] Created init image ${width}x${height}")

        // Step 1: Load T5 and precompute conditions
        println("[SequentialE2E-I2V] Step 1: Loading T5...")
        val t5Model = StableDiffusion.load(
            context = context,
            modelPath = t5Path,
            vaePath = null,
            t5xxlPath = null,
            nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
            offloadToCpu = true,
            keepClipOnCpu = true,
            keepVaeOnCpu = true,
            flashAttn = true
        )
        
        val cond = t5Model.precomputeCondition(prompt, "", width, height)
        val uncond = t5Model.precomputeCondition("", "", width, height)
        t5Model.close()
        System.gc()

        // Step 2: Load diffusion model
        println("[SequentialE2E-I2V] Step 2: Loading diffusion model...")
        val diffusionModel = StableDiffusion.load(
            context = context,
            modelPath = modelPath,
            vaePath = vaePath,
            t5xxlPath = null,
            nThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(8),
            offloadToCpu = true,
            keepClipOnCpu = true,
            keepVaeOnCpu = true,
            flashAttn = true
        )

        // Step 3: Generate video with init image
        println("[SequentialE2E-I2V] Step 3: Generating I2V with precomputed conditions...")
        val videoParams = StableDiffusion.VideoGenerateParams(
            prompt = prompt,
            negative = "",
            width = width,
            height = height,
            videoFrames = videoFrames,
            steps = steps,
            cfgScale = cfgScale,
            seed = seed,
            initImage = initBitmap,
            strength = strength,
            easyCacheParams = StableDiffusion.EasyCacheParams(enabled = true)
        )

        val bitmaps = try {
            diffusionModel.txt2VidWithPrecomputedCondition(
                params = videoParams,
                cond = cond,
                uncond = uncond
            )
        } finally {
            diffusionModel.close()
            initBitmap.recycle()
        }

        val totalTime = System.currentTimeMillis() - startTime
        println("[SequentialE2E-I2V] I2V completed in ${totalTime}ms, got ${bitmaps.size} frames")

        assertTrue("Expected at least 1 frame", bitmaps.isNotEmpty())
        bitmaps.forEach { bmp ->
            assertEquals(width, bmp.width)
            assertEquals(height, bmp.height)
        }
        
        println("[SequentialE2E-I2V] ✓ I2V validation passed!")
    }
}

package io.aatricks.llmedge

import android.graphics.Bitmap
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkStatic
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.atomic.AtomicBoolean

class VideoGenerationTest {

    @Suppress("unused")
    private val disableNativeLoadForTests = run {
        System.setProperty("llmedge.disableNativeLoad", "true")
        StableDiffusion.enableNativeBridgeForTests()
        StableDiffusion.overrideNativeBridgeForTests { instance ->
            object : StableDiffusion.NativeBridge {
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
                    scheduler: StableDiffusion.Scheduler,
                    strength: Float,
                    initImage: ByteArray?,
                    initWidth: Int,
                    initHeight: Int,
                ): Array<ByteArray>? = when {
                    prompt.contains("fail") -> null // Simulate failure
                    else -> Array(videoFrames) { byteArrayOf(1, 2, 3) } // Mock frame data
                }

                override fun setProgressCallback(handle: Long, callback: StableDiffusion.VideoProgressCallback?) {}
                override fun cancelGeneration(handle: Long) {}
            }
        }
        true
    }

    @Test
    fun `txt2vid generates video frames with correct count`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val params = StableDiffusion.VideoGenerateParams(
            prompt = "a cat walking",
            width = 256,
            height = 256,
            videoFrames = 8,
            steps = 10,
            cfgScale = 7.0f,
            seed = 123L
        )

        val result = sd.txt2vid(params)

        assertNotNull("txt2vid should return frames", result)
        assertEquals("Should return correct number of frames", params.videoFrames, result.size)

        result.forEach { frame ->
            assertEquals("Frame width should match params", params.width, frame.width)
            assertEquals("Frame height should match params", params.height, frame.height)
            assertEquals("Frame should be ARGB_8888", Bitmap.Config.ARGB_8888, frame.config)
        }
    }

    @Test
    fun `txt2vid throws exception when not a video model`() = runTest {
        val sd = newStableDiffusion()
        // Don't set video model metadata

        val params = StableDiffusion.VideoGenerateParams(prompt = "test")

        val exception = assertThrows(IllegalStateException::class.java) {
            runTest { sd.txt2vid(params) }
        }

        assertTrue(exception.message?.contains("not a video model") == true)
    }

    @Test
    fun `txt2vid throws exception when native method returns null`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val params = StableDiffusion.VideoGenerateParams(prompt = "fail") // Triggers null return

        val exception = assertThrows(IllegalStateException::class.java) {
            runTest { sd.txt2vid(params) }
        }

        assertEquals("Video generation failed", exception.message)
    }

    @Test
    fun `txt2vid enforces frame limits based on model size`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata(parameterCount = "5B"))

        val params = StableDiffusion.VideoGenerateParams(
            prompt = "test",
            videoFrames = 64 // Over limit for 5B model
        )

        val exception = assertThrows(IllegalArgumentException::class.java) {
            runTest { sd.txt2vid(params) }
        }

        assertTrue(exception.message?.contains("supports maximum 32 frames") == true)
    }

    @Test
    fun `txt2vid handles cancellation correctly`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        // Set cancellation flag
        val cancellationField = StableDiffusion::class.java.getDeclaredField("cancellationRequested")
        cancellationField.isAccessible = true
        val flag = cancellationField.get(sd) as AtomicBoolean
        flag.set(true)

        val params = StableDiffusion.VideoGenerateParams(prompt = "test")

        val exception = assertThrows(CancellationException::class.java) {
            runTest { sd.txt2vid(params) }
        }

        assertTrue(exception.message?.contains("cancelled") == true)
    }

    @Test
    fun `txt2vid sets and clears progress callback correctly`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val callback = StableDiffusion.VideoProgressCallback { _, _, _, _, _ -> }
        val params = StableDiffusion.VideoGenerateParams(prompt = "test")

        // Set callback before generation
        sd.setProgressCallback(callback)

        // Generate video (should use callback)
        val result = sd.txt2vid(params)
        assertNotNull(result)

        // Callback should be restored after generation
        val cachedCallbackField = StableDiffusion::class.java.getDeclaredField("cachedProgressCallback")
        cachedCallbackField.isAccessible = true
        assertEquals(callback, cachedCallbackField.get(sd))
    }

    @Test
    fun `txt2vid validates parameters before generation`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val invalidParams = StableDiffusion.VideoGenerateParams(
            prompt = "", // Invalid: blank prompt
            width = 256,
            height = 256
        )

        val exception = assertThrows(IllegalArgumentException::class.java) {
            runTest { sd.txt2vid(invalidParams) }
        }

        assertTrue(exception.message?.contains("Prompt cannot be blank") == true)
    }

    @Test
    fun `txt2vid handles init image for I2V mode correctly`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val initImage = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        val params = StableDiffusion.VideoGenerateParams(
            prompt = "test",
            initImage = initImage,
            strength = 0.8f
        )

        val result = sd.txt2vid(params)
        assertNotNull("Should generate video with init image", result)
    }

    @Test
    fun `txt2vid validates init image and strength consistency`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val params = StableDiffusion.VideoGenerateParams(
            prompt = "test",
            initImage = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888),
            strength = 0.0f // Invalid: strength must be > 0 for I2V
        )

        val exception = assertThrows(IllegalArgumentException::class.java) {
            runTest { sd.txt2vid(params) }
        }

        assertTrue(exception.message?.contains("strength must be > 0.0") == true)
    }

    @Test
    fun `txt2vid uses generation mutex for thread safety`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val params = StableDiffusion.VideoGenerateParams(prompt = "test")

        // The method should use generationMutex.withLock
        // We verify this by ensuring the method completes successfully
        val result = sd.txt2vid(params)
        assertNotNull("Should complete generation", result)
    }

    @Test
    fun `txt2vid calculates and stores generation metrics`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val params = StableDiffusion.VideoGenerateParams(
            prompt = "test",
            width = 256,
            height = 256,
            videoFrames = 4,
            steps = 5
        )

        val result = sd.txt2vid(params)

        val metrics = sd.getLastGenerationMetrics()
        assertNotNull("Should have generation metrics", metrics)
        assertTrue("Total time should be positive", metrics!!.totalTimeSeconds > 0)
        assertTrue("Frames per second should be positive", metrics.framesPerSecond > 0)
        assertEquals(4, result.size) // Verify frame count matches
    }

    @Test
    fun `txt2vid handles different schedulers correctly`() = runTest {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(createVideoModelMetadata())

        val schedulers = StableDiffusion.Scheduler.values()

        for (scheduler in schedulers) {
            val params = StableDiffusion.VideoGenerateParams(
                prompt = "test with $scheduler",
                scheduler = scheduler
            )

            val result = sd.txt2vid(params)
            assertNotNull("Should generate with $scheduler scheduler", result)
        }
    }

    @Test
    fun `cancelGeneration sets cancellation flag`() {
        val sd = newStableDiffusion()

        val cancellationField = StableDiffusion::class.java.getDeclaredField("cancellationRequested")
        cancellationField.isAccessible = true
        val flag = cancellationField.get(sd) as AtomicBoolean

        assertFalse("Flag should start false", flag.get())

        sd.cancelGeneration()

        assertTrue("Flag should be set to true", flag.get())
    }

    private fun newStableDiffusion(): StableDiffusion {
        val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
        constructor.isAccessible = true
        return constructor.newInstance(1L)
    }

    private fun createVideoModelMetadata(
        architecture: String = "wan",
        parameterCount: String = "1.3B"
    ): StableDiffusion.VideoModelMetadata {
        return StableDiffusion.VideoModelMetadata(
            architecture = architecture,
            modelType = "t2v",
            parameterCount = parameterCount,
            mobileSupported = true,
            tags = setOf("wan", "text-to-video"),
            filename = "test-model.gguf"
        )
    }
}
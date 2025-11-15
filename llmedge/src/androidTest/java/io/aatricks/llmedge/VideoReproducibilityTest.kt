package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.security.MessageDigest

/**
 * Reproducibility tests for video generation.
 * Tests that identical parameters produce identical outputs and that randomness works correctly.
 * 
 * NOTE: These tests require a real video model to be loaded. They will fail gracefully if
 * the model is not available or if native library is unavailable on the device.
 */
@RunWith(AndroidJUnit4::class)
class VideoReproducibilityTest : BaseVideoIntegrationTest() {
    private lateinit var context: Context

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
    }

    /**
     * T084: Test deterministic generation with fixed seed
     * Generate 2 videos with same seed, compare frame checksums byte-by-byte
     */
    @Test
    fun testDeterministicGenerationWithFixedSeed() = runBlocking {
        val sd = createStableDiffusion()

        // Use small parameters for faster test execution
        val params = StableDiffusion.VideoGenerateParams(
            prompt = "a cat walking",
            width = 256,
            height = 256,
            videoFrames = 4,
            steps = 10,
            seed = 42L // Fixed seed for reproducibility
        )

        // Generate first video
        val video1 = sd.txt2vid(params)
        assertNotNull("First video should not be null", video1)
        assertEquals("First video should have 4 frames", 4, video1.size)

        // Generate second video with same seed
        val video2 = sd.txt2vid(params)
        assertNotNull("Second video should not be null", video2)
        assertEquals("Second video should have 4 frames", 4, video2.size)

        // Compare frame by frame using checksums
        for (i in video1.indices) {
            val frame1Hash = bitmapToMd5(video1[i])
            val frame2Hash = bitmapToMd5(video2[i])
            assertEquals(
                "Frame $i should be identical between runs with same seed",
                frame1Hash,
                frame2Hash
            )
        }
        
        sd.close()
    }

    /**
     * T085: Test seed randomization
     * Use seed=-1 twice, verify different outputs
     */
    @Test
    fun testSeedRandomization() = runBlocking {
        val sd = createStableDiffusion()

        val params = StableDiffusion.VideoGenerateParams(
            prompt = "a cat walking",
            width = 256,
            height = 256,
            videoFrames = 4,
            steps = 10,
            seed = -1L // Random seed
        )

        // Generate two videos with random seeds
        val video1 = sd.txt2vid(params)
        assertNotNull("First video should not be null", video1)

        val video2 = sd.txt2vid(params)
        assertNotNull("Second video should not be null", video2)

        // At least one frame should be different (probabilistically certain with random seeds)
        var allFramesIdentical = true
        for (i in video1.indices) {
            val frame1Hash = bitmapToMd5(video1[i])
            val frame2Hash = bitmapToMd5(video2[i])
            if (frame1Hash != frame2Hash) {
                allFramesIdentical = false
                break
            }
        }

        assertFalse(
            "With random seeds, videos should be different (frames should not all be identical)",
            allFramesIdentical
        )
        
        sd.close()
    }

    /**
     * T086: Test deterministic generation across multiple runs
     * Fixed seed, verify identical output across 3 runs
     */
    @Test
    fun testDeterministicGenerationThreeRuns() = runBlocking {
        val sd = createStableDiffusion()

        val params = StableDiffusion.VideoGenerateParams(
            prompt = "a cat walking",
            width = 256,
            height = 256,
            videoFrames = 4,
            steps = 10,
            seed = 12345L
        )

        // Generate three videos
        val video1 = sd.txt2vid(params)
        val video2 = sd.txt2vid(params)
        val video3 = sd.txt2vid(params)

        assertNotNull("Video 1 should not be null", video1)
        assertNotNull("Video 2 should not be null", video2)
        assertNotNull("Video 3 should not be null", video3)

        // All three videos should have same frame count
        assertEquals(4, video1.size)
        assertEquals(4, video2.size)
        assertEquals(4, video3.size)

        // Compare all frames across all three runs
        for (i in video1.indices) {
            val frame1Hash = bitmapToMd5(video1[i])
            val frame2Hash = bitmapToMd5(video2[i])
            val frame3Hash = bitmapToMd5(video3[i])

            assertEquals(
                "Frame $i should be identical in run 1 and run 2",
                frame1Hash,
                frame2Hash
            )
            assertEquals(
                "Frame $i should be identical in run 1 and run 3",
                frame1Hash,
                frame3Hash
            )
        }
        
        sd.close()
    }

    /**
     * Helper function to compute MD5 hash of bitmap for comparison
     */
    private fun bitmapToMd5(bitmap: Bitmap): String {
        val buffer = ByteArray(bitmap.width * bitmap.height * 4)
        val intBuffer = java.nio.IntBuffer.allocate(bitmap.width * bitmap.height)
        bitmap.copyPixelsToBuffer(intBuffer)
        intBuffer.rewind()
        
        for (i in 0 until bitmap.width * bitmap.height) {
            val pixel = intBuffer.get()
            buffer[i * 4] = (pixel shr 24 and 0xFF).toByte()
            buffer[i * 4 + 1] = (pixel shr 16 and 0xFF).toByte()
            buffer[i * 4 + 2] = (pixel shr 8 and 0xFF).toByte()
            buffer[i * 4 + 3] = (pixel and 0xFF).toByte()
        }

        val md = MessageDigest.getInstance("MD5")
        val digest = md.digest(buffer)
        return digest.joinToString("") { "%02x".format(it) }
    }
}

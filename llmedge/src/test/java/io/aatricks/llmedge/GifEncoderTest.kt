package io.aatricks.llmedge

import android.graphics.BitmapFactory
import io.aatricks.llmedge.vision.ImageUtils
import java.io.File
import java.io.FileOutputStream
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Quick test to verify GIF encoding with existing frames. This can be run without regenerating
 * video.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class GifEncoderTest {

    @Test
    fun `re-encode existing frames to GIF`() {
        // Try multiple locations
        val workspaceRoot = File(System.getProperty("user.dir")).parentFile
        var framesDir = File("${workspaceRoot?.absolutePath}/llmedge/hq_frames")
        if (!framesDir.exists()) {
            framesDir = File("${System.getProperty("user.dir")}/hq_frames")
        }
        if (!framesDir.exists()) {
            framesDir = File("${System.getProperty("user.dir")}/llmedge/hq_frames")
        }

        println("[GifEncoderTest] Looking for frames in: ${framesDir.absolutePath}")
        Assume.assumeTrue(
                "Frames directory not found at ${framesDir.absolutePath}",
                framesDir.exists()
        )

        val frameFiles =
                framesDir.listFiles { f -> f.name.endsWith(".png") }?.sortedBy { it.name }
                        ?: emptyList()
        Assume.assumeTrue("No frames found", frameFiles.isNotEmpty())

        println("[GifEncoderTest] Found ${frameFiles.size} frames")

        // Load frames as bitmaps
        val frames =
                frameFiles.map { file ->
                    println("[GifEncoderTest] Loading ${file.name}")
                    BitmapFactory.decodeFile(file.absolutePath)
                }

        // Test with single frame first
        val singleFrameOutput =
                File(
                        "${workspaceRoot?.absolutePath ?: System.getProperty("user.dir")}/gif_single_frame.gif"
                )
        FileOutputStream(singleFrameOutput).use { fos ->
            ImageUtils.createAnimatedGif(
                    listOf(frames[0]),
                    delayMs = 100,
                    output = fos,
                    quality = 1
            ) // Best quality
        }
        println(
                "[GifEncoderTest] Single frame GIF: ${singleFrameOutput.absolutePath} (${singleFrameOutput.length()} bytes)"
        )

        // Create GIF with all frames using best quality
        val outputFile =
                File(
                        "${workspaceRoot?.absolutePath ?: System.getProperty("user.dir")}/gif_encoder_test.gif"
                )
        FileOutputStream(outputFile).use { fos ->
            ImageUtils.createAnimatedGif(
                    frames,
                    delayMs = 100,
                    output = fos,
                    quality = 1
            ) // Best quality
        }

        println("[GifEncoderTest] Saved GIF to ${outputFile.absolutePath}")
        println("[GifEncoderTest] GIF size: ${outputFile.length()} bytes")

        // Cleanup bitmaps
        frames.forEach { it.recycle() }

        assert(outputFile.exists()) { "GIF file was not created" }
        assert(outputFile.length() > 1000) { "GIF file is too small" }
    }
}

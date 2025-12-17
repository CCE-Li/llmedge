package io.aatricks.llmedge

import android.content.Context
import android.graphics.Bitmap
import io.aatricks.llmedge.StableDiffusion
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlinx.coroutines.runBlocking
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class VideoGenerationHQTest {

    @Test
    fun testTaehvHQ() = runBlocking {
        println("--- STARTING HIGH QUALITY TAEHV TEST ---")
        
        val modelPath = System.getProperty("LLMEDGE_TEST_MODEL_PATH") ?: System.getenv("LLMEDGE_TEST_MODEL_PATH")
        val t5Path = System.getProperty("LLMEDGE_TEST_T5_PATH") ?: System.getenv("LLMEDGE_TEST_T5_PATH")
        val taesdPath = System.getProperty("LLMEDGE_TEST_TAESD_PATH") ?: System.getenv("LLMEDGE_TEST_TAESD_PATH")
        
        if (modelPath == null || t5Path == null || taesdPath == null) {
            println("Skipping test: Missing paths")
            return@runBlocking
        }

        val context = org.robolectric.RuntimeEnvironment.getApplication() as Context

        try {
            val sd = StableDiffusion.load(
                context = context,
                modelPath = modelPath,
                vaePath = null,
                t5xxlPath = t5Path,
                taesdPath = taesdPath,
                nThreads = 8,
                offloadToCpu = true,
                keepClipOnCpu = true,
                keepVaeOnCpu = true,
                flashAttn = true,
                vaeDecodeOnly = true,
                sequentialLoad = false
            )
            
            val params = StableDiffusion.VideoGenerateParams(
                prompt = "A high quality close up of a magical glowing butterfly in a forest, highly detailed, cinematic",
                width = 256,
                height = 256,
                videoFrames = 9,
                steps = 30, // 30 steps for real quality
                cfgScale = 5.0f,
                seed = 12345
            )
            
            println("Generating HQ video (30 steps)... this will take a minute.")
            val frames = sd.txt2vid(params) { step, total, fCurrent, fTotal, tps ->
                if (step % 5 == 0) println("Progress: step $step/$total")
            }
            
            if (frames.isNotEmpty()) {
                val rootGif = File("../../../generated_video_hq.gif")
                java.io.FileOutputStream(rootGif).use { fos ->
                    io.aatricks.llmedge.vision.ImageUtils.createAnimatedGif(frames, 125, fos, 0, 1)
                }
                println("HQ GIF saved to ${rootGif.absolutePath}")
            }
            
            sd.close()
        } catch (e: Exception) {
            e.printStackTrace()
            throw e
        }
    }
}

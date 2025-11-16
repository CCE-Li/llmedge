package io.aatricks.llmedge

import org.junit.Assert.assertTrue
import org.junit.Test

class StableDiffusionVideoModelDetectorExtraTest {
    @Test
    fun `filename containing WAR casing doesn't prevent detection`() {
        val sd = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType).apply { isAccessible = true }
            .newInstance(1L)
        sd.updateModelMetadata(
            StableDiffusion.VideoModelMetadata(
                architecture = null,
                modelType = null,
                parameterCount = null,
                mobileSupported = true,
                tags = emptySet(),
                filename = "HUNYUAN_video_model.gguf",
            )
        )
        assertTrue(sd.isVideoModel())
    }

    @Test
    fun `tags case-insensitive detection`() {
        val sd = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType).apply { isAccessible = true }
            .newInstance(1L)
        sd.updateModelMetadata(
            StableDiffusion.VideoModelMetadata(
                architecture = "stable-diffusion-xl",
                modelType = null,
                parameterCount = null,
                mobileSupported = true,
                tags = setOf("Text-to-Video"),
                filename = "sdxl.gguf",
            ),
        )
        assertTrue(sd.isVideoModel())
    }
}

package io.aatricks.llmedge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class VideoGenerateParamsTest {

    @Test
    fun `validate succeeds for typical params`() {
        val params = StableDiffusion.VideoGenerateParams(
            prompt = "a cat walking",
            width = 512,
            height = 512,
            videoFrames = 16,
            steps = 20,
            cfgScale = 7.0f,
            strength = 0.8f,
        )

        assertTrue(params.validate().isSuccess)
    }

    @Test
    fun `blank prompt fails validation`() {
        val params = StableDiffusion.VideoGenerateParams(prompt = " ")

        assertValidationFails(params, "Prompt cannot be blank")
    }

    @Test
    fun `width must be multiple of 64`() {
        val params = StableDiffusion.VideoGenerateParams(
            prompt = "valid",
            width = 510,
        )

        assertValidationFails(params, "Width must be a multiple of 64")
    }

    @Test
    fun `height must be in supported range`() {
        val params = StableDiffusion.VideoGenerateParams(
            prompt = "valid",
            height = 128,
        )

        assertValidationFails(params, "Height must be a multiple of 64")
    }

    @Test
    fun `frame count must be between 4 and 64`() {
        val params = StableDiffusion.VideoGenerateParams(
            prompt = "valid",
            videoFrames = 2,
        )

        assertValidationFails(params, "Frame count must be between 4 and 64")
    }

    @Test
    fun `zero frame count fails validation`() {
        val params = StableDiffusion.VideoGenerateParams(
            prompt = "valid",
            videoFrames = 0,
        )

        assertValidationFails(params, "Frame count must be between 4 and 64")
    }

    @Test
    fun `steps must stay within 10 to 50`() {
        val params = StableDiffusion.VideoGenerateParams(
            prompt = "valid",
            steps = 9,
        )

        assertValidationFails(params, "Steps must be between 10 and 50")
    }

    @Test
    fun `cfg scale must stay within allowed range`() {
        val params = StableDiffusion.VideoGenerateParams(
            prompt = "valid",
            cfgScale = 0.5f,
        )

        assertValidationFails(params, "CFG scale must be between 1.0 and 15.0")
    }

    @Test
    fun `strength must stay within zero to one`() {
        val params = StableDiffusion.VideoGenerateParams(
            prompt = "valid",
            strength = 1.5f,
        )

        assertValidationFails(params, "Strength must be between 0.0 and 1.0")
    }

    @Test
    fun `seed below negative one fails validation`() {
        val params = StableDiffusion.VideoGenerateParams(
            prompt = "valid",
            seed = -5L,
        )

        assertValidationFails(params, "Seed must be -1 or non-negative")
    }

    @Test
    fun `withPrompt returns a copy with new prompt`() {
        val params = StableDiffusion.VideoGenerateParams(prompt = "first")
        val updated = params.withPrompt("second")

        assertEquals("second", updated.prompt)
        assertEquals(params.copy(prompt = "second"), updated)
    }

    @Test
    fun `default factory populates sane values`() {
        val defaults = StableDiffusion.VideoGenerateParams.default("hey")

        assertEquals("hey", defaults.prompt)
        assertEquals(512, defaults.width)
        assertEquals(16, defaults.videoFrames)
    }

    private fun assertValidationFails(
        params: StableDiffusion.VideoGenerateParams,
        expectedMessagePart: String,
    ) {
        val failure = params.validate().exceptionOrNull()
        assertNotNull("Expected validation failure", failure)
        assertTrue(failure is IllegalArgumentException)
        assertTrue(
            "Error message should mention '$expectedMessagePart' but was '${failure?.message}'",
            failure?.message?.contains(expectedMessagePart) == true,
        )
    }
}

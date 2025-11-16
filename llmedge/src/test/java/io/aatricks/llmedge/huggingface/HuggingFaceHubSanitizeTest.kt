package io.aatricks.llmedge.huggingface

import org.junit.Assert.assertEquals
import org.junit.Test

class HuggingFaceHubSanitizeTest {
    @Test
    fun `sanitize replaces slashes with underscores`() {
        val input = "unsloth/Qwen3-0.6B-GGUF"
        val sanitized = HuggingFaceHub.sanitize(input)
        assertEquals("unsloth_Qwen3-0.6B-GGUF", sanitized)
    }
}

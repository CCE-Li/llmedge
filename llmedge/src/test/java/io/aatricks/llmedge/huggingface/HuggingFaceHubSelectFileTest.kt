package io.aatricks.llmedge.huggingface

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class HuggingFaceHubSelectFileTest {
    @Test
    fun `selectFile returns null when no gguf file present`() {
        val files = listOf(
            HFModelTree.HFModelFile(type = "file", path = "weights.pt", size = 1234L),
            HFModelTree.HFModelFile(type = "file", path = "weights.safetensors", size = 2345L),
        )

        val selected = HuggingFaceHub.selectFile(files, filename = null, preferredQuantizations = listOf("Q4_0"))
        assertNull(selected)
    }

    @Test
    fun `selectFile picks file with null type`() {
        val files = listOf(
            HFModelTree.HFModelFile(type = null, path = "Qwen3-0.6B-FP16.gguf", size = 200L),
            HFModelTree.HFModelFile(type = null, path = "readme.txt", size = 50L),
        )
        val selected = HuggingFaceHub.selectFile(files, filename = null, preferredQuantizations = listOf("Q4_0"))
        assertEquals("Qwen3-0.6B-FP16.gguf", selected?.path)
    }

    @Test
    fun `selectFile honors filename argument with suffix match`() {
        val files = listOf(
            HFModelTree.HFModelFile(type = "file", path = "qwen3-0.6b.gguf", size = 1000L),
            HFModelTree.HFModelFile(type = "file", path = "qwen3-0.6b-Q4_K_M.gguf", size = 900L),
        )
        val selected = HuggingFaceHub.selectFile(files, filename = "qwen3-0.6b-Q4_K_M.gguf", preferredQuantizations = HuggingFaceHub.DEFAULT_QUANTIZATION_PRIORITIES)
        assertEquals("qwen3-0.6b-Q4_K_M.gguf", selected?.path)
    }

    @Test
    fun `selectFile prefers preferredQuantizations`() {
        val files = listOf(
            HFModelTree.HFModelFile(type = "file", path = "qwen3-0.6b-Q8_0.gguf", size = 1300L),
            HFModelTree.HFModelFile(type = "file", path = "qwen3-0.6b-Q4_0.gguf", size = 900L),
            HFModelTree.HFModelFile(type = "file", path = "qwen3-0.6b-Q4_K_M.gguf", size = 1100L),
        )
        // Pick q4_k_m first because it's earlier in the default priority list
        val selected = HuggingFaceHub.selectFile(files, filename = null, preferredQuantizations = HuggingFaceHub.DEFAULT_QUANTIZATION_PRIORITIES)
        assertEquals("qwen3-0.6b-Q4_K_M.gguf", selected?.path)
    }

    @Test
    fun `selectFile respects custom quantization priorities`() {
        val files = listOf(
            HFModelTree.HFModelFile(type = "file", path = "qwen3-0.6b-Q8_0.gguf", size = 1300L),
            HFModelTree.HFModelFile(type = "file", path = "qwen3-0.6b-Q4_0.gguf", size = 900L),
            HFModelTree.HFModelFile(type = "file", path = "qwen3-0.6b-Q4_K_M.gguf", size = 1100L),
        )
        val customPriorities = listOf("Q8_0", "Q4_0")
        val selected = HuggingFaceHub.selectFile(files, filename = null, preferredQuantizations = customPriorities)
        assertEquals("qwen3-0.6b-Q8_0.gguf", selected?.path)
    }
}


package io.aatricks.llmedge.huggingface

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class HuggingFaceHubPickRepoFileTest {
    @Test
    fun `pickRepoFile returns null with no matching extension`() {
        val files = listOf(
            HFModelTree.HFModelFile(type = "file", path = "weights.pt", size = 1234L),
            HFModelTree.HFModelFile(type = "file", path = "weights.safetensors", size = 2345L),
        )
        val selected = HuggingFaceHub.pickRepoFile(files, filename = null, allowedExtensions = listOf(".gguf"))
        assertNull(selected)
    }

    @Test
    fun `pickRepoFile picks largest allowed extension if no filename provided`() {
        val files = listOf(
            HFModelTree.HFModelFile(type = "file", path = "vae.safetensors", size = 12456L),
            HFModelTree.HFModelFile(type = "file", path = "weights.gguf", size = 42345L),
            HFModelTree.HFModelFile(type = "file", path = "small.gguf", size = 1024L),
        )
        val selected = HuggingFaceHub.pickRepoFile(files, filename = null, allowedExtensions = listOf(".safetensors", ".gguf"))
        assertEquals("weights.gguf", selected?.path)
    }

    @Test
    fun `pickRepoFile respects filename exact match`() {
        val files = listOf(
            HFModelTree.HFModelFile(type = "file", path = "weights.gguf", size = 42345L),
            HFModelTree.HFModelFile(type = "file", path = "small.gguf", size = 1024L),
        )
        val selected = HuggingFaceHub.pickRepoFile(files, filename = "small.gguf", allowedExtensions = listOf(".gguf"))
        assertEquals("small.gguf", selected?.path)
    }
}

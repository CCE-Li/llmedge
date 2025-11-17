package io.aatricks.llmedge.huggingface

import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Test

class HFModelTreeSerializationTest {
    @Test
    fun `deserialize HFModelFile with rfilename maps to path`() {
        val json = """
            {
              "type": "file",
              "oid": "012345",
              "size": 12345,
              "rfilename": "subdir/model.gguf"
            }
        """.trimIndent()

        val file = Json { ignoreUnknownKeys = true }.decodeFromString(HFModelTree.HFModelFile.serializer(), json)
        assertEquals("subdir/model.gguf", file.path)
        assertEquals("file", file.type)
        assertEquals(12345L, file.size)
    }
}

package io.aatricks.llmedge.huggingface

import android.content.Context
import android.content.res.AssetManager
import io.mockk.*
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.io.ByteArrayInputStream

class WanModelRegistryTest {

    @Test
    fun `loadFromAssets parses JSON correctly`() {
        val context = mockk<Context>()
        val assetManager = mockk<AssetManager>()

        val jsonContent = """
        [
            {
                "modelId": "wan2.1-t2v-1.3b",
                "filename": "wan2.1_t2v_1.3b_fp16.safetensors",
                "quantization": "fp16",
                "t5ModelId": "t5-xxl",
                "t5Filename": "t5-xxl_fp16.safetensors",
                "vaeFilename": "wan_vae_fp16.safetensors",
                "sizeBytes": 1073741824
            },
            {
                "modelId": "wan2.1-t2v-14b",
                "filename": "wan2.1_t2v_14b_fp8.safetensors",
                "quantization": "fp8",
                "sizeBytes": 2147483648
            }
        ]
        """.trimIndent()

        every { context.assets } returns assetManager
        val assetName = "wan-models/model-registry.json"
        every { assetManager.open(assetName) } answers { ByteArrayInputStream(jsonContent.toByteArray()) }

        val models = WanModelRegistry.loadFromAssets(context)

        assertEquals(2, models.size)

        val firstModel = models[0]
        assertEquals("wan2.1-t2v-1.3b", firstModel.modelId)
        assertEquals("wan2.1_t2v_1.3b_fp16.safetensors", firstModel.filename)
        assertEquals("fp16", firstModel.quantization)
        assertEquals("t5-xxl", firstModel.t5ModelId)
        assertEquals("t5-xxl_fp16.safetensors", firstModel.t5Filename)
        assertEquals("wan_vae_fp16.safetensors", firstModel.vaeFilename)
        assertEquals(1073741824L, firstModel.sizeBytes)

        val secondModel = models[1]
        assertEquals("wan2.1-t2v-14b", secondModel.modelId)
        assertEquals("wan2.1_t2v_14b_fp8.safetensors", secondModel.filename)
        assertEquals("fp8", secondModel.quantization)
        assertNull(secondModel.t5ModelId)
        assertNull(secondModel.t5Filename)
        assertNull(secondModel.vaeFilename)
        assertEquals(2147483648L, secondModel.sizeBytes)
    }

    @Test
    fun `findById returns exact match ignoring case`() {
        val context = mockk<Context>()
        val assetManager = mockk<AssetManager>()

        val jsonContent = """
        [
            {
                "modelId": "wan2.1-t2v-1.3b",
                "filename": "wan2.1_t2v_1.3b_fp16.safetensors"
            },
            {
                "modelId": "WAN2.1-I2V-5B",
                "filename": "wan2.1_i2v_5b_fp8.safetensors"
            }
        ]
        """.trimIndent()

        every { context.assets } returns assetManager
        val assetName = "wan-models/model-registry.json"
        every { assetManager.open(assetName) } answers { ByteArrayInputStream(jsonContent.toByteArray()) }

        val result1 = WanModelRegistry.findById(context, "wan2.1-t2v-1.3b")
        verify { assetManager.open(assetName) }
        assertEquals("wan2.1-t2v-1.3b", result1?.modelId)

        val result2 = WanModelRegistry.findById(context, "WAN2.1-I2V-5B")
        assertEquals("WAN2.1-I2V-5B", result2?.modelId)

        val result3 = WanModelRegistry.findById(context, "nonexistent")
        assertNull(result3)
    }

    @Test
    fun `findByModelIdPrefix returns first match with prefix ignoring case`() {
        val context = mockk<Context>()
        val assetManager = mockk<AssetManager>()

        val jsonContent = """
        [
            {
                "modelId": "wan2.1-t2v-1.3b",
                "filename": "wan2.1_t2v_1.3b_fp16.safetensors"
            },
            {
                "modelId": "wan2.1-t2v-5b",
                "filename": "wan2.1_t2v_5b_fp8.safetensors"
            },
            {
                "modelId": "hunyuan-video-t2v-720p",
                "filename": "hunyuan_video_t2v_720p_fp16.safetensors"
            }
        ]
        """.trimIndent()

        every { context.assets } returns assetManager
        val assetName = "wan-models/model-registry.json"
        every { assetManager.open(assetName) } answers { ByteArrayInputStream(jsonContent.toByteArray()) }

        val result1 = WanModelRegistry.findByModelIdPrefix(context, "wan2.1-t2v")
        verify { assetManager.open(assetName) }
        assertEquals("wan2.1-t2v-1.3b", result1?.modelId) // Should return first match

        val result2 = WanModelRegistry.findByModelIdPrefix(context, "WAN2.1-T2V")
        assertEquals("wan2.1-t2v-1.3b", result2?.modelId) // Case insensitive

        val result3 = WanModelRegistry.findByModelIdPrefix(context, "hunyuan")
        assertEquals("hunyuan-video-t2v-720p", result3?.modelId)

        val result4 = WanModelRegistry.findByModelIdPrefix(context, "nonexistent")
        assertNull(result4)
    }

    @Test
    fun `WanModelEntry data class handles null values correctly`() {
        val entry = WanModelEntry(
            modelId = "test-model",
            filename = "test.gguf",
            quantization = null,
            t5ModelId = null,
            t5Filename = null,
            vaeFilename = null,
            sizeBytes = null
        )

        assertEquals("test-model", entry.modelId)
        assertEquals("test.gguf", entry.filename)
        assertNull(entry.quantization)
        assertNull(entry.t5ModelId)
        assertNull(entry.t5Filename)
        assertNull(entry.vaeFilename)
        assertNull(entry.sizeBytes)
    }

    @Test
    fun `loadFromAssets handles empty JSON array`() {
        val context = mockk<Context>()
        val assetManager = mockk<AssetManager>()

        val jsonContent = "[]"

        every { context.assets } returns assetManager
        val assetName = "wan-models/model-registry.json"
        every { assetManager.open(assetName) } answers { ByteArrayInputStream(jsonContent.toByteArray()) }

        val models = WanModelRegistry.loadFromAssets(context)
        assertEquals(0, models.size)
    }
}
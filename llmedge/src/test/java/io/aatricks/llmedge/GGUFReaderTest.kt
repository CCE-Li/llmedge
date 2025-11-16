package io.aatricks.llmedge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class GGUFReaderTest {

    @Test
    fun `GGUFReader constructor initializes correctly`() {
        val reader = GGUFReader()
        assertNotNull(reader)
    }

    @Test
    fun `GGUFReader close works when not initialized`() {
        val reader = GGUFReader()
        // Should not throw exception
        reader.close()
    }

    @Test
    fun `GGUFReader getContextSize throws when not loaded`() {
        val reader = GGUFReader()
        try {
            reader.getContextSize()
            assert(false) { "Should have thrown AssertionError" }
        } catch (e: AssertionError) {
            assert(e.message?.contains("Use GGUFReader.load()") == true)
        }
    }

    @Test
    fun `GGUFReader getChatTemplate throws when not loaded`() {
        val reader = GGUFReader()
        try {
            reader.getChatTemplate()
            assert(false) { "Should have thrown AssertionError" }
        } catch (e: AssertionError) {
            assert(e.message?.contains("Use GGUFReader.load()") == true)
        }
    }

    @Test
    fun `GGUFReader getArchitecture throws when not loaded`() {
        val reader = GGUFReader()
        try {
            reader.getArchitecture()
            assert(false) { "Should have thrown AssertionError" }
        } catch (e: AssertionError) {
            assert(e.message?.contains("Use GGUFReader.load()") == true)
        }
    }

    @Test
    fun `GGUFReader getParameterCount throws when not loaded`() {
        val reader = GGUFReader()
        try {
            reader.getParameterCount()
            assert(false) { "Should have thrown AssertionError" }
        } catch (e: AssertionError) {
            assert(e.message?.contains("Use GGUFReader.load()") == true)
        }
    }

    @Test
    fun `GGUFReader getModelName throws when not loaded`() {
        val reader = GGUFReader()
        try {
            reader.getModelName()
            assert(false) { "Should have thrown AssertionError" }
        } catch (e: AssertionError) {
            assert(e.message?.contains("Use GGUFReader.load()") == true)
        }
    }

    @Test
    fun `GGUFReader operations work with mocked native bridge`() {
        GGUFReader.overrideNativeBridgeForTests { instance ->
            object : GGUFReader.NativeBridge {
                override fun getGGUFContextNativeHandle(modelPath: String): Long = 12345L
                override fun getContextSize(nativeHandle: Long): Long = 4096L
                override fun getChatTemplate(nativeHandle: Long): String = "{% for message in messages %}{{ message.content }}{% endfor %}"
                override fun getArchitecture(nativeHandle: Long): String = "llama"
                override fun getParameterCount(nativeHandle: Long): String = "7B"
                override fun getModelName(nativeHandle: Long): String = "Test Model"
                override fun releaseGGUFContext(nativeHandle: Long) {}
            }
        }

        try {
            val reader = GGUFReader()

            // Test successful load
            kotlinx.coroutines.runBlocking {
                reader.load("/fake/path/model.gguf")
            }

            // Test getters
            assertEquals(4096L, reader.getContextSize())
            assertEquals("{% for message in messages %}{{ message.content }}{% endfor %}", reader.getChatTemplate())
            assertEquals("llama", reader.getArchitecture())
            assertEquals("7B", reader.getParameterCount())
            assertEquals("Test Model", reader.getModelName())

            // Test close
            reader.close()

        } finally {
            GGUFReader.resetNativeBridgeForTests()
        }
    }

    @Test
    fun `GGUFReader returns null for missing metadata with mocked native bridge`() {
        GGUFReader.overrideNativeBridgeForTests { instance ->
            object : GGUFReader.NativeBridge {
                override fun getGGUFContextNativeHandle(modelPath: String): Long = 12345L
                override fun getContextSize(nativeHandle: Long): Long = -1L  // Indicates missing
                override fun getChatTemplate(nativeHandle: Long): String = ""  // Empty indicates missing
                override fun getArchitecture(nativeHandle: Long): String = ""
                override fun getParameterCount(nativeHandle: Long): String = ""
                override fun getModelName(nativeHandle: Long): String = ""
                override fun releaseGGUFContext(nativeHandle: Long) {}
            }
        }

        try {
            val reader = GGUFReader()

            kotlinx.coroutines.runBlocking {
                reader.load("/fake/path/model.gguf")
            }

            // Test null returns for missing metadata
            assertNull(reader.getContextSize())
            assertNull(reader.getChatTemplate())
            assertNull(reader.getArchitecture())
            assertNull(reader.getParameterCount())
            assertNull(reader.getModelName())

            reader.close()

        } finally {
            GGUFReader.resetNativeBridgeForTests()
        }
    }
}
package io.aatricks.llmedge.huggingface

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class WanModelRegistryAndroidTest {
    @Test
    fun registryLoadsAndFindsWanModel() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val entry = WanModelRegistry.findById(context, "wan/Wan2.1-T2V-1.3B")
        assertNotNull("Registry should contain Wan2.1-T2V-1.3B", entry)
        assertEquals("wan/Wan2.1-T2V-1.3B", entry?.modelId)
        // The registry contains multiple entries for the same modelId; first entry is Q4_K_M
        assertEquals("Wan2.1-T2V-1.3B-Q4_K_M.gguf", entry?.filename)
        assertEquals("city96/umt5-xxl-encoder-gguf", entry?.t5ModelId)
    }
}

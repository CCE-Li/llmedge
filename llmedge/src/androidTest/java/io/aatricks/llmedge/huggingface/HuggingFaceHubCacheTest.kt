package io.aatricks.llmedge.huggingface

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class HuggingFaceHubCacheTest {
    @Test
    fun clearCacheRemovesFiles() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val root = File(context.filesDir, "hf-models/test-model")
        root.mkdirs()
        val testFile = File(root, "dummy.txt")
        testFile.writeText("hello")
        assertTrue(testFile.exists())
        HuggingFaceHub.clearCache(context)
        assertFalse(File(context.filesDir, "hf-models").exists())
    }

    @Test
    fun listCachedModelsReportsDir() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val root = File(context.filesDir, "hf-models/example-model")
        root.mkdirs()
        val cached = HuggingFaceHub.listCachedModels(context)
        assertTrue("Cache should contain at least one directory", cached.any { it.name == "example-model" })
        HuggingFaceHub.clearCache(context)
    }
}

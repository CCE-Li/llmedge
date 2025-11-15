package io.aatricks.llmedge.huggingface

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class HuggingFaceHubWanAssetsOnDiskTest {
    @Test
    fun ensureWanAssetsOnDiskUnknownModelThrows() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        assertThrows(IllegalArgumentException::class.java) {
            runBlocking {
                HuggingFaceHub.ensureWanAssetsOnDisk(
                    context = context,
                    wanModelId = "wan/this-does-not-exist",
                    preferSystemDownloader = false,
                )
            }
        }
    }
}

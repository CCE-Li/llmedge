package io.aatricks.llmedge

import android.graphics.Color
import org.junit.After
import org.junit.Before

abstract class BaseVideoIntegrationTest {
    companion object {
        @Suppress("unused")
        private val disableNativeLoadForAndroidTests = run {
            System.setProperty("llmedge.disableNativeLoad", "true")
            true
        }
    }

    @Before
    fun baseSetUp() {
        StableDiffusion.enableNativeBridgeForTests()
    }

    @After
    fun baseTearDown() {
        StableDiffusion.resetNativeBridgeForTests()
    }

    protected fun createStableDiffusion(): StableDiffusion {
        val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
        constructor.isAccessible = true
        return constructor.newInstance(1L).apply {
            updateModelMetadata(
                StableDiffusion.VideoModelMetadata(
                    architecture = "wan",
                    modelType = "t2v",
                    parameterCount = "1.3B",
                    mobileSupported = true,
                    filename = "wan-test.gguf",
                    tags = setOf("wan", "video"),
                ),
            )
        }
    }

    protected fun buildFrames(frameCount: Int, width: Int, height: Int): Array<ByteArray> {
        return Array(frameCount) { index ->
            val color = Color.rgb(index * 0x10, index * 0x10, index * 0x10)
            val bytes = ByteArray(width * height * 3)
            var i = 0
            val r = Color.red(color).toByte()
            val g = Color.green(color).toByte()
            val b = Color.blue(color).toByte()
            while (i < bytes.size) {
                bytes[i++] = r
                bytes[i++] = g
                bytes[i++] = b
            }
            bytes
        }
    }
}

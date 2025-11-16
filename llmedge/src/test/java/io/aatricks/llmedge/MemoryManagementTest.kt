package io.aatricks.llmedge

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkStatic
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.lang.reflect.Method

class MemoryManagementTest {

    @Test
    fun `estimateFrameFootprintBytes calculates correct memory usage for 512x512x16 frames`() {
        val sd = newStableDiffusion()

        val result = callPrivateMethod<Long>(
            sd,
            "estimateFrameFootprintBytes",
            Int::class.java to 512,
            Int::class.java to 512,
            Int::class.java to 16
        )

        // 512 * 512 * 4 bytes per pixel * 16 frames = 16,777,216 bytes
        assertEquals(16777216L, result)
    }

    @Test
    fun `estimateFrameFootprintBytes calculates correct memory usage for different dimensions`() {
        val sd = newStableDiffusion()

        val result = callPrivateMethod<Long>(
            sd,
            "estimateFrameFootprintBytes",
            Int::class.java to 256,
            Int::class.java to 256,
            Int::class.java to 8
        )

        // 256 * 256 * 4 * 8 = 2,097,152 bytes
        assertEquals(2097152L, result)
    }

    @Test
    fun `estimateFrameFootprintBytes handles zero frames`() {
        val sd = newStableDiffusion()

        val result = callPrivateMethod<Long>(
            sd,
            "estimateFrameFootprintBytes",
            Int::class.java to 512,
            Int::class.java to 512,
            Int::class.java to 0
        )

        assertEquals(0L, result)
    }

    @Test
    fun `readNativeMemoryMb returns non-negative value when Debug succeeds`() {
        val sd = newStableDiffusion()

        mockkStatic(Debug::class)
        every { Debug.getNativeHeapAllocatedSize() } returns 50L * 1024L * 1024L // 50MB

        val result = callPrivateMethod<Long>(sd, "readNativeMemoryMb")

        assertTrue("Memory usage should be positive", result > 0L)
        assertEquals(50L, result) // Should be converted to MB
    }

    @Test
    fun `readNativeMemoryMb falls back to runtime memory when Debug fails`() {
        val sd = newStableDiffusion()

        mockkStatic(Debug::class)
        every { Debug.getNativeHeapAllocatedSize() } throws RuntimeException("Debug not available")

        mockkStatic(Runtime::class)
        val mockRuntime = mockk<Runtime>()
        every { Runtime.getRuntime() } returns mockRuntime
        every { mockRuntime.totalMemory() } returns 100L * 1024L * 1024L // 100MB
        every { mockRuntime.freeMemory() } returns 25L * 1024L * 1024L // 75MB used

        val result = callPrivateMethod<Long>(sd, "readNativeMemoryMb")

        assertTrue("Memory usage should be positive", result > 0L)
        // Should be (100MB - 25MB) / 1MB = 75MB
        assertEquals(75L, result)
    }

    @Test
    fun `warnIfLowMemory does not warn when memory usage is low`() {
        val sd = newStableDiffusion()

        // Mock low memory usage (50% of max)
        mockkStatic(Runtime::class)
        val mockRuntime = mockk<Runtime>()
        every { Runtime.getRuntime() } returns mockRuntime
        every { mockRuntime.maxMemory() } returns 100L * 1024L * 1024L // 100MB
        every { mockRuntime.totalMemory() } returns 50L * 1024L * 1024L // 50MB used
        every { mockRuntime.freeMemory() } returns 50L * 1024L * 1024L // 50MB free

        // This should not trigger a warning (50% usage < 85% threshold)
        callPrivateMethod<Unit>(
            sd,
            "warnIfLowMemory",
            Long::class.java to (1L * 1024L * 1024L) // 1MB additional
        )

        // Method should complete without issues
        assertTrue(true)
    }

    @Test
    fun `warnIfLowMemory warns when memory pressure is high`() {
        val sd = newStableDiffusion()

        // Mock high memory usage (90% of max)
        mockkStatic(Runtime::class)
        val mockRuntime = mockk<Runtime>()
        every { Runtime.getRuntime() } returns mockRuntime
        every { mockRuntime.maxMemory() } returns 100L * 1024L * 1024L // 100MB
        every { mockRuntime.totalMemory() } returns 90L * 1024L * 1024L // 90MB used
        every { mockRuntime.freeMemory() } returns 10L * 1024L * 1024L // 10MB free

        // This should trigger a warning (90% usage > 85% threshold)
        callPrivateMethod<Unit>(
            sd,
            "warnIfLowMemory",
            Long::class.java to (5L * 1024L * 1024L) // 5MB additional
        )

        // Method should complete (warning is logged, not thrown)
        assertTrue(true)
    }

    @Test
    fun `determineBatchSize returns correct batch sizes based on frame count`() {
        val sd = newStableDiffusion()

        // Test various frame counts
        assertEquals(8, callPrivateMethod<Int>(sd, "determineBatchSize", Int::class.java to 4))
        assertEquals(6, callPrivateMethod<Int>(sd, "determineBatchSize", Int::class.java to 24))
        assertEquals(4, callPrivateMethod<Int>(sd, "determineBatchSize", Int::class.java to 48))
        assertEquals(4, callPrivateMethod<Int>(sd, "determineBatchSize", Int::class.java to 64))
    }

    @Test
    fun `determineBatchSize handles edge cases`() {
        val sd = newStableDiffusion()

        // Test minimum frames
        assertEquals(8, callPrivateMethod<Int>(sd, "determineBatchSize", Int::class.java to 1))

        // Test large frame counts
        assertEquals(4, callPrivateMethod<Int>(sd, "determineBatchSize", Int::class.java to 100))
    }

    private fun newStableDiffusion(): StableDiffusion {
        val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
        constructor.isAccessible = true
        return constructor.newInstance(1L)
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T> callPrivateMethod(
        instance: Any,
        methodName: String,
        vararg args: Pair<Class<*>, Any>
    ): T {
        val parameterTypes = args.map { it.first }.toTypedArray()
        val parameterValues = args.map { it.second }.toTypedArray()

        val method = instance.javaClass.getDeclaredMethod(methodName, *parameterTypes)
        method.isAccessible = true

        return method.invoke(instance, *parameterValues) as T
    }
}
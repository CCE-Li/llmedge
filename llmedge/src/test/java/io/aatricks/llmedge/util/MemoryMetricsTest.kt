package io.aatricks.llmedge.util

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.robolectric.annotation.Config

@Config(sdk = [34])
class MemoryMetricsTest {

    @Test
    fun `MemoryMetrics object is accessible`() {
        assertNotNull(MemoryMetrics)
    }

    @Test
    fun `MemoryMetrics Snapshot data class works correctly`() {
        val snapshot = MemoryMetrics.Snapshot(
            availSystemMemBytes = 1000000L,
            totalSystemMemBytes = 2000000L,
            lowMemory = false,
            totalPssKb = 50000,
            dalvikPssKb = 20000,
            nativePssKb = 25000,
            otherPssKb = 5000
        )

        assertEquals(1000000L, snapshot.availSystemMemBytes)
        assertEquals(2000000L, snapshot.totalSystemMemBytes)
        assertEquals(false, snapshot.lowMemory)
        assertEquals(50000, snapshot.totalPssKb)
        assertEquals(20000, snapshot.dalvikPssKb)
        assertEquals(25000, snapshot.nativePssKb)
        assertEquals(5000, snapshot.otherPssKb)
    }


}
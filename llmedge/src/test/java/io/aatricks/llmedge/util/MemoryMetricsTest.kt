package io.aatricks.llmedge.util

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
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


    @Test
    fun `snapshot and toPretty produce sane output`() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val snap = MemoryMetrics.snapshot(context)
        // Basic invariants: total >= avail; PSS non-negative
        assertTrue(snap.totalSystemMemBytes >= snap.availSystemMemBytes)
        assertTrue(snap.totalPssKb >= 0)
        assertTrue(snap.nativePssKb >= 0)
        val pretty = snap.toPretty(context)
        assertTrue(pretty.contains("System memory"))
        assertTrue(pretty.contains("App PSS"))
    }
}
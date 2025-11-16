package io.aatricks.llmedge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SchedulerConversionTest {

    @Test
    fun `schedulerToNativeSampleMethod maps EULER_A correctly`() {
        assertEquals(0, schedulerToNativeSampleMethod(StableDiffusion.Scheduler.EULER_A))
    }

    @Test
    fun `schedulerToNativeSampleMethod maps DDIM correctly`() {
        assertEquals(10, schedulerToNativeSampleMethod(StableDiffusion.Scheduler.DDIM))
    }

    @Test
    fun `schedulerToNativeSampleMethod maps DDPM to EULER_A fallback`() {
        assertEquals(0, schedulerToNativeSampleMethod(StableDiffusion.Scheduler.DDPM))
    }

    @Test
    fun `schedulerToNativeSampleMethod maps LCM correctly`() {
        assertEquals(11, schedulerToNativeSampleMethod(StableDiffusion.Scheduler.LCM))
    }

    @Test
    fun `all schedulers have valid mappings`() {
        val schedulers = StableDiffusion.Scheduler.values()
        schedulers.forEach { scheduler ->
            val result = schedulerToNativeSampleMethod(scheduler)
            assertTrue("Scheduler $scheduler should map to non-negative value", result >= 0)
        }
    }

    @Test
    fun `scheduler enum has all expected values`() {
        val expectedSchedulers = arrayOf(
            StableDiffusion.Scheduler.EULER_A,
            StableDiffusion.Scheduler.DDIM,
            StableDiffusion.Scheduler.DDPM,
            StableDiffusion.Scheduler.LCM
        )

        val actualSchedulers = StableDiffusion.Scheduler.values()
        assertEquals(expectedSchedulers.size, actualSchedulers.size)

        expectedSchedulers.forEach { expected ->
            assertTrue("Scheduler $expected should exist", actualSchedulers.contains(expected))
        }
    }
}
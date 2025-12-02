package io.aatricks.llmedge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SchedulerConversionTest {

    @Test
    fun `SampleMethod enum has correct native IDs`() {
        assertEquals(0, StableDiffusion.SampleMethod.DEFAULT.id)
        assertEquals(1, StableDiffusion.SampleMethod.EULER.id)
        assertEquals(2, StableDiffusion.SampleMethod.HEUN.id)
        assertEquals(3, StableDiffusion.SampleMethod.DPM2.id)
        assertEquals(4, StableDiffusion.SampleMethod.DPMPP2S_A.id)
        assertEquals(5, StableDiffusion.SampleMethod.DPMPP2M.id)
        assertEquals(6, StableDiffusion.SampleMethod.DPMPP2MV2.id)
        assertEquals(7, StableDiffusion.SampleMethod.IPNDM.id)
        assertEquals(8, StableDiffusion.SampleMethod.IPNDM_V.id)
        assertEquals(9, StableDiffusion.SampleMethod.LCM.id)
        assertEquals(10, StableDiffusion.SampleMethod.DDIM_TRAILING.id)
        assertEquals(11, StableDiffusion.SampleMethod.TCD.id)
        assertEquals(12, StableDiffusion.SampleMethod.EULER_A.id)
    }

    @Test
    fun `Scheduler enum has correct native IDs`() {
        assertEquals(0, StableDiffusion.Scheduler.DEFAULT.id)
        assertEquals(1, StableDiffusion.Scheduler.DISCRETE.id)
        assertEquals(2, StableDiffusion.Scheduler.KARRAS.id)
        assertEquals(3, StableDiffusion.Scheduler.EXPONENTIAL.id)
        assertEquals(4, StableDiffusion.Scheduler.AYS.id)
        assertEquals(5, StableDiffusion.Scheduler.GITS.id)
        assertEquals(6, StableDiffusion.Scheduler.SGM_UNIFORM.id)
        assertEquals(7, StableDiffusion.Scheduler.SIMPLE.id)
        assertEquals(8, StableDiffusion.Scheduler.SMOOTHSTEP.id)
    }

    @Test
    fun `SampleMethod fromId returns correct enum`() {
        assertEquals(StableDiffusion.SampleMethod.DEFAULT, StableDiffusion.SampleMethod.fromId(0))
        assertEquals(StableDiffusion.SampleMethod.EULER, StableDiffusion.SampleMethod.fromId(1))
        assertEquals(StableDiffusion.SampleMethod.EULER_A, StableDiffusion.SampleMethod.fromId(12))
        assertEquals(StableDiffusion.SampleMethod.DEFAULT, StableDiffusion.SampleMethod.fromId(999)) // Invalid ID
    }

    @Test
    fun `Scheduler fromId returns correct enum`() {
        assertEquals(StableDiffusion.Scheduler.DEFAULT, StableDiffusion.Scheduler.fromId(0))
        assertEquals(StableDiffusion.Scheduler.KARRAS, StableDiffusion.Scheduler.fromId(2))
        assertEquals(StableDiffusion.Scheduler.SMOOTHSTEP, StableDiffusion.Scheduler.fromId(8))
        assertEquals(StableDiffusion.Scheduler.DEFAULT, StableDiffusion.Scheduler.fromId(999)) // Invalid ID
    }

    @Test
    fun `all SampleMethod values have valid IDs`() {
        val sampleMethods = StableDiffusion.SampleMethod.values()
        sampleMethods.forEach { method ->
            assertTrue("SampleMethod $method should have non-negative ID", method.id >= 0)
            assertTrue("SampleMethod $method should have ID less than enum count", method.id < sampleMethods.size)
        }
    }

    @Test
    fun `all Scheduler values have valid IDs`() {
        val schedulers = StableDiffusion.Scheduler.values()
        schedulers.forEach { scheduler ->
            assertTrue("Scheduler $scheduler should have non-negative ID", scheduler.id >= 0)
            assertTrue("Scheduler $scheduler should have ID less than enum count", scheduler.id < schedulers.size)
        }
    }

    @Test
    fun `SampleMethod enum has all expected values`() {
        val expectedMethods = arrayOf(
            StableDiffusion.SampleMethod.DEFAULT,
            StableDiffusion.SampleMethod.EULER,
            StableDiffusion.SampleMethod.HEUN,
            StableDiffusion.SampleMethod.DPM2,
            StableDiffusion.SampleMethod.DPMPP2S_A,
            StableDiffusion.SampleMethod.DPMPP2M,
            StableDiffusion.SampleMethod.DPMPP2MV2,
            StableDiffusion.SampleMethod.IPNDM,
            StableDiffusion.SampleMethod.IPNDM_V,
            StableDiffusion.SampleMethod.LCM,
            StableDiffusion.SampleMethod.DDIM_TRAILING,
            StableDiffusion.SampleMethod.TCD,
            StableDiffusion.SampleMethod.EULER_A
        )

        val actualMethods = StableDiffusion.SampleMethod.values()
        assertEquals(expectedMethods.size, actualMethods.size)

        expectedMethods.forEach { expected ->
            assertTrue("SampleMethod $expected should exist", actualMethods.contains(expected))
        }
    }

    @Test
    fun `Scheduler enum has all expected values`() {
        val expectedSchedulers = arrayOf(
            StableDiffusion.Scheduler.DEFAULT,
            StableDiffusion.Scheduler.DISCRETE,
            StableDiffusion.Scheduler.KARRAS,
            StableDiffusion.Scheduler.EXPONENTIAL,
            StableDiffusion.Scheduler.AYS,
            StableDiffusion.Scheduler.GITS,
            StableDiffusion.Scheduler.SGM_UNIFORM,
            StableDiffusion.Scheduler.SIMPLE,
            StableDiffusion.Scheduler.SMOOTHSTEP
        )

        val actualSchedulers = StableDiffusion.Scheduler.values()
        assertEquals(expectedSchedulers.size, actualSchedulers.size)

        expectedSchedulers.forEach { expected ->
            assertTrue("Scheduler $expected should exist", actualSchedulers.contains(expected))
        }
    }
}
package io.aatricks.llmedge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SmolLMThinkingModeTest {
    @Test
    fun `default thinking mode is DEFAULT and toggles correctly`() {
        val smol = SmolLM()
        assertEquals(SmolLM.ThinkingMode.DEFAULT, smol.getThinkingMode())

        smol.setThinkingEnabled(false)
        assertEquals(SmolLM.ThinkingMode.DISABLED, smol.getThinkingMode())
        assertFalse(smol.isThinkingEnabled())

        smol.setThinkingEnabled(true)
        assertEquals(SmolLM.ThinkingMode.DEFAULT, smol.getThinkingMode())
        assertTrue(smol.isThinkingEnabled())
    }

    @Test
    fun `setReasoningBudget zero disables thinking`() {
        val smol = SmolLM()
        smol.setReasoningBudget(0)
        assertEquals(SmolLM.ThinkingMode.DISABLED, smol.getThinkingMode())
        assertEquals(0, smol.getReasoningBudget())
    }
}

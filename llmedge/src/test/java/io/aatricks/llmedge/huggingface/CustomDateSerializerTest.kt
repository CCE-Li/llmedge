package io.aatricks.llmedge.huggingface

import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Test
import java.time.LocalDateTime
import java.time.ZoneOffset

class CustomDateSerializerTest {

    private val json = Json

    @Test
    fun `serialize LocalDateTime to UTC string format`() {
        val dateTime = LocalDateTime.of(2023, 12, 25, 15, 30, 45, 123000000)
        val expected = "2023-12-25T15:30:45.123Z"

        val serialized = json.encodeToString(CustomDateSerializer(), dateTime)
        assertEquals("\"$expected\"", serialized)
    }

    @Test
    fun `deserialize UTC string to LocalDateTime`() {
        val jsonString = "\"2023-12-25T15:30:45.123Z\""
        val expected = LocalDateTime.of(2023, 12, 25, 15, 30, 45, 123000000)

        val deserialized = json.decodeFromString(CustomDateSerializer(), jsonString)
        assertEquals(expected, deserialized)
    }

    @Test
    fun `serialize and deserialize round trip preserves millisecond precision`() {
        val original = LocalDateTime.of(2023, 12, 25, 15, 30, 45, 123000000) // 123ms = 123000000ns

        val serialized = json.encodeToString(CustomDateSerializer(), original)
        val deserialized = json.decodeFromString(CustomDateSerializer(), serialized)

        assertEquals(original, deserialized)
    }

    @Test
    fun `serialize midnight UTC time`() {
        val dateTime = LocalDateTime.of(2024, 1, 1, 0, 0, 0, 0)
        val expected = "2024-01-01T00:00:00.000Z"

        val serialized = json.encodeToString(CustomDateSerializer(), dateTime)
        assertEquals("\"$expected\"", serialized)
    }

    @Test
    fun `deserialize midnight UTC time`() {
        val jsonString = "\"2024-01-01T00:00:00.000Z\""
        val expected = LocalDateTime.of(2024, 1, 1, 0, 0, 0, 0)

        val deserialized = json.decodeFromString(CustomDateSerializer(), jsonString)
        assertEquals(expected, deserialized)
    }
}
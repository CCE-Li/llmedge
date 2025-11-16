package io.aatricks.llmedge.huggingface

import io.ktor.client.HttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import io.ktor.serialization.kotlinx.json.json
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import java.time.LocalDateTime

class HFModelInfoTest {

    private val json = Json { ignoreUnknownKeys = true }

    @Test
    fun `getModelInfo returns parsed ModelInfo on successful response`() = runTest {
        val mockResponse = """
        {
            "_id": "test-model-id",
            "id": "test-model",
            "modelId": "test-model",
            "author": "test-author",
            "private": false,
            "disabled": false,
            "tags": ["tag1", "tag2"],
            "downloads": 1000,
            "likes": 50,
            "lastModified": "2023-12-25T15:30:45.123Z",
            "createdAt": "2023-01-01T00:00:00.000Z"
        }
        """.trimIndent()

        val mockEngine = MockEngine { request ->
            respond(
                content = mockResponse,
                status = HttpStatusCode.OK,
                headers = headersOf("Content-Type", "application/json")
            )
        }

        val client = HttpClient(mockEngine) {
            install(ContentNegotiation) {
                json(json)
            }
        }

        val modelInfo = HFModelInfo(client)
        val result = modelInfo.getModelInfo("test-model")

        assertEquals("test-model-id", result.idInternal)
        assertEquals("test-model", result.id)
        assertEquals("test-model", result.modelId)
        assertEquals("test-author", result.author)
        assertEquals(false, result.isPrivate)
        assertEquals(false, result.disabled)
        assertEquals(listOf("tag1", "tag2"), result.tags)
        assertEquals(1000L, result.numDownloads)
        assertEquals(50L, result.numLikes)
        assertEquals(LocalDateTime.of(2023, 12, 25, 15, 30, 45, 123000000), result.lastModified)
        assertEquals(LocalDateTime.of(2023, 1, 1, 0, 0, 0, 0), result.createdAt)
    }

    @Test
    fun `getModelInfo includes authorization header when token provided`() = runTest {
        val mockEngine = MockEngine { request ->
            assertEquals("Bearer test-token", request.headers["Authorization"])
            respond(
                content = """{"_id": "test-id", "id": "test", "modelId": "test", "author": "test", "private": false, "disabled": false, "tags": [], "downloads": 0, "likes": 0, "lastModified": "2023-01-01T00:00:00.000Z", "createdAt": "2023-01-01T00:00:00.000Z"}""",
                status = HttpStatusCode.OK,
                headers = headersOf("Content-Type", "application/json")
            )
        }

        val client = HttpClient(mockEngine) {
            install(ContentNegotiation) {
                json(json)
            }
        }

        val modelInfo = HFModelInfo(client)
        modelInfo.getModelInfo("test-model", "test-token")
    }

    @Test
    fun `getModelInfo throws exception on 404 response`() = runTest {
        val mockEngine = MockEngine { request ->
            respond(
                content = "Not Found",
                status = HttpStatusCode.NotFound
            )
        }

        val client = HttpClient(mockEngine) {
            install(ContentNegotiation) {
                json(json)
            }
        }

        val modelInfo = HFModelInfo(client)
        try {
            modelInfo.getModelInfo("nonexistent-model")
            assertEquals("Expected exception to be thrown", "but no exception was thrown")
        } catch (e: IllegalArgumentException) {
            // Expected
            assertEquals("Hugging Face model 'nonexistent-model' not found or unavailable", e.message)
        }
    }

    @Test
    fun `getModelInfo throws IllegalArgumentException on 403 response`() = runTest {
        val mockEngine = MockEngine { request ->
            respond(
                content = "Forbidden",
                status = HttpStatusCode.Forbidden
            )
        }

        val client = HttpClient(mockEngine) {
            install(ContentNegotiation) {
                json(json)
            }
        }

        val modelInfo = HFModelInfo(client)
        val exception = assertThrows(IllegalArgumentException::class.java) {
            runTest { modelInfo.getModelInfo("private-model") }
        }

        assertEquals("Hugging Face model 'private-model' not found or unavailable", exception.message)
    }

    @Test
    fun `ModelInfo data class handles all fields correctly`() {
        val createdAt = LocalDateTime.of(2023, 1, 1, 12, 0, 0, 0)
        val lastModified = LocalDateTime.of(2023, 12, 31, 23, 59, 59, 999000000)

        val modelInfo = HFModelInfo.ModelInfo(
            idInternal = "_id123",
            id = "model-id",
            modelId = "full/model/id",
            author = "test-author",
            isPrivate = true,
            disabled = false,
            tags = listOf("tag1", "tag2", "tag3"),
            numDownloads = 5000L,
            numLikes = 250L,
            lastModified = lastModified,
            createdAt = createdAt
        )

        assertEquals("_id123", modelInfo.idInternal)
        assertEquals("model-id", modelInfo.id)
        assertEquals("full/model/id", modelInfo.modelId)
        assertEquals("test-author", modelInfo.author)
        assertEquals(true, modelInfo.isPrivate)
        assertEquals(false, modelInfo.disabled)
        assertEquals(listOf("tag1", "tag2", "tag3"), modelInfo.tags)
        assertEquals(5000L, modelInfo.numDownloads)
        assertEquals(250L, modelInfo.numLikes)
        assertEquals(lastModified, modelInfo.lastModified)
        assertEquals(createdAt, modelInfo.createdAt)
    }
}
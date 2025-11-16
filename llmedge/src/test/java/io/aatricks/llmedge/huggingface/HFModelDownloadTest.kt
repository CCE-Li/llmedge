package io.aatricks.llmedge.huggingface

import io.ktor.client.HttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.client.engine.mock.respondError
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class HFModelDownloadTest {
    @Test
    fun `downloadModelFile writes file to destination`() {
        runBlocking {
        val bytes = "hello".toByteArray()
        val engine = MockEngine { request ->
            respond(content = bytes, status = HttpStatusCode.OK, headers = headersOf(HttpHeaders.ContentLength, bytes.size.toString()))
        }
        val client = HttpClient(engine)
        val dl = HFModelDownload(client)

        val tmpDir = createTempDir(prefix = "hf-download-test")
        val destination = File(tmpDir, "model.gguf")
        val res = dl.downloadModelFile("user/repo", "main", "model.gguf", destination, token = null, onProgress = null)
        assertTrue(res.exists())
        val content = res.readBytes()
        assertArrayEquals(bytes, content)
        tmpDir.deleteRecursively()
        }
    }

    @Test
    fun `downloadModelFile throws on non-success responses`() {
        runBlocking {
        val engine = MockEngine { request ->
            respondError(HttpStatusCode.NotFound, "Not found")
        }
        val client = HttpClient(engine)
        val dl = HFModelDownload(client)

        val tmpDir = createTempDir(prefix = "hf-download-test")
        val destination = File(tmpDir, "model.gguf")
        val ex = org.junit.Assert.assertThrows(IllegalStateException::class.java) {
            runBlocking {
                dl.downloadModelFile("user/repo", "main", "model.gguf", destination, token = null, onProgress = null)
            }
        }
        assertTrue(ex.message?.contains("Not found") == true)
        tmpDir.deleteRecursively()
        }
    }

    @Test
    fun `downloadModelFile resumes when server returns 206`() {
        runBlocking {
        val fullContent = "abcdef".toByteArray()
        val totalLen = fullContent.size
        val engine = MockEngine { request ->
            val rangeHeader = request.headers[HttpHeaders.Range]
            if (rangeHeader != null && rangeHeader.startsWith("bytes=2-")) {
                // Return partial content (from offset 2 -> 'cdef')
                val chunk = fullContent.copyOfRange(2, fullContent.size)
                respond(content = chunk, status = HttpStatusCode.PartialContent, headers = headersOf(HttpHeaders.ContentLength, totalLen.toString()))
            } else {
                respond(content = fullContent, status = HttpStatusCode.OK, headers = headersOf(HttpHeaders.ContentLength, totalLen.toString()))
            }
        }
        val client = HttpClient(engine)
        val dl = HFModelDownload(client)

        val tmpDir = createTempDir(prefix = "hf-download-test")
        val destination = File(tmpDir, "model.gguf")
        val tempPart = File(destination.parentFile, destination.name + ".part")
        tempPart.parentFile?.mkdirs()
        tempPart.writeBytes(fullContent.copyOfRange(0, 2)) // write 'ab'

        val res = dl.downloadModelFile("user/repo", "main", "model.gguf", destination, token = null, onProgress = null)
        assertTrue(res.exists())
        assertEquals(totalLen.toLong(), res.length())
        val content = res.readBytes()
        assertArrayEquals(fullContent, content)
        tmpDir.deleteRecursively()
        }
    }

    @Test
    fun `downloadModelFile restarts when server ignores range and returns 200`() {
        runBlocking {
        val fullContent = "zx".toByteArray()
        val totalLen = fullContent.size
        val engine = MockEngine { request ->
            // Simulate server ignoring Range header and returning full content
            respond(content = fullContent, status = HttpStatusCode.OK, headers = headersOf(HttpHeaders.ContentLength, totalLen.toString()))
        }
        val client = HttpClient(engine)
        val dl = HFModelDownload(client)

        val tmpDir = createTempDir(prefix = "hf-download-test")
        val destination = File(tmpDir, "model.gguf")
        val tempPart = File(destination.parentFile, destination.name + ".part")
        tempPart.parentFile?.mkdirs()
        tempPart.writeBytes("old".toByteArray()) // write old content

        val res = dl.downloadModelFile("user/repo", "main", "model.gguf", destination, token = null, onProgress = null)
        assertTrue(res.exists())
        val content = res.readBytes()
        assertArrayEquals(fullContent, content)
        tmpDir.deleteRecursively()
        }
    }
}

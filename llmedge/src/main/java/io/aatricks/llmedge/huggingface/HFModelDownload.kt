package io.aatricks.llmedge.huggingface

import io.ktor.client.HttpClient
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsChannel
import io.ktor.client.statement.bodyAsText
import io.ktor.http.HttpHeaders
import io.ktor.http.URLProtocol
import io.ktor.http.isSuccess
import io.ktor.http.path
import io.ktor.utils.io.errors.IOException
import io.ktor.utils.io.readAvailable
import io.ktor.utils.io.jvm.javaio.toInputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

internal class HFModelDownload(
    private val client: HttpClient,
) {
    suspend fun downloadModelFile(
        modelId: String,
        revision: String,
        filePath: String,
        destination: File,
        token: String?,
        onProgress: ((downloaded: Long, totalBytes: Long?) -> Unit)?,
    ): File {
        destination.parentFile?.mkdirs()
        val tempFile = File(destination.parentFile, "${destination.name}.part")
        if (destination.exists()) {
            destination.delete()
        }
    val resumeStart = if (tempFile.exists()) tempFile.length() else 0L
    val response = requestFile(modelId, revision, filePath, token, resumeStart)
    val expectedLength = response.headers[HttpHeaders.ContentLength]?.toLongOrNull()

        if (!response.status.isSuccess()) {
            val errorBody = runCatching { response.bodyAsText() }.getOrNull()
            throw IllegalStateException(
                buildString {
                    append("Failed to download $filePath from $modelId (${response.status})")
                    if (!errorBody.isNullOrBlank()) {
                        append(": ")
                        append(errorBody)
                    }
                }
            )
        }

        var downloaded = resumeStart
        // If server returned full file despite providing a Range header, start fresh
        val responseStatusValue = response.status.value
        if (resumeStart > 0L && responseStatusValue == 200) {
            // Server ignored range — start over
            tempFile.delete()
            downloaded = 0L
            onProgress?.invoke(downloaded, expectedLength)
        }
        onProgress?.invoke(downloaded, expectedLength)

        val channel = response.bodyAsChannel()

        try {
            // Use InputStream adapter to avoid Ktor allocating large ByteBuffer objects on the JVM heap
            val inputStream = channel.toInputStream()
            withContext(Dispatchers.IO) {
                inputStream.use { ins ->
                    // If resuming, open in append mode
                    tempFile.outputStream().use { outputStream ->
                        // If resumeStart > 0, we are appending; else overwrite
                        if (resumeStart > 0L) {
                            outputStream.channel.position(resumeStart)
                        }
                        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                        while (true) {
                            val read = ins.read(buffer)
                            if (read <= 0) break
                            outputStream.write(buffer, 0, read)
                            downloaded += read
                            onProgress?.invoke(downloaded, expectedLength)
                        }
                        outputStream.flush()
                    }
                }
            }
        } catch (oom: OutOfMemoryError) {
            // Attempt graceful cleanup and rethrow to avoid leaving partial files behind
            try { channel.cancel(cause = oom) } catch (_: Throwable) {}
            tempFile.delete()
            throw oom
        } catch (ioe: IOException) {
            try { channel.cancel(cause = ioe) } catch (_: Throwable) {}
            tempFile.delete()
            throw ioe
        } finally {
            try { channel.cancel(cause = null) } catch (_: Throwable) {}
        }

        if (tempFile.renameTo(destination).not()) {
            tempFile.delete()
            throw IllegalStateException("Failed to move temporary file for $filePath")
        }
        return destination
    }

    private suspend fun requestFile(
        modelId: String,
        revision: String,
        filePath: String,
        token: String?,
        rangeStart: Long = 0L,
    ): HttpResponse {
        return client.get {
            url {
                protocol = URLProtocol.HTTPS
                host = "huggingface.co"
                val modelSegments = modelId.trim('/').split('/')
                val fileSegments = filePath.trim('/').split('/')
                path(*modelSegments.toTypedArray(), "resolve", revision, *fileSegments.toTypedArray())
            }
            token?.let { header(HttpHeaders.Authorization, "Bearer $it") }
            if (rangeStart > 0L) {
                header(HttpHeaders.Range, "bytes=$rangeStart-")
            }
        }
    }

    companion object {
        private const val DEFAULT_BUFFER_SIZE = 2 * 1024
    }
}

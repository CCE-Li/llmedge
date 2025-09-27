/*
 * Copyright (C) 2025 Shubham Panchal
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
        if (tempFile.exists()) {
            tempFile.delete()
        }
        if (destination.exists()) {
            destination.delete()
        }
    val response = requestFile(modelId, revision, filePath, token)
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

        onProgress?.invoke(0L, expectedLength)

        val channel = response.bodyAsChannel()

        try {
            withContext(Dispatchers.IO) {
                tempFile.outputStream().use { outputStream ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    var downloaded = 0L
                    while (!channel.isClosedForRead) {
                        val read = channel.readAvailable(buffer)
                        if (read == -1) break
                        if (read == 0) continue
                        outputStream.write(buffer, 0, read)
                        downloaded += read
                        onProgress?.invoke(downloaded, expectedLength)
                    }
                    outputStream.flush()
                }
            }
        } catch (ioe: IOException) {
            channel.cancel(cause = ioe)
            tempFile.delete()
            throw ioe
        } finally {
            channel.cancel(cause = null)
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
        }
    }

    companion object {
        private const val DEFAULT_BUFFER_SIZE = 8 * 1024
    }
}

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
import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.http.HttpHeaders
import io.ktor.http.isSuccess
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import java.time.LocalDateTime

internal class HFModelInfo(
    private val client: HttpClient,
) {
    @Serializable
    data class ModelInfo(
        @SerialName("_id") val idInternal: String,
        val id: String,
        val modelId: String,
        val author: String,
        @SerialName("private") val isPrivate: Boolean,
        val disabled: Boolean,
        val tags: List<String>,
        @SerialName("downloads") val numDownloads: Long,
        @SerialName("likes") val numLikes: Long,
        @Serializable(with = CustomDateSerializer::class) val lastModified: LocalDateTime,
        @Serializable(with = CustomDateSerializer::class) val createdAt: LocalDateTime,
    )

    suspend fun getModelInfo(modelId: String, token: String? = null): ModelInfo {
        val response =
            client.get(HFEndpoints.modelSpecsEndpoint(modelId)) {
                token?.let { header(HttpHeaders.Authorization, "Bearer $it") }
            }
        if (!response.status.isSuccess()) {
            throw IllegalArgumentException("Hugging Face model '$modelId' not found or unavailable")
        }
        return response.body()
    }
}

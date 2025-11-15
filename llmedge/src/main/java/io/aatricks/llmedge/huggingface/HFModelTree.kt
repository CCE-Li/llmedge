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

internal class HFModelTree(
    private val client: HttpClient,
) {
    @Serializable
    data class HFModelFile(
        val type: String? = null,
        val oid: String? = null,
        val size: Long? = null,
        @SerialName("rfilename") val path: String,
        @SerialName("lfs") val lfs: LfsMetadata? = null,
    ) {
        @Serializable
        data class LfsMetadata(
            val oid: String,
            val size: Long,
        )
    }

    suspend fun getModelFileTree(
        modelId: String,
        revision: String,
        token: String? = null,
    ): List<HFModelFile> {
        // Use model specs endpoint which returns all files in 'siblings'
        val response =
            client.get(HFEndpoints.modelSpecsEndpoint(modelId)) {
                token?.let { header(HttpHeaders.Authorization, "Bearer $it") }
            }
        if (!response.status.isSuccess()) {
            throw IllegalArgumentException("Hugging Face model '$modelId' not found")
        }
        @Serializable
        data class ModelSpecs(val siblings: List<HFModelFile>)
        val specs: ModelSpecs = response.body()
        return specs.siblings
    }
}

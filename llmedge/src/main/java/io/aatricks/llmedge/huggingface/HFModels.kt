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
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.json.Json
import okhttp3.ConnectionPool
import okhttp3.Dispatcher
import okhttp3.Protocol
import java.util.concurrent.TimeUnit

internal object HFModels {
    val client: HttpClient =
        HttpClient(OkHttp) {
            engine {
                config {
                    dispatcher(
                        Dispatcher().apply {
                            maxRequests = 2
                            maxRequestsPerHost = 1
                        }
                    )
                    protocols(listOf(Protocol.HTTP_1_1))
                    connectionPool(ConnectionPool(1, 5, TimeUnit.SECONDS))
                    retryOnConnectionFailure(false)
                }
            }
            install(ContentNegotiation) {
                json(Json { ignoreUnknownKeys = true })
            }
        }

    fun info(): HFModelInfo = HFModelInfo(client)

    fun tree(): HFModelTree = HFModelTree(client)

    fun search(): HFModelSearch = HFModelSearch(client)

    fun download(): HFModelDownload = HFModelDownload(client)
}

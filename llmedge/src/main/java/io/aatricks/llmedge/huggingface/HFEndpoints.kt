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

internal object HFEndpoints {
    private const val HF_BASE_ENDPOINT = "https://huggingface.co/api/models"

    val listModelsEndpoint: () -> String = { HF_BASE_ENDPOINT }

    val modelTreeEndpoint: (String, String) -> String = { modelId, revision ->
        "$HF_BASE_ENDPOINT/$modelId/tree/$revision"
    }

    val modelSpecsEndpoint: (String) -> String = { modelId -> "$HF_BASE_ENDPOINT/$modelId" }

    val fileDownloadEndpoint: (String, String, String) -> String =
        { modelId, revision, filePath ->
            "https://huggingface.co/$modelId/resolve/$revision/$filePath"
        }
}

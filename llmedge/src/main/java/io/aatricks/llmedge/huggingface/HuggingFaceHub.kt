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

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * High-level helper to discover and download GGUF models from Hugging Face.
 */
object HuggingFaceHub {
    data class ModelDownloadResult(
        val requestedModelId: String,
        val requestedRevision: String,
        val modelId: String,
        val revision: String,
        val file: File,
        val fileInfo: ModelFileMetadata,
        val fromCache: Boolean,
        val aliasApplied: Boolean,
    )

    data class ModelFileMetadata(
        val path: String,
        val sizeBytes: Long,
        val sha256: String?,
    )

    suspend fun ensureModelOnDisk(
        context: Context,
        modelId: String,
        revision: String = "main",
        preferredQuantizations: List<String> = DEFAULT_QUANTIZATION_PRIORITIES,
        filename: String? = null,
        token: String? = null,
        forceDownload: Boolean = false,
        preferSystemDownloader: Boolean = false,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): ModelDownloadResult =
        withContext(Dispatchers.IO) {
            val destinationRoot = File(context.filesDir, DEFAULT_MODELS_DIRECTORY)
            ensureModelOnDisk(
                destinationRoot = destinationRoot,
                modelId = modelId,
                revision = revision,
                preferredQuantizations = preferredQuantizations,
                filename = filename,
                token = token,
                forceDownload = forceDownload,
                preferSystemDownloader = preferSystemDownloader,
                systemDownloadContext = if (preferSystemDownloader) context else null,
                onProgress = onProgress,
            )
        }

    suspend fun ensureModelOnDisk(
        destinationRoot: File,
        modelId: String,
        revision: String = "main",
        preferredQuantizations: List<String> = DEFAULT_QUANTIZATION_PRIORITIES,
        filename: String? = null,
        token: String? = null,
        forceDownload: Boolean = false,
        preferSystemDownloader: Boolean = false,
        systemDownloadContext: Context? = null,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): ModelDownloadResult {
        val resolved = resolveModelReference(modelId, revision)
        val treeClient = HFModels.tree()
        val files = treeClient.getModelFileTree(resolved.modelId, resolved.revision, token)
        val modelFile = selectFile(files, filename, preferredQuantizations)
            ?: throw IllegalArgumentException("No GGUF file found for '$modelId' (revision '$revision')")

        val sanitizedModelId = sanitize(resolved.modelId)
        val revisionDir = File(destinationRoot, "${sanitizedModelId}/${resolved.revision}")
        val targetName = modelFile.path.substringAfterLast('/')
        val targetFile = File(revisionDir, targetName)
        val expectedSize = modelFile.lfs?.size ?: modelFile.size

        if (!forceDownload && targetFile.exists() && targetFile.length() == expectedSize) {
            return ModelDownloadResult(
                requestedModelId = modelId,
                requestedRevision = revision,
                modelId = resolved.modelId,
                revision = resolved.revision,
                file = targetFile,
                fileInfo = modelFile.toMetadata(),
                fromCache = true,
                aliasApplied = resolved.aliasApplied,
            )
        }

        revisionDir.mkdirs()
        val downloadUrl = HFEndpoints.fileDownloadEndpoint(resolved.modelId, resolved.revision, modelFile.path)

        val useSystemDownloader = preferSystemDownloader && systemDownloadContext != null

        if (useSystemDownloader) {
            val tempDir = systemDownloadContext?.getExternalFilesDir("hf-downloads")
            if (tempDir == null) {
                Log.w(LOG_TAG, "External downloads directory unavailable; falling back to in-app streaming")
            } else {
                val tempFile = File(tempDir, "${sanitize(resolved.modelId)}-${System.currentTimeMillis()}.tmp")
                val downloaded = SystemDownload.download(
                    context = systemDownloadContext,
                    url = downloadUrl,
                    token = token,
                    destination = tempFile,
                    displayName = targetName,
                    onProgress = onProgress,
                )
                if (targetFile.exists()) {
                    targetFile.delete()
                }
                downloaded.copyTo(targetFile, overwrite = true)
                downloaded.delete()
            }
        }

        if (!targetFile.exists()) {
            val downloader = HFModels.download()
            downloader.downloadModelFile(
                modelId = resolved.modelId,
                revision = resolved.revision,
                filePath = modelFile.path,
                destination = targetFile,
                token = token,
                onProgress = onProgress,
            )
        }

        if (expectedSize > 0 && targetFile.length() != expectedSize) {
            targetFile.delete()
            throw IllegalStateException("Downloaded file size mismatch for ${modelFile.path}")
        }

        return ModelDownloadResult(
            requestedModelId = modelId,
            requestedRevision = revision,
            modelId = resolved.modelId,
            revision = resolved.revision,
            file = targetFile,
            fileInfo = modelFile.toMetadata(),
            fromCache = false,
            aliasApplied = resolved.aliasApplied,
        )
    }

    fun sanitize(modelId: String): String = modelId.replace("/", "_")

    private fun resolveModelReference(modelId: String, revision: String): ResolvedModel {
        val normalized = normalizeModelId(modelId)
        MODEL_ALIASES[normalized]?.let { alias ->
            val resolvedRevision = alias.revision ?: revision
            return ResolvedModel(
                requestedModelId = modelId,
                requestedRevision = revision,
                modelId = alias.modelId,
                revision = resolvedRevision,
                aliasApplied = true,
            )
        }
        return ResolvedModel(
            requestedModelId = modelId,
            requestedRevision = revision,
            modelId = modelId,
            revision = revision,
            aliasApplied = false,
        )
    }

    private fun normalizeModelId(modelId: String): String =
        modelId
            .trim()
            .removePrefix("https://huggingface.co/")
            .removePrefix("hf://")
            .removeSuffix(".git")
            .lowercase()

    private fun selectFile(
        files: List<HFModelTree.HFModelFile>,
        filename: String?,
        preferredQuantizations: List<String>,
    ): HFModelTree.HFModelFile? {
        val candidates = files.filter { it.type == "file" && it.path.endsWith(".gguf", ignoreCase = true) }
        if (candidates.isEmpty()) {
            return null
        }

        if (!filename.isNullOrEmpty()) {
            candidates.firstOrNull { it.path.equals(filename, ignoreCase = true) }
                ?.let { return it }
            candidates.firstOrNull { it.path.endsWith(filename, ignoreCase = true) }
                ?.let { return it }
        }

        preferredQuantizations.forEach { marker ->
            candidates.firstOrNull { it.path.contains(marker, ignoreCase = true) }?.let { return it }
        }

        return candidates.minByOrNull { it.size }
    }

    private const val DEFAULT_MODELS_DIRECTORY = "hf-models"
    private const val LOG_TAG = "HuggingFaceHub"

    val DEFAULT_QUANTIZATION_PRIORITIES: List<String> = listOf(
        "Q2_K",
        "Q3_K_S",
        "Q3_K_M",
        "Q3_K_L",
        "Q4_K_S",
        "Q4_K_M",
        "Q4_K",
        "Q4_0",
        "Q5_K_S",
        "Q5_K_M",
        "Q5_K",
        "Q5_0",
        "Q8_0",
        ".gguf",
    )

    private fun HFModelTree.HFModelFile.toMetadata(): ModelFileMetadata =
        ModelFileMetadata(
            path = path,
            sizeBytes = lfs?.size ?: size,
            sha256 = lfs?.oid ?: oid,
        )

    private data class ResolvedModel(
        val requestedModelId: String,
        val requestedRevision: String,
        val modelId: String,
        val revision: String,
        val aliasApplied: Boolean,
    )

    private data class ModelAlias(
        val modelId: String,
        val revision: String? = null,
    )

    private val MODEL_ALIASES: Map<String, ModelAlias> = mapOf(
        "bartowski/tinyllama-1.1b-chat-v1.0-gguf" to ModelAlias("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"),
        "tinyllama-1.1b-chat-v1.0-gguf" to ModelAlias("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"),
    )
}

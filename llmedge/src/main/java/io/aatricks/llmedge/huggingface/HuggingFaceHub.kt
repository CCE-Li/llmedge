/*
 * Copyright (C) 2025 Aatricks
 *
 * Licensed under the GNU General Public License v3.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
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
        val expectedSize = modelFile.lfs?.size ?: modelFile.size ?: 0L

        if (!forceDownload && targetFile.exists() && expectedSize > 0 && targetFile.length() == expectedSize) {
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
            try {
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
            } catch (t: Throwable) {
                Log.w(LOG_TAG, "System download failed (${t.message}) - falling back to in-app downloader")
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

        if (expectedSize != null && expectedSize > 0 && targetFile.length() != expectedSize) {
            targetFile.delete()
            throw IllegalStateException("Downloaded file size mismatch for ${modelFile.path}")
        }

        // If file metadata includes sha256, validate to avoid corrupted downloads
        modelFile.lfs?.oid?.let { expectedSha ->
            try {
                val actualSha = computeSha256(targetFile)
                if (!actualSha.equals(expectedSha, ignoreCase = true)) {
                    targetFile.delete()
                    throw IllegalStateException("Downloaded file sha mismatch for ${modelFile.path}")
                }
            } catch (_: Throwable) {
                // If hashing fails for any reason, prefer to proceed rather than block; the next
                // consumer will fail later. We already validated size above.
            }
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

    suspend fun ensureWanAssetsOnDisk(
        context: Context,
        wanModelId: String,
        preferSystemDownloader: Boolean = true,
        token: String? = null,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): Triple<ModelDownloadResult, ModelDownloadResult?, ModelDownloadResult?> = withContext(Dispatchers.IO) {
        var registryEntry = WanModelRegistry.findById(context, wanModelId)
        if (registryEntry == null) {
            // Try prefix match (e.g. 'wan/Wan2.1-T2V-1.3B' vs 'Wan2.1-T2V-1.3B')
            val trimmed = wanModelId.removePrefix("wan/")
            registryEntry = WanModelRegistry.findByModelIdPrefix(context, trimmed)
        }
        registryEntry ?: throw IllegalArgumentException("Unknown Wan model $wanModelId")

        val modelRes = ensureModelOnDisk(
            context = context,
            modelId = registryEntry.modelId,
            filename = registryEntry.filename,
            token = token,
            preferSystemDownloader = preferSystemDownloader,
            onProgress = onProgress,
        )

        val vaeRes = registryEntry.vaeFilename?.let { vaeName ->
            ensureRepoFileOnDisk(
                context = context,
                modelId = registryEntry.modelId,
                filename = vaeName,
                allowedExtensions = listOf(".safetensors", ".pt"),
                token = token,
                preferSystemDownloader = preferSystemDownloader,
                onProgress = onProgress,
            )
        }

        val t5Res = registryEntry.t5ModelId?.let { t5ModelId ->
            val t5Filename = registryEntry.t5Filename
                ?: throw IllegalArgumentException("Registry entry for $wanModelId missing t5 filename")
            ensureRepoFileOnDisk(
                context = context,
                modelId = t5ModelId,
                filename = t5Filename,
                allowedExtensions = listOf(".gguf"),
                token = token,
                preferSystemDownloader = preferSystemDownloader,
                onProgress = onProgress,
            )
        }

        Triple(modelRes, vaeRes, t5Res)
    }

    /* Cache utilities */
    fun clearCache(context: Context) {
        val root = File(context.filesDir, DEFAULT_MODELS_DIRECTORY)
        if (root.exists()) {
            root.deleteRecursively()
        }
    }

    fun listCachedModels(context: Context): List<File> {
        val root = File(context.filesDir, DEFAULT_MODELS_DIRECTORY)
        return if (root.exists() && root.isDirectory) {
            root.listFiles()?.filter { it.isDirectory }?.toList() ?: emptyList()
        } else emptyList()
    }

    /**
     * Ensure an arbitrary file from a Hugging Face model repo is present on disk.
     * This is useful for files that are not GGUF models (for example VAE safetensors
     * or other checkpoints) where we want to specify an explicit filename or
     * fall back to a heuristic (largest file with allowed extensions).
     */
    suspend fun ensureRepoFileOnDisk(
        context: Context,
        modelId: String,
        revision: String = "main",
        filename: String? = null,
        allowedExtensions: List<String> = listOf(".safetensors", ".pt", ".ckpt", ".gguf", ".bin"),
        token: String? = null,
        forceDownload: Boolean = false,
        preferSystemDownloader: Boolean = false,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): ModelDownloadResult = withContext(Dispatchers.IO) {
        val destinationRoot = File(context.filesDir, DEFAULT_MODELS_DIRECTORY)
        ensureRepoFileOnDisk(
            destinationRoot = destinationRoot,
            modelId = modelId,
            revision = revision,
            filename = filename,
            allowedExtensions = allowedExtensions,
            token = token,
            forceDownload = forceDownload,
            preferSystemDownloader = preferSystemDownloader,
            systemDownloadContext = if (preferSystemDownloader) context else null,
            onProgress = onProgress,
        )
    }

    suspend fun ensureRepoFileOnDisk(
        destinationRoot: File,
        modelId: String,
        revision: String = "main",
        filename: String? = null,
        allowedExtensions: List<String> = listOf(".safetensors", ".pt", ".ckpt", ".gguf", ".bin"),
        token: String? = null,
        forceDownload: Boolean = false,
        preferSystemDownloader: Boolean = false,
        systemDownloadContext: Context? = null,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)? = null,
    ): ModelDownloadResult {
        val resolved = resolveModelReference(modelId, revision)
        val treeClient = HFModels.tree()
        val files = treeClient.getModelFileTree(resolved.modelId, resolved.revision, token)
        
        // Debug: log file count and sample paths
        android.util.Log.d("HuggingFaceHub", "getModelFileTree returned ${files.size} items for ${resolved.modelId}")
        files.take(5).forEach { android.util.Log.d("HuggingFaceHub", "  Sample file: ${it.path} (type=${it.type})") }

        // If a filename is provided, try to find it (exact or suffix match).
        val fileMatch = if (!filename.isNullOrEmpty()) {
            android.util.Log.d("HuggingFaceHub", "Searching for filename: $filename")
            files.firstOrNull { (it.type == "file" || it.type == null) && (it.path.equals(filename, ignoreCase = true) || it.path.endsWith(filename, ignoreCase = true)) }
        } else null

        val candidate = fileMatch ?: run {
            // Choose the largest file that ends with one of the allowed extensions
            val candidates = files.filter { (it.type == "file" || it.type == null) && allowedExtensions.any { ext -> it.path.endsWith(ext, ignoreCase = true) } }
            candidates.maxByOrNull { it.lfs?.size ?: it.size ?: 0L }
        }

        val modelFile = candidate ?: throw IllegalArgumentException("No file found for '$modelId' matching ${filename ?: allowedExtensions}")

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
            try {
                val tempDir = systemDownloadContext?.getExternalFilesDir("hf-downloads")
                if (tempDir == null) {
                    Log.w(LOG_TAG, "External downloads directory unavailable; falling back to in-app streaming")
                } else {
                    val tempFile = File(tempDir, "${sanitize(resolved.modelId)}-${System.currentTimeMillis()}.tmp")
                    val downloaded = SystemDownload.download(
                        context = systemDownloadContext!!,
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
            } catch (t: Throwable) {
                Log.w(LOG_TAG, "System download failed (${t.message}) - falling back to in-app downloader")
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

        if (expectedSize != null && expectedSize > 0 && targetFile.length() != expectedSize) {
            targetFile.delete()
            throw IllegalStateException("Downloaded file size mismatch for ${modelFile.path}")
        }

        modelFile.lfs?.oid?.let { expectedSha ->
            try {
                val actualSha = computeSha256(targetFile)
                if (!actualSha.equals(expectedSha, ignoreCase = true)) {
                    targetFile.delete()
                    throw IllegalStateException("Downloaded file sha mismatch for ${modelFile.path}")
                }
            } catch (_: Throwable) {
            }
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
        return ResolvedModel(
            requestedModelId = modelId,
            requestedRevision = revision,
            modelId = modelId,
            revision = revision,
            aliasApplied = false,
        )
    }

    private fun selectFile(
        files: List<HFModelTree.HFModelFile>,
        filename: String?,
        preferredQuantizations: List<String>,
    ): HFModelTree.HFModelFile? {
        // NOTE: The model specs endpoint (siblings list) does not populate a 'type' field.
        // Treat null type as a file entry.
        val candidates = files.filter { (it.type == "file" || it.type == null) && it.path.endsWith(".gguf", ignoreCase = true) }
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

        return candidates.minByOrNull { it.size ?: it.lfs?.size ?: Long.MAX_VALUE }
    }

    private const val DEFAULT_MODELS_DIRECTORY = "hf-models"
    private const val LOG_TAG = "HuggingFaceHub"

    val DEFAULT_QUANTIZATION_PRIORITIES: List<String> = listOf(
        "Q4_K_M",
        "Q4_K",
        "Q4_K_S",
        "Q4_0",
        "Q3_K_L",
        "Q5_K_S",
        "Q3_K_M",
        "Q5_K_M",
        "Q3_K_S",
        "Q2_K",
        "Q5_K",
        "Q5_0",
        "Q8_0",
        ".gguf",
    )

    private fun HFModelTree.HFModelFile.toMetadata(): ModelFileMetadata =
        ModelFileMetadata(
            path = path,
            sizeBytes = lfs?.size ?: size ?: 0L,
            sha256 = lfs?.oid ?: oid,
        )

    private data class ResolvedModel(
        val requestedModelId: String,
        val requestedRevision: String,
        val modelId: String,
        val revision: String,
        val aliasApplied: Boolean,
    )

    private fun computeSha256(file: File): String {
        try {
            val md = java.security.MessageDigest.getInstance("SHA-256")
            file.inputStream().use { fis ->
                val buffer = ByteArray(8 * 1024)
                var bytesRead = fis.read(buffer)
                while (bytesRead >= 0) {
                    md.update(buffer, 0, bytesRead)
                    bytesRead = fis.read(buffer)
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        } catch (t: Throwable) {
            throw t
        }
    }
}

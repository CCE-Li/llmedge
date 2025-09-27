package io.aatricks.llmedge.huggingface

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import java.io.File
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

internal object SystemDownload {

    suspend fun download(
        context: Context,
        url: String,
        token: String?,
        destination: File,
        displayName: String,
        onProgress: ((downloaded: Long, total: Long?) -> Unit)?,
    ): File {
        val downloadManager = context.getSystemService(DownloadManager::class.java)
            ?: throw IllegalStateException("DownloadManager service not available")

        destination.parentFile?.mkdirs()
        if (destination.exists()) {
            destination.delete()
        }

        val request = DownloadManager.Request(Uri.parse(url))
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(true)
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            .setDestinationUri(Uri.fromFile(destination))
            .setTitle(displayName)

        token?.let { request.addRequestHeader("Authorization", "Bearer $it") }

        val downloadId = downloadManager.enqueue(request)
        val query = DownloadManager.Query().setFilterById(downloadId)

        return suspendCancellableCoroutine { cont ->
            val monitorScope = CoroutineScope(Dispatchers.IO + Job())
            val monitorJob = monitorScope.launch {
                while (isActive && cont.isActive) {
                    try {
                        downloadManager.query(query).use { cursor ->
                            if (!cursor.moveToFirst()) {
                                return@use
                            }
                            val status = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
                            val downloaded = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
                            val total = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
                            onProgress?.invoke(downloaded, if (total > 0) total else null)

                            when (status) {
                                DownloadManager.STATUS_SUCCESSFUL -> {
                                    if (cont.isActive) {
                                        cont.resume(destination)
                                    }
                                    cancel()
                                    return@launch
                                }
                                DownloadManager.STATUS_FAILED -> {
                                    val reason = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON))
                                    if (cont.isActive) {
                                        cont.resumeWithException(IllegalStateException("System download failed (reason=$reason)"))
                                    }
                                    cancel()
                                    return@launch
                                }
                            }
                        }
                    } catch (t: Throwable) {
                        if (cont.isActive) {
                            cont.resumeWithException(t)
                        }
                        cancel()
                        return@launch
                    }
                    delay(500L)
                }
            }

            cont.invokeOnCancellation {
                monitorJob.cancel()
                monitorScope.cancel()
                downloadManager.remove(downloadId)
                destination.delete()
            }
        }
    }
}

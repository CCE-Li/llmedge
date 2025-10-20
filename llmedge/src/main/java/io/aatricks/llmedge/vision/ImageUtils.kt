/*
 * Copyright (C) 2024 Aatricks
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

package io.aatricks.llmedge.vision

import android.content.Context
import android.graphics.*
import android.media.ExifInterface
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.coroutines.CoroutineContext
import java.io.*

/**
 * Image processing helpers: decoding, scaling, basic OCR-friendly filtering and
 * file conversions used by the vision example.
 */
object ImageUtils {
    
    private const val MAX_DIMENSION = 1600
    private const val JPEG_QUALITY = 90
    
    /** Convert an ImageSource to a Bitmap. Throws IOException on failure. */
    private suspend fun <T> safeWithIO(block: suspend () -> T): T {
        return try {
            withContext(Dispatchers.IO) {
                block()
            }
        } catch (e: Throwable) {
            // Some runtime environments can (incorrectly) have Dispatchers.IO == null
            // which causes a NullPointerException inside withContext. Fall back to
            // executing the block on the current coroutine dispatcher.
            if (e is NullPointerException) {
                block()
            } else throw e
        }
    }

    suspend fun imageToBitmap(context: Context, source: ImageSource): Bitmap = safeWithIO {
        val bmp: Bitmap? = when (source) {
            is ImageSource.BitmapSource -> source.bitmap
            is ImageSource.FileSource -> BitmapFactory.decodeFile(source.file.absolutePath)
            is ImageSource.UriSource -> {
                context.contentResolver.openInputStream(source.uri)?.use { stream ->
                    BitmapFactory.decodeStream(stream)
                }
            }
            is ImageSource.ByteArraySource -> {
                BitmapFactory.decodeByteArray(source.bytes, 0, source.bytes.size)
            }
        }

        if (bmp == null) {
            val srcDesc = when (source) {
                is ImageSource.BitmapSource -> "BitmapSource"
                is ImageSource.FileSource -> "File(${source.file.absolutePath})"
                is ImageSource.UriSource -> "Uri(${source.uri})"
                is ImageSource.ByteArraySource -> "ByteArray(len=${source.bytes.size})"
                else -> "unknown"
            }
            throw IOException("Failed to decode image from source: $srcDesc")
        }

        bmp
    }
    
    /** Save an ImageSource to a temporary file. */
    suspend fun imageToFile(
        context: Context,
        source: ImageSource,
        filename: String = "temp_image.jpg"
    ): File = safeWithIO {
        val tempFile = File(context.cacheDir, filename)
        
        when (source) {
            is ImageSource.FileSource -> source.file
            is ImageSource.BitmapSource -> {
                saveBitmapToFile(source.bitmap, tempFile)
                tempFile
            }
            is ImageSource.UriSource -> {
                context.contentResolver.openInputStream(source.uri)?.use { input ->
                    tempFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
                tempFile
            }
            is ImageSource.ByteArraySource -> {
                tempFile.writeBytes(source.bytes)
                tempFile
            }
        }
    }
    
    /** Preprocess a bitmap for OCR/vision: scale and optional enhancement. */
    fun preprocessImage(
        bitmap: Bitmap,
        correctOrientation: Boolean = true,
        maxDimension: Int = MAX_DIMENSION,
        enhance: Boolean = false
    ): Bitmap {
        var result = bitmap
        
        // Scale down if needed
        if (bitmap.width > maxDimension || bitmap.height > maxDimension) {
            result = scaleBitmap(result, maxDimension)
        }
        
        // Apply OCR enhancements if requested
        if (enhance) {
            result = enhanceForOcr(result)
        }
        
        return result
    }
    
    /** Scale a bitmap to fit within max dimension while preserving aspect ratio. */
    private fun scaleBitmap(bitmap: Bitmap, maxDimension: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        
        if (width <= maxDimension && height <= maxDimension) {
            return bitmap
        }
        
        val scale = if (width > height) {
            maxDimension.toFloat() / width
        } else {
            maxDimension.toFloat() / height
        }
        
        val newWidth = (width * scale).toInt()
        val newHeight = (height * scale).toInt()
        
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }
    
    /** Enhance image for better OCR results (grayscale + contrast). */
    private fun enhanceForOcr(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        
        // Convert to grayscale
        val grayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscale)
        val paint = Paint()
        
        // Grayscale color matrix
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f)
        
        // Increase contrast
        val contrastMatrix = ColorMatrix(floatArrayOf(
            1.5f, 0f, 0f, 0f, -40f,
            0f, 1.5f, 0f, 0f, -40f,
            0f, 0f, 1.5f, 0f, -40f,
            0f, 0f, 0f, 1f, 0f
        ))
        
        colorMatrix.postConcat(contrastMatrix)
        paint.colorFilter = ColorMatrixColorFilter(colorMatrix)
        
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        
        return grayscale
    }
    
    /** Apply EXIF orientation correction to a bitmap. */
    fun applyExifOrientation(bitmap: Bitmap, imagePath: String): Bitmap {
        return try {
            val exif = ExifInterface(imagePath)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
            
            val matrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
                ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
                ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
                else -> return bitmap
            }
            
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } catch (e: Exception) {
            bitmap
        }
    }

    /** Decode a File into a Bitmap and apply EXIF orientation if available. */
    fun fileToBitmap(file: File): Bitmap {
        try {
            val bmp = BitmapFactory.decodeFile(file.absolutePath)
            return applyExifOrientation(bmp, file.absolutePath)
        } catch (e: Exception) {
            throw IOException("Failed to decode image file: ${file.absolutePath}", e)
        }
    }
    
    /** Save a bitmap to a file as JPEG. */
    private fun saveBitmapToFile(bitmap: Bitmap, file: File, quality: Int = JPEG_QUALITY) {
        file.outputStream().use { stream ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, quality, stream)
        }
    }
    
    /** Convert an ImageSource to a byte array. */
    suspend fun imageToByteArray(
        context: Context,
        source: ImageSource,
        format: Bitmap.CompressFormat = Bitmap.CompressFormat.JPEG,
        quality: Int = JPEG_QUALITY
    ): ByteArray = safeWithIO {
        when (source) {
            is ImageSource.ByteArraySource -> source.bytes
            is ImageSource.FileSource -> source.file.readBytes()
            is ImageSource.UriSource -> {
                context.contentResolver.openInputStream(source.uri)?.use { stream ->
                    stream.readBytes()
                } ?: throw IOException("Failed to open URI: ${source.uri}")
            }
            is ImageSource.BitmapSource -> {
                ByteArrayOutputStream().use { stream ->
                    source.bitmap.compress(format, quality, stream)
                    stream.toByteArray()
                }
            }
        }
    }
}
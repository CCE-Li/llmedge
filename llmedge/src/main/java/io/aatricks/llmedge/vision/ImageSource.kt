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

import android.graphics.Bitmap
import android.net.Uri
import java.io.File

/**
 * Represents an image source that can be processed by OCR engines or vision models.
 */
sealed class ImageSource {
    data class UriSource(val uri: Uri) : ImageSource()
    data class FileSource(val file: File) : ImageSource()
    data class BitmapSource(val bitmap: Bitmap) : ImageSource()
    data class ByteArraySource(val bytes: ByteArray, val format: String = "JPEG") : ImageSource() {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            
            other as ByteArraySource
            
            if (!bytes.contentEquals(other.bytes)) return false
            if (format != other.format) return false
            
            return true
        }
        
        override fun hashCode(): Int {
            var result = bytes.contentHashCode()
            result = 31 * result + format.hashCode()
            return result
        }
    }
}
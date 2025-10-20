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

/**
 * Parameters for OCR processing.
 *
 * @property language The language code for OCR (e.g., "eng" for English).
 * @property pageSegmentationMode Tesseract page segmentation mode (0-13). Default is 3 (fully automatic).
 * @property engineMode Tesseract OCR engine mode (0-3). Default is 3 (default).
 * @property enhance Whether to apply image enhancement before OCR.
 */
data class OcrParams(
    val language: String = "eng",
    val pageSegmentationMode: Int = 3,
    val engineMode: Int = 3,
    val enhance: Boolean = true
)

/**
 * Result from OCR processing.
 *
 * @property text The extracted text.
 * @property language The language used for OCR.
 * @property durationMs Processing time in milliseconds.
 * @property engine The OCR engine used (e.g., "tesseract", "mlkit").
 * @property confidence Optional confidence score (0.0-1.0).
 */
data class OcrResult(
    val text: String,
    val language: String,
    val durationMs: Long,
    val engine: String,
    val confidence: Float? = null
)

/**
 * Interface for OCR engines.
 */
interface OcrEngine {
    /**
     * Extract text from an image.
     *
     * @param image The image source to process.
     * @param params OCR processing parameters.
     * @return The OCR result containing extracted text and metadata.
     */
    suspend fun extractText(image: ImageSource, params: OcrParams = OcrParams()): OcrResult
    
    /**
     * Check if the OCR engine is available and ready to use.
     */
    suspend fun isAvailable(): Boolean
    
    /**
     * Get the name of this OCR engine.
     */
    fun getEngineName(): String
}
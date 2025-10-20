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
 * Mode for processing images.
 */
enum class VisionMode {
    /** Automatically choose, preferring vision model if available */
    AUTO_PREFER_VISION,
    
    /** Force vision model (error if unavailable) */
    FORCE_VISION,
    
    /** Automatically choose, preferring OCR */
    AUTO_PREFER_OCR,
    
    
    /** Force ML Kit OCR */
    FORCE_MLKIT
}

/**
 * Result from image understanding processing.
 *
 * @property text The extracted or generated text.
 * @property engine The engine used (e.g., "vision:model-name", "ocr:tesseract", "ocr:mlkit").
 * @property durationMs Processing time in milliseconds.
 * @property confidence Optional confidence score.
 */
data class ImageUnderstandingResult(
    val text: String,
    val engine: String,
    val durationMs: Long,
    val confidence: Float? = null
)

/**
 * Orchestrates image understanding through vision models or OCR with fallback logic.
 *
 * @property visionAnalyzer Optional vision model analyzer.
 * @property ocrEngines List of available OCR engines.
 */
class ImageUnderstanding(
    private val visionAnalyzer: VisionModelAnalyzer? = null,
    private val ocrEngines: List<OcrEngine> = emptyList()
) {
    
    /**
     * Process an image using the specified mode.
     *
     * @param image The image to process.
     * @param mode The processing mode.
     * @param prompt Optional prompt for vision models or to refine OCR output.
     * @return The result containing extracted/generated text and metadata.
     * @throws IllegalStateException if the requested mode is not available.
     */
    suspend fun process(
        image: ImageSource,
        mode: VisionMode = VisionMode.AUTO_PREFER_VISION,
        prompt: String? = null
    ): ImageUnderstandingResult {
        val startTime = System.currentTimeMillis()
        
        return when (mode) {
            VisionMode.FORCE_VISION -> {
                if (visionAnalyzer == null || !visionAnalyzer.hasVisionCapabilities()) {
                    throw IllegalStateException("Vision model not available or doesn't have vision capabilities")
                }
                processWithVision(image, prompt ?: "Describe what you see in this image.", startTime)
            }
            
            
            VisionMode.FORCE_MLKIT -> {
                val mlkitEngine = ocrEngines.find { it.getEngineName() == "mlkit" }
                    ?: throw IllegalStateException("ML Kit OCR engine not available")
                processWithOcr(mlkitEngine, image, startTime)
            }
            
            VisionMode.AUTO_PREFER_VISION -> {
                if (visionAnalyzer?.hasVisionCapabilities() == true) {
                    try {
                        processWithVision(image, prompt ?: "Describe what you see in this image.", startTime)
                    } catch (e: Exception) {
                        // Fall back to OCR
                        processWithBestOcr(image, startTime)
                    }
                } else {
                    processWithBestOcr(image, startTime)
                }
            }
            
            VisionMode.AUTO_PREFER_OCR -> {
                processWithBestOcr(image, startTime)
            }
        }
    }
    
    private suspend fun processWithVision(
        image: ImageSource,
        prompt: String,
        startTime: Long
    ): ImageUnderstandingResult {
        val result = visionAnalyzer!!.analyze(image, prompt)
        return ImageUnderstandingResult(
            text = result.text,
            engine = "vision:${visionAnalyzer.getModelId()}",
            durationMs = System.currentTimeMillis() - startTime,
            confidence = null
        )
    }
    
    private suspend fun processWithOcr(
        engine: OcrEngine,
        image: ImageSource,
        startTime: Long
    ): ImageUnderstandingResult {
        val result = engine.extractText(image)
        return ImageUnderstandingResult(
            text = result.text,
            engine = "ocr:${engine.getEngineName()}",
            durationMs = System.currentTimeMillis() - startTime,
            confidence = result.confidence
        )
    }
    
    private suspend fun processWithBestOcr(
        image: ImageSource,
        startTime: Long
    ): ImageUnderstandingResult {
        if (ocrEngines.isEmpty()) {
            throw IllegalStateException("No OCR engines available")
        }
        
        // Prefer ML Kit if available (faster)
        val mlkitEngine = ocrEngines.find { it.getEngineName() == "mlkit" }
        if (mlkitEngine != null && mlkitEngine.isAvailable()) {
            val result = processWithOcr(mlkitEngine, image, startTime)
            
            // If ML Kit returns low confidence or empty text, fall back to any other available OCR engine
            if (result.text.isBlank() || (result.confidence != null && result.confidence < 0.5f)) {
                for (engine in ocrEngines) {
                    if (engine.getEngineName() != "mlkit" && engine.isAvailable()) {
                        return processWithOcr(engine, image, startTime)
                    }
                }
            }
            
            return result
        }
        
        // Fall back to any available OCR engine
        for (engine in ocrEngines) {
            if (engine.isAvailable()) {
                return processWithOcr(engine, image, startTime)
            }
        }
        
        throw IllegalStateException("No OCR engines are available")
    }
    
    /**
     * Check what processing modes are available.
     */
    fun getAvailableModes(): Set<VisionMode> {
        val modes = mutableSetOf<VisionMode>()
        
        // Always add auto modes
        modes.add(VisionMode.AUTO_PREFER_VISION)
        modes.add(VisionMode.AUTO_PREFER_OCR)
        
        // Add specific modes based on availability
        if (visionAnalyzer?.hasVisionCapabilities() == true) {
            modes.add(VisionMode.FORCE_VISION)
        }
        
        
        if (ocrEngines.any { it.getEngineName() == "mlkit" }) {
            modes.add(VisionMode.FORCE_MLKIT)
        }
        
        return modes
    }
}
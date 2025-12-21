package io.aatricks.llmedge

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import java.io.File
import java.lang.ref.WeakReference
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * High-level manager for LLMEdge operations. Simplifies model loading, generation, and memory
 * management.
 */
object LLMEdgeManager {
        private const val TAG = "LLMEdgeManager"
        private const val MIN_AVAILABLE_MEMORY_MB = 2000L

        // Default Video Model (Wan 2.1)
        private const val DEFAULT_VIDEO_MODEL_ID = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
        private const val DEFAULT_VIDEO_MODEL_FILENAME = "wan2.1_t2v_1.3B_fp16.safetensors"
        private const val DEFAULT_VIDEO_VAE_ID = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
        private const val DEFAULT_VIDEO_VAE_FILENAME = "wan_2.1_vae.safetensors"
        private const val DEFAULT_VIDEO_T5XXL_ID = "city96/umt5-xxl-encoder-gguf"
        private const val DEFAULT_VIDEO_T5XXL_FILENAME = "umt5-xxl-encoder-Q3_K_S.gguf"

        // Default Image Model (MeinaMix)
        private const val DEFAULT_IMAGE_MODEL_ID = "Meina/MeinaMix"
        private const val DEFAULT_IMAGE_MODEL_FILENAME = "MeinaPastel - baked VAE.safetensors"

        // Default Text Model (SmolLM 135M)
        private const val DEFAULT_TEXT_MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct-GGUF"
        private const val DEFAULT_TEXT_MODEL_FILENAME = "smollm-135m-instruct.q4_k_m.gguf"

        // Default Vision Model (LLaVA Phi-3 Mini)
        private const val DEFAULT_VISION_MODEL_ID = "xtuner/llava-phi-3-mini-gguf"
        private const val DEFAULT_VISION_MODEL_FILENAME = "llava-phi-3-mini-int4.gguf"
        private const val DEFAULT_VISION_PROJ_FILENAME = "llava-phi-3-mini-mmproj-f16.gguf"

        // Default Whisper Model (Speech-to-Text)
        private const val DEFAULT_WHISPER_MODEL_ID = "ggerganov/whisper.cpp"
        private const val DEFAULT_WHISPER_MODEL_FILENAME = "ggml-tiny.bin"

        // Default Bark Model (Text-to-Speech)
        // Note: f16 models are slow on mobile (~10+ minutes). Quantized models not yet available.
        private const val DEFAULT_BARK_MODEL_ID = "Green-Sky/bark-ggml"
        private const val DEFAULT_BARK_MODEL_FILENAME = "bark-small_weights-f16.bin"

        private val textModelMutex = Mutex() // For SmolLM text generation
        private val diffusionModelMutex = Mutex() // For Stable Diffusion image/video generation
        private val whisperMutex = Mutex() // For Whisper speech-to-text
        private val barkMutex = Mutex() // For Bark text-to-speech

        // Phase 3: Model caching with LRU eviction
        private val textModelCache =
                ModelCache<io.aatricks.llmedge.SmolLM>(
                        maxCacheSize = 2,
                        maxMemoryMB = 2048L // 2GB max for text models
                )
        private val diffusionModelCache =
                ModelCache<StableDiffusion>(
                        maxCacheSize = 1,
                        maxMemoryMB = 4096L // 4GB max for diffusion models
                )

        @Volatile private var cachedModel: StableDiffusion? = null
        @Volatile private var cachedSmolLM: io.aatricks.llmedge.SmolLM? = null
        @Volatile private var cachedWhisper: io.aatricks.llmedge.Whisper? = null
        @Volatile private var cachedBark: io.aatricks.llmedge.BarkTTS? = null
        @Volatile private var isLoading = false
        private var contextRef: WeakReference<Context>? = null
        /** When true, prefer higher throughput and lower memory-safety heuristics. */
        @Volatile var preferPerformanceMode: Boolean = false

        // `preferPerformanceMode` property provides a runtime setter (no need for explicit wrapper)

        // Phase 3: Core topology for optimal threading
        private val coreInfo by lazy { CpuTopology.detectCoreTopology() }

        // Track currently loaded text model to allow switching
        private data class LoadedTextModelSpec(
                val modelId: String,
                val filename: String,
                val path: String?
        )
        private var currentTextModelSpec: LoadedTextModelSpec? = null

        // Track loaded diffusion model (image/video) to allow switching and cache removal
        private data class LoadedDiffusionModelSpec(
                val modelId: String,
                val filename: String,
                val path: String,
                val vaePath: String?,
                val t5xxlPath: String?,
                val taesdPath: String? = null,
                val flowShift: Float = Float.POSITIVE_INFINITY,
                val loraModelDir: String? = null,
                val loraApplyMode: StableDiffusion.LoraApplyMode =
                        StableDiffusion.LoraApplyMode.AUTO
        )
        private var currentDiffusionModelSpec: LoadedDiffusionModelSpec? = null

        // Track loaded Whisper model
        private data class LoadedWhisperModelSpec(
                val modelId: String,
                val filename: String,
                val path: String?
        )
        private var currentWhisperModelSpec: LoadedWhisperModelSpec? = null

        // Track loaded Bark model
        private data class LoadedBarkModelSpec(
                val modelId: String,
                val filename: String,
                val path: String?
        )
        private var currentBarkModelSpec: LoadedBarkModelSpec? = null

        data class ImageGenerationParams(
                val prompt: String,
                val negative: String = "",
                val width: Int = 512,
                val height: Int = 512,
                val steps: Int = 20,
                val cfgScale: Float = 7.0f,
                val seed: Long = -1L,
                val flashAttn: Boolean = true,
                val forceSequentialLoad: Boolean = false,
                val easyCache: StableDiffusion.EasyCacheParams = StableDiffusion.EasyCacheParams(),
                val loraModelDir: String? = null,
                val loraApplyMode: StableDiffusion.LoraApplyMode =
                        StableDiffusion.LoraApplyMode.AUTO
        )

        data class VideoGenerationParams(
                val prompt: String,
                val negative: String = "",
                val width: Int = 512,
                val height: Int = 512,
                val videoFrames: Int = 16,
                val steps: Int = 20,
                val cfgScale: Float = 7.0f,
                val seed: Long = -1L,
                val flowShift: Float = Float.POSITIVE_INFINITY,
                val flashAttn: Boolean = true,
                val forceSequentialLoad: Boolean = false,
                // I2V (Image-to-Video) parameters
                val initImage: ByteArray? = null,
                val initWidth: Int = 0,
                val initHeight: Int = 0,
                val strength: Float = 1.0f, // 1.0 = full T2V, 0.0 = init image dominates
                // Sampling configuration
                val sampleMethod: StableDiffusion.SampleMethod =
                        StableDiffusion.SampleMethod.DEFAULT,
                val scheduler: StableDiffusion.Scheduler = StableDiffusion.Scheduler.DEFAULT,
                // Easy cache & LoRA
                val easyCache: StableDiffusion.EasyCacheParams = StableDiffusion.EasyCacheParams(),
                val loraModelDir: String? = null,
                val loraApplyMode: StableDiffusion.LoraApplyMode =
                        StableDiffusion.LoraApplyMode.AUTO,
                val taehvPath: String? = null
        ) {
                /**
                 * Calculate the actual number of frames that will be generated. Wan model uses
                 * formula: actual_frames = (videoFrames-1)/4*4+1
                 */
                fun actualFrameCount(): Int = (videoFrames - 1) / 4 * 4 + 1
        }

        data class TextGenerationParams(
                val prompt: String,
                val systemPrompt: String = "You are a helpful assistant.",
                val modelId: String = DEFAULT_TEXT_MODEL_ID,
                val modelFilename: String = DEFAULT_TEXT_MODEL_FILENAME,
                val modelPath: String? = null, // Absolute path override
                val temperature: Float = 0.7f,
                val maxTokens: Int = -1, // -1 for infinite/until EOS
                val revision: String = "main",
                val thinkingMode: io.aatricks.llmedge.SmolLM.ThinkingMode =
                        io.aatricks.llmedge.SmolLM.ThinkingMode.DEFAULT,
                val reasoningBudget: Int? = null
        )

        data class VisionAnalysisParams(
                val image: Bitmap,
                val prompt: String,
                val modelId: String = DEFAULT_VISION_MODEL_ID,
                val modelFilename: String = DEFAULT_VISION_MODEL_FILENAME,
                val projFilename: String = DEFAULT_VISION_PROJ_FILENAME
        )

        /** Parameters for speech-to-text transcription. */
        data class TranscriptionParams(
                /** Audio samples as 32-bit float PCM at 16kHz mono */
                val audioSamples: FloatArray,
                /** Hugging Face model ID */
                val modelId: String = DEFAULT_WHISPER_MODEL_ID,
                /** Model filename */
                val modelFilename: String = DEFAULT_WHISPER_MODEL_FILENAME,
                /** Translate to English instead of transcribing */
                val translate: Boolean = false,
                /** Target language code (e.g., "en", "es"). null = auto-detect */
                val language: String? = null,
                /** Enable token-level timestamps for more precise timing */
                val tokenTimestamps: Boolean = false,
                /** Number of threads. 0 = auto */
                val nThreads: Int = 0
        ) {
                override fun equals(other: Any?): Boolean {
                        if (this === other) return true
                        if (javaClass != other?.javaClass) return false
                        other as TranscriptionParams
                        if (!audioSamples.contentEquals(other.audioSamples)) return false
                        if (modelId != other.modelId) return false
                        if (modelFilename != other.modelFilename) return false
                        if (translate != other.translate) return false
                        if (language != other.language) return false
                        if (tokenTimestamps != other.tokenTimestamps) return false
                        if (nThreads != other.nThreads) return false
                        return true
                }

                override fun hashCode(): Int {
                        var result = audioSamples.contentHashCode()
                        result = 31 * result + modelId.hashCode()
                        result = 31 * result + modelFilename.hashCode()
                        result = 31 * result + translate.hashCode()
                        result = 31 * result + (language?.hashCode() ?: 0)
                        result = 31 * result + tokenTimestamps.hashCode()
                        result = 31 * result + nThreads
                        return result
                }
        }

        /**
         * Parameters for streaming (real-time) speech-to-text transcription.
         *
         * This enables real-time captioning by processing audio in a sliding window:
         * - Audio is collected in chunks of `stepMs` milliseconds
         * - A window of `lengthMs` milliseconds is transcribed at each step
         * - `keepMs` of audio is retained from the previous window for context
         *
         * Recommended settings:
         * - Fast captioning: stepMs=1000, lengthMs=5000, keepMs=200
         * - Balanced: stepMs=3000, lengthMs=10000, keepMs=200 (default)
         * - High accuracy: stepMs=5000, lengthMs=15000, keepMs=500
         */
        data class StreamingTranscriptionParams(
                /** Hugging Face model ID */
                val modelId: String = DEFAULT_WHISPER_MODEL_ID,
                /** Model filename */
                val modelFilename: String = DEFAULT_WHISPER_MODEL_FILENAME,
                /**
                 * Duration in milliseconds of each step (how often transcription runs). Default:
                 * 3000ms
                 */
                val stepMs: Int = 3000,
                /** Length of the transcription window in milliseconds. Default: 10000ms */
                val lengthMs: Int = 10000,
                /** Audio from previous window to keep for context. Default: 200ms */
                val keepMs: Int = 200,
                /** Translate to English instead of transcribing */
                val translate: Boolean = false,
                /** Target language code (e.g., "en", "es"). null = auto-detect */
                val language: String? = null,
                /** Number of threads. 0 = auto */
                val nThreads: Int = 0,
                /**
                 * Voice Activity Detection threshold (0.0-1.0). Higher = more aggressive silence
                 * detection
                 */
                val vadThreshold: Float = 0.6f,
                /** Enable VAD to only transcribe when speech is detected */
                val useVad: Boolean = true
        )

        /**
         * Parameters for text-to-speech synthesis.
         *
         * **Warning:** Bark TTS with f16 models is very slow on mobile (~10+ minutes). Consider
         * using this for batch processing or on desktop/server environments only.
         */
        data class SpeechSynthesisParams(
                /** Text to synthesize */
                val text: String,
                /** Hugging Face model ID */
                val modelId: String = DEFAULT_BARK_MODEL_ID,
                /** Model filename */
                val modelFilename: String = DEFAULT_BARK_MODEL_FILENAME,
                /** Random seed for reproducibility (0 = random) */
                val seed: Int = 0,
                /** Sampling temperature for text/coarse encoders */
                val temperature: Float = 0.7f,
                /** Sampling temperature for fine encoder */
                val fineTemperature: Float = 0.5f,
                /** Number of threads. 0 = auto */
                val nThreads: Int = 0
        )

        /** Information about Vulkan GPU device capabilities */
        data class VulkanDeviceInfo(
                val deviceCount: Int,
                val totalMemoryMB: Long,
                val freeMemoryMB: Long,
                val deviceIndex: Int = 0
        )

        data class PerformanceSnapshot(
                val textMetrics: io.aatricks.llmedge.SmolLM.GenerationMetrics?,
                val diffusionMetrics: io.aatricks.llmedge.StableDiffusion.GenerationMetrics?,
                val timestamp: Long = System.currentTimeMillis()
        )

        /** Check if Vulkan GPU acceleration is available on this device */
        fun isVulkanAvailable(): Boolean {
                return try {
                        io.aatricks.llmedge.StableDiffusion.getVulkanDeviceCount() > 0
                } catch (e: Exception) {
                        false
                }
        }

        /**
         * Get Vulkan device information (memory, device count) Returns null if Vulkan is not
         * available
         */
        fun getVulkanDeviceInfo(): VulkanDeviceInfo? {
                return try {
                        val deviceCount = io.aatricks.llmedge.StableDiffusion.getVulkanDeviceCount()
                        if (deviceCount <= 0) return null

                        val memory = io.aatricks.llmedge.StableDiffusion.getVulkanDeviceMemory(0)
                        if (memory == null || memory.size < 2) return null

                        VulkanDeviceInfo(
                                deviceCount = deviceCount,
                                freeMemoryMB = memory[0] / (1024 * 1024),
                                totalMemoryMB = memory[1] / (1024 * 1024),
                                deviceIndex = 0
                        )
                } catch (e: Exception) {
                        null
                }
        }

        /** Get combined performance metrics from both text and diffusion models */
        fun getPerformanceSnapshot(): PerformanceSnapshot {
                return PerformanceSnapshot(
                        textMetrics = cachedSmolLM?.getLastGenerationMetrics(),
                        diffusionMetrics = cachedModel?.getLastGenerationMetrics()
                )
        }

        /** Log performance snapshot to Android logcat for debugging */
        fun logPerformanceSnapshot() {
                val snapshot = getPerformanceSnapshot()
                val vulkanInfo = getVulkanDeviceInfo()

                Log.i(TAG, "=== Performance Snapshot ===")

                snapshot.textMetrics?.let { metrics ->
                        Log.i(
                                TAG,
                                "Text Generation: ${metrics.tokensPerSecond} tok/s, " +
                                        "${metrics.tokenCount} tokens, ${metrics.elapsedMillis}ms"
                        )
                }

                snapshot.diffusionMetrics?.let { metrics ->
                        Log.i(
                                TAG,
                                "Diffusion: ${metrics.stepsPerSecond} steps/s, " +
                                        "${metrics.totalTimeSeconds}s total"
                        )
                }

                vulkanInfo?.let { info ->
                        Log.i(
                                TAG,
                                "Vulkan: ${info.deviceCount} device(s), " +
                                        "${info.freeMemoryMB}MB free / ${info.totalMemoryMB}MB total"
                        )
                }

                Log.i(TAG, "===========================")
        }

        /** Generates text using a local LLM. */
        suspend fun generateText(
                context: Context,
                params: TextGenerationParams,
                onProgress: ((String) -> Unit)? = null
        ): String =
                textModelMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)

                        // Unload heavy diffusion models if loaded to free up memory
                        unloadDiffusionModel()

                        val smol =
                                getOrLoadSmolLM(
                                        context,
                                        params.modelId,
                                        params.modelFilename,
                                        params.modelPath
                                )

                        // Reset and set system prompt if needed
                        // smol.addSystemPrompt(params.systemPrompt)

                        // Apply runtime params that might change per generation
                        smol.setThinkingMode(params.thinkingMode)
                        params.reasoningBudget?.let { smol.setReasoningBudget(it) }

                        if (onProgress != null) {
                                val sb = StringBuilder()
                                // We need to use flow for streaming and respect maxTokens
                                var tokenCount = 0
                                var flow = smol.getResponseAsFlow(params.prompt)
                                if (params.maxTokens > 0) {
                                        flow = flow.take(params.maxTokens)
                                }
                                flow.collect { token ->
                                        if (token != "[EOG]") {
                                                sb.append(token)
                                                onProgress(token)
                                                tokenCount++
                                        }
                                }
                                return@withLock sb.toString()
                        } else {
                                // Use Dispatchers.IO for blocking native JNI operations
                                // Dispatchers.Default has limited parallelism and is meant for
                                // CPU-bound work
                                return@withLock kotlinx.coroutines.withContext(
                                        kotlinx.coroutines.Dispatchers.IO
                                ) { smol.getResponse(params.prompt) }
                        }
                }

        fun getLastTextGenerationMetrics(): io.aatricks.llmedge.SmolLM.GenerationMetrics? {
                return cachedSmolLM?.getLastGenerationMetrics()
        }

        fun getLastDiffusionMetrics(): io.aatricks.llmedge.StableDiffusion.GenerationMetrics? {
                return cachedModel?.getLastGenerationMetrics()
        }

        /**
         * Returns the shared SmolLM instance, loading it if necessary. Useful for advanced use
         * cases like RAG.
         */
        suspend fun getSmolLM(context: Context): io.aatricks.llmedge.SmolLM {
                return textModelMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)
                        unloadDiffusionModel()
                        getOrLoadSmolLM(context, DEFAULT_TEXT_MODEL_ID, DEFAULT_TEXT_MODEL_FILENAME)
                }
        }

        /**
         * Returns the shared SmolLM instance WITHOUT loading a model. Useful for activities that
         * want to manage the loading process themselves (e.g. HuggingFaceDemoActivity).
         */
        suspend fun getSmolLMInstance(context: Context): io.aatricks.llmedge.SmolLM {
                return textModelMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)
                        unloadDiffusionModel()
                        val smol =
                                cachedSmolLM
                                        ?: io.aatricks.llmedge.SmolLM().also { cachedSmolLM = it }
                        return@withLock smol
                }
        }

        /** Extracts text from an image using OCR. */
        suspend fun extractText(context: Context, image: Bitmap): String =
                textModelMutex.withLock {
                        // Create a fresh engine instance and close it immediately after use to free
                        // resources
                        val engine = io.aatricks.llmedge.vision.ocr.MlKitOcrEngine(context)
                        try {
                                val result =
                                        engine.extractText(
                                                io.aatricks.llmedge.vision.ImageSource.BitmapSource(
                                                        image
                                                )
                                        )
                                return@withLock result.text
                        } finally {
                                engine.close()
                        }
                }

        // ============================================================
        // Speech-to-Text (Whisper)
        // ============================================================

        /**
         * Transcribe audio to text using Whisper.
         *
         * @param context Android context
         * @param params Transcription parameters including audio samples
         * @param onProgress Optional callback for transcription progress (0-100)
         * @return List of transcription segments with timing information
         */
        suspend fun transcribeAudio(
                context: Context,
                params: TranscriptionParams,
                onProgress: ((Int) -> Unit)? = null
        ): List<Whisper.TranscriptionSegment> =
                whisperMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)

                        val whisper =
                                getOrLoadWhisper(context, params.modelId, params.modelFilename)

                        // Set progress callback if provided
                        if (onProgress != null) {
                                whisper.setProgressCallback { progress -> onProgress(progress) }
                        }

                        try {
                                val whisperParams =
                                        Whisper.TranscribeParams(
                                                nThreads = params.nThreads,
                                                translate = params.translate,
                                                language = params.language,
                                                tokenTimestamps = params.tokenTimestamps
                                        )
                                return@withLock whisper.transcribe(
                                        params.audioSamples,
                                        whisperParams
                                )
                        } finally {
                                // Clear callback after use
                                whisper.setProgressCallback(null)
                        }
                }

        /**
         * Transcribe audio and return as a simple string (concatenated segments).
         *
         * @param context Android context
         * @param audioSamples Audio samples as 32-bit float PCM at 16kHz mono
         * @param language Target language code (null = auto-detect)
         * @return Transcribed text
         */
        suspend fun transcribeAudioToText(
                context: Context,
                audioSamples: FloatArray,
                language: String? = null
        ): String {
                val params = TranscriptionParams(audioSamples = audioSamples, language = language)
                val segments = transcribeAudio(context, params)
                return segments.joinToString(" ") { it.text.trim() }
        }

        /**
         * Detect the language of audio.
         *
         * @param context Android context
         * @param audioSamples Audio samples as 32-bit float PCM at 16kHz mono
         * @return Language code (e.g., "en", "es") or null if detection fails
         */
        suspend fun detectLanguage(context: Context, audioSamples: FloatArray): String? =
                whisperMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)
                        val whisper =
                                getOrLoadWhisper(
                                        context,
                                        DEFAULT_WHISPER_MODEL_ID,
                                        DEFAULT_WHISPER_MODEL_FILENAME
                                )
                        return@withLock whisper.detectLanguage(audioSamples)
                }

        /**
         * Transcribe audio and generate SRT subtitle content.
         *
         * @param context Android context
         * @param audioSamples Audio samples as 32-bit float PCM at 16kHz mono
         * @param language Target language code (null = auto-detect)
         * @return SRT subtitle content
         */
        suspend fun transcribeToSrt(
                context: Context,
                audioSamples: FloatArray,
                language: String? = null
        ): String {
                val params =
                        TranscriptionParams(
                                audioSamples = audioSamples,
                                language = language,
                                tokenTimestamps = true
                        )
                val segments = transcribeAudio(context, params)
                return segments.joinToString("\n") { it.toSrtEntry() }
        }

        // ============================================================
        // Streaming Transcription (Real-time Speech-to-Text)
        // ============================================================

        // Cached streaming transcriber for the current session
        @Volatile private var cachedStreamingTranscriber: Whisper.StreamingTranscriber? = null

        /**
         * Create a streaming transcriber for real-time audio transcription.
         *
         * The streaming transcriber uses a sliding window approach to provide near real-time
         * transcription as audio becomes available.
         *
         * Usage:
         * ```kotlin
         * val transcriber = LLMEdgeManager.createStreamingTranscriber(context)
         *
         * // Start collecting transcription results
         * launch {
         *     transcriber.start().collect { segment ->
         *         println("Transcribed: ${segment.text}")
         *     }
         * }
         *
         * // Feed audio from microphone or other source
         * audioRecorder.onAudioChunk { samples ->
         *     transcriber.feedAudio(samples)
         * }
         *
         * // When done
         * transcriber.stop()
         * LLMEdgeManager.stopStreamingTranscription()
         * ```
         *
         * @param context Android context
         * @param params Streaming transcription parameters
         * @return StreamingTranscriber instance
         */
        suspend fun createStreamingTranscriber(
                context: Context,
                params: StreamingTranscriptionParams = StreamingTranscriptionParams()
        ): Whisper.StreamingTranscriber =
                whisperMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)

                        val whisper =
                                getOrLoadWhisper(context, params.modelId, params.modelFilename)

                        val streamingParams =
                                Whisper.StreamingParams(
                                        stepMs = params.stepMs,
                                        lengthMs = params.lengthMs,
                                        keepMs = params.keepMs,
                                        translate = params.translate,
                                        language = params.language,
                                        nThreads = params.nThreads,
                                        vadThreshold = params.vadThreshold,
                                        useVad = params.useVad
                                )

                        val transcriber = whisper.createStreamingTranscriber(streamingParams)
                        cachedStreamingTranscriber = transcriber
                        return@withLock transcriber
                }

        /**
         * Start streaming transcription and return a Flow of segments.
         *
         * This is a convenience method that creates a streaming transcriber and starts it
         * immediately. For more control, use createStreamingTranscriber().
         *
         * @param context Android context
         * @param params Streaming transcription parameters
         * @return Flow of TranscriptionSegment as they are transcribed
         */
        suspend fun startStreamingTranscription(
                context: Context,
                params: StreamingTranscriptionParams = StreamingTranscriptionParams()
        ): kotlinx.coroutines.flow.Flow<Whisper.TranscriptionSegment> {
                val transcriber = createStreamingTranscriber(context, params)
                return transcriber.start()
        }

        /**
         * Feed audio samples to the active streaming transcriber.
         *
         * Audio should be:
         * - 16kHz sample rate
         * - Mono channel
         * - 32-bit float PCM (-1.0 to 1.0)
         *
         * @param samples Audio samples to process
         */
        suspend fun feedStreamingAudio(samples: FloatArray) {
                cachedStreamingTranscriber?.feedAudio(samples)
        }

        /** Get the current streaming transcriber, if one is active. */
        fun getStreamingTranscriber(): Whisper.StreamingTranscriber? {
                return cachedStreamingTranscriber
        }

        /** Stop and cleanup the streaming transcriber. */
        fun stopStreamingTranscription() {
                cachedStreamingTranscriber?.stop()
                cachedStreamingTranscriber = null
        }

        // ============================================================
        // Text-to-Speech (Bark)
        // ============================================================

        /**
         * Synthesize speech from text using Bark.
         *
         * **Warning:** Bark TTS with f16 models is very slow on mobile (~10+ minutes). This method
         * is best suited for batch processing or desktop/server environments.
         *
         * @param context Android context
         * @param params Speech synthesis parameters
         * @param onProgress Optional callback for generation progress (step, percentage)
         * @return AudioResult containing the generated audio samples
         */
        suspend fun synthesizeSpeech(
                context: Context,
                params: SpeechSynthesisParams,
                onProgress: ((BarkTTS.EncodingStep, Int) -> Unit)? = null
        ): BarkTTS.AudioResult =
                barkMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)

                        // Unload heavy models to free memory for Bark
                        unloadSmolLM()
                        unloadDiffusionModel()

                        val bark =
                                getOrLoadBark(
                                        context,
                                        params.modelId,
                                        params.modelFilename,
                                        params.seed,
                                        params.temperature,
                                        params.fineTemperature
                                )

                        // Set progress callback if provided
                        if (onProgress != null) {
                                bark.setProgressCallback { step, progress ->
                                        onProgress(step, progress)
                                }
                        }

                        try {
                                val barkParams = BarkTTS.GenerateParams(nThreads = params.nThreads)
                                return@withLock bark.generate(params.text, barkParams)
                        } finally {
                                // Clear callback after use
                                bark.setProgressCallback(null)
                        }
                }

        /**
         * Synthesize speech and save directly to a WAV file.
         *
         * **Warning:** Bark TTS with f16 models is very slow on mobile (~10+ minutes).
         *
         * @param context Android context
         * @param text Text to synthesize
         * @param outputFile File to save WAV audio
         * @param onProgress Optional callback for generation progress
         */
        suspend fun synthesizeSpeechToFile(
                context: Context,
                text: String,
                outputFile: File,
                onProgress: ((BarkTTS.EncodingStep, Int) -> Unit)? = null
        ) {
                val params = SpeechSynthesisParams(text = text)
                val audio = synthesizeSpeech(context, params, onProgress)
                barkMutex.withLock {
                        cachedBark?.saveAsWav(audio, outputFile.absolutePath)
                                ?: throw IllegalStateException("Bark model not loaded")
                }
        }

        // ============================================================
        // Speech Model Management
        // ============================================================

        private suspend fun getOrLoadWhisper(
                context: Context,
                modelId: String,
                filename: String
        ): Whisper {
                val existingSpec = currentWhisperModelSpec
                val cached = cachedWhisper

                // Check if we already have the right model loaded
                if (cached != null &&
                                existingSpec != null &&
                                existingSpec.modelId == modelId &&
                                existingSpec.filename == filename
                ) {
                        return cached
                }

                // Need to load a new model
                prepareMemoryForLoading()

                // Close existing model if different
                cached?.close()
                cachedWhisper = null
                currentWhisperModelSpec = null

                Log.d(TAG, "Loading Whisper model: $modelId/$filename")
                val modelFile = getFile(context, modelId, filename)

                val whisper =
                        Whisper.load(
                                modelPath = modelFile.absolutePath,
                                useGpu = false, // Whisper GPU not well supported on mobile
                                flashAttn = true
                        )

                cachedWhisper = whisper
                currentWhisperModelSpec =
                        LoadedWhisperModelSpec(modelId, filename, modelFile.absolutePath)

                Log.d(TAG, "Whisper model loaded: ${whisper.getModelType()}")
                return whisper
        }

        private suspend fun getOrLoadBark(
                context: Context,
                modelId: String,
                filename: String,
                seed: Int,
                temperature: Float,
                fineTemperature: Float
        ): BarkTTS {
                val existingSpec = currentBarkModelSpec
                val cached = cachedBark

                // Check if we already have the right model loaded
                if (cached != null &&
                                existingSpec != null &&
                                existingSpec.modelId == modelId &&
                                existingSpec.filename == filename
                ) {
                        return cached
                }

                // Need to load a new model
                prepareMemoryForLoading()

                // Close existing model if different
                cached?.close()
                cachedBark = null
                currentBarkModelSpec = null

                Log.d(TAG, "Loading Bark model: $modelId/$filename")
                val modelFile = getFile(context, modelId, filename)

                val bark =
                        BarkTTS.load(
                                modelPath = modelFile.absolutePath,
                                seed = seed,
                                temperature = temperature,
                                fineTemperature = fineTemperature
                        )

                cachedBark = bark
                currentBarkModelSpec =
                        LoadedBarkModelSpec(modelId, filename, modelFile.absolutePath)

                Log.d(TAG, "Bark model loaded (sample rate: ${bark.getSampleRate()}Hz)")
                return bark
        }

        /** Unload the cached Whisper model to free memory. */
        fun unloadWhisper() {
                cachedWhisper?.close()
                cachedWhisper = null
                currentWhisperModelSpec = null
        }

        /** Unload the cached Bark model to free memory. */
        fun unloadBark() {
                cachedBark?.close()
                cachedBark = null
                currentBarkModelSpec = null
        }

        /** Unload all speech models to free memory. */
        fun unloadSpeechModels() {
                unloadWhisper()
                unloadBark()
        }

        /** Analyzes an image using a Vision Language Model (VLM). */
        suspend fun analyzeImage(
                context: Context,
                params: VisionAnalysisParams,
                onProgress: ((String) -> Unit)? = null
        ): String =
                textModelMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)

                        // Unload heavy diffusion models
                        unloadDiffusionModel()
                        // Also unload cached SmolLM to free up memory for this heavy operation
                        unloadSmolLM()

                        // Ensure files
                        val modelFile = getFile(context, params.modelId, params.modelFilename)
                        val projFile = getFile(context, params.modelId, params.projFilename)

                        // Instantiate a fresh SmolLM for this specific vision task
                        // Use explicit Vulkan setting based on preference
                        // Backup logs show useVulkan=0 (false) was the working configuration.
                        // We force false here to strictly match the working backup state.
                        val smol = io.aatricks.llmedge.SmolLM(useVulkan = false)

                        try {
                                prepareMemoryForLoading()

                                // Prepare Vision Adapter
                                val adapter =
                                        io.aatricks.llmedge.vision.SmolLMVisionAdapter(
                                                context,
                                                smol
                                        )

                                try {
                                        // 1. Prepare temp files (matching backup behavior)
                                        onProgress?.invoke("Preparing image")
                                        val imageFile =
                                                File.createTempFile(
                                                        "vision_input",
                                                        ".jpg",
                                                        context.cacheDir
                                                )
                                        val embedFile =
                                                File.createTempFile(
                                                        "vision_prepared",
                                                        ".bin",
                                                        context.cacheDir
                                                )

                                        // Clean up any stale metadata for the new temp file
                                        // (unlikely but safe)
                                        val metaFile = File(embedFile.absolutePath + ".meta.json")
                                        if (metaFile.exists()) metaFile.delete()

                                        try {
                                                Log.d(TAG, "Vision: Preprocessing image...")
                                                // Preprocess and save image
                                                val scaled =
                                                        io.aatricks.llmedge.vision.ImageUtils
                                                                .preprocessImage(
                                                                        params.image,
                                                                        correctOrientation = true,
                                                                        maxDimension =
                                                                                672, // Aligned to
                                                                        // 2x 336px
                                                                        // (Phi-3
                                                                        // native tile
                                                                        // size)
                                                                        enhance = false
                                                                )
                                                imageFile.outputStream().use { out ->
                                                        scaled.compress(
                                                                Bitmap.CompressFormat.JPEG,
                                                                90,
                                                                out
                                                        )
                                                }

                                                // 2. Load Model FIRST (Lightweight load for
                                                // Projection validation)
                                                onProgress?.invoke("Loading vision model (init)")
                                                Log.d(
                                                        TAG,
                                                        "Vision: Loading vision model (stage 1) ${modelFile.absolutePath}"
                                                )

                                                // Load directly into smol for the projection phase
                                                smol.load(
                                                        modelPath = modelFile.absolutePath,
                                                        params =
                                                                io.aatricks.llmedge.SmolLM
                                                                        .InferenceParams(
                                                                                numThreads = 2,
                                                                                contextSize =
                                                                                        null, // Auto context for projection is fine
                                                                                storeChats =
                                                                                        false, // Minimal memory
                                                                                temperature = 0.0f,
                                                                                thinkingMode =
                                                                                        io.aatricks
                                                                                                .llmedge
                                                                                                .SmolLM
                                                                                                .ThinkingMode
                                                                                                .DISABLED
                                                                        )
                                                )

                                                // 3. Run Projector (With loaded model pointer)
                                                onProgress?.invoke("Encoding image")
                                                Log.d(
                                                        TAG,
                                                        "Vision: Initializing projector with mmproj=${projFile.absolutePath}"
                                                )
                                                val projector =
                                                        io.aatricks.llmedge.vision.Projector()

                                                // Retrieve ptr from the NOW LOADED model
                                                val modelPtr = smol.getNativeModelPointer()
                                                Log.d(
                                                        TAG,
                                                        "Vision: Text model ptr=0x${java.lang.Long.toHexString(modelPtr)}"
                                                )

                                                projector.init(projFile.absolutePath, modelPtr)

                                                Log.d(
                                                        TAG,
                                                        "Vision: Encoding image to ${embedFile.absolutePath}"
                                                )
                                                val ok =
                                                        projector.encodeImageToFile(
                                                                imageFile.absolutePath,
                                                                embedFile.absolutePath
                                                        )
                                                projector.close()

                                                Log.d(TAG, "Vision: Projector returned $ok")

                                                if (!ok ||
                                                                !File(
                                                                                embedFile
                                                                                        .absolutePath +
                                                                                        ".meta.json"
                                                                        )
                                                                        .exists()
                                                ) {
                                                        Log.w(
                                                                TAG,
                                                                "Vision: Projection failed or metadata missing. Fallback to using raw image file."
                                                        )
                                                        imageFile.copyTo(
                                                                embedFile,
                                                                overwrite = true
                                                        )
                                                }

                                                // 4. Reload Model for Analysis (Inference
                                                // Configuration)
                                                // We reload with storeChats=true and larger
                                                // context, matching the working backup state
                                                onProgress?.invoke(
                                                        "Loading vision model (inference)"
                                                )
                                                Log.d(
                                                        TAG,
                                                        "Vision: Reloading vision model (stage 2) for inference"
                                                )

                                                adapter.loadVisionModel(
                                                        modelFile.absolutePath,
                                                        null,
                                                        io.aatricks.llmedge.SmolLM.InferenceParams(
                                                                numThreads = 2,
                                                                contextSize = 4096L, // Required for
                                                                // vision tokens
                                                                storeChats =
                                                                        true, // Backup used default
                                                                // (true) for
                                                                // inference
                                                                temperature =
                                                                        0.6f, // Increase temp to
                                                                // 0.6f to break
                                                                // repetition loops
                                                                thinkingMode =
                                                                        io.aatricks.llmedge.SmolLM
                                                                                .ThinkingMode
                                                                                .DISABLED
                                                        )
                                                )

                                                // 5. Run Analysis
                                                onProgress?.invoke("Running vision analysis")

                                                // Pass the EMBEDDING/BIN file as source
                                                val imageSource =
                                                        io.aatricks.llmedge.vision.ImageSource
                                                                .FileSource(embedFile)

                                                val result =
                                                        adapter.analyze(
                                                                imageSource,
                                                                params.prompt,
                                                                io.aatricks.llmedge.vision
                                                                        .VisionParams()
                                                        )
                                                Log.d(
                                                        TAG,
                                                        "Vision: Analysis complete. Response length=${result.text.length}"
                                                )
                                                return@withLock result.text
                                        } finally {
                                                // Cleanup
                                                if (imageFile.exists()) imageFile.delete()
                                                if (embedFile.exists()) embedFile.delete()
                                                if (metaFile.exists()) metaFile.delete()
                                                adapter.close()
                                        }
                                } catch (e: Exception) {
                                        Log.e(TAG, "Vision analysis failed", e)
                                        throw e
                                }
                        } finally {
                                smol.close()
                        }
                }
        /**
         * Generates an image using the default or configured model. Automatically handles
         * sequential loading for low-memory devices.
         */
        suspend fun generateImage(
                context: Context,
                params: ImageGenerationParams,
                onProgress: ((String, Int, Int) -> Unit)? = null
        ): Bitmap? =
                diffusionModelMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)
                        unloadSmolLM() // Free up memory from LLM
                        // Sequential load logic is handled inside getOrLoadImageModel via
                        // auto-detection if null,
                        // or we can force it here if needed. Current implementation uses default
                        // (null).

                        val model =
                                getOrLoadImageModel(
                                        context,
                                        params.flashAttn,
                                        params.width,
                                        params.height,
                                        onProgress,
                                        loraModelDir = params.loraModelDir,
                                        loraApplyMode = params.loraApplyMode
                                )

                        // Auto-detect and enable EasyCache if supported
                        val easyCacheSupported = model.isEasyCacheSupported()
                        val finalEasyCacheParams =
                                if (easyCacheSupported) {
                                        params.easyCache.copy(enabled = true)
                                } else {
                                        params.easyCache.copy(enabled = false)
                                }

                        // Use txt2img(GenerateParams) which returns Bitmap directly
                        val sdParams =
                                StableDiffusion.GenerateParams(
                                        prompt = params.prompt,
                                        negative = params.negative,
                                        width = params.width,
                                        height = params.height,
                                        steps = params.steps,
                                        cfgScale = params.cfgScale,
                                        seed = params.seed,
                                        easyCacheParams = finalEasyCacheParams
                                )
                        return model.txt2img(sdParams)
                }

        /** Generates a video using the default or configured model. */
        suspend fun generateVideo(
                context: Context,
                params: VideoGenerationParams,
                onProgress: ((String, Int, Int) -> Unit)? = null
        ): List<Bitmap> =
                diffusionModelMutex.withLock {
                        contextRef = WeakReference(context.applicationContext)
                        unloadSmolLM() // Free up memory from LLM

                        val isLowMem = isLowMemoryDevice(context)
                        val useSequential = params.forceSequentialLoad || isLowMem
                        Log.i(
                                TAG,
                                "generateVideo: preferPerformanceMode=$preferPerformanceMode, isLowMem=$isLowMem, forceSequential=${params.forceSequentialLoad}, useSequential=$useSequential, taehvPath=${params.taehvPath ?: "(none)"}, hasInitImage=${params.initImage != null}"
                        )

                        if (useSequential) {
                                return generateVideoSequentially(context, params, onProgress)
                        } else {
                                val model =
                                        getOrLoadVideoModel(
                                                context,
                                                params.flashAttn,
                                                params.flowShift,
                                                onProgress,
                                                sequentialLoad = useSequential,
                                                loraModelDir = params.loraModelDir,
                                                loraApplyMode = params.loraApplyMode,
                                                taehvPath = params.taehvPath,
                                                vaeDecodeOnly = params.initImage == null
                                        )
                                val sdParams =
                                        StableDiffusion.VideoGenerateParams(
                                                prompt = params.prompt,
                                                negative = params.negative,
                                                width = params.width,
                                                height = params.height,
                                                videoFrames = params.videoFrames,
                                                steps = params.steps,
                                                cfgScale = params.cfgScale,
                                                seed = params.seed,
                                                initImage =
                                                        params.initImage?.let { bytes ->
                                                                // Convert RGB byte array to Bitmap
                                                                if (params.initWidth > 0 &&
                                                                                params.initHeight >
                                                                                        0
                                                                ) {
                                                                        val pixels =
                                                                                IntArray(
                                                                                        params.initWidth *
                                                                                                params.initHeight
                                                                                )
                                                                        for (i in pixels.indices) {
                                                                                val r =
                                                                                        bytes[i * 3]
                                                                                                .toInt() and
                                                                                                0xFF
                                                                                val g =
                                                                                        bytes[
                                                                                                        i *
                                                                                                                3 +
                                                                                                                1]
                                                                                                .toInt() and
                                                                                                0xFF
                                                                                val b =
                                                                                        bytes[
                                                                                                        i *
                                                                                                                3 +
                                                                                                                2]
                                                                                                .toInt() and
                                                                                                0xFF
                                                                                pixels[i] =
                                                                                        (0xFF shl
                                                                                                24) or
                                                                                                (r shl
                                                                                                        16) or
                                                                                                (g shl
                                                                                                        8) or
                                                                                                b
                                                                        }
                                                                        android.graphics.Bitmap
                                                                                .createBitmap(
                                                                                        params.initWidth,
                                                                                        params.initHeight,
                                                                                        android.graphics
                                                                                                .Bitmap
                                                                                                .Config
                                                                                                .ARGB_8888
                                                                                )
                                                                                .apply {
                                                                                        setPixels(
                                                                                                pixels,
                                                                                                0,
                                                                                                params.initWidth,
                                                                                                0,
                                                                                                0,
                                                                                                params.initWidth,
                                                                                                params.initHeight
                                                                                        )
                                                                                }
                                                                } else null
                                                        },
                                                strength = params.strength,
                                                sampleMethod = params.sampleMethod,
                                                scheduler = params.scheduler,
                                                easyCacheParams = params.easyCache
                                        )

                                // We need a progress callback wrapper
                                val progressWrapper =
                                        StableDiffusion.VideoProgressCallback {
                                                step,
                                                totalSteps,
                                                currentFrame,
                                                totalFrames,
                                                _ ->
                                                onProgress?.invoke(
                                                        "Generating frame $currentFrame/$totalFrames",
                                                        step,
                                                        totalSteps
                                                )
                                        }
                                model.setProgressCallback(progressWrapper)

                                val framesBytes = model.txt2vid(sdParams)
                                return framesBytes
                        }
                }

        private suspend fun generateImageSequentially(
                context: Context,
                params: ImageGenerationParams,
                onProgress: ((String, Int, Int) -> Unit)?
        ): Bitmap? {
                // ... Implementation similar to video sequential load ...
                // 1. Load T5 -> Encode
                // 2. Unload T5
                // 3. Load Diffusion -> Generate (using new txt2ImgWithPrecomputedCondition)

                // Ensure files
                // Update memory provider for diffusion cache
                if (!preferPerformanceMode) {
                        diffusionModelCache.systemMemoryProvider = {
                                val am =
                                        context.getSystemService(Context.ACTIVITY_SERVICE) as
                                                ActivityManager
                                val mi = ActivityManager.MemoryInfo()
                                am.getMemoryInfo(mi)
                                mi.availMem / (1024L * 1024L)
                        }
                } else {
                        // In performance mode, avoid using system memory to aggressively evict
                        // models so
                        // we can keep heavy models resident for throughput
                        diffusionModelCache.systemMemoryProvider = null
                }
                ensureVideoFiles(context, onProgress)

                // Load T5
                prepareMemoryForLoading()
                var t5Model: StableDiffusion? = null
                var cond: StableDiffusion.PrecomputedCondition?
                var uncond: StableDiffusion.PrecomputedCondition?

                try {
                        val t5File =
                                getFile(
                                        context,
                                        DEFAULT_VIDEO_T5XXL_ID,
                                        DEFAULT_VIDEO_T5XXL_FILENAME
                                )
                        t5Model =
                                StableDiffusion.load(
                                        context = context,
                                        modelPath = t5File.absolutePath,
                                        vaePath = null,
                                        t5xxlPath = null,
                                        nThreads =
                                                CpuTopology.getOptimalThreadCount(
                                                        CpuTopology.TaskType.PROMPT_PROCESSING
                                                ),
                                        offloadToCpu = true,
                                        keepClipOnCpu = true,
                                        keepVaeOnCpu = true,
                                        flashAttn = params.flashAttn
                                )

                        cond =
                                t5Model.precomputeCondition(
                                        params.prompt,
                                        params.negative,
                                        params.width,
                                        params.height
                                )
                        uncond =
                                if (params.negative.isNotEmpty()) {
                                        t5Model.precomputeCondition(
                                                params.negative,
                                                "",
                                                params.width,
                                                params.height
                                        )
                                } else {
                                        t5Model.precomputeCondition(
                                                "",
                                                "",
                                                params.width,
                                                params.height
                                        )
                                }
                } finally {
                        t5Model?.close()
                }

                // Load Diffusion
                prepareMemoryForLoading()
                var diffusionModel: StableDiffusion? = null
                try {
                        // Model and VAE files are handled in getOrLoadImageModel

                        diffusionModel =
                                getOrLoadImageModel(
                                        context = context,
                                        flashAttn = params.flashAttn,
                                        width = params.width,
                                        height = params.height,
                                        onProgress = onProgress,
                                        sequentialLoad = true,
                                        loraModelDir = params.loraModelDir,
                                        loraApplyMode = params.loraApplyMode
                                )

                        val bytes =
                                diffusionModel.txt2ImgWithPrecomputedCondition(
                                        prompt = params.prompt,
                                        negative = params.negative,
                                        width = params.width,
                                        height = params.height,
                                        steps = params.steps,
                                        cfg = params.cfgScale,
                                        seed = params.seed,
                                        cond = cond,
                                        uncond = uncond
                                )
                        // Convert raw RGB bytes to Bitmap - BitmapFactory.decodeByteArray expects
                        // encoded PNG/JPEG, not raw RGB pixels
                        return bytes?.let { rgbBytesToBitmap(it, params.width, params.height) }
                } finally {
                        diffusionModel?.close()
                }
        }

        private suspend fun generateVideoSequentially(
                context: Context,
                params: VideoGenerationParams,
                onProgress: ((String, Int, Int) -> Unit)?
        ): List<Bitmap> {
                // Skip downloading default VAE if custom TAEHV is provided
                ensureVideoFiles(context, onProgress, skipVae = params.taehvPath != null)

                // Check available memory before loading T5 (~6GB model)
                val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                val memInfo = ActivityManager.MemoryInfo()
                am.getMemoryInfo(memInfo)
                val availMemMB = memInfo.availMem / (1024L * 1024L)
                val requiredMemMB = 6000L // T5 Q3_K_S needs ~6GB

                if (availMemMB < requiredMemMB) {
                        Log.e(
                                TAG,
                                "Insufficient memory for video generation: available=${availMemMB}MB, required=${requiredMemMB}MB"
                        )
                        throw OutOfMemoryError(
                                "Video generation requires ~${requiredMemMB}MB free RAM, but only ${availMemMB}MB is available. " +
                                        "Please close other apps and try again, or use a device with more RAM."
                        )
                }

                Log.i(
                        TAG,
                        "Memory check passed: available=${availMemMB}MB, required=${requiredMemMB}MB"
                )

                prepareMemoryForLoading()
                var t5Model: StableDiffusion? = null
                var cond: StableDiffusion.PrecomputedCondition?
                var uncond: StableDiffusion.PrecomputedCondition?

                try {
                        val t5File =
                                getFile(
                                        context,
                                        DEFAULT_VIDEO_T5XXL_ID,
                                        DEFAULT_VIDEO_T5XXL_FILENAME
                                )
                        t5Model =
                                StableDiffusion.load(
                                        context = context,
                                        modelPath = t5File.absolutePath,
                                        vaePath = null,
                                        t5xxlPath = null,
                                        nThreads =
                                                CpuTopology.getOptimalThreadCount(
                                                        CpuTopology.TaskType.PROMPT_PROCESSING
                                                ),
                                        offloadToCpu = true,
                                        keepClipOnCpu = true,
                                        keepVaeOnCpu = true,
                                        flashAttn = params.flashAttn
                                )

                        cond =
                                t5Model.precomputeCondition(
                                        params.prompt,
                                        params.negative,
                                        params.width,
                                        params.height
                                )
                        uncond =
                                if (params.negative.isNotEmpty()) {
                                        t5Model.precomputeCondition(
                                                params.negative,
                                                "",
                                                params.width,
                                                params.height
                                        )
                                } else {
                                        t5Model.precomputeCondition(
                                                "",
                                                "",
                                                params.width,
                                                params.height
                                        )
                                }
                } finally {
                        t5Model?.close()
                }

                prepareMemoryForLoading()
                var diffusionModel: StableDiffusion? = null
                try {
                        // Model and VAE files are handled in getOrLoadVideoModel

                        diffusionModel =
                                getOrLoadVideoModel(
                                        context = context,
                                        flashAttn = params.flashAttn,
                                        flowShift = params.flowShift,
                                        onProgress = onProgress,
                                        sequentialLoad = true,
                                        loadT5 = false,
                                        loraModelDir = params.loraModelDir,
                                        loraApplyMode = params.loraApplyMode,
                                        taehvPath = params.taehvPath,
                                        vaeDecodeOnly = params.initImage == null
                                )

                        val sdParams =
                                StableDiffusion.VideoGenerateParams(
                                        prompt = params.prompt,
                                        negative = params.negative,
                                        width = params.width,
                                        height = params.height,
                                        videoFrames = params.videoFrames,
                                        steps = params.steps,
                                        cfgScale = params.cfgScale,
                                        seed = params.seed,
                                        initImage =
                                                params.initImage?.let { bytes ->
                                                        // Convert RGB byte array to Bitmap for I2V
                                                        if (params.initWidth > 0 &&
                                                                        params.initHeight > 0
                                                        ) {
                                                                val pixels =
                                                                        IntArray(
                                                                                params.initWidth *
                                                                                        params.initHeight
                                                                        )
                                                                for (i in pixels.indices) {
                                                                        val r =
                                                                                bytes[i * 3]
                                                                                        .toInt() and
                                                                                        0xFF
                                                                        val g =
                                                                                bytes[i * 3 + 1]
                                                                                        .toInt() and
                                                                                        0xFF
                                                                        val b =
                                                                                bytes[i * 3 + 2]
                                                                                        .toInt() and
                                                                                        0xFF
                                                                        pixels[i] =
                                                                                (0xFF shl 24) or
                                                                                        (r shl
                                                                                                16) or
                                                                                        (g shl 8) or
                                                                                        b
                                                                }
                                                                Bitmap.createBitmap(
                                                                        pixels,
                                                                        params.initWidth,
                                                                        params.initHeight,
                                                                        Bitmap.Config.ARGB_8888
                                                                )
                                                        } else null
                                                },
                                        strength = params.strength
                                )

                        val progressWrapper =
                                StableDiffusion.VideoProgressCallback {
                                        step,
                                        totalSteps,
                                        currentFrame,
                                        totalFrames,
                                        _ ->
                                        onProgress?.invoke(
                                                "Generating frame $currentFrame/$totalFrames",
                                                step,
                                                totalSteps
                                        )
                                }

                        return diffusionModel.txt2VidWithPrecomputedCondition(
                                params = sdParams,
                                cond = cond,
                                uncond = uncond,
                                onProgress = progressWrapper
                        )
                } finally {
                        diffusionModel?.close()
                }
        }

        fun cancelGeneration() {
                Log.d(TAG, "cancelGeneration invoked: cancelling any active generation")
                // Cancel diffusion model generation if running
                cachedModel?.cancelGeneration()
                // Also attempt to stop SmolLM completion if active
                try {
                        cachedSmolLM?.stopCompletion()
                        Log.d(TAG, "LLMEdgeManager.cancelGeneration: SmolLM.stopCompletion invoked")
                } catch (e: Throwable) {
                        Log.w(TAG, "SmolLM.stopCompletion failed: ${'$'}{e.message}")
                }
        }

        /** Debug helper: returns current video model's T5 path, if any. */
        fun getLoadedVideoModelT5Path(): String? {
                return currentDiffusionModelSpec?.t5xxlPath
        }

        private suspend fun getOrLoadImageModel(
                context: Context,
                flashAttn: Boolean,
                width: Int = 512,
                height: Int = 512,
                onProgress: ((String, Int, Int) -> Unit)?,
                sequentialLoad: Boolean? = null,
                loraModelDir: String? = null,
                loraApplyMode: StableDiffusion.LoraApplyMode = StableDiffusion.LoraApplyMode.AUTO
        ): StableDiffusion {
                // Check if we already have the correct IMAGE model loaded
                val spec = currentDiffusionModelSpec
                cachedModel?.let {
                        // Only return cached model if it's specifically an image model (not video)
                        if (spec != null &&
                                        spec.filename == DEFAULT_IMAGE_MODEL_FILENAME &&
                                        spec.vaePath ==
                                                null && // Image models don't use separate VAE
                                        spec.t5xxlPath == null && // Image models don't use T5
                                        spec.flowShift == Float.POSITIVE_INFINITY &&
                                        spec.loraModelDir == loraModelDir &&
                                        spec.loraApplyMode == loraApplyMode
                        ) {
                                return it
                        }
                        // Wrong model type loaded - unload it first
                        unloadDiffusionModel()
                }

                if (!preferPerformanceMode) {
                        diffusionModelCache.systemMemoryProvider = {
                                val am =
                                        context.getSystemService(Context.ACTIVITY_SERVICE) as
                                                ActivityManager
                                val mi = ActivityManager.MemoryInfo()
                                am.getMemoryInfo(mi)
                                mi.availMem / (1024L * 1024L)
                        }
                } else {
                        diffusionModelCache.systemMemoryProvider = null
                }
                ensureImageFiles(context, onProgress)
                prepareMemoryForLoading()

                // Phase 3: Adaptive flash attention based on image dimensions
                // When flashAttn=true (default), let the helper decide based on dimensions.
                // When flashAttn=false, force disable flash attention.
                // This ensures small images (128x128) don't use flash attention which can be
                // inefficient on mobile GPUs that may lack proper hardware support
                // (coopmat2/subgroup_shuffle).
                val adaptiveFlashAttn =
                        FlashAttentionHelper.shouldUseFlashAttention(
                                width = width,
                                height = height,
                                forceEnable = if (flashAttn) null else false
                        )
                Log.i(
                        TAG,
                        "Loading image model with flash_attn=$adaptiveFlashAttn " +
                                "(requested=$flashAttn, dimensions=${width}x${height}, " +
                                "seqLen=${(width / 8) * (height / 8)}, sequentialLoad=${sequentialLoad})"
                )

                val modelFile =
                        getFile(context, DEFAULT_IMAGE_MODEL_ID, DEFAULT_IMAGE_MODEL_FILENAME)

                // Build cache key
                val cacheKey =
                        makeDiffusionCacheKey(
                                modelFile.absolutePath,
                                null,
                                null,
                                null,
                                Float.POSITIVE_INFINITY,
                                loraModelDir,
                                loraApplyMode
                        )

                // Check cache
                diffusionModelCache.get(cacheKey)?.let { cached ->
                        Log.i(TAG, "Loaded Image model from cache: $cacheKey")
                        cachedModel = cached
                        currentDiffusionModelSpec =
                                LoadedDiffusionModelSpec(
                                        modelId = DEFAULT_IMAGE_MODEL_ID,
                                        filename = DEFAULT_IMAGE_MODEL_FILENAME,
                                        path = modelFile.absolutePath,
                                        vaePath = null,
                                        t5xxlPath = null,
                                        loraModelDir = loraModelDir,
                                        loraApplyMode = loraApplyMode
                                )
                        return cached
                }

                val loadStart = System.currentTimeMillis()
                // Let StableDiffusion.load() auto-detect the best backend.
                // On devices with slow Vulkan implementations (e.g., Samsung Xclipse 920),
                // CPU backend with sequential load can be 5x faster than Vulkan.
                // The auto-detection enables CPU backend on low-memory devices which is
                // often the better choice for mobile diffusion workloads.
                // Fix: Respect explicit sequentialLoad=true request (e.g. from generateImageSequentially)
                val finalSequentialLoad =
                        if (sequentialLoad == true) true
                        else if (preferPerformanceMode) null
                        else sequentialLoad
                Log.i(
                        TAG,
                        "StableDiffusion.load(image) called with finalSequentialLoad=${finalSequentialLoad}, forceVulkan=${preferPerformanceMode}, offloadToCpu=false, flashAttn=$adaptiveFlashAttn"
                )
                val model =
                        StableDiffusion.load(
                                context = context,
                                modelPath = modelFile.absolutePath,
                                nThreads =
                                        CpuTopology.getOptimalThreadCount(
                                                CpuTopology.TaskType.DIFFUSION
                                        ),
                                offloadToCpu = false,
                                sequentialLoad = finalSequentialLoad,
                                forceVulkan = preferPerformanceMode,
                                preferPerformanceMode = preferPerformanceMode,
                                flashAttn = adaptiveFlashAttn,
                                vaeDecodeOnly = true,
                                loraModelDir = loraModelDir,
                                loraApplyMode = loraApplyMode
                                // sequentialLoad defaults to null, allowing auto-detection
                                )
                val loadTime = System.currentTimeMillis() - loadStart
                // Use file size as cache size estimate to avoid re-parsing the model file.
                val modelSize = modelFile.length()
                Log.i(
                        TAG,
                        "Loaded image model in ${loadTime}ms (size=${modelSize / 1024 / 1024}MB)"
                )
                // StableDiffusion.load() already performs estimation internally if needed for
                // Vulkan VRAM heuristics, so we don't need to call it again here.

                diffusionModelCache.put(cacheKey, model, modelSize, loadTime)
                cachedModel = model
                currentDiffusionModelSpec =
                        LoadedDiffusionModelSpec(
                                modelId = DEFAULT_IMAGE_MODEL_ID,
                                filename = DEFAULT_IMAGE_MODEL_FILENAME,
                                path = modelFile.absolutePath,
                                vaePath = null,
                                t5xxlPath = null,
                                loraModelDir = loraModelDir,
                                loraApplyMode = loraApplyMode
                        )
                return model
        }

        private suspend fun getOrLoadVideoModel(
                context: Context,
                flashAttn: Boolean,
                flowShift: Float,
                onProgress: ((String, Int, Int) -> Unit)?,
                sequentialLoad: Boolean? = null,
                loadT5: Boolean = true,
                loraModelDir: String? = null,
                loraApplyMode: StableDiffusion.LoraApplyMode = StableDiffusion.LoraApplyMode.AUTO,
                taehvPath: String? = null,
                vaeDecodeOnly: Boolean = true
        ): StableDiffusion {
                // Check if we already have the correct VIDEO model loaded
                val spec = currentDiffusionModelSpec

                // If taehvPath is provided, it is treated as a tiny autoencoder (TAEHV/TAESD)
                // and must be passed via `taesdPath` (NOT `vaePath`).
                val usingCustomTae = taehvPath != null

                cachedModel?.let {
                        // Only return cached model if it's specifically a video model with VAE and
                        // T5
                        // or if loaded without T5 when `loadT5=false`.
                        val t5Match =
                                if (loadT5) spec?.t5xxlPath != null else spec?.t5xxlPath == null

                        // Check VAE/TAESD match
                        val vaeMatch =
                                if (usingCustomTae) {
                                        spec?.taesdPath == taehvPath
                                } else {
                                        // If we didn't request a custom VAE, allow the cached one
                                        // if it's the default.
                                        // We can check the filename if path is absolute.
                                        spec?.vaePath?.endsWith(DEFAULT_VIDEO_VAE_FILENAME) == true
                                }

                        if (spec != null &&
                                        spec.filename == DEFAULT_VIDEO_MODEL_FILENAME &&
                                        vaeMatch && // Video models require VAE or TAESD
                                        t5Match &&
                                        spec.flowShift == flowShift &&
                                        spec.loraModelDir == loraModelDir &&
                                        spec.loraApplyMode == loraApplyMode
                        ) {
                                return it
                        }
                        // Wrong model type loaded - unload it first
                        unloadDiffusionModel()
                }

                // Set memory provider for cache before loading
                diffusionModelCache.systemMemoryProvider = {
                        val am =
                                context.getSystemService(Context.ACTIVITY_SERVICE) as
                                        ActivityManager
                        val mi = ActivityManager.MemoryInfo()
                        am.getMemoryInfo(mi)
                        mi.availMem / (1024L * 1024L)
                }

                ensureVideoFiles(context, onProgress, skipVae = usingCustomTae)
                prepareMemoryForLoading()

                val modelFile =
                        getFile(context, DEFAULT_VIDEO_MODEL_ID, DEFAULT_VIDEO_MODEL_FILENAME)

                val vaePath: String?
                val taesdPath: String?
                if (usingCustomTae) {
                        val file = File(taehvPath!!)
                        if (!file.exists() || !file.isFile || file.length() <= 0L) {
                                throw java.io.FileNotFoundException(
                                        "TAEHV/TAESD file not found or empty: ${file.absolutePath}"
                                )
                        }
                        // Treat taehvPath strictly as a tiny autoencoder (TAEHV/TAESD).
                        // Some TAEHV safetensors can be >100MB, so file-size heuristics are not
                        // reliable and can cause native crashes if misrouted via `vaePath`.
                        vaePath = null
                        taesdPath = file.absolutePath
                        Log.i(
                                TAG,
                                "Using custom TAEHV/TAESD: path=${file.absolutePath}, sizeMB=${file.length() / (1024L * 1024L)}"
                        )
                } else {
                        vaePath =
                                getFile(context, DEFAULT_VIDEO_VAE_ID, DEFAULT_VIDEO_VAE_FILENAME)
                                        .absolutePath
                        taesdPath = null
                }

                val t5File =
                        if (loadT5)
                                getFile(
                                        context,
                                        DEFAULT_VIDEO_T5XXL_ID,
                                        DEFAULT_VIDEO_T5XXL_FILENAME
                                )
                        else null
                if (!loadT5) {
                        Log.i(
                                TAG,
                                "Not loading T5 encoder into video model (loadT5=false) to reduce peak memory during sequential load"
                        )
                }

                val cacheKey =
                        makeDiffusionCacheKey(
                                modelFile.absolutePath,
                                vaePath,
                                t5File?.absolutePath,
                                taesdPath,
                                flowShift,
                                loraModelDir,
                                loraApplyMode
                        )
                diffusionModelCache.get(cacheKey)?.let { cached ->
                        Log.i(TAG, "Loaded Video model from cache: $cacheKey")
                        cachedModel = cached
                        currentDiffusionModelSpec =
                                LoadedDiffusionModelSpec(
                                        modelId = DEFAULT_VIDEO_MODEL_ID,
                                        filename = DEFAULT_VIDEO_MODEL_FILENAME,
                                        path = modelFile.absolutePath,
                                        vaePath = vaePath,
                                        t5xxlPath = t5File?.absolutePath,
                                        taesdPath = taesdPath,
                                        flowShift = flowShift,
                                        loraModelDir = loraModelDir,
                                        loraApplyMode = loraApplyMode
                                )
                        return cached
                }

                val loadStart = System.currentTimeMillis()
                val finalSequentialLoadV =
                        if (sequentialLoad == true) true
                        else if (preferPerformanceMode) null
                        else sequentialLoad
                val finalKeepClipOnCpu = if (preferPerformanceMode) false else true
                // Keep TAEHV on CPU to prevent crashes on some devices, otherwise respect performance mode
                val finalKeepVaeOnCpu = if (usingCustomTae || !preferPerformanceMode) true else false
                // Log runtime ABI details to help diagnose device-specific native crashes
                try {
                        val is64 = android.os.Process.is64Bit()
                        Log.i(
                                TAG,
                                "Runtime ABI: is64Bit=$is64, primaryAbi=${android.os.Build.SUPPORTED_ABIS.firstOrNull() ?: "?"}"
                        )
                } catch (_: Throwable) {
                        // best-effort only
                }
                Log.i(
                        TAG,
                        "StableDiffusion.load(video) called with finalSequentialLoad=${finalSequentialLoadV}, forceVulkan=${preferPerformanceMode}, offloadToCpu=false, keepClipOnCpu=${finalKeepClipOnCpu}, keepVaeOnCpu=${finalKeepVaeOnCpu}, flashAttn=$flashAttn, vae=$vaePath, taesd=$taesdPath"
                )
                val model =
                        StableDiffusion.load(
                                context = context,
                                modelPath = modelFile.absolutePath,
                                vaePath = vaePath,
                                t5xxlPath = t5File?.absolutePath,
                                taesdPath = taesdPath,
                                nThreads =
                                        CpuTopology.getOptimalThreadCount(
                                                CpuTopology.TaskType.DIFFUSION
                                        ),
                                offloadToCpu = false,
                                sequentialLoad = finalSequentialLoadV,
                                forceVulkan = preferPerformanceMode,
                                preferPerformanceMode = preferPerformanceMode,
                                keepClipOnCpu = finalKeepClipOnCpu,
                                keepVaeOnCpu = finalKeepVaeOnCpu,
                                flashAttn = flashAttn,
                                vaeDecodeOnly = vaeDecodeOnly,
                                flowShift = flowShift,
                                loraModelDir = loraModelDir,
                                loraApplyMode = loraApplyMode
                        )
                val loadTime = System.currentTimeMillis() - loadStart
                val modelSize = modelFile.length()
                Log.i(
                        TAG,
                        "Loaded video model in ${loadTime}ms (size=${modelSize / 1024 / 1024}MB) sequentialLoad=${sequentialLoad}"
                )
                // Use file size as cache size estimate to avoid re-parsing the model file.

                diffusionModelCache.put(cacheKey, model, modelSize, loadTime)
                cachedModel = model
                currentDiffusionModelSpec =
                        LoadedDiffusionModelSpec(
                                modelId = DEFAULT_VIDEO_MODEL_ID,
                                filename = DEFAULT_VIDEO_MODEL_FILENAME,
                                path = modelFile.absolutePath,
                                vaePath = vaePath,
                                t5xxlPath = t5File?.absolutePath,
                                taesdPath = taesdPath,
                                flowShift = flowShift,
                                loraModelDir = loraModelDir,
                                loraApplyMode = loraApplyMode
                        )
                Log.i(
                        TAG,
                        "Loaded Video model from cache: $cacheKey, sequentialLoad=${sequentialLoad}"
                )
                return model
        }

        private suspend fun getOrLoadSmolLM(
                context: Context,
                modelId: String,
                filename: String,
                absolutePath: String? = null
        ): io.aatricks.llmedge.SmolLM {
                // Check existing cache first
                cachedSmolLM?.let {
                        // Check if the loaded model matches the request
                        val spec = currentTextModelSpec
                        if (spec != null &&
                                        spec.modelId == modelId &&
                                        spec.filename == filename &&
                                        spec.path == absolutePath
                        ) {
                                return it
                        }
                        // If not match, properly unload and remove from cache
                        unloadSmolLM()
                }

                val finalPath =
                        if (absolutePath != null) {
                                File(absolutePath)
                        } else {
                                getFile(context, modelId, filename)
                        }

                // Update memory provider for the cache based on context
                textModelCache.systemMemoryProvider = {
                        val am =
                                context.getSystemService(Context.ACTIVITY_SERVICE) as
                                        ActivityManager
                        val mi = ActivityManager.MemoryInfo()
                        am.getMemoryInfo(mi)
                        mi.availMem / (1024L * 1024L)
                }

                // Phase 3: Check model cache
                val cacheKey = finalPath.absolutePath
                textModelCache.get(cacheKey)?.let { cachedModel ->
                        Log.i(TAG, "Loaded SmolLM from cache: $cacheKey")
                        cachedSmolLM = cachedModel
                        currentTextModelSpec = LoadedTextModelSpec(modelId, filename, absolutePath)
                        return cachedModel
                }

                // Phase 3: Use core-aware threading
                var optimalThreads =
                        if (preferPerformanceMode) {
                                CpuTopology.getOptimalThreadCount(
                                        CpuTopology.TaskType.PROMPT_PROCESSING
                                )
                        } else {
                                // Conservative threading for stability/background use
                                2
                        }

                // If the model file is large, use more conservative settings to avoid high memory
                // and concurrency which can cause native failures on some devices.
                val modelSizeMB = finalPath.length() / (1024L * 1024L)
                var overrideContextSize: Long? = null
                if (modelSizeMB >= 250L) {
                        Log.w(
                                TAG,
                                "Large model detected: ${modelSizeMB}MB. Forcing conservative settings: threads=1, context=2048"
                        )
                        optimalThreads = 1
                        overrideContextSize = 2048L
                }
                Log.i(
                        TAG,
                        "Loading SmolLM with $optimalThreads threads (${coreInfo}), vulkan=$preferPerformanceMode"
                )
                val loadStart = System.currentTimeMillis()

                // Initialize SmolLM with Vulkan setting based on performance mode
                // If preferPerformanceMode is false, useVulkan=false (CPU only) to avoid hangs on
                // some devices
                val smol = io.aatricks.llmedge.SmolLM(useVulkan = preferPerformanceMode)

                // Help clear heap before loading large model
                prepareMemoryForLoading()

                smol.load(
                        modelPath = finalPath.absolutePath,
                        params =
                                io.aatricks.llmedge.SmolLM.InferenceParams(
                                        numThreads = optimalThreads,
                                        contextSize = overrideContextSize
                                )
                )

                val loadTime = System.currentTimeMillis() - loadStart
                val modelSize = finalPath.length()

                // Cache the loaded model
                textModelCache.put(cacheKey, smol, modelSize, loadTime)

                cachedSmolLM = smol
                currentTextModelSpec = LoadedTextModelSpec(modelId, filename, absolutePath)
                return smol
        }

        /**
         * Downloads a model from Hugging Face with progress updates. Useful for activities that
         * need to show download progress before generation.
         *
         * Uses Android's system DownloadManager by default to avoid heap memory issues with large
         * model files. The system downloader streams directly to disk without using the app's Java
         * heap.
         */
        suspend fun downloadModel(
                context: Context,
                modelId: String,
                filename: String?,
                revision: String = "main",
                preferSystemDownloader: Boolean = true,
                onProgress: ((Long, Long?) -> Unit)? = null
        ): File {
                return HuggingFaceHub.ensureModelOnDisk(
                                context = context,
                                modelId = modelId,
                                revision = revision,
                                filename = filename,
                                preferSystemDownloader = preferSystemDownloader,
                                onProgress = onProgress
                        )
                        .file
        }

        private fun unloadDiffusionModel() {
                // Prefer removing from cache (which will also close the model) if we have a cache
                // key; otherwise, fall back to closing the cached instance.
                val spec = currentDiffusionModelSpec
                if (spec != null) {
                        val key =
                                makeDiffusionCacheKey(
                                        spec.path,
                                        spec.vaePath,
                                        spec.t5xxlPath,
                                        spec.taesdPath,
                                        spec.flowShift,
                                        spec.loraModelDir,
                                        spec.loraApplyMode
                                )
                        diffusionModelCache.remove(key)
                        currentDiffusionModelSpec = null
                        cachedModel = null
                } else {
                        cachedModel?.close()
                        cachedModel = null
                }
        }

        private fun unloadSmolLM() {
                // Remove from cache if we have a spec (this will also close the model)
                val spec = currentTextModelSpec
                if (spec != null) {
                        val cacheKey = spec.path ?: getModelCacheKey(spec.modelId, spec.filename)
                        textModelCache.remove(cacheKey)
                }
                // Clear local references
                cachedSmolLM = null
                currentTextModelSpec = null
        }

        private fun getModelCacheKey(modelId: String, filename: String): String {
                return "$modelId/$filename"
        }

        private suspend fun ensureVideoFiles(
                context: Context,
                onProgress: ((String, Int, Int) -> Unit)?,
                skipVae: Boolean = false
        ) {
                val hfCallback: ((Long, Long?) -> Unit)? =
                        onProgress?.let { genCb ->
                                { downloaded: Long, total: Long? ->
                                        genCb(
                                                "Downloading video asset: $downloaded/${total ?: "?"}",
                                                0,
                                                0
                                        )
                                }
                        }

                HuggingFaceHub.ensureRepoFileOnDisk(
                        context,
                        DEFAULT_VIDEO_MODEL_ID,
                        "main",
                        DEFAULT_VIDEO_MODEL_FILENAME,
                        emptyList(),
                        null,
                        false,
                        true,
                        hfCallback
                )
                if (!skipVae) {
                        HuggingFaceHub.ensureRepoFileOnDisk(
                                context,
                                DEFAULT_VIDEO_VAE_ID,
                                "main",
                                DEFAULT_VIDEO_VAE_FILENAME,
                                emptyList(),
                                null,
                                false,
                                true,
                                hfCallback
                        )
                }
                HuggingFaceHub.ensureRepoFileOnDisk(
                        context,
                        DEFAULT_VIDEO_T5XXL_ID,
                        "main",
                        DEFAULT_VIDEO_T5XXL_FILENAME,
                        emptyList(),
                        null,
                        false,
                        true,
                        hfCallback
                )
        }

        private suspend fun ensureImageFiles(
                context: Context,
                onProgress: ((String, Int, Int) -> Unit)?
        ) {
                val hfCallback: ((Long, Long?) -> Unit)? =
                        onProgress?.let { genCb ->
                                { downloaded: Long, total: Long? ->
                                        genCb(
                                                "Downloading image asset: $downloaded/${total ?: "?"}",
                                                0,
                                                0
                                        )
                                }
                        }

                HuggingFaceHub.ensureRepoFileOnDisk(
                        context,
                        DEFAULT_IMAGE_MODEL_ID,
                        "main",
                        DEFAULT_IMAGE_MODEL_FILENAME,
                        emptyList(),
                        null,
                        false,
                        true,
                        hfCallback
                )
        }

        private suspend fun getFile(context: Context, repoId: String, filename: String): File {
                return HuggingFaceHub.ensureRepoFileOnDisk(
                                context,
                                repoId,
                                "main",
                                filename,
                                emptyList(),
                                null,
                                false,
                                true,
                                null
                        )
                        .file
        }

        private fun isLowMemoryDevice(context: Context): Boolean {
                val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                val memInfo = ActivityManager.MemoryInfo()
                am.getMemoryInfo(memInfo)
                val totalRamGB = memInfo.totalMem / (1024L * 1024L * 1024L)
                return totalRamGB < 8
        }

        private fun prepareMemoryForLoading() {
                // Native memory is freed immediately on close()
                // Try to nudge the VM to free and compact memory before loading large models in
                // conservative mode. If preferPerformanceMode is enabled, skip the GC hint to
                // avoid unnecessary pauses and let the OS manage memory instead.
                if (!preferPerformanceMode) {
                        try {
                                // Log current memory
                                val rt = Runtime.getRuntime()
                                val used = (rt.totalMemory() - rt.freeMemory()) / (1024L * 1024L)
                                val max = rt.maxMemory() / (1024L * 1024L)
                                Log.d(
                                        TAG,
                                        "Preparing memory: heap_used=${used}MB heap_max=${max}MB"
                                )
                                // Ask for GC; this is a hint to ART and may help on
                                // memory-constrained devices
                                System.gc()
                        } catch (e: Exception) {
                                // no-op
                        }
                }
        }

        /**
         * Build a cache key for diffusion models using model path + optional VAE + T5 path + LoRA.
         */
        private fun makeDiffusionCacheKey(
                modelPath: String,
                vaePath: String?,
                t5Path: String?,
                taesdPath: String? = null,
                flowShift: Float = Float.POSITIVE_INFINITY,
                loraModelDir: String? = null,
                loraApplyMode: StableDiffusion.LoraApplyMode = StableDiffusion.LoraApplyMode.AUTO
        ): String {
                return listOf(
                                modelPath,
                                vaePath ?: "",
                                t5Path ?: "",
                                taesdPath ?: "",
                                flowShift.toString(),
                                loraModelDir ?: "",
                                loraApplyMode.name
                        )
                        .joinToString("|")
        }

        /**
         * Convert raw RGB byte array to Bitmap. The native txt2img/txt2ImgWithPrecomputedCondition
         * return raw RGB bytes (3 bytes per pixel), not encoded image formats like PNG/JPEG.
         */
        // Reuse pixel buffers across conversions to reduce GC pressure
        private val pixelBufferThreadLocal = ThreadLocal<IntArray>()

        private fun rgbBytesToBitmap(rgb: ByteArray, width: Int, height: Int): Bitmap {
                val total = width * height
                var pixels = pixelBufferThreadLocal.get()
                if (pixels == null || pixels.size < total) {
                        pixels = IntArray(total)
                        pixelBufferThreadLocal.set(pixels)
                }
                return io.aatricks.llmedge.vision.ImageUtils.rgbBytesToBitmap(
                        rgb,
                        width,
                        height,
                        pixels
                )
        }
}

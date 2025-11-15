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

package io.aatricks.llmedge

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable

class GGUFReader : Closeable {
    companion object {
        init {
            System.loadLibrary("ggufreader")
        }
    }

    private var nativeHandle: Long = 0L

    suspend fun load(modelPath: String) =
        withContext(Dispatchers.IO) {
            if (nativeHandle != 0L) {
                close()
            }
            nativeHandle = getGGUFContextNativeHandle(modelPath)
        }

    fun getContextSize(): Long? {
        assert(nativeHandle != 0L) { "Use GGUFReader.load() to initialize the reader" }
        val contextSize = getContextSize(nativeHandle)
        return if (contextSize == -1L) {
            null
        } else {
            contextSize
        }
    }

    fun getChatTemplate(): String? {
        assert(nativeHandle != 0L) { "Use GGUFReader.load() to initialize the reader" }
        val chatTemplate = getChatTemplate(nativeHandle)
        return chatTemplate.ifEmpty {
            null
        }
    }

    /**
     * Read the model architecture from GGUF metadata (e.g., "wan", "llama", "stable-diffusion")
     */
    fun getArchitecture(): String? {
        assert(nativeHandle != 0L) { "Use GGUFReader.load() to initialize the reader" }
        val arch = getArchitecture(nativeHandle)
        return arch.ifEmpty { null }
    }

    /**
     * Read parameter count metadata (e.g., "1.3B", "5B", "14B")
     * Falls back to estimating from model file size if not in metadata
     */
    fun getParameterCount(): String? {
        assert(nativeHandle != 0L) { "Use GGUFReader.load() to initialize the reader" }
        val params = getParameterCount(nativeHandle)
        return params.ifEmpty { null }
    }

    /**
     * Read model name from GGUF metadata
     */
    fun getModelName(): String? {
        assert(nativeHandle != 0L) { "Use GGUFReader.load() to initialize the reader" }
        val name = getModelName(nativeHandle)
        return name.ifEmpty { null }
    }

    override fun close() {
        if (nativeHandle != 0L) {
            releaseGGUFContext(nativeHandle)
            nativeHandle = 0L
        }
    }

    /**
     * Returns the native handle (pointer to gguf_context created on the native side)
     */
    private external fun getGGUFContextNativeHandle(modelPath: String): Long

    /**
     * Read the context size (in no. of tokens) from the GGUF file, given the native handle
     */
    private external fun getContextSize(nativeHandle: Long): Long

    /**
     * Read the chat template from the GGUF file, given the native handle
     */
    private external fun getChatTemplate(nativeHandle: Long): String

    /**
     * Read the architecture from GGUF metadata
     */
    private external fun getArchitecture(nativeHandle: Long): String

    /**
     * Read parameter count from GGUF metadata
     */
    private external fun getParameterCount(nativeHandle: Long): String

    /**
     * Read model name from GGUF metadata
     */
    private external fun getModelName(nativeHandle: Long): String

    private external fun releaseGGUFContext(nativeHandle: Long)
}

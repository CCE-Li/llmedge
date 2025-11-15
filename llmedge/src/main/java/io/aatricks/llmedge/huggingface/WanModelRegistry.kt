package io.aatricks.llmedge.huggingface

import android.content.Context
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.io.InputStream

@Serializable
data class WanModelEntry(
    val modelId: String,
    val filename: String,
    val quantization: String? = null,
    val t5ModelId: String? = null,
    val t5Filename: String? = null,
    val vaeFilename: String? = null,
    val sizeBytes: Long? = null,
)

object WanModelRegistry {
    private val json = Json { ignoreUnknownKeys = true }

    fun loadFromAssets(context: Context): List<WanModelEntry> {
        val assetName = "wan-models/model-registry.json"
        val input: InputStream = context.assets.open(assetName)
        val text = input.bufferedReader().use { it.readText() }
        return json.decodeFromString(text)
    }

    fun findById(context: Context, modelId: String): WanModelEntry? =
        loadFromAssets(context).firstOrNull { it.modelId.equals(modelId, ignoreCase = true) }

    fun findByModelIdPrefix(context: Context, modelIdPrefix: String): WanModelEntry? =
        loadFromAssets(context).firstOrNull { it.modelId.startsWith(modelIdPrefix, ignoreCase = true) }
}

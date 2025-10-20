package io.aatricks.llmedge.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions
import kotlinx.coroutines.tasks.await
import java.io.File

/**
 * Lightweight image description engine intended as a fast, on-device fallback
 * when a vision-capable LLM is not available. Combines ML Kit Image Labeling
 * with simple image statistics to produce a concise description.
 */
object LocalImageDescriber {

    data class Description(
        val labels: List<String>,
        val dominantColor: String?,
        val brightness: String?,
        val size: Pair<Int, Int>?,
        val summary: String
    )

    suspend fun describe(context: Context, source: ImageSource): Description {
        val bmp = ImageUtils.imageToBitmap(context, source)
        val size = bmp.width to bmp.height
        val thumb = if (bmp.width * bmp.height > 512 * 512) {
            Bitmap.createScaledBitmap(bmp, 512, (512f * bmp.height / bmp.width).toInt().coerceAtLeast(1), true)
        } else bmp

        val labels = runImageLabels(context, thumb)
        val (brightnessLabel, dominantHex) = computeQuickStats(thumb)

        val top = labels.take(3)
        val parts = mutableListOf<String>()
        if (top.isNotEmpty()) parts += top.joinToString(", ")
        if (brightnessLabel != null) parts += brightnessLabel
        if (dominantHex != null) parts += "dominant $dominantHex"
        val summary = if (parts.isEmpty()) "No salient objects detected" else parts.joinToString("; ")

        if (thumb !== bmp) thumb.recycle()
        return Description(labels = labels, dominantColor = dominantHex, brightness = brightnessLabel, size = size, summary = summary)
    }

    private suspend fun runImageLabels(context: Context, bitmap: Bitmap): List<String> {
        val image = InputImage.fromBitmap(bitmap, 0)
        val labeler = ImageLabeling.getClient(ImageLabelerOptions.DEFAULT_OPTIONS)
        val results = labeler.process(image).await()
        return results
            .filter { it.confidence >= 0.5f }
            .sortedByDescending { it.confidence }
            .map { it.text.lowercase() }
    }

    private fun computeQuickStats(bitmap: Bitmap): Pair<String?, String?> {
        val w = bitmap.width
        val h = bitmap.height
        val step = maxOf(1, (w * h) / 250_000) // sample at most ~250k pixels
        var lumaSum = 0L
        var rSum = 0L
        var gSum = 0L
        var bSum = 0L
        var count = 0
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        var i = 0
        while (i < pixels.size) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            lumaSum += (0.2126 * r + 0.7152 * g + 0.0722 * b).toInt()
            rSum += r
            gSum += g
            bSum += b
            count++
            i += step
        }
        if (count == 0) return null to null
        val luma = (lumaSum / count).toInt()
        val brightness = when {
            luma < 60 -> "dark"
            luma < 140 -> "moderate brightness"
            else -> "bright"
        }
        val rAvg = (rSum / count).toInt()
        val gAvg = (gSum / count).toInt()
        val bAvg = (bSum / count).toInt()
        val hex = String.format("#%02X%02X%02X", rAvg, gAvg, bAvg)
        return brightness to hex
    }
}

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

    /** Convert raw RGB bytes (R,G,B 3 bytes per pixel) to a Bitmap. */
    fun rgbBytesToBitmap(rgb: ByteArray, width: Int, height: Int, pixels: IntArray? = null): Bitmap {
        val total = width * height
        val pixelArray = if (pixels == null || pixels.size < total) IntArray(total) else pixels
        var idx = 0
        var p = 0
        while (idx + 2 < rgb.size && p < total) {
            val r = (rgb[idx].toInt() and 0xFF)
            val g = (rgb[idx + 1].toInt() and 0xFF)
            val b = (rgb[idx + 2].toInt() and 0xFF)
            pixelArray[p] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            idx += 3
            p += 1
        }
        return Bitmap.createBitmap(pixelArray, 0, width, width, height, Bitmap.Config.ARGB_8888)
    }

    /**
     * Creates an animated GIF from a list of Bitmaps.
     * Uses a simple GIF89a encoder with global color table.
     * 
     * @param frames List of Bitmap frames (should be same dimensions)
     * @param delayMs Delay between frames in milliseconds
     * @param output Output stream to write GIF data
     * @param loop Number of times to loop (0 = infinite)
     */
    fun createAnimatedGif(
        frames: List<Bitmap>,
        delayMs: Int = 100,
        output: OutputStream,
        loop: Int = 0
    ) {
        if (frames.isEmpty()) return
        
        val width = frames[0].width
        val height = frames[0].height
        val encoder = GifEncoder()
        encoder.start(output)
        encoder.setRepeat(loop)
        encoder.setDelay(delayMs)
        encoder.setQuality(10) // Lower = better quality but slower
        
        for (frame in frames) {
            // Resize if dimensions don't match
            val resized = if (frame.width != width || frame.height != height) {
                Bitmap.createScaledBitmap(frame, width, height, true)
            } else frame
            
            encoder.addFrame(resized)
            
            if (resized !== frame) {
                resized.recycle()
            }
        }
        
        encoder.finish()
    }

    /**
     * Simple GIF encoder based on NeuQuant neural network quantization.
     * Adapted from open source implementations.
     */
    private class GifEncoder {
        private var width = 0
        private var height = 0
        private var out: OutputStream? = null
        private var pixels: ByteArray? = null
        private var indexedPixels: ByteArray? = null
        private var colorDepth = 8
        private var colorTab: ByteArray? = null
        private var usedEntry = BooleanArray(256)
        private var palSize = 7
        private var dispose = -1
        private var repeat = -1
        private var delay = 0
        private var sample = 10
        private var firstFrame = true

        fun setDelay(ms: Int) { delay = ms / 10 }
        fun setRepeat(iter: Int) { repeat = iter }
        fun setQuality(quality: Int) { sample = quality.coerceIn(1, 30) }

        fun start(os: OutputStream) {
            out = os
            writeString("GIF89a")
        }

        fun addFrame(im: Bitmap): Boolean {
            if (out == null) return false
            
            width = im.width
            height = im.height
            getImagePixels(im)
            analyzePixels()
            
            if (firstFrame) {
                writeLSD()
                writePalette()
                if (repeat >= 0) writeNetscapeExt()
            }
            
            writeGraphicCtrlExt()
            writeImageDesc()
            if (!firstFrame) writePalette()
            writePixels()
            firstFrame = false
            return true
        }

        fun finish(): Boolean {
            out?.write(0x3b) // GIF trailer
            out?.flush()
            return true
        }

        private fun getImagePixels(image: Bitmap) {
            val w = image.width
            val h = image.height
            pixels = ByteArray(w * h * 3)
            val data = IntArray(w * h)
            image.getPixels(data, 0, w, 0, 0, w, h)
            
            for (i in data.indices) {
                val px = data[i]
                val j = i * 3
                pixels!![j] = ((px shr 16) and 0xff).toByte()
                pixels!![j + 1] = ((px shr 8) and 0xff).toByte()
                pixels!![j + 2] = (px and 0xff).toByte()
            }
        }

        private fun analyzePixels() {
            val len = pixels!!.size
            val nPix = len / 3
            indexedPixels = ByteArray(nPix)
            
            val nq = NeuQuant(pixels!!, len, sample)
            colorTab = nq.process()
            
            for (i in 0 until nPix) {
                val index = nq.map(
                    pixels!![i * 3].toInt() and 0xff,
                    pixels!![i * 3 + 1].toInt() and 0xff,
                    pixels!![i * 3 + 2].toInt() and 0xff
                )
                usedEntry[index] = true
                indexedPixels!![i] = index.toByte()
            }
            pixels = null
            colorDepth = 8
            palSize = 7
        }

        private fun writeLSD() {
            writeShort(width)
            writeShort(height)
            out?.write(0x80 or 0x70 or palSize)
            out?.write(0)
            out?.write(0)
        }

        private fun writeNetscapeExt() {
            out?.write(0x21)
            out?.write(0xff)
            out?.write(11)
            writeString("NETSCAPE2.0")
            out?.write(3)
            out?.write(1)
            writeShort(repeat)
            out?.write(0)
        }

        private fun writePalette() {
            out?.write(colorTab!!, 0, colorTab!!.size)
            val n = 3 * 256 - colorTab!!.size
            for (i in 0 until n) out?.write(0)
        }

        private fun writeGraphicCtrlExt() {
            out?.write(0x21)
            out?.write(0xf9)
            out?.write(4)
            val transp = 0
            var disp = if (dispose >= 0) dispose else 0
            disp = disp shl 2
            out?.write(disp or transp)
            writeShort(delay)
            out?.write(0)
            out?.write(0)
        }

        private fun writeImageDesc() {
            out?.write(0x2c)
            writeShort(0)
            writeShort(0)
            writeShort(width)
            writeShort(height)
            if (firstFrame) out?.write(0) else out?.write(0x80 or palSize)
        }

        private fun writePixels() {
            val encoder = LZWEncoder(width, height, indexedPixels!!, colorDepth)
            encoder.encode(out!!)
        }

        private fun writeShort(value: Int) {
            out?.write(value and 0xff)
            out?.write((value shr 8) and 0xff)
        }

        private fun writeString(s: String) {
            for (c in s) out?.write(c.code)
        }
    }

    /**
     * NeuQuant Neural-Net Quantization Algorithm
     * Adapted from Anthony Dekker's NeuQuant algorithm
     */
    private class NeuQuant(private val thepicture: ByteArray, private val lengthcount: Int, private val samplefac: Int) {
        private val netsize = 256
        private val prime1 = 499
        private val prime2 = 491
        private val prime3 = 487
        private val prime4 = 503
        private val maxnetpos = netsize - 1
        private val netbiasshift = 4
        private val ncycles = 100
        private val intbiasshift = 16
        private val intbias = 1 shl intbiasshift
        private val gammashift = 10
        private val betashift = 10
        private val beta = intbias shr betashift
        private val betagamma = intbias shl (gammashift - betashift)
        private val initrad = netsize shr 3
        private val radiusbiasshift = 6
        private val radiusbias = 1 shl radiusbiasshift
        private val initradius = initrad * radiusbias
        private val radiusdec = 30
        private val alphabiasshift = 10
        private val initalpha = 1 shl alphabiasshift
        private val radbiasshift = 8
        private val radbias = 1 shl radbiasshift
        private val alpharadbshift = alphabiasshift + radbiasshift
        private val alpharadbias = 1 shl alpharadbshift

        private val network = Array(netsize) { IntArray(4) }
        private val netindex = IntArray(256)
        private val bias = IntArray(netsize)
        private val freq = IntArray(netsize)
        private val radpower = IntArray(initrad)

        init {
            for (i in 0 until netsize) {
                val p = network[i]
                p[0] = (i shl (netbiasshift + 8)) / netsize
                p[1] = (i shl (netbiasshift + 8)) / netsize
                p[2] = (i shl (netbiasshift + 8)) / netsize
                freq[i] = intbias / netsize
                bias[i] = 0
            }
        }

        fun process(): ByteArray {
            learn()
            unbiasnet()
            inxbuild()
            return colorMap()
        }

        private fun colorMap(): ByteArray {
            val map = ByteArray(3 * netsize)
            val index = IntArray(netsize)
            for (i in 0 until netsize) index[network[i][3]] = i
            var k = 0
            for (i in 0 until netsize) {
                val j = index[i]
                map[k++] = network[j][0].toByte()
                map[k++] = network[j][1].toByte()
                map[k++] = network[j][2].toByte()
            }
            return map
        }

        private fun inxbuild() {
            var previouscol = 0
            var startpos = 0
            for (i in 0 until netsize) {
                val p = network[i]
                var smallpos = i
                var smallval = p[1]
                for (j in i + 1 until netsize) {
                    val q = network[j]
                    if (q[1] < smallval) {
                        smallpos = j
                        smallval = q[1]
                    }
                }
                val q = network[smallpos]
                if (i != smallpos) {
                    var j = q[0]; q[0] = p[0]; p[0] = j
                    j = q[1]; q[1] = p[1]; p[1] = j
                    j = q[2]; q[2] = p[2]; p[2] = j
                    j = q[3]; q[3] = p[3]; p[3] = j
                }
                if (smallval != previouscol) {
                    netindex[previouscol] = (startpos + i) shr 1
                    for (j in previouscol + 1 until smallval) netindex[j] = i
                    previouscol = smallval
                    startpos = i
                }
            }
            netindex[previouscol] = (startpos + maxnetpos) shr 1
            for (j in previouscol + 1 until 256) netindex[j] = maxnetpos
        }

        private fun learn() {
            val lengthcount = this.lengthcount
            val alphadec = 30 + ((samplefac - 1) / 3)
            val samplepixels = lengthcount / (3 * samplefac)
            var delta = samplepixels / ncycles
            var alpha = initalpha
            var radius = initradius

            var rad = radius shr radiusbiasshift
            if (rad <= 1) rad = 0
            for (i in 0 until rad) radpower[i] = alpha * (((rad * rad - i * i) * radbias) / (rad * rad))

            val step = when {
                lengthcount < 500009 -> 3 * prime1
                lengthcount % prime1 != 0 -> 3 * prime1
                lengthcount % prime2 != 0 -> 3 * prime2
                lengthcount % prime3 != 0 -> 3 * prime3
                else -> 3 * prime4
            }

            var pix = 0
            for (i in 0 until samplepixels) {
                val b = (thepicture[pix].toInt() and 0xff) shl netbiasshift
                val g = (thepicture[pix + 1].toInt() and 0xff) shl netbiasshift
                val r = (thepicture[pix + 2].toInt() and 0xff) shl netbiasshift
                var j = contest(b, g, r)
                altersingle(alpha, j, b, g, r)
                if (rad != 0) alterneigh(rad, j, b, g, r)
                pix += step
                if (pix >= lengthcount) pix -= lengthcount
                if (delta == 0) delta = 1
                if (i % delta == 0) {
                    alpha -= alpha / alphadec
                    radius -= radius / radiusdec
                    rad = radius shr radiusbiasshift
                    if (rad <= 1) rad = 0
                    for (k in 0 until rad) radpower[k] = alpha * (((rad * rad - k * k) * radbias) / (rad * rad))
                }
            }
        }

        fun map(b: Int, g: Int, r: Int): Int {
            var bestd = 1000
            var best = -1
            var i = netindex[g]
            var j = i - 1
            while (i < netsize || j >= 0) {
                if (i < netsize) {
                    val p = network[i]
                    var dist = p[1] - g
                    if (dist >= bestd) i = netsize
                    else {
                        i++
                        if (dist < 0) dist = -dist
                        var a = p[0] - b; if (a < 0) a = -a
                        dist += a
                        if (dist < bestd) {
                            a = p[2] - r; if (a < 0) a = -a
                            dist += a
                            if (dist < bestd) { bestd = dist; best = p[3] }
                        }
                    }
                }
                if (j >= 0) {
                    val p = network[j]
                    var dist = g - p[1]
                    if (dist >= bestd) j = -1
                    else {
                        j--
                        if (dist < 0) dist = -dist
                        var a = p[0] - b; if (a < 0) a = -a
                        dist += a
                        if (dist < bestd) {
                            a = p[2] - r; if (a < 0) a = -a
                            dist += a
                            if (dist < bestd) { bestd = dist; best = p[3] }
                        }
                    }
                }
            }
            return best
        }

        private fun unbiasnet() {
            for (i in 0 until netsize) {
                network[i][0] = network[i][0] shr netbiasshift
                network[i][1] = network[i][1] shr netbiasshift
                network[i][2] = network[i][2] shr netbiasshift
                network[i][3] = i
            }
        }

        private fun alterneigh(rad: Int, i: Int, b: Int, g: Int, r: Int) {
            var lo = i - rad; if (lo < -1) lo = -1
            var hi = i + rad; if (hi > netsize) hi = netsize
            var j = i + 1
            var k = i - 1
            var m = 1
            while (j < hi || k > lo) {
                val a = radpower[m++]
                if (j < hi) {
                    val p = network[j++]
                    p[0] -= (a * (p[0] - b)) / alpharadbias
                    p[1] -= (a * (p[1] - g)) / alpharadbias
                    p[2] -= (a * (p[2] - r)) / alpharadbias
                }
                if (k > lo) {
                    val p = network[k--]
                    p[0] -= (a * (p[0] - b)) / alpharadbias
                    p[1] -= (a * (p[1] - g)) / alpharadbias
                    p[2] -= (a * (p[2] - r)) / alpharadbias
                }
            }
        }

        private fun altersingle(alpha: Int, i: Int, b: Int, g: Int, r: Int) {
            val n = network[i]
            n[0] -= (alpha * (n[0] - b)) / initalpha
            n[1] -= (alpha * (n[1] - g)) / initalpha
            n[2] -= (alpha * (n[2] - r)) / initalpha
        }

        private fun contest(b: Int, g: Int, r: Int): Int {
            var bestd = Int.MAX_VALUE.inv()
            var bestbiasd = bestd
            var bestpos = -1
            var bestbiaspos = bestpos
            for (i in 0 until netsize) {
                val n = network[i]
                var dist = n[0] - b; if (dist < 0) dist = -dist
                var a = n[1] - g; if (a < 0) a = -a
                dist += a
                a = n[2] - r; if (a < 0) a = -a
                dist += a
                if (dist < bestd) { bestd = dist; bestpos = i }
                val biasdist = dist - ((bias[i]) shr (intbiasshift - netbiasshift))
                if (biasdist < bestbiasd) { bestbiasd = biasdist; bestbiaspos = i }
                val betafreq = (freq[i] shr betashift)
                freq[i] -= betafreq
                bias[i] += (betafreq shl gammashift)
            }
            freq[bestpos] += beta
            bias[bestpos] -= betagamma
            return bestbiaspos
        }
    }

    /**
     * LZW encoder for GIF
     */
    private class LZWEncoder(private val imgW: Int, private val imgH: Int, private val pixAry: ByteArray, private val initCodeSize: Int) {
        private val EOF = -1
        private var curPixel = 0
        private var codeSize = 0
        private var clearCode = 0
        private var EOFCode = 0
        private var freeEnt = 0
        private var maxCode = 0
        private var aCount = 0
        private val HSIZE = 5003
        private val htab = IntArray(HSIZE)
        private val codetab = IntArray(HSIZE)
        private val accum = ByteArray(256)
        private var gInitBits = 0
        private var curAccum = 0
        private var curBits = 0
        private val masks = intArrayOf(0x0000, 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF,
            0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF)

        fun encode(os: OutputStream) {
            os.write(initCodeSize)
            curPixel = 0
            compress(initCodeSize + 1, os)
            os.write(0)
        }

        private fun compress(initBits: Int, outs: OutputStream) {
            gInitBits = initBits
            clearCode = 1 shl (initBits - 1)
            EOFCode = clearCode + 1
            freeEnt = clearCode + 2
            aCount = 0
            var ent = nextPixel()
            var hshift = 0
            var fcode = HSIZE
            while (fcode < 65536) { hshift++; fcode *= 2 }
            hshift = 8 - hshift
            val hSizeReg = HSIZE
            clearHash(hSizeReg)
            output(clearCode, outs)

            var c: Int
            while (nextPixel().also { c = it } != EOF) {
                fcode = (c shl 12) + ent
                var i = (c shl hshift) xor ent
                if (htab[i] == fcode) { ent = codetab[i]; continue }
                else if (htab[i] >= 0) {
                    var disp = hSizeReg - i
                    if (i == 0) disp = 1
                    do { i -= disp; if (i < 0) i += hSizeReg }
                    while (htab[i] >= 0 && htab[i] != fcode)
                }
                if (htab[i] == fcode) { ent = codetab[i]; continue }
                output(ent, outs)
                ent = c
                if (freeEnt < 4096) {
                    codetab[i] = freeEnt++
                    htab[i] = fcode
                } else clearHash(hSizeReg).also { output(clearCode, outs) }
            }
            output(ent, outs)
            output(EOFCode, outs)
        }

        private fun clearHash(hsize: Int) {
            for (i in 0 until hsize) htab[i] = -1
            freeEnt = clearCode + 2
            codeSize = gInitBits
            maxCode = (1 shl codeSize) - 1
        }

        private fun nextPixel(): Int {
            if (curPixel >= pixAry.size) return EOF
            return pixAry[curPixel++].toInt() and 0xff
        }

        private fun output(code: Int, outs: OutputStream) {
            curAccum = curAccum and masks[curBits]
            curAccum = if (curBits > 0) curAccum or (code shl curBits) else code
            curBits += codeSize
            while (curBits >= 8) {
                accum[aCount++] = (curAccum and 0xff).toByte()
                if (aCount >= 254) flushChar(outs)
                curAccum = curAccum shr 8
                curBits -= 8
            }
            if (freeEnt > maxCode || code == clearCode) {
                if (code == clearCode) { codeSize = gInitBits; maxCode = (1 shl codeSize) - 1 }
                else { codeSize++; maxCode = if (codeSize == 12) 4096 else (1 shl codeSize) - 1 }
            }
            if (code == EOFCode) {
                while (curBits > 0) {
                    accum[aCount++] = (curAccum and 0xff).toByte()
                    if (aCount >= 254) flushChar(outs)
                    curAccum = curAccum shr 8
                    curBits -= 8
                }
                flushChar(outs)
            }
        }

        private fun flushChar(outs: OutputStream) {
            if (aCount > 0) {
                outs.write(aCount)
                outs.write(accum, 0, aCount)
                aCount = 0
            }
        }
    }
}
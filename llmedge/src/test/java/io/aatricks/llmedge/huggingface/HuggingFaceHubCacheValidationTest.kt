package io.aatricks.llmedge.huggingface

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.security.MessageDigest

class HuggingFaceHubCacheValidationTest {
    private fun computeSha256String(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { fis ->
            val buffer = ByteArray(8 * 1024)
            var bytesRead = fis.read(buffer)
            while (bytesRead >= 0) {
                md.update(buffer, 0, bytesRead)
                bytesRead = fis.read(buffer)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    @Test
    fun `size match returns true`() {
        val temp = File.createTempFile("hf-test", ".tmp")
        temp.writeText("abcdef")
        assertTrue(HuggingFaceHub.isFileValidCached(temp, temp.length(), null))
    }

    @Test
    fun `size mismatch returns false`() {
        val temp = File.createTempFile("hf-test", ".tmp")
        temp.writeText("12345")
        assertFalse(HuggingFaceHub.isFileValidCached(temp, temp.length() + 10, null))
    }

    @Test
    fun `sha match returns true`() {
        val temp = File.createTempFile("hf-test", ".tmp")
        temp.writeText("match-sha-test")
        val sha = computeSha256String(temp)
        assertTrue(HuggingFaceHub.isFileValidCached(temp, null, sha))
    }

    @Test
    fun `sha prefix accepted`() {
        val temp = File.createTempFile("hf-test", ".tmp")
        temp.writeText("prefix-sha-test")
        val sha = computeSha256String(temp)
        // Some metadata uses a 'sha256:' prefix — ensure the normalization accepts this form.
        assertTrue(HuggingFaceHub.isFileValidCached(temp, null, "sha256:$sha"))
    }

    @Test
    fun `sha match overrides size mismatch`() {
        val temp = File.createTempFile("hf-test", ".tmp")
        temp.writeText("override-test")
        val sha = computeSha256String(temp)
        // Provide a size that does not match, but the SHA is correct — SHA should be authoritative.
        assertTrue(HuggingFaceHub.isFileValidCached(temp, temp.length() + 10, sha))
    }

    @Test
    fun `sha mismatch returns false`() {
        val temp = File.createTempFile("hf-test", ".tmp")
        temp.writeText("sha-fail")
        val sha = "deadbeef"
        assertFalse(HuggingFaceHub.isFileValidCached(temp, null, sha))
    }

    @Test
    fun `no size or sha uses non-empty file fallback`() {
        val temp = File.createTempFile("hf-test", ".tmp")
        temp.writeText("anything")
        assertTrue(HuggingFaceHub.isFileValidCached(temp, null, null))
        // empty file returns false
        val empty = File.createTempFile("hf-empty", ".tmp")
        empty.writeText("")
        assertFalse(HuggingFaceHub.isFileValidCached(empty, null, null))
    }
}

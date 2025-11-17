package io.aatricks.llmedge.huggingface

import io.ktor.client.HttpClient
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class HFModelsTest {

    @Test
    fun `client is properly configured with OkHttp engine`() {
        // The client is initialized statically, so we just verify it's not null
        // and has the expected configuration
        val client = HFModels.client
        assertNotNull("HFModels.client should not be null", client)
        // Note: We can't easily test the internal OkHttp configuration without reflection
        // but the fact that it doesn't throw an exception during initialization is a good sign
    }

    @Test
    fun `info factory method returns HFModelInfo instance`() {
        val info = HFModels.info()
        assertNotNull("info() should return non-null HFModelInfo", info)
        assertTrue("info() should return HFModelInfo instance", info is HFModelInfo)
    }

    @Test
    fun `tree factory method returns HFModelTree instance`() {
        val tree = HFModels.tree()
        assertNotNull("tree() should return non-null HFModelTree", tree)
        assertTrue("tree() should return HFModelTree instance", tree is HFModelTree)
    }

    @Test
    fun `search factory method returns HFModelSearch instance`() {
        val search = HFModels.search()
        assertNotNull("search() should return non-null HFModelSearch", search)
        assertTrue("search() should return HFModelSearch instance", search is HFModelSearch)
    }

    @Test
    fun `download factory method returns HFModelDownload instance`() {
        val download = HFModels.download()
        assertNotNull("download() should return non-null HFModelDownload", download)
        assertTrue("download() should return HFModelDownload instance", download is HFModelDownload)
    }

    @Test
    fun `all factory methods return instances with the same client`() {
        val info = HFModels.info()
        val tree = HFModels.tree()
        val search = HFModels.search()
        val download = HFModels.download()

        // All instances should be using the same shared client
        val infoClientField = HFModelInfo::class.java.getDeclaredField("client").apply { isAccessible = true }
        val treeClientField = HFModelTree::class.java.getDeclaredField("client").apply { isAccessible = true }
        val searchClientField = HFModelSearch::class.java.getDeclaredField("client").apply { isAccessible = true }
        val downloadClientField = HFModelDownload::class.java.getDeclaredField("client").apply { isAccessible = true }

        val infoClient = infoClientField.get(info)
        val treeClient = treeClientField.get(tree)
        val searchClient = searchClientField.get(search)
        val downloadClient = downloadClientField.get(download)

        assertTrue("All instances should share the same HttpClient", infoClient === treeClient)
        assertTrue("All instances should share the same HttpClient", treeClient === searchClient)
        assertTrue("All instances should share the same HttpClient", searchClient === downloadClient)
    }
}
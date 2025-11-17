import org.gradle.testing.jacoco.plugins.JacocoTaskExtension
import org.gradle.testing.jacoco.tasks.JacocoReport
/*
 * Copyright (C) 2024 Shubham Panchal
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

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    id("org.jetbrains.kotlin.plugin.serialization") version "2.0.0"
    id("jacoco")
}

android {
    // Keeping the Kotlin package as-is to avoid JNI breakage; namespace can be branded separately if desired
    namespace = "io.aatricks.llmedge"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        minSdk = 30  // Vulkan 1.2 requires API 30+ (Android 11)
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
        externalNativeBuild {
            cmake {
                cppFlags += listOf()
                // allow compiling 16 KB page-aligned shared libraries
                // https://developer.android.com/guide/practices/page-sizes#compile-r27
                arguments += listOf("-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON")
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                arguments += "-DSD_VULKAN=ON"
                arguments += "-DGGML_VULKAN=ON"
                arguments += "-DWAN_SUPPORT=ON"

                // (debugging) uncomment the following line to enable debug builds
                // and attach hardware-assisted address sanitizer
                // arguments += "-DCMAKE_BUILD_TYPE=Debug"
                // arguments += listOf("-DANDROID_SANITIZE=hwaddress")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
        testOptions {
            unitTests.all {
                it.systemProperty("llmedge.disableNativeLoad", "true")
            }
            managedDevices {
                devices {
                    maybeCreate<com.android.build.api.dsl.ManagedVirtualDevice>("pixel6api33").apply {
                        device = "Pixel 6"
                        apiLevel = 33
                        systemImageSource = "aosp-atd"
                    }
                }
            }
        }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
            // Handle duplicate files from JavaCPP
            pickFirsts += "META-INF/native-image/**"
            pickFirsts += "META-INF/maven/**"
            pickFirsts += "META-INF/INDEX.LIST"
            pickFirsts += "META-INF/LICENSE"
            pickFirsts += "META-INF/LICENSE.txt"
            pickFirsts += "META-INF/NOTICE"
            pickFirsts += "META-INF/NOTICE.txt"
            pickFirsts += "META-INF/LICENSE.md"
            pickFirsts += "META-INF/LICENSE-notice.md"
        }
    }
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.1")
    // Sentence Embeddings (on-device) - provides ONNX-based sentence-transformers
    implementation("io.gitlab.shubham0204:sentence-embeddings:v6")

    // Hugging Face Hub client (Ktor + JSON serialization)
    implementation("io.ktor:ktor-client-core:2.3.12")
    implementation("io.ktor:ktor-client-okhttp:2.3.12")
    implementation("io.ktor:ktor-client-content-negotiation:2.3.12")
    implementation("io.ktor:ktor-serialization-kotlinx-json:2.3.12")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")

    // PDF parsing on Android (Apache 2.0)
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")

    // JSON serialization (for simple local embedding index persistence)
    implementation("com.google.code.gson:gson:2.11.0")
    
    // OCR support - Google ML Kit Text Recognition (Tesseract removed)
    
    // OCR support - Google ML Kit Text Recognition
    implementation("com.google.mlkit:text-recognition:16.0.0")
    // Image labeling for fast local description
    implementation("com.google.mlkit:image-labeling:17.0.7")
    // Optional: Additional language support for ML Kit
    // implementation("com.google.mlkit:text-recognition-chinese:16.0.0")
    // implementation("com.google.mlkit:text-recognition-japanese:16.0.0")
    // implementation("com.google.mlkit:text-recognition-korean:16.0.0")
    
    // Coroutines support for ML Kit
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.10.1")

    testImplementation(libs.junit)
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.1")
    testImplementation("org.jetbrains.kotlin:kotlin-reflect:2.0.0")
    testImplementation("io.mockk:mockk:1.13.12")
    testImplementation("io.ktor:ktor-client-mock:2.3.12")
    testImplementation("org.robolectric:robolectric:4.13")
    testImplementation("androidx.test:core:1.6.1")

    androidTestImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
    androidTestImplementation("io.mockk:mockk-android:1.13.12")
}

jacoco {
    toolVersion = "0.8.11"
}

tasks.withType<Test>().configureEach {
    extensions.configure(JacocoTaskExtension::class) {
        isIncludeNoLocationClasses = true
        excludes = listOf("jdk.internal.*")
    }
}

val jacocoTestReport by tasks.registering(JacocoReport::class) {
    dependsOn("testDebugUnitTest")

    reports {
        xml.required.set(true)
        html.required.set(true)

        val reportsDir = layout.buildDirectory.dir("reports/jacoco/jacocoTestReport")
        xml.outputLocation.set(reportsDir.map { it.file("jacocoTestReport.xml") })
        html.outputLocation.set(reportsDir.map { it.dir("html") })
    }

    val fileFilter = listOf(
        "**/R.class",
        "**/R$*.class",
        "**/BuildConfig.*",
        "**/Manifest*.*",
        "**/*Test*.*",
        "**/*$*Companion*",
        // Exclude non-core Android-facing packages from coverage to focus on core library logic
        // These areas involve platform services (ML Kit OCR, Camera, DownloadManager) and
        // external integrations that are not meaningful for unit test coverage targets.
        "**/io/aatricks/llmedge/vision/**",
        "**/io/aatricks/llmedge/vision/ocr/**",
        "**/io/aatricks/llmedge/rag/**",
        "**/io/aatricks/llmedge/huggingface/**",
    )

    val classTrees = listOf(
        layout.buildDirectory.dir("intermediates/javac/debug/classes"),
        layout.buildDirectory.dir("tmp/kotlin-classes/debug")
    ).map { dirProvider ->
        dirProvider.map { dir ->
            fileTree(dir) {
                // Only include core library package and exclude Android/platform-heavy subpackages
                include("io/aatricks/llmedge/**")
                exclude(fileFilter)
            }
        }
    }

    classDirectories.setFrom(classTrees)
    sourceDirectories.setFrom(files("src/main/java", "src/main/kotlin"))
    val coverageData = layout.buildDirectory.asFileTree.matching {
        include("jacoco/testDebugUnitTest.exec")
        include("outputs/unit_test_code_coverage/**/testDebugUnitTest.exec")
        include("outputs/unit_test_code_coverage/**/*.ec")
    }
    executionData.setFrom(coverageData)
}

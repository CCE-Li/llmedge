#include "sd_jni_internal.h"

#include <jni.h>

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SD_TEST_JAVA_CLASS_DIR
#define SD_TEST_JAVA_CLASS_DIR "."
#endif

extern "C" {
JNIEXPORT jlong JNICALL Java_io_aatricks_llmedge_StableDiffusion_nativeCreate(
        JNIEnv* env, jclass clazz, jstring jModelPath, jstring jVaePath,
        jint nThreads, jboolean offloadToCpu, jboolean keepClipOnCpu, jboolean keepVaeOnCpu);
JNIEXPORT void JNICALL Java_io_aatricks_llmedge_StableDiffusion_nativeDestroy(JNIEnv* env, jobject thiz, jlong handlePtr);
JNIEXPORT jobjectArray JNICALL Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2Vid(
        JNIEnv* env, jobject thiz, jlong handlePtr,
        jstring jPrompt, jstring jNegative,
        jint width, jint height,
        jint videoFrames, jint steps, jfloat cfg, jlong seed,
        jbyteArray jInitImage, jint initWidth, jint initHeight);
JNIEXPORT void JNICALL Java_io_aatricks_llmedge_StableDiffusion_nativeSetProgressCallback(
        JNIEnv* env, jobject thiz, jlong handlePtr, jobject progressCallback);
JNIEXPORT void JNICALL Java_io_aatricks_llmedge_StableDiffusion_nativeCancelGeneration(
        JNIEnv* env, jobject thiz, jlong handlePtr);
}

static std::atomic<int> g_frameBufferFrees{0};
static std::atomic<int> g_frameArrayFrees{0};

void sd_jni_notify_frame_buffer_freed(const void* ptr) {
    if (ptr) {
        g_frameBufferFrees.fetch_add(1);
    }
}

void sd_jni_notify_frame_array_freed(const void* ptr) {
    if (ptr) {
        g_frameArrayFrees.fetch_add(1);
    }
}

static void reset_free_counters() {
    g_frameBufferFrees.store(0);
    g_frameArrayFrees.store(0);
}

class JvmHolder {
public:
    JvmHolder() {
        JavaVMInitArgs vm_args{};
        vm_args.version = JNI_VERSION_1_8;

        std::string classPathOption = std::string("-Djava.class.path=") + SD_TEST_JAVA_CLASS_DIR;
        options_.push_back(classPathOption);

        std::vector<JavaVMOption> vmOptions;
        for (auto& opt : options_) {
            JavaVMOption option{};
            option.optionString = opt.data();
            vmOptions.push_back(option);
        }
        vm_args.nOptions = static_cast<jint>(vmOptions.size());
        vm_args.options = vmOptions.data();
        vm_args.ignoreUnrecognized = JNI_TRUE;

        JNIEnv* envLocal = nullptr;
        if (JNI_CreateJavaVM(&vm_, reinterpret_cast<void**>(&envLocal), &vm_args) != JNI_OK) {
            throw std::runtime_error("Failed to create JVM for native tests");
        }
        env_ = envLocal;
    }

    ~JvmHolder() {
        if (vm_) {
            vm_->DestroyJavaVM();
        }
    }

    JNIEnv* env() const { return env_; }
    JavaVM* vm() const { return vm_; }

private:
    JavaVM* vm_ = nullptr;
    JNIEnv* env_ = nullptr;
    std::vector<std::string> options_;
};

static bool test_nativeTxt2Vid_memory(JNIEnv* env) {
    reset_free_counters();
    jstring modelPath = env->NewStringUTF("stub-model.gguf");
    jlong handle = Java_io_aatricks_llmedge_StableDiffusion_nativeCreate(
            env, nullptr, modelPath, nullptr, 4, JNI_FALSE, JNI_FALSE, JNI_FALSE);
    env->DeleteLocalRef(modelPath);
    if (handle == 0) {
        std::cerr << "nativeCreate returned null handle" << std::endl;
        return false;
    }

    jstring prompt = env->NewStringUTF("hello world");
    jobjectArray frames = Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2Vid(
            env, nullptr, handle, prompt, nullptr,
            256, 256, 4, 12, 7.5f, 42L,
            nullptr, 0, 0);
    env->DeleteLocalRef(prompt);

    bool success = true;
    if (!frames) {
        std::cerr << "nativeTxt2Vid returned null" << std::endl;
        success = false;
    } else {
        const jsize frameCount = env->GetArrayLength(frames);
        if (frameCount != 4) {
            std::cerr << "Expected 4 frames, got " << frameCount << std::endl;
            success = false;
        }
        const jsize expectedBytes = 256 * 256 * 3;
        for (jsize i = 0; i < frameCount; ++i) {
            auto frameBytes = static_cast<jbyteArray>(env->GetObjectArrayElement(frames, i));
            if (!frameBytes) {
                std::cerr << "Frame " << i << " is null" << std::endl;
                success = false;
                continue;
            }
            jsize length = env->GetArrayLength(frameBytes);
            if (length != expectedBytes) {
                std::cerr << "Frame " << i << " expected " << expectedBytes
                          << " bytes, got " << length << std::endl;
                success = false;
            }
            env->DeleteLocalRef(frameBytes);
        }
        env->DeleteLocalRef(frames);
    }

    Java_io_aatricks_llmedge_StableDiffusion_nativeDestroy(env, nullptr, handle);

    if (g_frameBufferFrees.load() != 4) {
        std::cerr << "Expected 4 frame buffer frees, got " << g_frameBufferFrees.load() << std::endl;
        success = false;
    }
    if (g_frameArrayFrees.load() != 1) {
        std::cerr << "Expected 1 frame array free, got " << g_frameArrayFrees.load() << std::endl;
        success = false;
    }
    return success;
}

static bool test_progress_callback_bridge(JNIEnv* env) {
    jstring modelPath = env->NewStringUTF("stub-model.gguf");
    jlong handlePtr = Java_io_aatricks_llmedge_StableDiffusion_nativeCreate(
            env, nullptr, modelPath, nullptr, 2, JNI_FALSE, JNI_FALSE, JNI_FALSE);
    env->DeleteLocalRef(modelPath);
    if (handlePtr == 0) {
        std::cerr << "Failed to create handle for progress test" << std::endl;
        return false;
    }

    jclass callbackClass = env->FindClass("io/aatricks/llmedge/NativeTestProgressCallback");
    if (!callbackClass) {
        std::cerr << "Unable to locate NativeTestProgressCallback" << std::endl;
        Java_io_aatricks_llmedge_StableDiffusion_nativeDestroy(env, nullptr, handlePtr);
        return false;
    }
    jmethodID ctor = env->GetMethodID(callbackClass, "<init>", "()V");
    jobject callbackInstance = env->NewObject(callbackClass, ctor);
    if (!callbackInstance) {
        std::cerr << "Failed to instantiate progress callback" << std::endl;
        Java_io_aatricks_llmedge_StableDiffusion_nativeDestroy(env, nullptr, handlePtr);
        return false;
    }

    Java_io_aatricks_llmedge_StableDiffusion_nativeSetProgressCallback(env, nullptr, handlePtr, callbackInstance);

    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    handle->totalFrames = 4;
    handle->stepsPerFrame = 3;
    handle->totalSteps = handle->totalFrames * handle->stepsPerFrame;
    handle->currentFrame = 0;

    try {
        sd_video_progress_wrapper(5, handle->totalSteps, 1.5f, handle);
    } catch (const std::exception& ex) {
        std::cerr << "Progress wrapper threw unexpectedly: " << ex.what() << std::endl;
        Java_io_aatricks_llmedge_StableDiffusion_nativeDestroy(env, nullptr, handlePtr);
        return false;
    }

    jfieldID callCountField = env->GetFieldID(callbackClass, "callCount", "I");
    jfieldID lastFrameField = env->GetFieldID(callbackClass, "lastFrame", "I");
    jint callCount = env->GetIntField(callbackInstance, callCountField);
    jint reportedFrame = env->GetIntField(callbackInstance, lastFrameField);
    if (callCount == 0 || reportedFrame != 1) {
        std::cerr << "Progress callback did not update fields as expected" << std::endl;
        Java_io_aatricks_llmedge_StableDiffusion_nativeDestroy(env, nullptr, handlePtr);
        return false;
    }

    handle->cancellationRequested.store(true);
    bool cancellationRaised = false;
    try {
        sd_video_progress_wrapper(6, handle->totalSteps, 2.0f, handle);
    } catch (const std::runtime_error&) {
        cancellationRaised = true;
    }
    if (!cancellationRaised) {
        std::cerr << "Expected cancellation exception" << std::endl;
        Java_io_aatricks_llmedge_StableDiffusion_nativeDestroy(env, nullptr, handlePtr);
        return false;
    }

    Java_io_aatricks_llmedge_StableDiffusion_nativeCancelGeneration(env, nullptr, handlePtr);
    Java_io_aatricks_llmedge_StableDiffusion_nativeDestroy(env, nullptr, handlePtr);
    env->DeleteLocalRef(callbackInstance);
    return true;
}

int main() {
    try {
        JvmHolder jvm;
        JNIEnv* env = jvm.env();
        bool txt2vidResult = test_nativeTxt2Vid_memory(env);
        bool progressResult = test_progress_callback_bridge(env);
        if (!txt2vidResult || !progressResult) {
            std::cerr << "video_jni_tests FAILED" << std::endl;
            return 1;
        }
        std::cout << "video_jni_tests PASSED" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception during tests: " << ex.what() << std::endl;
        return 1;
    }
}

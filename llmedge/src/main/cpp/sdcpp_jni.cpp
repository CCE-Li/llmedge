#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <algorithm>

#if __has_include(<android/log.h>)
#include <android/log.h>
#else
#include <cstdio>
#define ANDROID_LOG_DEBUG 3
#define ANDROID_LOG_INFO 4
#define ANDROID_LOG_WARN 5
#define ANDROID_LOG_ERROR 6
inline int __android_log_print(int, const char*, const char* format, ...) {
    (void)format;
    return 0;
}
#endif

#include "stable-diffusion.h"
#include "sd_jni_internal.h"

#define LOG_TAG "SmolSD"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)

SD_JNI_INTERNAL void throwJavaException(JNIEnv* env, const char* className, const char* message) {
    if (!env) return;
    jclass exClass = env->FindClass(className);
    if (!exClass) return;
    env->ThrowNew(exClass, message);
}

SD_JNI_INTERNAL void clearProgressCallback(JNIEnv* env, SdHandle* handle) {
    if (!handle) return;
    if (handle->progressCallbackGlobalRef && env) {
        env->DeleteGlobalRef(handle->progressCallbackGlobalRef);
    }
    handle->progressCallbackGlobalRef = nullptr;
    handle->progressMethodID = nullptr;
    handle->currentFrame = 0;
    handle->totalFrames = 0;
    handle->stepsPerFrame = 0;
    handle->totalSteps = 0;
    handle->cancellationRequested.store(false);
}

SD_JNI_INTERNAL void sd_video_progress_wrapper(int step, int steps, float time, void* data) {
    auto* handle = static_cast<SdHandle*>(data);
    if (!handle) return;

    if (handle->cancellationRequested.load()) {
        throw std::runtime_error("Video generation cancelled");
    }

    if (!handle->progressCallbackGlobalRef || !handle->jvm || !handle->progressMethodID) {
        return;
    }

    JNIEnv* env = nullptr;
    bool detach = false;
    jint envStatus = handle->jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (envStatus == JNI_EDETACHED) {
#if defined(__ANDROID__)
        if (handle->jvm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
#else
        if (handle->jvm->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr) != JNI_OK) {
#endif
            return;
        }
        detach = true;
    } else if (envStatus != JNI_OK) {
        return;
    }

    const int totalFrames = handle->totalFrames > 0 ? handle->totalFrames : 1;
    const int totalSteps = handle->totalSteps > 0 ? handle->totalSteps : steps;
    const int stepsPerFrame = handle->stepsPerFrame > 0
            ? handle->stepsPerFrame
            : (totalFrames > 0 ? std::max(steps / totalFrames, 1) : steps);

    if (stepsPerFrame > 0) {
        int inferredFrame = step / stepsPerFrame;
        if (inferredFrame >= totalFrames) inferredFrame = totalFrames - 1;
        handle->currentFrame = inferredFrame;
    }

    env->CallVoidMethod(
            handle->progressCallbackGlobalRef,
            handle->progressMethodID,
            static_cast<jint>(std::min(step, totalSteps)),
            static_cast<jint>(totalSteps),
            static_cast<jint>(handle->currentFrame),
            static_cast<jint>(handle->totalFrames),
            static_cast<jfloat>(time));

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    if (detach) {
        handle->jvm->DetachCurrentThread();
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeCheckBindings(JNIEnv*, jclass) {
    return JNI_TRUE;
}

static void sd_android_log_cb(enum sd_log_level_t level, const char* text, void* data) {
    (void)data;
    if (!text) return;
    switch (level) {
        case SD_LOG_DEBUG: __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "%s", text); break;
        case SD_LOG_INFO:  __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, "%s", text); break;
        case SD_LOG_WARN:  __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, "%s", text); break;
        case SD_LOG_ERROR: __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "%s", text); break;
        default: __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "%s", text); break;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeCreate(
    JNIEnv* env, jclass clazz,
        jstring jModelPath,
        jstring jVaePath,
        jstring jT5xxlPath,
        jint nThreads,
        jboolean offloadToCpu,
        jboolean keepClipOnCpu,
        jboolean keepVaeOnCpu) {
    (void)clazz;
    const char* modelPath = jModelPath ? env->GetStringUTFChars(jModelPath, nullptr) : nullptr;
    const char* vaePath   = jVaePath   ? env->GetStringUTFChars(jVaePath,   nullptr) : nullptr;
    const char* t5xxlPath = jT5xxlPath ? env->GetStringUTFChars(jT5xxlPath, nullptr) : nullptr;

    sd_set_log_callback(sd_android_log_cb, nullptr);

    ALOGI("Initializing Stable Diffusion with:");
    ALOGI("  offloadToCpu=%s, keepClipOnCpu=%s, keepVaeOnCpu=%s",
          offloadToCpu ? "true" : "false",
          keepClipOnCpu ? "true" : "false",
          keepVaeOnCpu ? "true" : "false");

    sd_ctx_params_t p{};
    sd_ctx_params_init(&p);
    p.model_path = modelPath ? modelPath : "";
    p.vae_path = vaePath ? vaePath : "";
    p.t5xxl_path = t5xxlPath ? t5xxlPath : "";
    p.free_params_immediately = true;
    p.n_threads = nThreads > 0 ? nThreads : get_num_physical_cores();
    p.offload_params_to_cpu = offloadToCpu;
    p.keep_clip_on_cpu = keepClipOnCpu;
    p.keep_vae_on_cpu = keepVaeOnCpu;

    sd_ctx_t* ctx = new_sd_ctx(&p);

    if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
    if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
    if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);

    if (!ctx) {
        ALOGE("Failed to create sd_ctx");
        return 0;
    }

    auto* handle = new SdHandle();
    handle->ctx = ctx;
    if (env) {
        env->GetJavaVM(&handle->jvm);
    }
    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeDestroy(JNIEnv* env, jobject, jlong handlePtr) {
    if (handlePtr == 0) return;
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    clearProgressCallback(env, handle);
    sd_set_progress_callback(nullptr, nullptr);
    if (handle->ctx) {
        free_sd_ctx(handle->ctx);
        handle->ctx = nullptr;
    }
    delete handle;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2Img(
        JNIEnv* env, jobject thiz, jlong handlePtr,
        jstring jPrompt, jstring jNegative,
        jint width, jint height,
        jint steps, jfloat cfg, jlong seed) {
    (void)thiz;
    if (handlePtr == 0) {
        ALOGE("StableDiffusion not initialized");
        return nullptr;
    }
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    const char* prompt = jPrompt ? env->GetStringUTFChars(jPrompt, nullptr) : "";
    const char* negative = jNegative ? env->GetStringUTFChars(jNegative, nullptr) : "";

    sd_sample_params_t sample{};
    sd_sample_params_init(&sample);
    if (steps > 0) sample.sample_steps = steps;
    sample.guidance.txt_cfg = cfg > 0 ? cfg : 7.0f;

    sd_img_gen_params_t gen{};
    sd_img_gen_params_init(&gen);
    gen.prompt = prompt;
    gen.negative_prompt = negative;
    gen.width = width;
    gen.height = height;
    gen.sample_params = sample;
    gen.seed = seed;
    gen.batch_count = 1;

    sd_image_t* out = generate_image(handle->ctx, &gen);

    if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
    if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);

    if (!out || !out[0].data) {
        ALOGE("generate_image failed");
        return nullptr;
    }

    const size_t byteCount = (size_t)out[0].width * out[0].height * out[0].channel;
    jbyteArray jbytes = env->NewByteArray((jsize)byteCount);
    if (!jbytes) {
        free(out[0].data);
        free(out);
        return nullptr;
    }
    env->SetByteArrayRegion(jbytes, 0, (jsize)byteCount, reinterpret_cast<jbyte*>(out[0].data));

    // Free native buffers allocated by stable-diffusion.cpp
    free(out[0].data);
    free(out);

    return jbytes;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2Vid(
        JNIEnv* env, jobject thiz, jlong handlePtr,
        jstring jPrompt, jstring jNegative,
        jint width, jint height,
        jint videoFrames, jint steps, jfloat cfg, jlong seed,
        jint jScheduler, jfloat jStrength,
        jbyteArray jInitImage, jint initWidth, jint initHeight) {
    (void)thiz;
    if (handlePtr == 0) {
        throwJavaException(env, "java/lang/IllegalStateException", "StableDiffusion not initialized");
        return nullptr;
    }
    if (width <= 0 || height <= 0 || videoFrames <= 0) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "Invalid video dimensions or frame count");
        return nullptr;
    }

    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    handle->cancellationRequested.store(false);
    handle->totalFrames = std::max(1, static_cast<int>(videoFrames));
    handle->currentFrame = 0;

    const char* prompt = jPrompt ? env->GetStringUTFChars(jPrompt, nullptr) : "";
    const char* negative = jNegative ? env->GetStringUTFChars(jNegative, nullptr) : "";

    auto releaseStrings = [&]() {
        if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
        if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);
    };

    sd_sample_params_t sample{};
    sd_sample_params_init(&sample);
    if (steps > 0) sample.sample_steps = steps;
    if (cfg > 0.f) sample.guidance.txt_cfg = cfg;

    sd_vid_gen_params_t gen{};
    sd_vid_gen_params_init(&gen);
    gen.prompt = prompt;
    gen.negative_prompt = negative;
    gen.width = width;
    gen.height = height;
    gen.video_frames = videoFrames;
    gen.sample_params = sample;
    gen.seed = seed;
    // Map scheduler enum if provided
    if (jScheduler >= 0) {
        enum scheduler_t s = static_cast<enum scheduler_t>(jScheduler);
        gen.sample_params.scheduler = s;
    }
    gen.strength = jStrength;

    std::vector<uint8_t> initImageData;
    if (jInitImage != nullptr) {
        jsize initSize = env->GetArrayLength(jInitImage);
        if (initSize > 0) {
            initImageData.resize(static_cast<size_t>(initSize));
            env->GetByteArrayRegion(jInitImage, 0, initSize,
                                    reinterpret_cast<jbyte*>(initImageData.data()));
            if (env->ExceptionCheck()) {
                releaseStrings();
                return nullptr;
            }
            gen.init_image.width = initWidth;
            gen.init_image.height = initHeight;
            gen.init_image.channel = 3;
            gen.init_image.data = initImageData.data();
        }
    }

    handle->stepsPerFrame = sample.sample_steps > 0 ? sample.sample_steps : 0;
    handle->totalSteps = handle->stepsPerFrame * handle->totalFrames;

    // Ensure progress callback is wired for cancellation even if Kotlin-side callback is null.
    if (!handle->progressCallbackGlobalRef) {
        sd_set_progress_callback(sd_video_progress_wrapper, handle);
    }

    sd_image_t* frames = nullptr;
    int numFrames = 0;
    try {
        frames = generate_video(handle->ctx, &gen, &numFrames);
    } catch (const std::exception& e) {
        releaseStrings();
        const char* clazz = handle->cancellationRequested.load()
                ? "java/util/concurrent/CancellationException"
                : "java/lang/RuntimeException";
        throwJavaException(env, clazz, e.what());
        return nullptr;
    }

    releaseStrings();

    if (!frames || numFrames <= 0) {
        if (frames) {
            sd_jni_notify_frame_array_freed(frames);
            free(frames);
        }
        throwJavaException(env, "java/lang/IllegalStateException", "Video generation failed");
        if (!handle->progressCallbackGlobalRef) {
            sd_set_progress_callback(nullptr, nullptr);
        }
        return nullptr;
    }

    jclass byteArrayClass = env->FindClass("[B");
    if (!byteArrayClass) {
        for (int i = 0; i < numFrames; ++i) {
            if (frames[i].data) {
                sd_jni_notify_frame_buffer_freed(frames[i].data);
                free(frames[i].data);
            }
        }
        sd_jni_notify_frame_array_freed(frames);
        free(frames);
        if (!handle->progressCallbackGlobalRef) {
            sd_set_progress_callback(nullptr, nullptr);
        }
        return nullptr;
    }

    jobjectArray result = env->NewObjectArray(numFrames, byteArrayClass, nullptr);
    if (!result) {
        for (int i = 0; i < numFrames; ++i) {
            if (frames[i].data) {
                sd_jni_notify_frame_buffer_freed(frames[i].data);
                free(frames[i].data);
            }
        }
        sd_jni_notify_frame_array_freed(frames);
        free(frames);
        throwJavaException(env, "java/lang/OutOfMemoryError", "Unable to allocate video frame array");
        if (!handle->progressCallbackGlobalRef) {
            sd_set_progress_callback(nullptr, nullptr);
        }
        return nullptr;
    }

    for (int i = 0; i < numFrames; ++i) {
        if (!frames[i].data) {
            for (int j = i; j < numFrames; ++j) {
                if (frames[j].data) {
                    sd_jni_notify_frame_buffer_freed(frames[j].data);
                    free(frames[j].data);
                }
            }
            sd_jni_notify_frame_array_freed(frames);
            free(frames);
            throwJavaException(env, "java/lang/IllegalStateException", "Missing frame data");
            if (!handle->progressCallbackGlobalRef) {
                sd_set_progress_callback(nullptr, nullptr);
            }
            return nullptr;
        }
        const size_t byteCount = static_cast<size_t>(frames[i].width) * frames[i].height * frames[i].channel;
        jbyteArray frameBytes = env->NewByteArray(static_cast<jsize>(byteCount));
        if (!frameBytes) {
            for (int j = i; j < numFrames; ++j) {
                if (frames[j].data) {
                    sd_jni_notify_frame_buffer_freed(frames[j].data);
                    free(frames[j].data);
                }
            }
            sd_jni_notify_frame_array_freed(frames);
            free(frames);
            throwJavaException(env, "java/lang/OutOfMemoryError", "Unable to allocate frame buffer");
            if (!handle->progressCallbackGlobalRef) {
                sd_set_progress_callback(nullptr, nullptr);
            }
            return nullptr;
        }
        env->SetByteArrayRegion(frameBytes, 0, static_cast<jsize>(byteCount),
                                 reinterpret_cast<jbyte*>(frames[i].data));
        if (env->ExceptionCheck()) {
            env->DeleteLocalRef(frameBytes);
            for (int j = i; j < numFrames; ++j) {
                if (frames[j].data) {
                    sd_jni_notify_frame_buffer_freed(frames[j].data);
                    free(frames[j].data);
                }
            }
            sd_jni_notify_frame_array_freed(frames);
            free(frames);
            if (!handle->progressCallbackGlobalRef) {
                sd_set_progress_callback(nullptr, nullptr);
            }
            return nullptr;
        }
        env->SetObjectArrayElement(result, i, frameBytes);
        env->DeleteLocalRef(frameBytes);
        sd_jni_notify_frame_buffer_freed(frames[i].data);
        free(frames[i].data);
    }

    sd_jni_notify_frame_array_freed(frames);
    free(frames);
    if (!handle->progressCallbackGlobalRef) {
        sd_set_progress_callback(nullptr, nullptr);
    }
    handle->cancellationRequested.store(false);
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeSetProgressCallback(
        JNIEnv* env, jobject, jlong handlePtr, jobject progressCallback) {
    if (handlePtr == 0) {
        throwJavaException(env, "java/lang/IllegalStateException", "StableDiffusion not initialized");
        return;
    }
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    if (!handle->jvm && env) {
        env->GetJavaVM(&handle->jvm);
    }

    if (!progressCallback) {
        clearProgressCallback(env, handle);
        sd_set_progress_callback(nullptr, nullptr);
        return;
    }

    clearProgressCallback(env, handle);

    jclass callbackClass = env->GetObjectClass(progressCallback);
    if (!callbackClass) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "Invalid progress callback instance");
        return;
    }
    jmethodID methodId = env->GetMethodID(callbackClass, "onProgress", "(IIIIF)V");
    env->DeleteLocalRef(callbackClass);
    if (!methodId) {
        throwJavaException(env, "java/lang/NoSuchMethodError", "onProgress method not found");
        return;
    }
    jobject globalRef = env->NewGlobalRef(progressCallback);
    if (!globalRef) {
        throwJavaException(env, "java/lang/OutOfMemoryError", "Unable to hold progress callback reference");
        return;
    }

    handle->progressCallbackGlobalRef = globalRef;
    handle->progressMethodID = methodId;
    handle->cancellationRequested.store(false);
    sd_set_progress_callback(sd_video_progress_wrapper, handle);
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeCancelGeneration(
        JNIEnv* env, jobject, jlong handlePtr) {
    (void)env;
    if (handlePtr == 0) return;
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    handle->cancellationRequested.store(true);
}

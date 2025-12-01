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
#include <cstdarg>
#define ANDROID_LOG_DEBUG 3
#define ANDROID_LOG_INFO 4
#define ANDROID_LOG_WARN 5
#define ANDROID_LOG_ERROR 6
inline int __android_log_print(int level, const char* tag, const char* format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[%s] ", tag);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    fflush(stderr);
    va_end(args);
    
    // Also print to stdout for good measure
    va_start(args, format);
    fprintf(stdout, "[%s] ", tag);
    vfprintf(stdout, format, args);
    fprintf(stdout, "\n");
    fflush(stdout);
    va_end(args);
    return 0;
}
#endif

#define GGML_MAX_NAME 128
#include "stable-diffusion.h"
#include "sd_jni_internal.h"
#if defined(SD_USE_VULKAN)
#include "ggml-vulkan.h"
#endif
#include "model.h"
#include "conditioner.hpp"
#include "ggml-backend.h"

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

extern "C" JNIEXPORT jint JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeGetVulkanDeviceCount(JNIEnv* env, jclass clazz) {
    (void)env;
    (void)clazz;
#ifdef SD_USE_VULKAN
    return (jint)ggml_backend_vk_get_device_count();
#else
    return 0;
#endif
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeGetVulkanDeviceMemory(JNIEnv* env, jclass clazz, jint deviceIndex) {
    (void)clazz;
#ifdef SD_USE_VULKAN
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_vk_get_device_memory((int)deviceIndex, &free_mem, &total_mem);
    jlongArray arr = env->NewLongArray(2);
    if (!arr) return nullptr;
    jlong vals[2];
    vals[0] = (jlong)free_mem;
    vals[1] = (jlong)total_mem;
    env->SetLongArrayRegion(arr, 0, 2, vals);
    return arr;
#else
    jlongArray arr = env->NewLongArray(2);
    if (!arr) return nullptr;
    jlong vals[2] = {0, 0};
    env->SetLongArrayRegion(arr, 0, 2, vals);
    return arr;
#endif
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeEstimateModelParamsMemory(JNIEnv* env, jclass clazz, jstring jModelPath, jint deviceIndex) {
    (void)clazz;
    if (!jModelPath) return (jlong)-1;
    const char* modelPath = env->GetStringUTFChars(jModelPath, nullptr);
    if (!modelPath) return (jlong)-1;
    ModelLoader model_loader;
    bool ok = model_loader.init_from_file(modelPath);
    if (!ok) {
        env->ReleaseStringUTFChars(jModelPath, modelPath);
        return (jlong)-1;
    }
    ggml_backend_t backend = nullptr;
#ifdef SD_USE_VULKAN
    if (deviceIndex >= 0 && ggml_backend_vk_get_device_count() > deviceIndex) {
        backend = ggml_backend_vk_init(deviceIndex);
    }
#endif
    int64_t params_mem = model_loader.get_params_mem_size(backend, GGML_TYPE_COUNT);
    if (backend) ggml_backend_free(backend);
    env->ReleaseStringUTFChars(jModelPath, modelPath);
    return (jlong)params_mem;
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeEstimateModelParamsMemoryDetailed(JNIEnv* env, jclass clazz, jstring jModelPath, jint deviceIndex) {
    (void)clazz;
    if (!jModelPath) return nullptr;
    const char* modelPath = env->GetStringUTFChars(jModelPath, nullptr);
    if (!modelPath) return nullptr;
    ModelLoader model_loader;
    bool ok = model_loader.init_from_file(modelPath);
    if (!ok) {
        env->ReleaseStringUTFChars(jModelPath, modelPath);
        return nullptr;
    }
    ggml_backend_t backend = nullptr;
#ifdef SD_USE_VULKAN
    if (deviceIndex >= 0 && ggml_backend_vk_get_device_count() > deviceIndex) {
        backend = ggml_backend_vk_init(deviceIndex);
    }
#endif
    // Detailed per-prefix parameter memory estimation API was removed upstream.
    // As a best-effort fallback, return zeros for subcomponents and provide the total.
    // Callers currently only rely on the total to make offload decisions.
    jlong clip = (jlong)0;
    jlong diffusion = (jlong)0;
    jlong vae = (jlong)0;
    jlong control = (jlong)0;
    jlong pmid = (jlong)0;
    jlong total = (jlong)model_loader.get_params_mem_size(backend, GGML_TYPE_COUNT);
    if (backend) ggml_backend_free(backend);
    env->ReleaseStringUTFChars(jModelPath, modelPath);
    jlong vals[6] = {clip, diffusion, vae, control, pmid, total};
    jlongArray arr = env->NewLongArray(6);
    if (!arr) return nullptr;
    env->SetLongArrayRegion(arr, 0, 6, vals);
    return arr;
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
        jboolean keepVaeOnCpu,
        jboolean flashAttn,
        jfloat flowShift,
        jstring jLoraModelDir, jint jLoraApplyMode) {
    (void)clazz;
    FILE* f = fopen("/tmp/sdcpp_log.txt", "a");
    if (f) {
        fprintf(f, "[SmolSD] nativeCreate ENTERED\n");
        fclose(f);
    }
    fprintf(stderr, "[SmolSD] nativeCreate ENTERED\n"); fflush(stderr);
    const char* modelPath = jModelPath ? env->GetStringUTFChars(jModelPath, nullptr) : nullptr;
    const char* vaePath   = jVaePath   ? env->GetStringUTFChars(jVaePath,   nullptr) : nullptr;
    const char* t5xxlPath = jT5xxlPath ? env->GetStringUTFChars(jT5xxlPath, nullptr) : nullptr;

    sd_set_log_callback(sd_android_log_cb, nullptr);

    ALOGI("Initializing Stable Diffusion with:");
    ALOGI("  modelPath=%s", modelPath ? modelPath : "NULL");
    ALOGI("  vaePath=%s", vaePath ? vaePath : "NULL");
    ALOGI("  t5xxlPath=%s", t5xxlPath ? t5xxlPath : "NULL");
    ALOGI("  offloadToCpu=%s, keepClipOnCpu=%s, keepVaeOnCpu=%s, flashAttn=%s",
          offloadToCpu ? "true" : "false",
          keepClipOnCpu ? "true" : "false",
          keepVaeOnCpu ? "true" : "false",
          flashAttn ? "true" : "false");

    sd_ctx_params_t p{};
    sd_ctx_params_init(&p);
    p.model_path = modelPath ? modelPath : "";
    p.vae_path = vaePath ? vaePath : "";
    // Pass through nullptr when no T5XXL path is provided. Some pipelines
    // distinguish between null and empty string; using nullptr ensures the
    // text encoder is selected correctly for SD 1.x models.
    p.t5xxl_path = t5xxlPath; // keep null if not provided
    p.free_params_immediately = true;
    p.n_threads = nThreads > 0 ? nThreads : get_num_physical_cores();
    p.offload_params_to_cpu = offloadToCpu;
    p.keep_clip_on_cpu = keepClipOnCpu;
    p.keep_vae_on_cpu = keepVaeOnCpu;
    p.diffusion_flash_attn = flashAttn;
    p.flow_shift = flowShift;
    if (jLoraModelDir) {
        const char* loraPath = env->GetStringUTFChars(jLoraModelDir, nullptr);
        if (loraPath) {
            p.lora_model_dir = strdup(loraPath);
            env->ReleaseStringUTFChars(jLoraModelDir, loraPath);
        }
    }
    p.lora_apply_mode = static_cast<enum lora_apply_mode_t>(jLoraApplyMode);

    sd_ctx_t* ctx = new_sd_ctx(&p);

    if (!ctx) {
        // Fallback: Check if we can load as T5-only context
        // This is a hack to support sequential loading where we only want the text encoder
        if (modelPath && !vaePath && !t5xxlPath) {
             ALOGI("Attempting to load as T5-only context: %s", modelPath);
             // We need to manually instantiate T5CLIPEmbedder
             // But we need a backend.
             // And ModelLoader.
             
             // We can't easily do this inside nativeCreate because we need to return a jlong handle
             // and we need to store the T5 object in it.
             
             // Let's try to instantiate T5CLIPEmbedder here.
             ModelLoader model_loader;
             if (model_loader.init_from_file(modelPath, "text_encoders.t5xxl.transformer.")) {
                 ALOGI("ModelLoader initialized for T5");
                 model_loader.convert_tensors_name();
                 
                 ggml_backend_t backend = nullptr;
                 #ifdef SD_USE_VULKAN
                 if (ggml_backend_vk_get_device_count() > 0) {
                     backend = ggml_backend_vk_init(0);
                 }
                 #endif
                 if (!backend) {
                     backend = ggml_backend_cpu_init();
                 }
                 
                 if (!backend) {
                     ALOGE("Vulkan backend not available and CPU backend init failed/missing");
                     if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
                     if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
                     if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
                     return 0;
                 }
                 ALOGI("Backend initialized for T5");
                 
                 // T5CLIPEmbedder(backend, offload, storage, use_mask, mask_pad, is_umt5)
                 bool is_umt5 = std::string(modelPath).find("umt5") != std::string::npos;
                 ALOGI("Creating T5CLIPEmbedder (is_umt5=%d)", is_umt5);
                 
                 auto* t5 = new T5CLIPEmbedder(backend, offloadToCpu, model_loader.get_tensor_storage_map(), false, 0, is_umt5);
                 ALOGI("Allocating params buffer for T5");
                 t5->alloc_params_buffer();
                 
                 // Load weights
                 std::map<std::string, struct ggml_tensor*> tensors;
                 t5->get_param_tensors(tensors);
                 ALOGI("Got param tensors for T5: %zu tensors", tensors.size());
                 
                 std::set<std::string> ignore_tensors;
                 ALOGI("Loading tensors for T5");
                 model_loader.load_tensors(tensors, ignore_tensors, get_num_physical_cores());
                 
                 auto* handle = new SdHandle();
                 handle->ctx = nullptr;
                 handle->t5_ctx = t5;
                 if (env) {
                     env->GetJavaVM(&handle->jvm);
                 }
                 ALOGI("T5-only context created successfully");
                 
                 if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
                 if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
                 if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
                 
                 return reinterpret_cast<jlong>(handle);
             } else {
                 ALOGE("Failed to init ModelLoader for T5");
             }
        }
        
        ALOGE("Failed to create sd_ctx");
        if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
        if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
        if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);
        return 0;
    }

    if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
    if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);
    if (jT5xxlPath) env->ReleaseStringUTFChars(jT5xxlPath, t5xxlPath);

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
    if (handle->t5_ctx) {
        auto* t5 = static_cast<T5CLIPEmbedder*>(handle->t5_ctx);
        t5->free_params_buffer();
        delete t5;
        handle->t5_ctx = nullptr;
    }
    delete handle;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeIsEasyCacheSupported(JNIEnv* env, jobject, jlong handlePtr) {
    if (handlePtr == 0) return JNI_FALSE;
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    if (!handle->ctx) return JNI_FALSE;
    return sd_is_easycache_supported(handle->ctx) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2Img(
    JNIEnv* env, jobject thiz, jlong handlePtr,
    jstring jPrompt, jstring jNegative,
    jint width, jint height,
    jint steps, jfloat cfg, jlong seed,
    jboolean jEasyCacheEnabled, jfloat jEasyCacheReuseThreshold, jfloat jEasyCacheStartPercent, jfloat jEasyCacheEndPercent) {
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
    gen.easycache.enabled = jEasyCacheEnabled ? true : false;
    gen.easycache.reuse_threshold = (float)jEasyCacheReuseThreshold;
    gen.easycache.start_percent = (float)jEasyCacheStartPercent;
    gen.easycache.end_percent = (float)jEasyCacheEndPercent;

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
    jbyteArray jInitImage, jint initWidth, jint initHeight,
    jboolean jEasyCacheEnabled, jfloat jEasyCacheReuseThreshold, jfloat jEasyCacheStartPercent, jfloat jEasyCacheEndPercent) {
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
    
    // Resolve default sample method, as generate_video doesn't do it
    if (gen.sample_params.sample_method == SAMPLE_METHOD_DEFAULT) {
        gen.sample_params.sample_method = sd_get_default_sample_method(handle->ctx);
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

            gen.easycache.enabled = jEasyCacheEnabled ? true : false;
            gen.easycache.reuse_threshold = (float)jEasyCacheReuseThreshold;
            gen.easycache.start_percent = (float)jEasyCacheStartPercent;
            gen.easycache.end_percent = (float)jEasyCacheEndPercent;
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

// JNI wrapper: precompute condition for a given prompt & video params
extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativePrecomputeCondition(
        JNIEnv* env, jobject thiz, jlong handlePtr,
        jstring jPrompt, jstring jNegative,
        jint width, jint height, jint clipSkip) {
    (void)thiz;
    
    const char* prompt = jPrompt ? env->GetStringUTFChars(jPrompt, nullptr) : "";
    const char* negative = jNegative ? env->GetStringUTFChars(jNegative, nullptr) : "";

    sd_vid_gen_params_t gen{};
    sd_vid_gen_params_init(&gen);
    gen.prompt = prompt;
    gen.negative_prompt = negative;
    gen.width = width;
    gen.height = height;
    gen.clip_skip = clipSkip;

    sd_condition_raw_t* cond = nullptr;
    
    if (handlePtr != 0) {
        // Legacy path: use existing context
        auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
        if (handle->ctx) {
            try {
                cond = sd_precompute_condition(handle->ctx, &gen);
            } catch (const std::exception& e) {
                if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
                if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);
                throwJavaException(env, "java/lang/RuntimeException", e.what());
                return nullptr;
            }
        } else if (handle->t5_ctx) {
            // T5-only path
            auto* t5 = static_cast<T5CLIPEmbedder*>(handle->t5_ctx);
            try {
                // T5CLIPEmbedder::get_learned_condition returns SDCondition
                // SDCondition has ggml_tensor* c_crossattn, c_vector, c_concat
                
                // We need a work context
                struct ggml_init_params params;
                params.mem_size   = 1024 * 1024 * 1024; // 1GB for intermediate tensors?
                params.mem_buffer = nullptr;
                params.no_alloc   = false;
                struct ggml_context* work_ctx = ggml_init(params);
                
                ConditionerParams cparams;
                cparams.text = prompt;
                cparams.clip_skip = clipSkip;
                cparams.width = width;
                cparams.height = height;
                
                SDCondition sd_cond = t5->get_learned_condition(work_ctx, get_num_physical_cores(), cparams);
                
                // Convert SDCondition to sd_condition_raw_t
                cond = (sd_condition_raw_t*)calloc(1, sizeof(sd_condition_raw_t));
                
                auto tensor_to_raw = [](struct ggml_tensor* t, sd_tensor_raw_t& raw) {
                    if (!t) return;
                    raw.ndims = ggml_n_dims(t);
                    for(int i=0; i<4; ++i) raw.ne[i] = t->ne[i];
                    size_t size = ggml_nbytes(t);
                    raw.data = (float*)malloc(size);
                    memcpy(raw.data, t->data, size);
                };
                
                if (sd_cond.c_crossattn) tensor_to_raw(sd_cond.c_crossattn, cond->c_crossattn);
                if (sd_cond.c_vector) tensor_to_raw(sd_cond.c_vector, cond->c_vector);
                if (sd_cond.c_concat) tensor_to_raw(sd_cond.c_concat, cond->c_concat);
                
                ggml_free(work_ctx);
                
            } catch (const std::exception& e) {
                if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
                if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);
                throwJavaException(env, "java/lang/RuntimeException", e.what());
                return nullptr;
            }
        } else {
            throwJavaException(env, "java/lang/IllegalStateException", "Invalid handle state");
            return nullptr;
        }
    } else {
        throwJavaException(env, "java/lang/IllegalStateException", "StableDiffusion not initialized");
        return nullptr;
    }

    if (jPrompt) env->ReleaseStringUTFChars(jPrompt, prompt);
    if (jNegative) env->ReleaseStringUTFChars(jNegative, negative);

    if (!cond) {
        return nullptr;
    }
    
    // We'll return an Object[] = {float[] cross, int[] cross_dims, float[] vec, int[] vec_dims, float[] concat, int[] concat_dims}
    jclass objClass = env->FindClass("java/lang/Object");
    if (!objClass) {
        sd_free_condition(cond);
        throwJavaException(env, "java/lang/RuntimeException", "Unable to find java/lang/Object");
        return nullptr;
    }

    jobjectArray result = env->NewObjectArray(6, objClass, nullptr);
    if (!result) {
        sd_free_condition(cond);
        throwJavaException(env, "java/lang/OutOfMemoryError", "Unable to allocate result array");
        return nullptr;
    }

    // Helper lambda to push arrays
    auto push_tensor = [&](const sd_tensor_raw_t* t, int data_index, int dims_index) {
        if (t == nullptr || t->ndims == 0 || t->data == nullptr) {
            env->SetObjectArrayElement(result, data_index, nullptr);
            env->SetObjectArrayElement(result, dims_index, nullptr);
            return;
        }
        size_t count = 1;
        for (int i = 0; i < t->ndims; ++i) count *= (size_t)t->ne[i];

        jfloatArray floatArr = env->NewFloatArray(static_cast<jsize>(count));
        if (!floatArr) {
            env->SetObjectArrayElement(result, data_index, nullptr);
        } else {
            env->SetFloatArrayRegion(floatArr, 0, static_cast<jsize>(count), reinterpret_cast<const jfloat*>(t->data));
            env->SetObjectArrayElement(result, data_index, floatArr);
            env->DeleteLocalRef(floatArr);
        }

        jintArray dimsArr = env->NewIntArray(t->ndims);
        if (!dimsArr) {
            env->SetObjectArrayElement(result, dims_index, nullptr);
        } else {
            jint dims[4] = {0,0,0,0};
            for (int i = 0; i < t->ndims && i < 4; ++i) dims[i] = t->ne[i];
            env->SetIntArrayRegion(dimsArr, 0, t->ndims, dims);
            env->SetObjectArrayElement(result, dims_index, dimsArr);
            env->DeleteLocalRef(dimsArr);
        }
    };

    // The condition struct uses sd_tensor_raw_t members; pass by reference to the lambda
    push_tensor(&cond->c_crossattn, 0, 1);
    push_tensor(&cond->c_vector, 2, 3);
    push_tensor(&cond->c_concat, 4, 5);

    // Free native cond buffers
    sd_free_condition(cond);

    return result;
}


// Helper to reconstruct sd_condition_raw_t from Java Object[]
static sd_condition_raw_t* reconstruct_condition(JNIEnv* env, jobjectArray condArr) {
    if (!condArr) return nullptr;
    
    // Layout: [float[] cross, int[] crossDims, float[] vector, int[] vectorDims, float[] concat, int[] concatDims]
    if (env->GetArrayLength(condArr) < 6) return nullptr;

    auto* cond = (sd_condition_raw_t*)calloc(1, sizeof(sd_condition_raw_t));

    auto extract_tensor = [&](int data_idx, int dims_idx, sd_tensor_raw_t& raw) {
        jfloatArray dataArr = (jfloatArray)env->GetObjectArrayElement(condArr, data_idx);
        jintArray dimsArr = (jintArray)env->GetObjectArrayElement(condArr, dims_idx);

        if (dataArr && dimsArr) {
            jsize dataLen = env->GetArrayLength(dataArr);
            jsize dimsLen = env->GetArrayLength(dimsArr);
            
            raw.ndims = std::min((int)dimsLen, 4);
            jint* dims = env->GetIntArrayElements(dimsArr, nullptr);
            for(int i=0; i<raw.ndims; ++i) raw.ne[i] = dims[i];
            env->ReleaseIntArrayElements(dimsArr, dims, JNI_ABORT);
            
            raw.data = (float*)malloc(dataLen * sizeof(float));
            jfloat* data = env->GetFloatArrayElements(dataArr, nullptr);
            memcpy(raw.data, data, dataLen * sizeof(float));
            env->ReleaseFloatArrayElements(dataArr, data, JNI_ABORT);
        } else {
            raw.ndims = 0;
            raw.data = nullptr;
        }
    };

    extract_tensor(0, 1, cond->c_crossattn);
    extract_tensor(2, 3, cond->c_vector);
    extract_tensor(4, 5, cond->c_concat);

    return cond;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2ImgWithPrecomputedCondition(
    JNIEnv* env, jobject thiz, jlong handlePtr,
    jstring jPrompt, jstring jNegative,
    jint width, jint height,
    jint steps, jfloat cfg, jlong seed,
    jobjectArray condArr, jobjectArray uncondArr,
    jboolean jEasyCacheEnabled, jfloat jEasyCacheReuseThreshold, jfloat jEasyCacheStartPercent, jfloat jEasyCacheEndPercent) {
    (void)thiz;
    if (handlePtr == 0) {
        ALOGE("StableDiffusion not initialized");
        return nullptr;
    }
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    
    // Reconstruct conditions
    sd_condition_raw_t* cond = reconstruct_condition(env, condArr);
    sd_condition_raw_t* uncond = reconstruct_condition(env, uncondArr);
    
    if (!cond) {
        ALOGE("Failed to reconstruct condition");
        if (uncond) sd_free_condition(uncond);
        return nullptr;
    }

    sd_sample_params_t sample{};
    sd_sample_params_init(&sample);
    if (steps > 0) sample.sample_steps = steps;
    sample.guidance.txt_cfg = cfg > 0 ? cfg : 7.0f;

    sd_img_gen_params_t gen{};
    sd_img_gen_params_init(&gen);
    gen.width = width;
    gen.height = height;
    gen.sample_params = sample;
    gen.seed = seed;
    gen.batch_count = 1;
    gen.easycache.enabled = jEasyCacheEnabled ? true : false;
    gen.easycache.reuse_threshold = (float)jEasyCacheReuseThreshold;
    gen.easycache.start_percent = (float)jEasyCacheStartPercent;
    gen.easycache.end_percent = (float)jEasyCacheEndPercent;

    sd_image_t* out = sd_generate_image_with_precomputed_condition(handle->ctx, &gen, cond, uncond);

    // Cleanup reconstructed conditions
    sd_free_condition(cond);
    if (uncond) sd_free_condition(uncond);

    if (!out || !out[0].data) {
        ALOGE("generate_image failed");
        if (out) free(out);
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

    free(out[0].data);
    free(out);

    return jbytes;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeTxt2VidWithPrecomputedCondition(
        JNIEnv* env, jobject thiz, jlong handlePtr,
        jstring jPrompt, jstring jNegative,
        jint width, jint height,
        jint videoFrames, jint steps, jfloat cfg, jlong seed,
        jint jScheduler, jfloat jStrength,
        jbyteArray jInitImage, jint initWidth, jint initHeight,
        jobjectArray condArr, jobjectArray uncondArr) {
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
    
    // Resolve default sample method, as sd_generate_video_with_precomputed_condition doesn't do it
    if (gen.sample_params.sample_method == SAMPLE_METHOD_DEFAULT) {
        gen.sample_params.sample_method = sd_get_default_sample_method(handle->ctx);
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

    // Convert condArr/uncondArr to sd_condition_raw_t structures
    auto build_condition_from_objarr = [&](jobjectArray arr) -> sd_condition_raw_t* {
        if (!arr) return nullptr;
        jsize arrlen = env->GetArrayLength(arr);
        if (arrlen < 6) return nullptr;

        sd_condition_raw_t* cond = (sd_condition_raw_t*)calloc(1, sizeof(sd_condition_raw_t));
        if (!cond) return nullptr;

        // Helper to fill sd_tensor_raw_t* from (float[], int[])
        auto build_tensor = [&](int dataIndex, int dimsIndex) -> sd_tensor_raw_t* {
            jobject dataObj = env->GetObjectArrayElement(arr, dataIndex);
            jobject dimsObj = env->GetObjectArrayElement(arr, dimsIndex);
            if (!dataObj || !dimsObj) {
                if (dataObj) env->DeleteLocalRef(dataObj);
                if (dimsObj) env->DeleteLocalRef(dimsObj);
                return nullptr;
            }
            jfloatArray jFloatArr = static_cast<jfloatArray>(dataObj);
            jintArray jDimsArr = static_cast<jintArray>(dimsObj);
            jsize dataLen = env->GetArrayLength(jFloatArr);
            jsize dimsLen = env->GetArrayLength(jDimsArr);
            if (dimsLen <= 0 || dimsLen > 4) {
                env->DeleteLocalRef(dataObj);
                env->DeleteLocalRef(dimsObj);
                return nullptr;
            }
            sd_tensor_raw_t* t = (sd_tensor_raw_t*)calloc(1, sizeof(sd_tensor_raw_t));
            t->ndims = dimsLen;
            for (int i = 0; i < 4; ++i) t->ne[i] = 0;
            jint dims[4] = {0,0,0,0};
            env->GetIntArrayRegion(jDimsArr, 0, dimsLen, dims);
            size_t expectedLen = 1;
            for (int i = 0; i < dimsLen; ++i) {
                t->ne[i] = dims[i];
                expectedLen *= (size_t)dims[i];
            }
            // If expectedLen doesn't match dataLen, still copy as-is but it's an error case
            t->data = (float*)malloc(sizeof(float) * static_cast<size_t>(dataLen));
            if (!t->data) {
                free(t);
                env->DeleteLocalRef(dataObj);
                env->DeleteLocalRef(dimsObj);
                return nullptr;
            }
            // If sizes mismatch we still copy as many elements as present in the array
            env->GetFloatArrayRegion(jFloatArr, 0, dataLen, reinterpret_cast<jfloat*>(t->data));
            env->DeleteLocalRef(dataObj);
            env->DeleteLocalRef(dimsObj);
            return t;
        };

        sd_tensor_raw_t* t_cross = build_tensor(0, 1);
        sd_tensor_raw_t* t_vector = build_tensor(2, 3);
        sd_tensor_raw_t* t_concat = build_tensor(4, 5);
        if (t_cross) { cond->c_crossattn = *t_cross; free(t_cross); }
        if (t_vector) { cond->c_vector = *t_vector; free(t_vector); }
        if (t_concat) { cond->c_concat = *t_concat; free(t_concat); }

        return cond;
    };

    sd_condition_raw_t* cond_use = build_condition_from_objarr(condArr);
    sd_condition_raw_t* uncond_use = build_condition_from_objarr(uncondArr);

    handle->stepsPerFrame = sample.sample_steps > 0 ? sample.sample_steps : 0;
    handle->totalSteps = handle->stepsPerFrame * handle->totalFrames;

    // Ensure progress callback is wired for cancellation even if Kotlin-side callback is null.
    if (!handle->progressCallbackGlobalRef) {
        sd_set_progress_callback(sd_video_progress_wrapper, handle);
    }

    sd_image_t* frames = nullptr;
    int numFrames = 0;
    try {
        ALOGI("Calling sd_generate_video_with_precomputed_condition...");
        frames = sd_generate_video_with_precomputed_condition(handle->ctx, &gen, cond_use, uncond_use, &numFrames);
        ALOGI("sd_generate_video_with_precomputed_condition returned %d frames", numFrames);
    } catch (const std::exception& e) {
        ALOGE("Exception in sd_generate_video_with_precomputed_condition: %s", e.what());
        releaseStrings();
        if (cond_use) {
            if (cond_use->c_crossattn.data) free(cond_use->c_crossattn.data);
            if (cond_use->c_vector.data) free(cond_use->c_vector.data);
            if (cond_use->c_concat.data) free(cond_use->c_concat.data);
            free(cond_use);
        }
        if (uncond_use) {
            if (uncond_use->c_crossattn.data) free(uncond_use->c_crossattn.data);
            if (uncond_use->c_vector.data) free(uncond_use->c_vector.data);
            if (uncond_use->c_concat.data) free(uncond_use->c_concat.data);
            free(uncond_use);
        }
        const char* clazz = handle->cancellationRequested.load()
                ? "java/util/concurrent/CancellationException"
                : "java/lang/RuntimeException";
        throwJavaException(env, clazz, e.what());
        return nullptr;
    }

    releaseStrings();

    // Free condition buffers built from Java arrays
    if (cond_use) {
        if (cond_use->c_crossattn.data) { free(cond_use->c_crossattn.data); cond_use->c_crossattn.data = nullptr; }
        if (cond_use->c_vector.data)   { free(cond_use->c_vector.data);   cond_use->c_vector.data   = nullptr; }
        if (cond_use->c_concat.data)   { free(cond_use->c_concat.data);   cond_use->c_concat.data   = nullptr; }
        free(cond_use);
    }
    if (uncond_use) {
        if (uncond_use->c_crossattn.data) { free(uncond_use->c_crossattn.data); uncond_use->c_crossattn.data = nullptr; }
        if (uncond_use->c_vector.data)   { free(uncond_use->c_vector.data);   uncond_use->c_vector.data   = nullptr; }
        if (uncond_use->c_concat.data)   { free(uncond_use->c_concat.data);   uncond_use->c_concat.data   = nullptr; }
        free(uncond_use);
    }

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

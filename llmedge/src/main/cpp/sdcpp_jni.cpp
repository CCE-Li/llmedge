#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <android/log.h>

#include "stable-diffusion.h"

#define LOG_TAG "SmolSD"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)

struct SdHandle {
    sd_ctx_t* ctx = nullptr;
    int last_width = 0;
    int last_height = 0;
};

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
        jint nThreads,
        jboolean offloadToCpu,
        jboolean keepClipOnCpu,
        jboolean keepVaeOnCpu) {
    (void)clazz;
    const char* modelPath = jModelPath ? env->GetStringUTFChars(jModelPath, nullptr) : nullptr;
    const char* vaePath   = jVaePath   ? env->GetStringUTFChars(jVaePath,   nullptr) : nullptr;

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
    p.free_params_immediately = true;
    p.n_threads = nThreads > 0 ? nThreads : get_num_physical_cores();
    p.offload_params_to_cpu = offloadToCpu;
    p.keep_clip_on_cpu = keepClipOnCpu;
    p.keep_vae_on_cpu = keepVaeOnCpu;

    sd_ctx_t* ctx = new_sd_ctx(&p);

    if (jModelPath) env->ReleaseStringUTFChars(jModelPath, modelPath);
    if (jVaePath)   env->ReleaseStringUTFChars(jVaePath, vaePath);

    if (!ctx) {
        ALOGE("Failed to create sd_ctx");
        return 0;
    }

    auto* handle = new SdHandle();
    handle->ctx = ctx;
    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_StableDiffusion_nativeDestroy(JNIEnv*, jobject, jlong handlePtr) {
    if (handlePtr == 0) return;
    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
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

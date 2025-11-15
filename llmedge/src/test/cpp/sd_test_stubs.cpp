#include "stable-diffusion.h"

#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {

static sd_log_cb_t g_log_cb = nullptr;
static void* g_log_user_data = nullptr;
static sd_progress_cb_t g_progress_cb = nullptr;
static void* g_progress_user_data = nullptr;

void sd_set_log_callback(sd_log_cb_t cb, void* data) {
    g_log_cb = cb;
    g_log_user_data = data;
    if (g_log_cb) {
        g_log_cb(SD_LOG_INFO, "sd_set_log_callback invoked", g_log_user_data);
    }
}

void sd_set_progress_callback(sd_progress_cb_t cb, void* data) {
    g_progress_cb = cb;
    g_progress_user_data = data;
}

void sd_set_preview_callback(sd_preview_cb_t, enum preview_t, int, bool, bool) {
    // Not used in tests.
}

int32_t get_num_physical_cores() {
    return 4;
}

const char* sd_get_system_info() {
    return "sd_test_stubs";
}

const char* sd_type_name(enum sd_type_t) { return "stub"; }
enum sd_type_t str_to_sd_type(const char*) { return SD_TYPE_F16; }
const char* sd_rng_type_name(enum rng_type_t) { return "stub"; }
enum rng_type_t str_to_rng_type(const char*) { return STD_DEFAULT_RNG; }
const char* sd_sample_method_name(enum sample_method_t) { return "stub"; }
enum sample_method_t str_to_sample_method(const char*) { return SAMPLE_METHOD_DEFAULT; }
const char* sd_schedule_name(enum scheduler_t) { return "stub"; }
enum scheduler_t str_to_schedule(const char*) { return DEFAULT; }
const char* sd_prediction_name(enum prediction_t) { return "stub"; }
enum prediction_t str_to_prediction(const char*) { return DEFAULT_PRED; }
const char* sd_preview_name(enum preview_t) { return "stub"; }
enum preview_t str_to_preview(const char*) { return PREVIEW_NONE; }
const char* sd_lora_apply_mode_name(enum lora_apply_mode_t) { return "stub"; }
enum lora_apply_mode_t str_to_lora_apply_mode(const char*) { return LORA_APPLY_AUTO; }

void sd_ctx_params_init(sd_ctx_params_t* params) {
    std::memset(params, 0, sizeof(sd_ctx_params_t));
}

char* sd_ctx_params_to_str(const sd_ctx_params_t*) { return nullptr; }

typedef struct sd_ctx_t {
    int dummy;
} sd_ctx_t_impl;

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t*) {
    return reinterpret_cast<sd_ctx_t*>(new sd_ctx_t_impl{1});
}

void free_sd_ctx(sd_ctx_t* ctx) {
    delete reinterpret_cast<sd_ctx_t_impl*>(ctx);
}

enum sample_method_t sd_get_default_sample_method(const sd_ctx_t*) {
    return SAMPLE_METHOD_DEFAULT;
}

void sd_sample_params_init(sd_sample_params_t* params) {
    std::memset(params, 0, sizeof(sd_sample_params_t));
    params->sample_steps = 0;
}

char* sd_sample_params_to_str(const sd_sample_params_t*) { return nullptr; }

void sd_img_gen_params_init(sd_img_gen_params_t* params) {
    std::memset(params, 0, sizeof(sd_img_gen_params_t));
    params->sample_params.sample_steps = 0;
}

char* sd_img_gen_params_to_str(const sd_img_gen_params_t*) { return nullptr; }

void sd_vid_gen_params_init(sd_vid_gen_params_t* params) {
    std::memset(params, 0, sizeof(sd_vid_gen_params_t));
}

char* sd_vid_gen_params_to_str(const sd_vid_gen_params_t*) { return nullptr; }

static void fill_image(sd_image_t& image, int width, int height, int channel, uint8_t seed) {
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.channel = static_cast<uint32_t>(channel);
    const size_t byteCount = static_cast<size_t>(width) * height * channel;
    image.data = static_cast<uint8_t*>(std::malloc(byteCount));
    for (size_t i = 0; i < byteCount; ++i) {
        image.data[i] = static_cast<uint8_t>(seed + (i % 253));
    }
}

sd_image_t* generate_image(sd_ctx_t*, const sd_img_gen_params_t* params) {
    auto* images = static_cast<sd_image_t*>(std::malloc(sizeof(sd_image_t)));
    fill_image(images[0], params->width > 0 ? params->width : 256,
               params->height > 0 ? params->height : 256,
               3, 42);
    return images;
}

sd_image_t* generate_video(sd_ctx_t*, const sd_vid_gen_params_t* params, int* num_frames_out) {
    const int frames = params->video_frames > 0 ? params->video_frames : 4;
    *num_frames_out = frames;
    auto* images = static_cast<sd_image_t*>(std::malloc(sizeof(sd_image_t) * frames));
    const int steps = params->sample_params.sample_steps > 0 ? params->sample_params.sample_steps : 10;

    for (int i = 0; i < frames; ++i) {
        fill_image(images[i], params->width > 0 ? params->width : 256,
                   params->height > 0 ? params->height : 256,
                   3, static_cast<uint8_t>(i));
        if (g_progress_cb) {
            const int frameBase = i * steps;
            for (int s = 0; s < steps; ++s) {
                g_progress_cb(frameBase + s, frames * steps, 0.1f * (frameBase + s), g_progress_user_data);
            }
        }
    }
    return images;
}

upscaler_ctx_t* new_upscaler_ctx(const char*, bool, bool, int) { return nullptr; }
void free_upscaler_ctx(upscaler_ctx_t*) {}
sd_image_t upscale(upscaler_ctx_t*, sd_image_t input_image, uint32_t) { return input_image; }
int get_upscale_factor(upscaler_ctx_t*) { return 1; }

bool convert(const char*, const char*, const char*, enum sd_type_t, const char*) { return true; }

bool preprocess_canny(sd_image_t, float, float, float, float, bool) { return true; }

}  // extern "C"

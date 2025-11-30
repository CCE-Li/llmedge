#include "LLMInference.h"
#include <android/log.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>

#define TAG "[SmolLMAndroid-Cpp]"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

void
LLMInference::loadModel(const char *model_path, float minP, float temperature, bool storeChats, long contextSize,
                        const char *chatTemplate, int nThreads, bool useMmap, bool useMlock, bool useVulkan) {
    LOGi("loading model with"
         "\n\tmodel_path = %s"
         "\n\tminP = %f"
         "\n\ttemperature = %f"
         "\n\tstoreChats = %d"
         "\n\tcontextSize = %li"
         "\n\tchatTemplate = %s"
         "\n\tnThreads = %d"
         "\n\tuseMmap = %d"
         "\n\tuseMlock = %d"
         "\n\tuseVulkan = %d",
         model_path, minP, temperature, storeChats, contextSize, chatTemplate, nThreads, useMmap, useMlock, useVulkan);

    // load dynamic backends
    ggml_backend_load_all();

    // create an instance of llama_model
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = useMmap;
    model_params.use_mlock = useMlock;
    if (useVulkan) {
        model_params.n_gpu_layers = 99;
    }
    _model = llama_model_load_from_file(model_path, model_params);
    if (!_model) {
        LOGe("failed to load model from %s", model_path);
        throw std::runtime_error("loadModel() failed");
    }

    // create an instance of llama_context
    llama_context_params ctx_params = llama_context_default_params();
    const long safeContext = std::clamp(contextSize, 1L, static_cast<long>(std::numeric_limits<uint32_t>::max()));
    if (safeContext != contextSize) {
        LOGi("contextSize %ld adjusted to %ld to fit llama context limits", contextSize, safeContext);
    }
    ctx_params.n_ctx = static_cast<uint32_t>(safeContext);
    // Optimal batch sizes are typically 512-2048 for modern ARM CPUs
    // Larger batches waste memory and reduce cache efficiency
    ctx_params.n_batch = std::min(static_cast<int>(safeContext), 512);
    ctx_params.n_threads = nThreads;
    ctx_params.no_perf = true; // disable performance metrics
    _ctx = llama_init_from_model(_model, ctx_params);
    if (!_ctx) {
        LOGe("llama_new_context_with_model() returned null)");
        throw std::runtime_error("llama_new_context_with_model() returned null");
    }

    // create an instance of llama_sampler
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true; // disable performance metrics
    _sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    _formattedMessages = std::vector<char>(llama_n_ctx(_ctx));
    _messages.clear();

    if (chatTemplate == nullptr) {
        _chatTemplate = llama_model_chat_template(_model, nullptr);
    } else {
        _chatTemplate = strdup(chatTemplate);
    }
    this->_storeChats = storeChats;
    _disableThinking = false;
    _reasoningBudget = -1;
}

void
LLMInference::addChatMessage(const char *message, const char *role) {
    _messages.push_back({strdup(role), strdup(message)});
}

float
LLMInference::getResponseGenerationTime() const {
    return getResponseTokensPerSecond();
}

float
LLMInference::getResponseTokensPerSecond() const {
    if (_responseGenerationTime <= 0 || _responseNumTokens <= 0) {
        return 0.f;
    }
    return (_responseNumTokens * 1e6f) / static_cast<float>(_responseGenerationTime);
}

long
LLMInference::getResponseTokenCount() const {
    return _responseNumTokens;
}

int64_t
LLMInference::getResponseGenerationTimeMicros() const {
    return _responseGenerationTime;
}

int
LLMInference::getContextSizeUsed() const {
    return _nCtxUsed;
}

void
LLMInference::startCompletion(const char *query) {
    if (!_storeChats) {
        for (auto it = _messages.begin(); it != _messages.end();) {
            if (std::strcmp(it->role, "system") != 0) {
                free(const_cast<char *>(it->role));
                free(const_cast<char *>(it->content));
                it = _messages.erase(it);
            } else {
                ++it;
            }
        }
        _prevLen = 0;
        _formattedMessages.assign(llama_n_ctx(_ctx), 0);
    }
    _responseGenerationTime = 0;
    _responseNumTokens = 0;
    _response.clear();
    _cacheResponseTokens.clear();
    std::string finalQuery = query ? std::string(query) : std::string();
    const bool suppressThinking = _disableThinking || _reasoningBudget == 0;
    if (suppressThinking && finalQuery.find("/no_think") == std::string::npos) {
        if (!finalQuery.empty()) {
            finalQuery.insert(0, "/no_think\n");
        } else {
            finalQuery = "/no_think";
        }
    }
    addChatMessage(finalQuery.c_str(), "user");
    // apply the chat-template
    int newLen = llama_chat_apply_template(_chatTemplate, _messages.data(), _messages.size(), true,
                                           _formattedMessages.data(), _formattedMessages.size());
    if (newLen > (int) _formattedMessages.size()) {
        // resize the output buffer `_formattedMessages`
        // and re-apply the chat template
        _formattedMessages.resize(newLen);
        newLen = llama_chat_apply_template(_chatTemplate, _messages.data(), _messages.size(), true,
                                           _formattedMessages.data(), _formattedMessages.size());
    }
    if (newLen < 0) {
        throw std::runtime_error("llama_chat_apply_template() in LLMInference::startCompletion() failed");
    }
    std::string prompt(_formattedMessages.begin() + _prevLen, _formattedMessages.begin() + newLen);
    // Only add special tokens (like BOS) if we are at the start of the context
    bool add_special = (_prevLen == 0); 
    _promptTokens = common_tokenize(llama_model_get_vocab(_model), prompt, add_special, true);
    if (_promptTokens.empty()) {
        LOGe("tokenize() returned no tokens for prompt; aborting completion");
        throw std::runtime_error("empty prompt tokenization");
    }
    if (_promptTokens.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        LOGe("prompt token count %zu exceeds int32 range", _promptTokens.size());
        throw std::runtime_error("prompt too long for llama_batch");
    }

    if (_batch == nullptr) {
        _batch = new llama_batch();
    }
    std::memset(_batch, 0, sizeof(llama_batch));
    _batch->token = _promptTokens.data();
    _batch->n_tokens = static_cast<int32_t>(_promptTokens.size());

    // Fix KV cache reuse
    int n_past = 0;
    if (_storeChats && _prevLen > 0) {
         // We are appending to existing context
         // Retrieve current KV cache position
         // seq_id 0 is assumed
         int max_seq_pos = llama_memory_seq_pos_max(llama_get_memory(_ctx), 0);
         if (max_seq_pos >= 0) {
             n_past = max_seq_pos + 1;
         }
    } else {
         // New conversation or no history storage -> overwrite from 0
         n_past = 0;
         // Clear KV cache to ensure fresh start
         llama_memory_seq_rm(llama_get_memory(_ctx), -1, -1, -1);
    }
    
    LOGi("startCompletion: n_past=%d, n_tokens=%d, prevLen=%d", n_past, _batch->n_tokens, _prevLen);

    _batchPos.resize(_promptTokens.size());
    for (size_t i = 0; i < _promptTokens.size(); ++i) {
        _batchPos[i] = n_past + i;
    }
    _batch->pos = _batchPos.data();
}

// taken from:
// https://github.com/ggerganov/llama.cpp/blob/master/examples/llama.android/llama/src/main/cpp/llama-android.cpp#L38
bool
LLMInference::_isValidUtf8(const char *response) {
    if (!response) {
        return true;
    }
    const unsigned char *bytes = (const unsigned char *) response;
    int num;
    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

std::string
LLMInference::completionLoop() {
    if (_batch == nullptr || _batch->n_tokens <= 0) {
        LOGe("completionLoop invoked with empty llama_batch");
        throw std::runtime_error("llama batch missing tokens");
    }
    // check if the length of the inputs to the model
    // have exceeded the context size of the model
    uint32_t contextSize = llama_n_ctx(_ctx);
    _nCtxUsed = llama_memory_seq_pos_max(llama_get_memory(_ctx), 0) + 1;
    if (_nCtxUsed + _batch->n_tokens > contextSize) {
        throw std::runtime_error("context size reached");
    }

    auto start = ggml_time_us();
    // run the model
    if (llama_decode(_ctx, *_batch) < 0) {
        throw std::runtime_error("llama_decode() failed");
    }

    // sample a token and check if it is an EOG (end of generation token)
    // convert the integer token to its corresponding word-piece
    _currToken = llama_sampler_sample(_sampler, _ctx, -1);
    if (llama_vocab_is_eog(llama_model_get_vocab(_model), _currToken)) {
        if (_storeChats) {
            addChatMessage(_response.c_str(), "assistant");
        }
        _response.clear();
        _cacheResponseTokens.clear();
        return "[EOG]";
    }
    std::string piece = common_token_to_piece(_ctx, _currToken, true);
    auto end = ggml_time_us();
    _responseGenerationTime += (end - start);
    _responseNumTokens += 1;
    _cacheResponseTokens += piece;

    // re-init the batch with the newly predicted token
    // key, value pairs of all previous tokens have been cached
    // in the KV cache
    _batch->token = &_currToken;
    _batch->n_tokens = 1;
    
    // Set position for the next token (append to KV cache)
    int n_past_next = llama_memory_seq_pos_max(llama_get_memory(_ctx), 0) + 1;
    if (_batchPos.size() < 1) _batchPos.resize(1);
    _batchPos[0] = n_past_next;
    _batch->pos = _batchPos.data();

    _batch->seq_id = nullptr;
    _batch->n_seq_id = nullptr;
    _batch->logits = nullptr;

    if (_isValidUtf8(_cacheResponseTokens.c_str())) {
        _response += _cacheResponseTokens;
        std::string valid_utf8_piece = _cacheResponseTokens;
        _cacheResponseTokens.clear();
        return valid_utf8_piece;
    }

    return "";
}

void
LLMInference::stopCompletion() {
    if (_storeChats) {
        _prevLen = llama_chat_apply_template(_chatTemplate, _messages.data(), _messages.size(), false, nullptr, 0);
        if (_prevLen < 0) {
            throw std::runtime_error("llama_chat_apply_template() in LLMInference::stopCompletion() failed");
        }
    } else {
        _prevLen = 0;
    }
    _response.clear();
    _cacheResponseTokens.clear();
}

void
LLMInference::setReasoningOptions(bool disableThinking, int reasoningBudget) {
    const bool requestedNoThink = disableThinking || reasoningBudget == 0;
    _disableThinking = requestedNoThink;
    _reasoningBudget = reasoningBudget;
    LOGi("Reasoning controls: disableThinking=%d, reasoningBudget=%d", _disableThinking, _reasoningBudget);
}

LLMInference::~LLMInference() {
    // free memory held by the message text in messages
    // (as we had used strdup() to create a malloc'ed copy)
    for (llama_chat_message &message: _messages) {
        free(const_cast<char *>(message.role));
        free(const_cast<char *>(message.content));
    }
    llama_free(_ctx);
    llama_model_free(_model);
    delete _batch;
    llama_sampler_free(_sampler);
}

// Safe accessors used by JNI/native glue. Return internal pointers; caller must not free.
llama_model* LLMInference::getModel() {
    return _model;
}

llama_context* LLMInference::getContext() {
    return _ctx;
}

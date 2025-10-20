#pragma once
#include "llama.h"
#include "common.h"
#include <string>
#include <vector>

class LLMInference {
    // llama.cpp-specific types
    llama_context* _ctx = nullptr;
    llama_model*   _model = nullptr;
    llama_sampler* _sampler = nullptr;
    llama_token    _currToken = 0;
    llama_batch*   _batch = nullptr;

    // container to store user/assistant messages in the chat
    std::vector<llama_chat_message> _messages;
    // stores the string generated after applying
    // the chat-template to all messages in `_messages`
    std::vector<char> _formattedMessages;
    // stores the tokens for the last query
    // appended to `_messages`
    std::vector<llama_token> _promptTokens;
    int                      _prevLen = 0;
    const char*              _chatTemplate;

    // stores the complete response for the given query
    std::string _response;
    std::string _cacheResponseTokens;
    // whether to cache previous messages in `_messages`
    bool _storeChats = true;
    bool _disableThinking = false;
    int  _reasoningBudget = -1;

    // response generation metrics
    int64_t _responseGenerationTime = 0;
    long    _responseNumTokens      = 0;

    // length of context window consumed during the conversation
    int _nCtxUsed = 0;

    bool _isValidUtf8(const char* response);

  public:
    void loadModel(const char* modelPath, float minP, float temperature, bool storeChats, long contextSize,
                   const char* chatTemplate, int nThreads, bool useMmap, bool useMlock, bool useVulkan);

    void addChatMessage(const char* message, const char* role);

    float getResponseGenerationTime() const;

    float getResponseTokensPerSecond() const;

    long getResponseTokenCount() const;

    int64_t getResponseGenerationTimeMicros() const;

    int getContextSizeUsed() const;

    void startCompletion(const char* query);

    std::string completionLoop();

    void stopCompletion();

    void setReasoningOptions(bool disableThinking, int reasoningBudget);

    ~LLMInference();

    // Expose internal model/context for JNI integrations (safe accessor)
    // These return the raw pointers managed by this instance. Caller must not free them.
    llama_model* getModel();
    llama_context* getContext();
};
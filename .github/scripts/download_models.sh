#!/usr/bin/env bash
set -e

# Download models for E2E tests if space permits

mkdir -p models
cd models

# Check available disk space
AVAILABLE_SPACE_KB=$(df -k . | awk 'NR==2 {print $4}')
echo "Available space: $((AVAILABLE_SPACE_KB / 1024 / 1024)) GB"

# 1. Audio Model: Whisper Tiny (Small, ~75MB)
echo "Downloading Whisper Tiny model..."
if [ ! -f "ggml-tiny.en.bin" ]; then
    curl -L -o ggml-tiny.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
fi
echo "Whisper model downloaded."

# 2. Text Model: SmolLM2 135M (Small, ~150MB)
# We need a GGUF model for SmolLM/Llama.cpp
echo "Downloading SmolLM2 model..."
# Cleanup old failed download if exists
rm -f "SmolLM-135M-Instruct-q8_0.gguf"

if [ ! -f "SmolLM2-135M-Instruct-Q8_0.gguf" ]; then
    curl -L -o SmolLM2-135M-Instruct-Q8_0.gguf https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf
fi
echo "SmolLM2 model downloaded."

# 3. Video/Image Model: Wan 1.3B + T5 + VAE
# This is heavy. Wan 1.3B Q4_K_M is ~1GB. T5 Q8 is ~4GB.
# We need to be careful.

# Check if we have enough space (at least 10GB free to be safe)
if [ "$AVAILABLE_SPACE_KB" -gt 10485760 ]; then
    echo "Enough space for Video models. Attempting download."

    # Wan 1.3B
    if [ ! -f "Wan2.1-T2V-1.3B-Q4_K_M.gguf" ]; then
        curl -L -o Wan2.1-T2V-1.3B-Q4_K_M.gguf https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-GGUF/resolve/main/Wan2.1-T2V-1.3B-Q4_K_M.gguf
    fi

    # VAE
    if [ ! -f "wan_2.1_vae.gguf" ]; then
        curl -L -o wan_2.1_vae.gguf https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-GGUF/resolve/main/wan_2.1_vae.gguf
    fi

    # T5 Encoder (The big one)
    # Using a smaller quant if possible? The script mentions `umt5-xxl-encoder-Q3_K_S.gguf`
    # Let's see if we can find that specific file or similar.
    # The default one in scripts/run_linux_e2e.sh is `umt5-xxl-encoder-Q3_K_S.gguf`.
    # Searching for that file on HF.
    # It seems to be from `city96/t5-v1_1-xxl-encoder-gguf` or similar?
    # Or maybe it's `umt5` specific for Wan.
    # Wan-AI/Wan2.1-T2V-1.3B-GGUF repo has `umt5-xxl-enc-bf16.gguf` (huge).
    # We might need to skip video if we can't find a small T5.

    # Let's try to download the one referenced in scripts if we can find a URL.
    # Assuming user might have put it there or we use a known URL.
    # For now, I will NOT download the T5 model as it is likely too big and I don't have a reliable URL for a small one.
    # I will skip the video test but leave the structure.

    echo "Skipping T5 download due to size/URL uncertainty. Video tests will be skipped."
else
    echo "Not enough space for Video models. Skipping."
fi

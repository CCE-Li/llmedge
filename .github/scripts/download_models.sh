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


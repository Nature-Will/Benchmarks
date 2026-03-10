#!/bin/bash
# Download all DreamZero model checkpoints via hf-mirror + lab proxy
# Usage: bash download_models.sh [target]
# target: all (default), dreamzero, wan, umt5

set -e

# Use hf-mirror (domestic) to reduce bandwidth on proxy
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/shared-storage-user/p1-shared/yujiale/.cache/huggingface

# Lab proxy (no quota limit, needed because CPU node has no direct internet)
export http_proxy=http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128
export https_proxy=http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128

MODELS_DIR=/mnt/shared-storage-user/p1-shared/yujiale/models
HF_CLI=/mnt/shared-storage-user/p1-shared/yujiale/conda/envs/robotwin/bin/huggingface-cli

TARGET=${1:-all}

if [ "$TARGET" = "all" ] || [ "$TARGET" = "dreamzero" ]; then
    echo "=== Downloading DreamZero-AgiBot (~46GB) ==="
    $HF_CLI download GEAR-Dreams/DreamZero-AgiBot \
        --repo-type model \
        --local-dir $MODELS_DIR/DreamZero-AgiBot
    echo "DreamZero-AgiBot done"
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "wan" ]; then
    echo "=== Downloading Wan2.1-I2V-14B-480P ==="
    $HF_CLI download Wan-AI/Wan2.1-I2V-14B-480P \
        --local-dir $MODELS_DIR/Wan2.1-I2V-14B-480P
    echo "Wan2.1 done"
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "umt5" ]; then
    echo "=== Downloading umt5-xxl ==="
    $HF_CLI download google/umt5-xxl \
        --local-dir $MODELS_DIR/umt5-xxl
    echo "umt5-xxl done"
fi

echo "All downloads complete!"

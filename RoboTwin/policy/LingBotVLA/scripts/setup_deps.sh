#!/bin/bash
set -e

# Proxy setup
export http_proxy="http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128"
export https_proxy="http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export no_proxy="10.0.0.0/8,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,127.0.0.1,localhost,.pjlab.org.cn,.h.pjlab.org.cn"
export NO_PROXY="$no_proxy"
export HF_HOME="/mnt/shared-storage-user/p1-shared/yujiale/.cache/huggingface"
export XDG_CACHE_HOME="/mnt/shared-storage-user/p1-shared/yujiale/.cache"

MODEL_DIR="/mnt/shared-storage-user/p1-shared/yujiale/models"
mkdir -p "$MODEL_DIR"

echo "=== Step 1: Install lerobot ==="
pip install lerobot --index-url https://pypi.org/simple/ 2>&1 | tail -5
echo ""

echo "=== Step 2: Install lingbot-vla deps ==="
pip install datasets==3.6.0 numpydantic msgpack websockets --index-url https://pypi.org/simple/ 2>&1 | tail -5
echo ""

echo "=== Step 3: Download Qwen2.5-VL-3B-Instruct ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', local_dir='$MODEL_DIR/Qwen2.5-VL-3B-Instruct')
print('Qwen download complete!')
"
echo ""

echo "=== Step 4: Download lingbot-vla-4b ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('robbyant/lingbot-vla-4b', local_dir='$MODEL_DIR/lingbot-vla-4b')
print('lingbot-vla-4b download complete!')
"
echo ""

echo "=== All done! ==="

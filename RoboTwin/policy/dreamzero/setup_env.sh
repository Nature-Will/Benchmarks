#!/bin/bash
# DreamZero environment setup script
# Run: source setup_env.sh

# Proxy
export http_proxy=http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128
export https_proxy=$http_proxy
export no_proxy="10.0.0.0/8,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,127.0.0.1,localhost,.pjlab.org.cn,.h.pjlab.org.cn"
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NO_PROXY=$no_proxy

# Cache
export XDG_CACHE_HOME=/mnt/shared-storage-user/p1-shared/yujiale/.cache
export TMPDIR=/mnt/shared-storage-user/p1-shared/yujiale/.tmp
export HF_HOME=/mnt/shared-storage-user/p1-shared/yujiale/.cache/huggingface
mkdir -p $TMPDIR $HF_HOME

# Conda
source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh

echo "Proxy and conda initialized"

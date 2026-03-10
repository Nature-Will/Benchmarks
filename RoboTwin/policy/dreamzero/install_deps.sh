#!/bin/bash
# Install DreamZero dependencies
# Usage: bash install_deps.sh

set -e

export http_proxy=http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128
export https_proxy=http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128

source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/dreamzero

cd /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/dreamzero

echo "Installing DreamZero package..."
pip install -e . --no-cache-dir --index-url https://pypi.org/simple/

echo "Done!"

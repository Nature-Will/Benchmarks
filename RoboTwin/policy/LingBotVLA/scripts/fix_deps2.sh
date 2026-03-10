#!/bin/bash
set -e
export http_proxy="http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128"
export https_proxy="http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128"

echo "Installing remaining deps..."
pip install "torchdata>=0.8.0" "blobfile>=3.0.0" "safetensors" "packaging" --index-url https://pypi.org/simple/ 2>&1 | tail -5
echo "Done!"

#!/bin/bash
set -e

export http_proxy="http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128"
export https_proxy="http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

echo "Installing transformers 4.51.3..."
pip install "transformers==4.51.3" --index-url https://pypi.org/simple/ 2>&1 | tail -5

echo ""
echo "Installing torchcodec 0.6.0..."
pip install "torchcodec==0.6.0" --index-url https://pypi.org/simple/ 2>&1 | tail -3

echo ""
echo "Done!"

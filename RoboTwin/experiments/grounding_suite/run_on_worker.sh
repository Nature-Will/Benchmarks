#!/bin/bash
set -e

WORKER_ID=${1:-9f8hv}
EXP_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="${EXP_DIR}/outputs/remote_launch"
TARGET="ws-f9d64708a8234207-worker-${WORKER_ID}.lihaozhan+root.ailab-p1.pod@h.pjlab.org.cn"

mkdir -p "${OUT_DIR}"

env -u LD_LIBRARY_PATH ssh -o StrictHostKeyChecking=no -i ~/.ssh/pjlab_id_rsa -CAXY "${TARGET}" \
  "cd ${EXP_DIR} && mkdir -p outputs/remote_launch && nohup bash -lc 'export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}; export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:\$PATH; export CUDA_HOME=/usr/local/cuda; export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics; bash ${EXP_DIR}/run_all.sh' > ${EXP_DIR}/outputs/remote_launch/${WORKER_ID}.log 2>&1 & echo \$!"

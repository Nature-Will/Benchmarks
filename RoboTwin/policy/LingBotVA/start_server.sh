#!/bin/bash
# LingBot-VA Server on GPU node
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_HOME=/usr/local/cuda

source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/lingbot-va

cd /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVA/lingbot-va

START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
NGPU=${NGPU:-1}

save_root='visualization/'
mkdir -p $save_root

echo "=== Starting LingBot-VA Server on $NGPU GPU(s), port $START_PORT ==="

python -m torch.distributed.run \
    --nproc_per_node $NGPU \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root

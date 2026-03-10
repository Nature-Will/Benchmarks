#!/bin/bash
# DreamZero evaluation with a trained RoboTwin checkpoint
#
# Usage:
#   bash scripts/run_eval_trained_gpu.sh <checkpoint_step>
#   bash scripts/run_eval_trained_gpu.sh 5000
#   bash scripts/run_eval_trained_gpu.sh 3000 1  # with seed=1
#   TASK_NAME=turn_switch TASK_CONFIG=demo_clean bash scripts/run_eval_trained_gpu.sh 5000
#
# Architecture:
#   - GPU 0,1: DreamZero inference server (torchrun distributed, LoRA checkpoint)
#   - GPU 2:   RoboTwin eval loop (SAPIEN simulation + policy client)

CKPT_STEP=${1:?Usage: $0 <checkpoint_step> [seed]}
SEED=${2:-0}

# ============ GPU Node Environment Init ============
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/cuda/bin:/opt/conda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export XDG_CACHE_HOME=/mnt/shared-storage-user/p1-shared/yujiale/.cache
export TMPDIR=/mnt/shared-storage-user/p1-shared/yujiale/.tmp
export HF_HOME=/mnt/shared-storage-user/p1-shared/yujiale/.cache/huggingface
mkdir -p $TMPDIR $HF_HOME

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh

# ============ Config ============
DREAMZERO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROBOTWIN_DIR=/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin

MODEL_ROOT=${MODEL_ROOT:-$DREAMZERO_DIR/checkpoints/dreamzero_robotwin_50tasks_lora}
MODEL_PATH=$MODEL_ROOT/checkpoint-$CKPT_STEP
SERVER_PORT=5000
SERVER_GPUS="0,1"
NUM_SERVER_GPUS=2
EVAL_GPU=2

TASK_NAME=${TASK_NAME:-beat_block_hammer}
TASK_CONFIG=${TASK_CONFIG:-demo_randomized}
CKPT_SETTING=${CKPT_SETTING:-DreamZero-RoboTwin50-ckpt${CKPT_STEP}}

LOG_DIR=$DREAMZERO_DIR/logs/eval_ckpt${CKPT_STEP}
mkdir -p $LOG_DIR

echo "============================================"
echo "DreamZero Trained Model Evaluation"
echo "============================================"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Checkpoint: $MODEL_PATH"
echo "Task: $TASK_NAME ($TASK_CONFIG)"
echo "Seed: $SEED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | head -8
echo "============================================"

# Validate checkpoint exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Checkpoint not found at $MODEL_PATH"
    echo "Available checkpoints:"
    ls -d $MODEL_ROOT/checkpoint-* 2>/dev/null
    exit 1
fi

# ============ Step 1: Start DreamZero Inference Server ============
echo ""
echo ">>> Step 1: Starting DreamZero inference server on GPUs $SERVER_GPUS..."
echo "  Model: $MODEL_PATH"

cd $DREAMZERO_DIR
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/dreamzero

# Kill any existing server
pkill -f "socket_test_robotwin" 2>/dev/null || true
sleep 3

CUDA_VISIBLE_DEVICES=$SERVER_GPUS \
torchrun --nproc_per_node=$NUM_SERVER_GPUS --standalone \
    socket_test_robotwin.py \
    --port $SERVER_PORT \
    --enable-dit-cache \
    --model-path $MODEL_PATH \
    > $LOG_DIR/server.log 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"
echo "Server log: $LOG_DIR/server.log"

# Wait for server to be ready
echo "Waiting for server to be ready..."
MAX_WAIT=600
WAITED=0
while ! python -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(('localhost', $SERVER_PORT))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: Server failed to start within ${MAX_WAIT}s"
        echo "=== Server log (last 50 lines) ==="
        tail -50 $LOG_DIR/server.log
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    echo "  Waiting... (${WAITED}s elapsed)"
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died"
        echo "=== Server log (last 100 lines) ==="
        tail -100 $LOG_DIR/server.log
        exit 1
    fi
done

echo "Server ready! (took ${WAITED}s)"

# ============ Step 2: Run Evaluation ============
echo ""
echo ">>> Step 2: Running evaluation on GPU $EVAL_GPU..."
echo "  Task: $TASK_NAME"
echo "  Config: $TASK_CONFIG"
echo "  Seed: $SEED"
echo ""

cd $ROBOTWIN_DIR
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/robotwin

echo "Starting eval_policy.py..."
CUDA_VISIBLE_DEVICES=$EVAL_GPU \
python -u script/eval_policy.py \
    --config policy/dreamzero/deploy_policy.yml \
    --overrides \
    --task_name $TASK_NAME \
    --task_config $TASK_CONFIG \
    --ckpt_setting $CKPT_SETTING \
    --seed $SEED \
    --policy_name dreamzero \
    --server_port $SERVER_PORT \
    2>&1 | tee $LOG_DIR/eval.log

EVAL_EXIT=${PIPESTATUS[0]}
echo ""
echo "eval_policy.py exited with code: $EVAL_EXIT"

# ============ Step 3: Cleanup ============
echo ""
echo ">>> Cleanup: stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo ""
echo "============================================"
echo "Evaluation complete! Exit code: $EVAL_EXIT"
echo "Checkpoint: checkpoint-$CKPT_STEP"
echo "Config: $TASK_CONFIG"
echo "Logs: $LOG_DIR/"
echo "============================================"

# Quick success rate summary if eval results exist
RESULT_DIR=$ROBOTWIN_DIR/eval_result/$TASK_NAME/dreamzero/$TASK_CONFIG
if [ -d "$RESULT_DIR" ]; then
    echo ""
    echo "=== Results ==="
    python3 -c "
import os, glob
result_dir = '$RESULT_DIR'
for d in sorted(glob.glob(os.path.join(result_dir, '*/'))):
    name = os.path.basename(d.rstrip('/'))
    vids = glob.glob(os.path.join(d, 'episode*.mp4'))
    if not vids:
        continue
    total = len(vids)
    # In RoboTwin, successful episodes end early (< max_steps)
    # Check by video frame count or log parsing
    print(f'{name}: {total} episodes evaluated')
" 2>/dev/null || true
fi

exit $EVAL_EXIT

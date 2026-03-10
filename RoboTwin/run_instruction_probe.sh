#!/bin/bash
# Top-level launcher for Dim-I Instruction Following Probe
# Runs both Pi0.5 and LingBot-VA on a single GPU node
set -e

# --- Environment setup ---
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export CUDA_HOME=/usr/local/cuda
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export PYTHONUNBUFFERED=1

BASE=/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin

apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1 || true

echo "============================================"
echo "  Dim-I Instruction Following Probe"
echo "  beat_block_hammer | demo_clean | 3 ep/type"
echo "  Models: Pi0.5 (15K), LingBot-VA (posttrain)"
echo "============================================"
echo ""

# ===== Phase 1: Pi0.5 =====
echo "===== Phase 1: Pi0.5 15K checkpoint ====="
bash ${BASE}/policy/pi05/run_instruction_probe.sh 0
echo "===== Pi0.5 done ====="
echo ""

# ===== Phase 2: LingBot-VA =====
echo "===== Phase 2: LingBot-VA (posttrain-robotwin) ====="
echo "Starting LingBot-VA server in background..."
nohup bash ${BASE}/policy/LingBotVA/start_server.sh > ${BASE}/policy/LingBotVA/server_probe.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start (checking port 29056)..."
for i in $(seq 1 120); do
    if grep -q "server listening" ${BASE}/policy/LingBotVA/server_probe.log 2>/dev/null; then
        echo "Server ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: Server did not start within 10 minutes"
        cat ${BASE}/policy/LingBotVA/server_probe.log
        exit 1
    fi
    sleep 5
done

# Run LingBot-VA probe
bash ${BASE}/policy/LingBotVA/run_instruction_probe_va.sh

# Stop server
echo "Stopping LingBot-VA server..."
kill $SERVER_PID 2>/dev/null || true
echo "===== LingBot-VA done ====="

echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "============================================"
echo "  Pi0.5 results:     eval_result/beat_block_hammer/pi05/demo_clean/probe_*/..."
echo "  LingBot-VA results: policy/LingBotVA/lingbot-va/results_instruction_probe/..."
echo "============================================"

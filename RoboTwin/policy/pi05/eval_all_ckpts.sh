#!/bin/bash
# Evaluate all checkpoints in parallel on separate GPUs

TASK=beat_block_hammer
CONFIG=demo_randomized
TRAIN_CONFIG=pi05_aloha_full_base
MODEL=beat_block_hammer
SEED=0

CKPTS=(5000 10000 15000 20000)
GPUS=(0 1 2 3)

mkdir -p ../../eval_logs

for i in "${!CKPTS[@]}"; do
    echo "Launching eval for checkpoint ${CKPTS[$i]} on GPU ${GPUS[$i]}"
    bash eval.sh $TASK $CONFIG $TRAIN_CONFIG $MODEL $SEED ${GPUS[$i]} ${CKPTS[$i]} \
        > ../../eval_logs/ckpt_${CKPTS[$i]}.log 2>&1 &
done

echo "All 4 evaluations launched. Monitoring..."
wait
echo "All evaluations complete."

echo ""
echo "========== Results Summary =========="
for ckpt in "${CKPTS[@]}"; do
    echo "--- Checkpoint ${ckpt} ---"
    tail -5 ../../eval_logs/ckpt_${ckpt}.log 2>/dev/null || echo "No log found"
    echo ""
done

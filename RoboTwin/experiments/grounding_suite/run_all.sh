#!/bin/bash
set -e

EXP_DIR=$(cd "$(dirname "$0")" && pwd)

echo "============================================"
echo "  Dim-I Grounding Reproduction Suite"
echo "============================================"
echo "1) Behavior probe: Pi0.5 + LingBot-VA"
echo "2) Bottleneck: Pi0.5"
echo "3) Bottleneck: LingBot-VA"
echo "============================================"

bash "${EXP_DIR}/run_behavior_probe.sh"
bash "${EXP_DIR}/run_pi05_bottleneck.sh"
bash "${EXP_DIR}/run_lingbotva_bottleneck.sh"

echo "============================================"
echo "  ALL REPRODUCTION EXPERIMENTS COMPLETE"
echo "============================================"

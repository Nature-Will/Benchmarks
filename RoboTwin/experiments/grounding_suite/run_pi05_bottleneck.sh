#!/bin/bash
set -e

EXP_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${EXP_DIR}/../.." && pwd)
OUT_DIR="${EXP_DIR}/outputs/pi05_bottleneck"

mkdir -p "${OUT_DIR}"

cd "${ROOT}"
bash "${ROOT}/run_bottleneck.sh" 2>&1 | tee "${OUT_DIR}/run.log"

if [ -d "${ROOT}/reports/attention_analysis" ]; then
  mkdir -p "${OUT_DIR}/native_outputs"
  cp -f "${ROOT}"/reports/attention_analysis/* "${OUT_DIR}/native_outputs/" 2>/dev/null || true
fi

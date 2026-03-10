#!/bin/bash
set -e

EXP_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${EXP_DIR}/../.." && pwd)
OUT_DIR="${EXP_DIR}/outputs/behavior_probe"

mkdir -p "${OUT_DIR}"

cd "${ROOT}"
bash "${ROOT}/run_instruction_probe.sh" 2>&1 | tee "${OUT_DIR}/run.log"

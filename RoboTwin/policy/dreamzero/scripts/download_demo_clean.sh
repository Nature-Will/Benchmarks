#!/bin/bash
# Download demo_clean data from HuggingFace for all 50 RoboTwin tasks
#
# Downloads aloha-agilex_clean_50.zip per task and extracts the full dataset
# directly into RoboTwin/data/{task}/demo_clean/.
#
# Total download: ~11.5GB (50 tasks × ~229MB each)
#
# Usage (CPU node):
#   bash scripts/download_demo_clean.sh

set -euo pipefail

ROBOTWIN_DATA_DIR=${ROBOTWIN_DATA_DIR:-"/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/data"}
HF_CACHE_DIR=${HF_CACHE_DIR:-"/tmp/robotwin_hf_clean"}
HF_REPO="TianxingChen/RoboTwin2.0"
HF_CLI=${HF_CLI:-"/mnt/shared-storage-user/p1-shared/yujiale/conda/envs/robotwin/bin/huggingface-cli"}
MAX_RETRIES=${MAX_RETRIES:-3}
ATTEMPT_TIMEOUT=${ATTEMPT_TIMEOUT:-900}
HF_PROXY_MODE=${HF_PROXY_MODE:-us}
HF_ENDPOINT_MODE=${HF_ENDPOINT_MODE:-official}
LAB_PROXY_SOURCE=${LAB_PROXY_SOURCE:-/root/.bashrc}

export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-120}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

configure_endpoint() {
    case "$HF_ENDPOINT_MODE" in
        official)
            unset HF_ENDPOINT
            ;;
        mirror)
            export HF_ENDPOINT=https://hf-mirror.com
            ;;
        *)
            echo "Unknown HF_ENDPOINT_MODE=$HF_ENDPOINT_MODE (expected: official, mirror)"
            return 1
            ;;
    esac
}

configure_proxy() {
    local proxy_url=""

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

    case "$HF_PROXY_MODE" in
        none)
            return 0
            ;;
        us)
            proxy_url="${HF_PROXY_URL:-http://127.0.0.1:9050}"
            ;;
        custom)
            proxy_url="${HF_PROXY_URL:-}"
            ;;
        lab)
            proxy_url="${HF_PROXY_URL:-$(grep -o 'http://[^ ]*@proxy\.h\.pjlab\.org\.cn:23128' "$LAB_PROXY_SOURCE" 2>/dev/null | head -1 || true)}"
            ;;
        *)
            echo "Unknown HF_PROXY_MODE=$HF_PROXY_MODE (expected: none, us, lab, custom)"
            return 1
            ;;
    esac

    if [ -n "$proxy_url" ]; then
        export http_proxy="$proxy_url"
        export https_proxy="$proxy_url"
        export HTTP_PROXY="$proxy_url"
        export HTTPS_PROXY="$proxy_url"
        echo "Proxy mode: $HF_PROXY_MODE"
    else
        echo "Proxy mode: none (no proxy URL found)"
    fi
}

configure_endpoint
configure_proxy

DEFAULT_TASKS=(
    adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
    click_alarmclock click_bell dump_bin_bigbin grab_roller
    handover_block handover_mic hanging_mug lift_pot
    move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad
    open_laptop open_microwave pick_diverse_bottles pick_dual_bottles
    place_a2b_left place_a2b_right place_bread_basket place_bread_skillet
    place_burger_fries place_can_basket place_cans_plasticbox place_container_plate
    place_dual_shoes place_empty_cup place_fan place_mouse_pad
    place_object_basket place_object_scale place_object_stand place_phone_stand
    place_shoe press_stapler put_bottles_dustbin put_object_cabinet
    rotate_qrcode scan_object shake_bottle shake_bottle_horizontally
    stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two
    stamp_seal turn_switch
)

if [ -n "${TASKS:-}" ]; then
    # shellcheck disable=SC2206
    ALL_TASKS=(${TASKS})
else
    ALL_TASKS=("${DEFAULT_TASKS[@]}")
fi

echo "=== Download demo_clean datasets for ${#ALL_TASKS[@]} tasks ==="
echo "Output: $ROBOTWIN_DATA_DIR/{task}/demo_clean/"
echo "HF endpoint mode: $HF_ENDPOINT_MODE ${HF_ENDPOINT:+($HF_ENDPOINT)}"
echo "HF CLI: $HF_CLI"
echo ""

if [ ! -x "$HF_CLI" ]; then
    echo "HF CLI not found or not executable: $HF_CLI"
    exit 1
fi

DOWNLOADED=0
SKIPPED=0
FAILED=0

download_task() {
    local task="$1"
    local download_dir="$2"
    local attempt

    for attempt in $(seq 1 "$MAX_RETRIES"); do
        if timeout "$ATTEMPT_TIMEOUT" "$HF_CLI" download "$HF_REPO" \
            --repo-type dataset \
            --include "dataset/$task/aloha-agilex_clean_50.zip" \
            --local-dir "$download_dir" 2>&1 | tail -2; then
            return 0
        fi
        echo "  [WARN] download attempt ${attempt}/${MAX_RETRIES} failed or timed out for $task"
        sleep $((attempt * 5))
    done

    return 1
}

for task in "${ALL_TASKS[@]}"; do
    TASK_DIR="$ROBOTWIN_DATA_DIR/$task/demo_clean"
    TRAJ_DIR="$TASK_DIR/_traj_data"
    DATA_DIR="$TASK_DIR/data"
    SEED_FILE="$TASK_DIR/seed.txt"

    # Skip if already has the full extracted dataset.
    if [ -d "$TRAJ_DIR" ] && [ -d "$DATA_DIR" ] && [ "$(ls "$TRAJ_DIR"/*.pkl 2>/dev/null | wc -l)" -ge 50 ] && [ "$(ls "$DATA_DIR"/episode*.hdf5 2>/dev/null | wc -l)" -ge 50 ] && [ -f "$SEED_FILE" ]; then
        echo "[SKIP] $task - already has full demo_clean dataset"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[DOWNLOAD] $task ..."

    DOWNLOAD_DIR="$HF_CACHE_DIR/$task"
    mkdir -p "$DOWNLOAD_DIR"

    if ! download_task "$task" "$DOWNLOAD_DIR"; then
        echo "  [FAIL] Download failed for $task"
        FAILED=$((FAILED + 1))
        continue
    fi

    ZIP_FILE="$DOWNLOAD_DIR/dataset/$task/aloha-agilex_clean_50.zip"
    if [ ! -f "$ZIP_FILE" ]; then
        echo "  [FAIL] Zip not found at $ZIP_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Extract to temp
    TEMP_DIR="/tmp/robotwin_extract_$task"
    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"

    if ! unzip -q "$ZIP_FILE" -d "$TEMP_DIR"; then
        echo "  [FAIL] Unzip failed for $task"
        FAILED=$((FAILED + 1))
        rm -rf "$TEMP_DIR"
        continue
    fi

    # Find extracted dataset root
    DATASET_SRC=$(find "$TEMP_DIR" -maxdepth 2 -type d -name "aloha-agilex_clean_50" | head -1)
    if [ -z "$DATASET_SRC" ]; then
        echo "  [FAIL] Extracted dataset root not found for $task"
        FAILED=$((FAILED + 1))
        rm -rf "$TEMP_DIR"
        continue
    fi

    mkdir -p "$TASK_DIR"
    cp -a "$DATASET_SRC"/. "$TASK_DIR"/

    NUM_PKL=$(find "$TRAJ_DIR" -maxdepth 1 -type f -name 'episode*.pkl' 2>/dev/null | wc -l)
    NUM_HDF5=$(find "$DATA_DIR" -maxdepth 1 -type f -name 'episode*.hdf5' 2>/dev/null | wc -l)

    echo "  [OK] $task - pkl=$NUM_PKL, hdf5=$NUM_HDF5"
    DOWNLOADED=$((DOWNLOADED + 1))

    rm -rf "$TEMP_DIR"
    # Clean up downloaded zip to save space
    rm -rf "$DOWNLOAD_DIR"
done

# Cleanup
rm -rf "$HF_CACHE_DIR"

echo ""
echo "=== Download Complete ==="
echo "Downloaded: $DOWNLOADED"
echo "Skipped: $SKIPPED"
echo "Failed: $FAILED"
echo "Total: ${#ALL_TASKS[@]}"

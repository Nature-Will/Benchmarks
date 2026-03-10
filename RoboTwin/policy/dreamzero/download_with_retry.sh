#!/bin/bash
# Download models with automatic stall detection and retry.
# Usage: bash download_with_retry.sh [target]
# target: all (default), dreamzero, wan, umt5

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/shared-storage-user/p1-shared/yujiale/.cache/huggingface
export http_proxy=http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128
export https_proxy=http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128

MODELS_DIR=/mnt/shared-storage-user/p1-shared/yujiale/models
HF_CLI=/mnt/shared-storage-user/p1-shared/yujiale/conda/envs/robotwin/bin/huggingface-cli
STALL_TIMEOUT=120  # seconds without progress before restart

TARGET=${1:-all}

download_one() {
    # Downloads a single model with stall detection + auto-retry.
    # Runs in foreground (call with & to parallelize).
    local NAME=$1
    local REPO=$2
    local LOCAL_DIR=$3
    shift 3
    local EXTRA_ARGS="$@"

    echo "[$NAME] Starting download..."
    while true; do
        # Start download in background
        $HF_CLI download $REPO $EXTRA_ARGS --local-dir "$LOCAL_DIR" &
        local PID=$!
        echo "[$NAME] PID=$PID, current size: $(du -sh "$LOCAL_DIR" 2>/dev/null | awk '{print $1}')"

        # Monitor for stall
        local STALL_SECS=0
        local LAST_SIZE=$(du -sb "$LOCAL_DIR" 2>/dev/null | awk '{print $1}')
        LAST_SIZE=${LAST_SIZE:-0}

        while kill -0 $PID 2>/dev/null; do
            sleep 30
            local CUR_SIZE=$(du -sb "$LOCAL_DIR" 2>/dev/null | awk '{print $1}')
            CUR_SIZE=${CUR_SIZE:-0}

            if [ "$CUR_SIZE" != "$LAST_SIZE" ]; then
                # Progress!
                STALL_SECS=0
                LAST_SIZE=$CUR_SIZE
                echo "[$NAME] Progress: $(du -sh "$LOCAL_DIR" 2>/dev/null | awk '{print $1}')"
            else
                STALL_SECS=$((STALL_SECS + 30))
                if [ $STALL_SECS -ge $STALL_TIMEOUT ]; then
                    echo "[$NAME] Stalled for ${STALL_SECS}s. Killing PID $PID and restarting..."
                    kill $PID 2>/dev/null
                    wait $PID 2>/dev/null
                    sleep 3
                    break  # break inner while, retry in outer while true
                fi
            fi
        done

        # If we got here without breaking (process exited on its own)
        if ! kill -0 $PID 2>/dev/null; then
            wait $PID 2>/dev/null
            local EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                echo "[$NAME] DONE! Final size: $(du -sh "$LOCAL_DIR" 2>/dev/null | awk '{print $1}')"
                return 0
            fi
            # Non-zero exit and not from our kill — could be transient error
            if [ $STALL_SECS -lt $STALL_TIMEOUT ]; then
                echo "[$NAME] Process exited with code $EXIT_CODE (not stall). Retrying in 10s..."
                sleep 10
            fi
            # If it was our kill (stall), just loop immediately
        fi
    done
}

# Launch downloads
PIDS=""

if [ "$TARGET" = "all" ] || [ "$TARGET" = "dreamzero" ]; then
    download_one "dreamzero" "GEAR-Dreams/DreamZero-AgiBot" "$MODELS_DIR/DreamZero-AgiBot" --repo-type model &
    PIDS="$PIDS $!"
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "wan" ]; then
    download_one "wan" "Wan-AI/Wan2.1-I2V-14B-480P" "$MODELS_DIR/Wan2.1-I2V-14B-480P" &
    PIDS="$PIDS $!"
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "umt5" ]; then
    download_one "umt5" "google/umt5-xxl" "$MODELS_DIR/umt5-xxl" &
    PIDS="$PIDS $!"
fi

echo "All downloads launched (PIDs:$PIDS). Waiting..."

# Wait for all to finish
FAIL=0
for PID in $PIDS; do
    wait $PID || FAIL=$((FAIL+1))
done

if [ $FAIL -eq 0 ]; then
    echo "=== ALL DOWNLOADS COMPLETE ==="
else
    echo "=== $FAIL download(s) failed ==="
    exit 1
fi

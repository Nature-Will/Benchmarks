#!/bin/bash
# Wrapper to run client in background
pkill -f eval_polict_client 2>/dev/null
sleep 1
nohup bash /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVA/start_client.sh results/ beat_block_hammer 100 \
    > /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVA/client.log 2>&1 &
echo "client PID: $!"
sleep 5
tail -20 /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVA/client.log

#!/bin/bash
set -o pipefail
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/cuda/bin:/opt/conda/bin:$PATH
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
mkdir -p /usr/share/vulkan/icd.d/ 2>/dev/null
echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json 2>/dev/null || true

source /mnt/shared-storage-user/p1-shared/yujiale/conda/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/p1-shared/yujiale/conda/envs/robotwin
POLICY_DIR=/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVLA
mkdir -p "$POLICY_DIR/logs"

echo "=== Test 1: SAPIEN only (no model) ==="
cd /mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin
python -u -X faulthandler -c "
import sys, os
sys.path.append('./')
sys.path.append('./script')
sys.path.append('./policy')
sys.path.append('./description/utils')

from test_render import Sapien_TEST
Sapien_TEST()
print('Render OK')

from envs import CONFIGS_PATH
import importlib, yaml
envs_module = importlib.import_module('envs.beat_block_hammer')
env_class = getattr(envs_module, 'beat_block_hammer')
TASK_ENV = env_class()

with open('./task_config/demo_clean.yml', 'r') as f:
    args = yaml.safe_load(f)
args['task_name'] = 'beat_block_hammer'
args['task_config'] = 'demo_clean'
args['ckpt_setting'] = '0'

embodiment_config_path = os.path.join(CONFIGS_PATH, '_embodiment_config.yml')
with open(embodiment_config_path) as f:
    _embodiment_types = yaml.safe_load(f)
with open(CONFIGS_PATH + '_camera_config.yml') as f:
    _camera_config = yaml.safe_load(f)

embodiment_type = args.get('embodiment')
args['left_robot_file'] = _embodiment_types[embodiment_type[0]]['file_path']
args['right_robot_file'] = _embodiment_types[embodiment_type[0]]['file_path']
args['dual_arm_embodied'] = True
args['left_embodiment_config'] = yaml.safe_load(open(os.path.join(args['left_robot_file'], 'config.yml')))
args['right_embodiment_config'] = yaml.safe_load(open(os.path.join(args['right_robot_file'], 'config.yml')))
head_camera_type = args['camera']['head_camera_type']
args['head_camera_h'] = _camera_config[head_camera_type]['h']
args['head_camera_w'] = _camera_config[head_camera_type]['w']
args['eval_mode'] = True

print('Setting up demo (no model loaded)...')
TASK_ENV.setup_demo(now_ep_num=0, seed=100000, is_test=True, **args)
print('SAPIEN setup_demo SUCCESS (no model)')
TASK_ENV.close_env()

print()
print('=== Test 2: Load model THEN setup_demo ==='  )
import torch
print(f'Loading model to GPU... (torch {torch.__version__}, CUDA {torch.cuda.is_available()})')
os.environ['QWEN25_PATH'] = '/mnt/shared-storage-user/p1-shared/yujiale/models/Qwen2.5-VL-3B-Instruct'
sys.path.insert(0, 'policy/LingBotVLA/lingbot-vla')
from deploy.lingbot_robotwin_policy import QwenPiServer
server = QwenPiServer(
    path_to_pi_model='/mnt/shared-storage-user/p1-shared/yujiale/models/lingbot-vla-4b-root/checkpoints/base/hf_ckpt',
    use_length=50, chunk_ret=True, use_bf16=True,
)
print('Model loaded, now trying setup_demo...')
TASK_ENV2 = env_class()
TASK_ENV2.setup_demo(now_ep_num=0, seed=100000, is_test=True, **args)
print('SAPIEN setup_demo SUCCESS (with model)')
TASK_ENV2.close_env()
print('ALL TESTS PASSED')
" 2>&1 | tee "$POLICY_DIR/logs/test_sapien_result.log"
echo "Exit code: ${PIPESTATUS[0]}"

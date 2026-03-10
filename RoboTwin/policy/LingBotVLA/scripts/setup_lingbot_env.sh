#!/bin/bash
set -e

export http_proxy="http://lihaozhan:CbvmmmgYaKySXGl8AGZn3YpOsCNK8MNXrWFjEM4VAxrocePHGGApT59sebHX@proxy.h.pjlab.org.cn:23128"
export https_proxy="$http_proxy"
export no_proxy="10.0.0.0/8,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,127.0.0.1,localhost,.pjlab.org.cn,.h.pjlab.org.cn"

CONDA_DIR=/mnt/shared-storage-user/p1-shared/yujiale/conda
ENV_DIR=$CONDA_DIR/envs/lingbot

source $CONDA_DIR/etc/profile.d/conda.sh

if [ -d "$ENV_DIR" ]; then
    echo "Environment already exists at $ENV_DIR"
    conda activate $ENV_DIR
else
    echo "=== Creating Python 3.11 environment ==="
    conda create -p $ENV_DIR python=3.11 -y
    conda activate $ENV_DIR
fi

echo "Python: $(python --version)"

# Install same sapien/mplib as pi05 (Python 3.11, known working)
echo "=== Installing sapien + mplib ==="
pip install sapien==3.0.0b1 mplib==0.2.1 --index-url https://pypi.org/simple/ 2>&1 | tail -3

# Install torch (same as robotwin env for GPU compat)
echo "=== Installing PyTorch ==="
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3

# Install transformers, lerobot, and lingbot-vla deps
echo "=== Installing transformers + lerobot + deps ==="
pip install transformers==4.51.3 lerobot==0.4.4 --index-url https://pypi.org/simple/ 2>&1 | tail -3
pip install torchdata blobfile ipdb numpydantic datasets --index-url https://pypi.org/simple/ 2>&1 | tail -3

# Install lingbot-vla
echo "=== Installing lingbot-vla ==="
LINGBOT_DIR=/mnt/shared-storage-user/p1-shared/yujiale/code/benchmarks/RoboTwin/policy/LingBotVLA/lingbot-vla
pip install -e $LINGBOT_DIR --no-deps --index-url https://pypi.org/simple/ 2>&1 | tail -3

# Install remaining RoboTwin deps (sapien already installed)
echo "=== Installing RoboTwin deps ==="
pip install opencv-python h5py pyyaml pillow einops warp-lang --index-url https://pypi.org/simple/ 2>&1 | tail -3

# Create lerobot.common compatibility shim
echo "=== Creating lerobot.common compat shim ==="
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p $SITE_PACKAGES/lerobot/common/datasets
mkdir -p $SITE_PACKAGES/lerobot/common/policies/pi0
cat > $SITE_PACKAGES/lerobot/common/__init__.py << 'SHIMEOF'
SHIMEOF
cat > $SITE_PACKAGES/lerobot/common/datasets/__init__.py << 'SHIMEOF'
SHIMEOF
cat > $SITE_PACKAGES/lerobot/common/datasets/lerobot_dataset.py << 'SHIMEOF'
from lerobot.datasets.lerobot_dataset import *
from lerobot.datasets.lerobot_dataset import LeRobotDataset
SHIMEOF
cat > $SITE_PACKAGES/lerobot/common/policies/__init__.py << 'SHIMEOF'
SHIMEOF
cat > $SITE_PACKAGES/lerobot/common/policies/pretrained.py << 'SHIMEOF'
from lerobot.policies.pretrained import *
from lerobot.policies.pretrained import PreTrainedPolicy
SHIMEOF
cat > $SITE_PACKAGES/lerobot/common/policies/pi0/__init__.py << 'SHIMEOF'
SHIMEOF
cat > $SITE_PACKAGES/lerobot/common/policies/pi0/configuration_pi0.py << 'SHIMEOF'
from lerobot.policies.pi0.configuration_pi0 import *
from lerobot.policies.pi0.configuration_pi0 import PI0Config
SHIMEOF

echo "=== Verifying ==="
python -c "
import sapien; print('sapien:', sapien.__version__)
import mplib; print('mplib:', mplib.__version__)
import torch; print('torch:', torch.__version__)
import transformers; print('transformers:', transformers.__version__)
import lingbotvla; print('lingbotvla: OK')
print('All imports OK!')
"

echo "=== Done ==="

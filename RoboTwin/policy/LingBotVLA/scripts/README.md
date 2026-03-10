# LingBotVLA Scripts

Top-level layout:

- `../deploy_policy.py` and `../deploy_policy.yml`: RoboTwin policy entrypoints.
- `../scripts/`: launch, training, evaluation, env setup, and debug shell scripts.
- `../tools/`: dataset conversion, normalization, and plotting utilities.
- `../logs/`: local run logs.
- `../figures/`: generated plots.

Common entrypoints:

- `run_lingbot_train.sh`: launch multi-GPU training.
- `run_lingbot_eval.sh`: run a single evaluation.
- `run_lingbot_eval_all.sh`: evaluate checkpoints sequentially.
- `run_lingbot_eval_all_ckpts.sh`: evaluate checkpoints in parallel.
- `run_lingbot_debug_eval.sh`: run debug evaluation.
- `train.sh`: full train pipeline for a small task/config setting.
- `train_randomized500.sh`: full train pipeline for the 500-episode randomized dataset.
- `setup_lingbot_env.sh`: create and install the LingBot conda environment.
- `install_curobo_gpu.sh`: install CuRobo into the LingBot environment.
- `test_sapien_only.sh`: debug SAPIEN and model loading behavior.

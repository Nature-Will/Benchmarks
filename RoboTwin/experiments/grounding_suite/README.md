# Grounding Suite

这个目录是这轮指令跟随/grounding 复现实验的统一入口。

目录约定：

- `run_all.sh`
  从头复现实验总入口。
- `run_behavior_probe.sh`
  复现 Pi0.5 + LingBot-VA 的行为层 instruction probe。
- `run_pi05_bottleneck.sh`
  复现 Pi0.5 的 bottleneck/linear probe 分析，并把结果拷回本目录。
- `run_lingbotva_bottleneck.sh`
  运行 LingBot-VA 的新 bottleneck/linear probe 分析。
- `verify_lingbotva_bottleneck.py`
  LingBot-VA 的离线分析脚本。
- `run_on_worker.sh`
  远端 worker 启动入口，默认 `9f8hv`。
- `outputs/`
  本目录下统一存放日志和汇总后的实验产物。

说明：

- Pi0.5 的底层分析代码仍然位于 `policy/pi05/verify_grounding_bottleneck.py`。
- LingBot-VA 的新分析逻辑集中在本目录，不再散落在 repo 根目录或 `policy/` 下。

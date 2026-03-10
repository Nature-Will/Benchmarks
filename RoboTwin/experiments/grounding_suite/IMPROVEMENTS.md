# Better Experiments

下面这些改进建议，目标都是把“模型看到了语言差异”与“语言差异是否真的控制了动作”区分得更清楚。

## 1. Held-out Probe

当前 probe 基本是可分性诊断，容易被 token 位置或固定 prompt 模板放大。

建议：

- 训练集用一组 prompt 模板，测试集换同义改写。
- 训练集用一组初始化图像，测试集换另一组初始化图像。
- 对 action-side probe 增加 held-out denoising step。

这样能回答“信息是否稳定泛化”，而不仅是“是否能在原样本上线性恢复”。

## 2. Causal Ablation

不只看表示差异，还直接做因果干预。

建议：

- 把 prompt embedding 置零，测行为变化。
- 用 `seen` 的视觉输入配 `do_nothing` 的 prompt embedding，直接看动作是否切换。
- 对 action-side hidden state 做投影消融：去掉最能区分指令的线性方向，再看输出动作是否变化。

这比单纯的 cosine / probe 更能回答“语言差异有没有真的被用到”。

## 3. Trajectory Divergence

不要只看某一层某一步，而是看完整去噪轨迹是否分叉。

建议：

- 比较不同指令下每个 denoising step 的 action latent 距离。
- 汇总成 trajectory divergence curve。
- 单独统计“早期分叉”和“晚期被拉回”的情况。

这有助于区分“从头就没用语言”和“中间短暂用过但最后被视觉先验压回去”。

## 4. Counterfactual Ranking

把问题改成排序而不是二分类。

建议：

- 对同一视觉场景，准备多个互斥动作描述。
- 用 action output 去匹配哪个 prompt 最一致。
- 看模型是否只认“任务身份”，还是能区分“敲击 / 静止 / 放下 / 错抓”等细粒度意图。

## 5. Multi-Task Stress Test

当前单任务环境下，视觉捷径太强。

建议：

- 在相同视觉对象上设计多个有效动作目标。
- 或者在相同动作模板上更换目标物体/目标位置。
- 把 instruction sensitivity score 作为主指标，不只看 success rate。

## 6. Unified Metrics

后续所有模型建议统一输出下面几类指标：

- behavior sensitivity: `seen_success - do_nothing_success`
- text-side separability: text probe / cosine
- action-side separability: action probe / cosine
- causal sensitivity: prompt ablation / prompt swap 后动作变化
- trajectory divergence: 跨 step 的 action latent 距离

这样 Pi0.5、LingBot-VA 以及后续模型就能做横向对比。

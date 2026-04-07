# 强化学习 - 策略梯度

- UID: `MATH-0412`
- 学科: `数学`
- 分类: `强化学习`
- 源序号: `412`
- 目标目录: `Algorithms/数学-强化学习-0412-强化学习_-_策略梯度`

## R01

策略梯度（Policy Gradient）是一类直接优化策略参数 `theta` 的强化学习方法，不先学状态价值表，而是直接让策略 `pi_theta(a|s)` 的期望回报变高。

本目录给出一个最小可运行 MVP：在自定义离散环境 `LineWorld` 上，用 REINFORCE（蒙特卡洛策略梯度）训练一个小型 PyTorch 策略网络。

## R02

要解决的问题：

给定随机策略 `pi_theta(a|s)`，最大化目标函数

`J(theta) = E_{tau ~ pi_theta}[sum_{t=0}^{T-1} gamma^t r_t]`

其中 `tau` 是轨迹（状态-动作-奖励序列）。

MVP 任务是让智能体从中间状态出发，尽量向右走到终点（奖励 `+1`），并避免走到左端终点（奖励 `-1`）。

## R03

为什么选这个设定：
- 环境小、可控，不依赖 `gym` 等额外包，保证目录自包含可运行。
- 奖励是稀疏终止奖励，能体现策略梯度对“整条轨迹回报”的优化本质。
- 通过概率策略采样，能直观看到“随机探索 -> 概率偏置 -> 收敛”的过程。

## R04

REINFORCE 核心梯度恒等式：

`nabla_theta J(theta) = E[ sum_t nabla_theta log pi_theta(a_t|s_t) * G_t ]`

其中 `G_t = sum_{k=t}^{T-1} gamma^{k-t} r_k` 是从时刻 `t` 开始的折扣回报。

在实现里使用一个标量移动平均基线 `b` 降低方差，更新信号变为：

`A_t = G_t - b`

损失写成最小化形式：

`L(theta) = - sum_t log pi_theta(a_t|s_t) * A_t`

## R05

算法流程（高层伪代码）：

```text
初始化策略网络 pi_theta 和优化器
for episode in 1..N:
  用当前 pi_theta 采样一条完整轨迹
  计算每个时间步的折扣回报 G_t
  计算优势 A_t = G_t - baseline
  loss = -sum_t log pi_theta(a_t|s_t) * A_t
  反向传播并更新 theta
  baseline <- EMA(episode_return)
训练后做采样评估和贪心评估
```

## R06

时间复杂度（单回合）近似为：

- 采样交互：`O(T * C_pi)`，`T` 为回合步长，`C_pi` 为一次前向策略网络成本。
- 回报计算：`O(T)`。
- 反向传播：`O(T * C_pi)`（对轨迹中每步 log-prob 求梯度）。

总计约 `O(T * C_pi)`；训练 `N` 回合为 `O(N * T * C_pi)`。

## R07

空间复杂度：

- 轨迹缓存 `log_probs`、`rewards`：`O(T)`。
- 网络参数：`O(P)`（`P` 为参数量）。
- 训练日志表（pandas）：`O(N)`。

MVP 规模很小（7 个状态、2 个动作、16 隐层），内存开销低。

## R08

关键参数（`PGConfig` 默认值）：

- `n_states=7`：状态数（必须奇数，起点在中间）。
- `max_steps=12`：单回合最大步数，防止无限循环。
- `gamma=0.99`：折扣因子。
- `lr=0.02`：Adam 学习率。
- `train_episodes=600`：训练回合数。
- `baseline_momentum=0.90`：回报基线的 EMA 动量。
- `grad_clip=5.0`：梯度裁剪上限。
- `seed=2026`：随机种子，保障可复现。

## R09

实现边界与假设：

- 只演示离散动作、离散状态的随机策略梯度。
- 不实现 Actor-Critic、GAE、PPO 等增强技巧。
- 不依赖任何强化学习黑盒库（如 Stable-Baselines）。
- 目标是教学型最小实现，优先可读、可审计而非性能极限。

## R10

`demo.py` 的函数职责：

- `LineWorldEnv`：定义环境动力学与终止奖励。
- `PolicyNet`：`state_index -> action logits` 的策略网络。
- `discounted_returns`：倒序计算 `G_t`。
- `sample_action`：按 `Categorical` 分布采样动作并返回 `log_prob`。
- `rollout_episode`：采样一整条轨迹。
- `train_reinforce`：执行 REINFORCE 更新并记录训练日志。
- `evaluate_policy`：评估采样策略和贪心策略表现。
- `print_training_summary`：打印 head/tail 训练轨迹与关键指标。
- `main`：整合训练前评估、训练、训练后评估。

## R11

运行方式（无交互输入）：

```bash
uv run python Algorithms/数学-强化学习-0412-强化学习_-_策略梯度/demo.py
```

脚本会自动训练并输出训练前后指标和训练轨迹摘要。

## R12

输出字段说明：

- `avg_return`：评估回合平均总回报。
- `success_rate`：到达右端终点（回报 `>0`）的比例。
- `avg_steps`：每回合平均步数。
- `start_right_prob`：起始状态下选择“向右”动作的概率。
- `return_ma20`：最近 20 回合平均回报。
- `success_ma20`：最近 20 回合成功率均值。

通常训练后应看到 `avg_return` 与 `success_rate` 明显上升。

## R13

正确性快速检查：

- 训练后 `post_eval_greedy.success_rate` 应接近 `1.0`。
- `return_ma20` 在后期通常高于前期。
- 起始状态的 `start_right_prob` 应向 `1` 靠近。
- 全流程不应出现 `NaN/Inf`（loss、概率、回报均有限）。

## R14

常见失败模式与排查：

- 学习率过大：loss 振荡或策略塌缩，成功率波动大。
- 回合数太少：策略还未稳定偏向右侧。
- `gamma` 太小：过度短视，学习信号变弱。
- 无基线时方差大：训练速度慢、曲线抖动大。
- 不做梯度裁剪时：偶发极端轨迹可能导致梯度过大。

## R15

与相近方法对比：

- 对比 Q-learning：
  - Q-learning 学的是动作价值再贪心选动作；
  - 策略梯度直接学随机策略参数。
- 对比 Actor-Critic：
  - REINFORCE 用整回合回报，方差较高；
  - Actor-Critic 用价值函数作 baseline，通常更稳定。
- 对比 PPO：
  - PPO 在策略更新上做了约束，工程稳定性更好；
  - 本 MVP 更适合教学与源码透明。

## R16

典型应用场景：

- 需要输出随机策略分布（非单一贪心动作）的任务。
- 离散动作控制问题的算法教学基线。
- 更复杂策略优化（Actor-Critic/PPO）之前的可验证起点。

## R17

可扩展方向：

- 把标量基线升级为状态价值网络（Actor-Critic）。
- 支持 batch 轨迹并行采样，降低梯度估计方差。
- 使用 GAE、熵正则、学习率调度改进训练稳定性。
- 扩展到连续动作（高斯策略）与更复杂环境。

## R18

`demo.py` 源码级流程（8 步）：

1. `main()` 构建 `PGConfig` 并设置随机种子，先对随机初始化策略做一次预评估。
2. `train_reinforce()` 初始化 `LineWorldEnv`、`PolicyNet` 和 `Adam` 优化器。
3. 每个训练回合调用 `rollout_episode()`：循环执行 `sample_action()`，拿到 `log_prob`、奖励序列和回合总回报。
4. `discounted_returns()` 反向遍历奖励，显式计算每个时间步的 `G_t`（不是黑盒 RL API）。
5. 使用移动平均基线 `b` 构造优势 `A_t = G_t - b`，并组装 `loss = -sum(log_prob_t * A_t)`。
6. 执行 `loss.backward()`，再做 `clip_grad_norm_` 与 `optimizer.step()` 完成参数更新。
7. 把 `return/loss/start_right_prob` 写入 pandas 日志，并计算 `return_ma20`、`success_ma20` 作为训练趋势指标。
8. 训练结束后，`evaluate_policy()` 分别做“采样评估”和“贪心评估”，`print_training_summary()` 输出前后对比与收敛信号。

以上步骤完整展开了策略梯度实现链路：环境交互、回报计算、梯度构造、参数更新与评估，全程可在源码中逐行追踪。

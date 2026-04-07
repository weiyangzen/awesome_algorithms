# 强化学习 - PPO

- UID: `MATH-0414`
- 学科: `数学`
- 分类: `强化学习`
- 源序号: `414`
- 目标目录: `Algorithms/数学-强化学习-0414-强化学习_-_PPO`

## R01

PPO（Proximal Policy Optimization）是主流的策略梯度强化学习算法，用于在“可学习速度”与“更新稳定性”之间做折中。

核心思想：
- 保留旧策略 `pi_theta_old` 采样得到的轨迹；
- 用概率比 `r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)` 衡量新旧策略偏移；
- 用 clip 机制限制每次策略更新幅度，防止策略一步走太远导致性能崩塌。

本目录给出一个从零实现的最小可运行 PPO-Clip MVP：
- 自定义小环境 `LineWorld`；
- PyTorch 实现 Actor/Critic；
- GAE 优势估计 + clipped surrogate objective；
- 无黑盒 RL 框架依赖。

## R02

简要发展脉络：
- 早期策略梯度（REINFORCE）无偏但方差较大；
- TRPO 引入“信赖域”约束提升稳定性，但实现复杂、计算代价高；
- PPO 通过剪切目标函数近似信赖域效果，保留了较好的稳定性与更简单实现；
- 之后 PPO 成为大量工业/研究基线算法之一。

因此 PPO 常被视作“工程友好的稳定策略梯度框架”。

## R03

PPO 主要解决的问题：
- 纯策略梯度更新步长敏感，容易因为一次大更新导致策略退化；
- 价值学习与策略学习耦合时，策略振荡会放大训练不稳定。

PPO 的解法是把“改进策略”和“限制偏移”同时写进目标函数：
- 通过优势函数推动改进；
- 通过 ratio clipping 限制新旧策略比值偏离。

本 MVP 在离散动作环境中演示该机制，目标是让智能体从中间出发稳定走向右端终点（`+1`）。

## R04

本实现使用 PPO-Clip + GAE 的标准骨架。

记：
- `r_t(theta) = exp(log pi_theta(a_t|s_t) - log pi_theta_old(a_t|s_t))`
- `A_t` 为优势估计（GAE）

策略目标（最大化）等价最小化损失：

`L_actor = -E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]`

价值损失：

`L_critic = E[(V_phi(s_t) - R_t)^2]`

总目标（分开优化）：
- Actor 额外带熵正则：`L_actor_total = L_actor - c_ent * H(pi_theta)`
- Critic 使用系数缩放：`c_v * L_critic`

其中 `R_t = A_t + V(s_t)`，`A_t` 用 GAE 回推得到。

## R05

高层算法流程：

```text
初始化 Actor/Critic 与优化器
for iteration in 1..K:
  用旧策略采样一批完整轨迹（on-policy）
  用奖励 + Critic 值估计回推 GAE 优势 A_t 与 returns R_t
  对优势做标准化
  for epoch in 1..E:
    打乱并切分 mini-batch
    更新 Actor: clipped surrogate + entropy bonus
    更新 Critic: MSE(V(s), R_t)
训练结束后做贪心评估并打印统计
```

## R06

复杂度估计（记批次步数为 `B`，每次前向/反向网络成本为 `C_net`）：

- 采样阶段：`O(B * C_net)`
- 更新阶段：`O(E * B * C_net)`（`E` 为 update epochs）
- 总体训练（`K` 次迭代）：`O(K * (1+E) * B * C_net)`

空间复杂度：
- 轨迹缓存（状态、动作、log_prob、returns、advantages）约 `O(B)`
- 模型参数 `O(|theta| + |phi|)`

本任务规模很小，CPU 即可快速跑完。

## R07

`demo.py` 的实验配置（`PPOConfig` 默认）：

- 环境：`LineWorld(n_states=9, max_steps=24, step_penalty=-0.01)`
- 折扣与优势：`gamma=0.99`, `gae_lambda=0.95`
- PPO：`clip_eps=0.2`, `update_epochs=8`, `minibatch_size=64`
- 优化：`actor_lr=3e-3`, `critic_lr=6e-3`
- 正则：`entropy_coef=1e-2`, `value_coef=0.5`, `max_grad_norm=0.5`
- 训练长度：`steps_per_batch=512`, `train_iterations=60`
- 评估：`eval_episodes=300`

技术栈：`numpy + pandas + scipy + scikit-learn + PyTorch`。

## R08

建议前置知识：

- MDP 与 on-policy 采样概念；
- 策略梯度与 `log pi(a|s)` 梯度形式；
- 优势函数与 GAE 的偏差-方差折中；
- PyTorch 反向传播、`Categorical(logits=...)` 的用法；
- mini-batch SGD 的基本训练流程。

## R09

本实现的边界与假设：

- 只演示离散动作 PPO-Clip，不覆盖连续动作高斯策略；
- 环境是教学型小 MDP，不追求复杂任务表现；
- 使用 on-policy 轨迹，不复用历史 batch；
- 目标是“可读、可审计、可运行”，不是大规模训练框架。

## R10

脚本内置了基础正确性检查：

1. 训练日志中的 `actor_loss` 与 `critic_loss` 必须是有限数值；
2. 贪心评估成功率需达到阈值（`>= 0.90`）；
3. 起始状态向右动作概率需明显占优（`>= 0.80`）；
4. 价值函数应与状态从左到右趋势大体一致（Spearman 相关阈值）。

这些断言用于确保“算法链路确实在工作”，而非仅代码可执行。

## R11

数值稳定性策略：

- Actor 使用 `Categorical(logits=...)`，避免手动 softmax+log 的数值误差；
- 优势先标准化，减小批间尺度波动；
- ratio 用 clip 限制在 `[1-eps, 1+eps]` 邻域；
- Actor/Critic 分开优化，且使用梯度裁剪；
- 用熵正则避免策略过快塌缩为确定性单动作。

## R12

关键超参数含义：

- `clip_eps`：越小越保守，越大越激进；
- `gae_lambda`：越大越接近长时优势，方差更高；
- `entropy_coef`：控制探索强度；
- `steps_per_batch`：越大梯度估计越稳但更新频率下降；
- `update_epochs`：过小学习不足，过大易过拟合旧 batch；
- `actor_lr / critic_lr`：学习速率匹配很关键，Critic 太慢会拖累策略更新质量。

调参建议：优先观察 `success_ma10`、`approx_kl`、`clip_fraction` 三者联动。

## R13

理论直觉：

- PPO 不是严格信赖域优化，但通过 clip 近似了“限制策略偏移”的作用；
- GAE 在偏差和方差之间做连续可调折中（由 `lambda` 控制）；
- Critic 提供低方差学习信号，Actor 在受约束目标下稳步改进。

实践上通常追求“稳定提高成功率”，而非形式化全局最优证明。

## R14

常见失效模式与修复：

- 策略几乎不学习：
  适度增大 `actor_lr` 或减小 `clip_eps` 之外的过强正则。
- 策略振荡明显：
  降低 `actor_lr`，增大 `steps_per_batch`，观察 `approx_kl` 是否过大。
- Critic 漂移导致优势噪声大：
  调整 `critic_lr`，必要时降低 `value_coef`。
- 过早塌缩为单动作：
  增大 `entropy_coef`，并检查奖励设计是否过于单一。

## R15

工程落地建议：

- 明确分层：环境、采样、优势估计、策略更新、评估分函数实现；
- 固定随机种子并记录关键曲线（回报、成功率、KL、clip_fraction）；
- 保留断言与统计摘要，便于自动化回归验证；
- 迁移复杂任务时，可按顺序加：并行采样、学习率调度、归一化器、早停策略。

## R16

与相关算法对比：

- REINFORCE：更新无约束，方差大；PPO 更稳。
- Actor-Critic（一步 TD）：实现更简单但更新约束弱；PPO 通常更鲁棒。
- TRPO：理论约束更强但实现复杂；PPO 在工程上更轻量。
- SAC：离策略且样本效率高，但实现与调试复杂度更高。

## R17

运行方式（无交互输入）：

```bash
cd Algorithms/数学-强化学习-0414-强化学习_-_PPO
uv run python demo.py
```

预期输出包含：
- 训练迭代数与滑动均值指标（`return_ma10`, `success_ma10`）；
- 贪心评估平均回报、成功率、95% 置信区间；
- 起始状态向右动作概率；
- 价值趋势相关系数；
- 最近 8 行训练日志表；
- 最后打印 `All checks passed.`。

## R18

`demo.py` 源码级算法流（9 步）：

1. `StateFeaturizer` 用 `StandardScaler` 对 one-hot 状态做标准化，形成网络输入特征。  
2. `collect_rollout` 用当前策略与环境交互，显式记录 `state/action/reward/done/old_log_prob/value`。  
3. 同函数内对整批轨迹做反向回推，按 GAE 公式计算 `advantages`，再得 `returns = advantages + values`。  
4. `ppo_update` 将 batch 张量化并标准化优势，准备 mini-batch 优化。  
5. Actor 前向计算 `new_log_prob` 与 `ratio`，构造 `min(ratio*A, clip(ratio)*A)` 的 clipped surrogate 损失。  
6. Actor 分支加入熵正则后反向传播并做梯度裁剪，完成策略参数更新。  
7. Critic 分支用 `MSE(V(s), returns)` 更新价值网络，同样执行梯度裁剪。  
8. 每轮记录 `actor_loss/critic_loss/entropy/approx_kl/clip_fraction`，并用 `pandas` 计算滑动均值趋势。  
9. `evaluate_policy` 用贪心动作做独立评估，`scipy.stats` 计算 95% CI，并通过断言闭环验证训练有效性。  

全流程没有调用黑盒 PPO 训练 API，关键更新步骤都在源码中逐行可追踪。

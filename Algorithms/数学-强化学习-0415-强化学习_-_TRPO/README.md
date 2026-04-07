# 强化学习 - TRPO

- UID: `MATH-0415`
- 学科: `数学`
- 分类: `强化学习`
- 源序号: `415`
- 目标目录: `Algorithms/数学-强化学习-0415-强化学习_-_TRPO`

## R01

TRPO（Trust Region Policy Optimization，信赖域策略优化）是策略梯度家族中的“约束优化”方法：  
它不是直接做一次大步梯度更新，而是在“平均 KL 散度不超过阈值”的信赖域内改进策略。

本目录提供一个最小可运行 MVP：在自定义离散环境 `LineWorld` 上，用 PyTorch 从零实现 TRPO 核心更新链路（Fisher 向量积、共轭梯度、回溯线搜索）。

## R02

TRPO 的核心优化问题可以写成：

`max_theta E_t[ (pi_theta(a_t|s_t) / pi_old(a_t|s_t)) * A_t ]`

约束：

`E_t[ D_KL( pi_old(.|s_t) || pi_theta(.|s_t) ) ] <= delta`

其中：
- `A_t` 是优势函数估计（本实现使用 GAE）；
- `delta` 是最大平均 KL（本实现默认 `max_kl=0.01`）。

这个约束的目的不是追求单步最优，而是避免策略更新过猛导致性能崩溃。

## R03

为什么要 TRPO（相对普通策略梯度）：

- 普通策略梯度对学习率敏感，步长稍大容易策略退化；
- TRPO 显式控制“新旧策略距离”，提高训练稳定性；
- 在函数逼近场景中，TRPO 常作为 PPO 等后续方法的理论和工程基础。

本 MVP 重点展示“可信更新机制”，不是追求最大样本效率。

## R04

本目录实现了一个离散动作版 TRPO 流程：

1. 用当前策略采样一批轨迹（on-policy）。
2. 用值函数估计和 GAE 计算 `advantages` 与 `returns`。
3. 拟合值网络（最小化 MSE）。
4. 对策略目标 `L(theta)` 求梯度 `g`。
5. 用 KL 的 Hessian-向量积构造 Fisher 近似 `H v`。
6. 用共轭梯度近似解 `H x = g`。
7. 按 `sqrt(2*delta/(x^T H x))` 缩放步长。
8. 用回溯线搜索选取满足 KL 约束且目标提升的参数更新。

这就是 TRPO 的最小骨架。

## R05

关键数学组件（对应 `demo.py`）：

- 代理目标（surrogate objective）：
  `L = E[ ratio_t * A_t ]`，`ratio_t = exp(logp_new - logp_old)`。
- 平均 KL：
  `KL = E[ D_KL(old || new) ]`。
- Fisher 向量积：
  `Hv = ∇_theta (∇_theta KL · v)`，再加阻尼 `damping * v`。
- 共轭梯度：
  迭代求解 `H^-1 g`，避免显式构建大 Hessian 矩阵。
- 线搜索：
  多次缩小步长，直到同时满足
  1) `KL <= max_kl`；2) surrogate 有改进。

## R06

时间复杂度（每次训练迭代，粗略）：

- 采样与前向：`O(T * C_pi)`，`T` 为 batch 总步数；
- 值网络拟合：`O(value_updates * T * C_v)`；
- TRPO 更新：
  - 共轭梯度 `cg_iters` 次，每次一次 Fisher 向量积；
  - 每次 FVP 需要两次自动求导（一次一阶、一次二阶向量积）。

综合上可近似为：

`O(T*(C_pi + value_updates*C_v) + cg_iters*C_fvp + backtrack_iters*C_eval)`

小环境下 CPU 可在秒级完成。

## R07

空间复杂度：

- 轨迹缓存：`O(T)`（状态、动作、log_prob、logits、优势、回报）；
- 模型参数：`O(P_pi + P_v)`；
- 日志表：`O(train_iters)`。

本实现网络规模很小（两层 MLP），主要内存占用来自 batch 轨迹张量，仍属于轻量级。

## R08

默认超参数（`TRPOConfig`）：

- `n_states=9`，`max_steps=20`：环境规模；
- `train_iters=35`，`batch_size=600`：训练迭代与每轮采样量；
- `gamma=0.99`，`lam=0.97`：折扣和 GAE 参数；
- `max_kl=0.01`：信赖域约束；
- `cg_iters=10`，`damping=0.1`：共轭梯度/Fisher 稳定化参数；
- `backtrack_iters=10`，`backtrack_coeff=0.8`：线搜索策略；
- `value_lr=1e-2`，`value_updates=25`：值函数拟合强度；
- `seed=415`：复现实验。

## R09

实现边界与假设：

- 仅实现离散动作 TRPO（Categorical policy）；
- 仅 on-policy 采样，不用经验回放；
- 不调用黑盒 RL 框架（如 stable-baselines 的 TRPO API）；
- 环境是教学型 `LineWorld`，用于透明展示算法机制。

因此它是“结构正确且可运行”的 MVP，不是生产级强化学习框架。

## R10

`demo.py` 中的关键函数职责：

- `collect_batch`：采样轨迹并计算 GAE 优势与回报；
- `surrogate_loss`：构造 TRPO 代理目标；
- `mean_kl`：计算新旧策略平均 KL；
- `conjugate_gradient`：求解 `H^-1 g`；
- `trpo_step`：完成一次 TRPO 策略更新（含线搜索）；
- `train_value_function`：拟合价值网络；
- `evaluate_policy`：贪心/采样评估；
- `main`：训练闭环与打印摘要。

函数划分与理论步骤一一对应，便于审计。

## R11

运行方式（无交互）：

```bash
uv run python Algorithms/数学-强化学习-0415-强化学习_-_TRPO/demo.py
```

脚本会自动完成：
- 训练前评估；
- 多轮 TRPO 训练；
- 训练后贪心与采样评估；
- 打印最后若干轮训练统计并执行断言检查。

## R12

输出指标说明：

- `batch_mean_return`：每轮采样批次的回合平均回报；
- `batch_success_rate`：批次内成功到达右端终点比例；
- `mean_kl`：策略更新后平均 KL（应被限制在阈值附近）；
- `line_search_accept`：该轮线搜索是否接受更新；
- `start_right_prob`：起始状态选择“向右”动作概率；
- `return_ma5` / `success_ma5`：5 轮滑动均值。

这些指标可同时观测“性能增长”和“约束是否生效”。

## R13

正确性快速检查：

1. 训练后 `post_success` 应显著高于训练前（本脚本断言 `>=0.90`）。
2. `mean_kl` 不应长期大幅超出 `max_kl`。
3. `start_right_prob` 应从接近随机逐步向 1 靠近。
4. 关键统计值应全是有限数（无 `NaN/Inf`）。

脚本末尾包含对应断言，便于自动化验证。

## R14

常见失败模式与排查建议：

- `max_kl` 太小：学习过慢，几乎不更新；
- `max_kl` 太大：更新过激，性能波动；
- `damping` 太小：Fisher 近奇异，共轭梯度不稳定；
- `batch_size` 太小：优势噪声大，线搜索接受率低；
- `value_updates` 太少：优势估计劣化，策略梯度方向变差。

排查优先级通常是：先看 KL 与接受率，再调 batch 与 damping。

## R15

与相近算法对比：

- REINFORCE：
  - 无约束，更新简单但方差大、稳定性弱；
  - TRPO 在目标改进上更保守、更稳。
- Actor-Critic（一步式）：
  - 仍用一阶优化器，依赖学习率调参；
  - TRPO 用约束步，减少“灾难性一步”概率。
- PPO：
  - 用 clip 目标近似信赖域，工程实现更简洁；
  - TRPO 约束更“显式和几何化”，但实现复杂。

## R16

典型应用场景：

- 需要较稳定策略更新的 on-policy 控制任务；
- 希望研究“自然梯度/信赖域”思想的教学与实验；
- 作为理解 PPO、NPG 等算法的理论过渡。

在大规模工程里，PPO 更常见；但 TRPO 对理解策略优化几何结构很关键。

## R17

可扩展方向：

- 将值函数拟合从简单回归升级为 minibatch + early stopping；
- 引入并行环境采样提升吞吐；
- 支持连续动作高斯策略（对角协方差）；
- 在策略网络中加入归一化、残差结构；
- 与 PPO 对照实验，比较同环境下稳定性与样本效率。

这些扩展都可在当前脚本结构上渐进实现。

## R18

`demo.py` 源码级 TRPO 流程（9 步）：

1. `main()` 构建 `TRPOConfig`、环境、策略网络和值网络，并做训练前评估。  
2. `collect_batch()` 按当前策略采样轨迹，逐步缓存 `state/action/log_prob/logits/reward/value`。  
3. 在每条轨迹末端用 bootstrap 值构造 `delta_t`，通过 `discount_cumsum()` 得到 GAE 优势与 returns。  
4. `train_value_function()` 用 MSE 对值网络做多步拟合，提供更平滑的优势基线。  
5. `surrogate_loss()` 显式计算 `ratio * advantage`，得到 TRPO 的代理目标梯度 `g`。  
6. `trpo_step()` 内部用 `mean_kl()` + 自动求导实现 Fisher 向量积 `Hv`，再用 `conjugate_gradient()` 近似求 `H^-1 g`。  
7. 根据 `max_kl` 计算理论步长缩放，得到 `full_step`，避免超出信赖域。  
8. 对候选参数执行回溯线搜索：仅当 `KL <= max_kl` 且 surrogate 改进时才接受更新，否则回滚到旧参数。  
9. 训练循环记录 `KL/回报/成功率/起始动作概率`，最终做训练后评估并执行断言闭环。  

以上 9 步完全在源码中展开，没有依赖黑盒 RL 训练器，能够逐行追踪 TRPO 的核心机制。

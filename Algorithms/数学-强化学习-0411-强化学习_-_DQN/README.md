# 强化学习 - DQN

- UID: `MATH-0411`
- 学科: `数学`
- 分类: `强化学习`
- 源序号: `411`
- 目标目录: `Algorithms/数学-强化学习-0411-强化学习_-_DQN`

## R01

DQN（Deep Q-Network）是把 Q-learning 与神经网络结合的离策略强化学习方法：

- 用参数化函数 `Q_theta(s,a)` 近似动作价值；
- 用经验回放（Replay Buffer）打破样本时序相关性；
- 用目标网络（Target Network）稳定 Bellman 目标。

本目录给出一个最小可运行 MVP：在自定义离散环境 `LineWorld` 中，使用 PyTorch 从零实现 DQN 的训练、评估与基础正确性检查。

## R02

要解决的问题是：在马尔可夫决策过程（MDP）中找到近似最优动作价值函数，使得策略

`pi(s) = argmax_a Q_theta(s,a)`

能够最大化期望折扣回报。

本 MVP 的任务目标是：智能体从中间状态出发，学习持续向右移动到达终点（奖励 `+1`），并避免到达左端终点（奖励 `-1`）。

## R03

为什么选 DQN 做这个示例：

- 它是值函数深度强化学习的经典起点，机制清晰（Bellman 目标 + 神经网络拟合）。
- 在离散动作场景下实现成本低、教学价值高。
- 通过 replay 与 target network，能直接观察“稳定化技巧”对训练曲线的影响。

此外，本实现不依赖 Gym 或 Stable-Baselines 等黑盒训练框架，关键更新都在源码里显式展开。

## R04

核心公式（Q-learning 目标）：

`y = r + gamma * (1-done) * max_{a'} Q_{theta^-}(s', a')`

`L(theta) = E[(Q_theta(s,a) - y)^2]`

在 `demo.py` 中使用 Huber 损失（`smooth_l1_loss`）替代纯 MSE，提高训练稳健性：

`L_huber = smooth_l1(Q_theta(s,a), y)`

动作选择采用 epsilon-greedy：

- 以概率 `epsilon` 随机探索；
- 以概率 `1-epsilon` 执行 `argmax_a Q_theta(s,a)`。

## R05

算法流程（高层伪代码）：

```text
初始化在线 Q 网络 Q_theta、目标网络 Q_theta^-、经验回放池 D
for episode in 1..N:
  reset 环境
  while not done:
    按 epsilon-greedy 选择动作 a
    执行动作得到 (s, a, r, s', done)，写入 D
    若 D 样本量足够：
      从 D 随机采样 batch
      计算 y = r + gamma*(1-done)*max_a' Q_theta^-(s',a')
      最小化 Huber(Q_theta(s,a), y)
    每隔 K 步把 Q_theta 参数复制到 Q_theta^-
训练后用贪心策略评估
```

## R06

时间复杂度（单步更新）近似为：

- 环境交互：`O(1)`（本例为离散小环境）；
- 网络前向/反向：`O(B * C_q)`，`B` 是 batch size，`C_q` 是一次 Q 网络前向成本；
- 目标网络同步：每 `K` 步一次，摊销后成本较低。

整体训练复杂度可记为 `O(T * B * C_q)`，其中 `T` 为总交互步数。

## R07

空间复杂度主要由两部分组成：

- 回放池：`O(capacity)` 条转移（每条包含 `s,a,r,s',done`）；
- 网络参数：`O(P)`，在线网络与目标网络约 `2P`。

在本 MVP 默认配置下（回放池 4000、网络 32 隐层），内存占用很小。

## R08

关键参数（见 `DQNConfig`）：

- `gamma=0.98`：折扣因子。
- `lr=3e-3`：Adam 学习率。
- `batch_size=32`：每次回放采样批大小。
- `replay_capacity=4000`：经验池容量。
- `warmup_steps=64`：开始梯度更新前最小样本量。
- `target_update_interval=50`：目标网络硬更新周期（按环境步）。
- `epsilon_start=1.0`、`epsilon_end=0.05`、`epsilon_decay_steps=1200`：探索率线性退火。
- `train_episodes=500`：训练轮数。
- `grad_clip=5.0`：梯度裁剪上限。
- `seed=411`：随机种子。

## R09

实现边界与假设：

- 只覆盖离散动作 DQN，不包含 Double DQN、Dueling DQN、Prioritized Replay。
- 环境为低维可观测状态，使用 one-hot 编码。
- 目标是“教学型最小闭环”，优先透明实现而非性能极限。
- 训练与评估均为单进程 CPU，可直接运行。

## R10

`demo.py` 主要模块职责：

- `LineWorld`：定义状态转移、终止条件与奖励函数。
- `ReplayBuffer`：维护经验池并执行随机无放回采样。
- `QNetwork`：从 one-hot 状态输出两个动作的 Q 值。
- `epsilon_by_step` / `select_action`：实现探索-利用策略。
- `optimize_dqn`：构造 Bellman 目标并执行一次梯度更新。
- `train_dqn`：组织完整训练循环与目标网络同步。
- `evaluate_policy`：使用贪心策略评估平均回报与成功率。
- `main`：串联训练前评估、训练、训练后评估、日志与断言。

## R11

运行方式（无交互输入）：

```bash
uv run python Algorithms/数学-强化学习-0411-强化学习_-_DQN/demo.py
```

脚本会自动完成训练并打印训练摘要、评估结果与检查结论。

## R12

输出字段说明：

- `return`：每回合累计回报。
- `success`：是否到达右端终点（1 表示成功）。
- `loss`：该回合内 DQN 更新损失均值。
- `epsilon`：该回合末探索率。
- `return_ma20`：最近 20 回合平均回报。
- `success_ma20`：最近 20 回合平均成功率。

额外会输出训练前/训练后的贪心评估：`avg_return`、`success_rate`、`avg_steps`。

## R13

快速正确性检查（脚本内置断言）：

- 训练日志必须为有限值（无 `NaN/Inf`）。
- 训练后贪心成功率 `post_success_rate` 应不低于训练前。
- 在该简单环境下，训练后成功率应达到较高水平（默认阈值 `>=0.90`）。
- 起始状态下 `Q(start,right)` 应高于 `Q(start,left)`，体现策略偏好正确。

## R14

常见失败模式与排查：

- 探索不足：`epsilon` 衰减过快，容易早熟到次优策略。
- 目标震荡：学习率偏大或目标网络更新过于频繁。
- 回放池过小：样本多样性不足，更新高相关导致不稳定。
- 奖励信号弱：如果步惩罚过重或终止奖励设计不合理，学习可能停滞。
- 训练回合过少：Q 值尚未传播到起始状态，评估表现偏弱。

## R15

与相近算法对比：

- 对比表格 Q-learning：
  DQN 用神经网络近似 `Q(s,a)`，可扩展到更高维状态；表格法只能处理小状态空间。
- 对比策略梯度：
  DQN 学价值函数再贪心决策；策略梯度直接优化策略分布。
- 对比 Actor-Critic：
  DQN 是纯价值法；Actor-Critic 同时学习策略与价值，通常在连续动作更自然。
- 对比 Double DQN：
  本实现未处理最大化偏差；Double DQN 通过动作选择/评估分离缓解过估计。

## R16

典型应用场景：

- 离散动作控制任务（如简化游戏决策、离散调度）。
- 需要值函数可解释性（比较各动作 Q 值）的小中型问题。
- 作为进阶值函数算法（Double/Dueling/Distributional DQN）的教学与验证基线。

## R17

可扩展方向：

- 升级为 Double DQN，降低过估计偏差。
- 引入 Prioritized Replay，提高样本效率。
- 使用 Dueling 网络结构分离状态价值与优势。
- 增加软更新（Polyak）替代硬同步，平滑目标网络演化。
- 扩展到向量化环境并行采样，提升训练吞吐。

## R18

`demo.py` 源码级算法流（8 步）：

1. `main()` 构造 `DQNConfig`，先初始化一个未训练 `QNetwork` 做贪心基线评估。  
2. `train_dqn()` 中创建 `LineWorld`、在线网络 `q_net`、目标网络 `target_net`，并复制初始参数。  
3. 每个环境步调用 `epsilon_by_step()` 计算当前探索率，再由 `select_action()` 执行 epsilon-greedy 选动作。  
4. 与环境交互得到 `(s,a,r,s',done)`，写入 `ReplayBuffer.append()`。  
5. `optimize_dqn()` 从回放池随机采样 batch，显式计算 `q_sa` 与 Bellman 目标 `td_target`。  
6. 使用 `smooth_l1_loss(q_sa, td_target)` 反向传播，执行梯度裁剪与 `Adam.step()` 更新在线网络。  
7. 按固定步频把在线网络参数复制到目标网络，实现稳定的 bootstrap 目标。  
8. 训练结束后 `evaluate_policy()` 用纯贪心动作评估表现，并在 `main()` 输出日志与断言完成闭环验证。  

以上 8 步完整覆盖 DQN 的关键机制：探索、回放采样、目标构造、梯度更新、目标网络同步与独立评估，全程可在源码逐行追踪。

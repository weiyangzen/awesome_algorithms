# 强化学习 - Actor-Critic

- UID: `MATH-0413`
- 学科: `数学`
- 分类: `强化学习`
- 源序号: `413`
- 目标目录: `Algorithms/数学-强化学习-0413-强化学习_-_Actor-Critic`

## R01

Actor-Critic（演员-评论家）是把“策略函数（Actor）”与“价值函数（Critic）”联合学习的一类强化学习方法：

- Actor 学习策略 `pi_theta(a|s)`，决定在状态 `s` 下如何选动作；
- Critic 学习价值估计（常见为 `V_w(s)` 或 `Q_w(s,a)`），评估 Actor 当前行为好坏；
- Critic 给 Actor 提供低方差学习信号（优势函数或 TD 误差），Actor 据此更新策略。

最典型的一步式形式：

- TD 目标：`y_t = r_t + gamma * V_w(s_{t+1})`
- 优势近似：`A_t ≈ y_t - V_w(s_t)`
- 策略梯度更新方向：`nabla_theta J(theta) ≈ E[nabla_theta log pi_theta(a_t|s_t) * A_t]`

## R02

发展脉络可概括为：

- 1980s-1990s：策略梯度与时序差分（TD）思想逐步融合；
- 1999 年前后：Konda & Tsitsiklis 系统讨论了 Actor-Critic 的收敛性质（函数逼近与两时间尺度思想）；
- 2010s：A3C / A2C 把 Actor-Critic 推向深度强化学习主流；
- 近年：PPO、SAC 等主流算法都可视作 Actor-Critic 框架上的稳定化和工程化演进。

因此 Actor-Critic 更像“方法家族”，而非单一公式。

## R03

它主要解决两类核心矛盾：

- 纯价值法（如 Q-learning）在连续动作或高维策略参数化下不自然；
- 纯策略梯度（REINFORCE）虽直接优化策略，但梯度方差通常较大、样本效率偏低。

Actor-Critic 用 Critic 做“基线/优势估计”，在保持策略优化灵活性的同时，显著降低梯度方差，提升学习稳定性。

## R04

本目录 `demo.py` 使用的一步式 on-policy Actor-Critic 核心流程：

1. 在状态 `s_t` 用 Actor 采样动作 `a_t ~ pi_theta(.|s_t)`；
2. 与环境交互得到 `(r_t, s_{t+1}, done)`；
3. Critic 计算 `V_w(s_t)`，并构造 `y_t = r_t + gamma * (1-done) * V_w(s_{t+1})`；
4. 得到优势 `A_t = y_t - V_w(s_t)`；
5. Actor 最小化 `L_actor = -log pi_theta(a_t|s_t) * stop_grad(A_t) - beta * H(pi_theta(.|s_t))`；
6. Critic 最小化 `L_critic = 0.5 * (y_t - V_w(s_t))^2`；
7. 按 episode 重复直到策略在评估中收敛。

## R05

若每步前向反向开销记为 `C_net`，每回合平均长度 `T`，训练回合数 `E`：

- 时间复杂度约为 `O(E * T * C_net)`；
- 空间复杂度约为 `O(|theta| + |w|)`（在线一步更新，不存长轨迹）；
- 与蒙特卡洛策略梯度相比，样本效率更高；与离策略大回放方法相比，实现更轻量。

本 MVP 环境很小，CPU 下几秒内即可完成训练与评估。

## R06

`demo.py` 的最小实验设计：

- 环境：自定义 `LineWorld`（7 个离散状态，左终点奖励 `-1`，右终点奖励 `+1`）；
- 状态表示：one-hot 向量；
- Actor 网络：`Linear(7,32)-Tanh-Linear(32,2)` 输出动作 logits；
- Critic 网络：`Linear(7,32)-Tanh-Linear(32,1)` 输出状态价值；
- 学习规则：一步 TD 优势 + 策略梯度 + 熵正则；
- 输出：训练统计、滑动均值、200 回合确定性评估成功率。

该设置避免外部环境依赖（如 Gym），强调 Actor-Critic 机制本身。

## R07

优点：

- 兼具策略优化灵活性与价值评估降方差能力；
- 可自然扩展到连续动作（高斯策略等）；
- 框架通用，易与熵正则、GAE、clip 目标等结合。

局限：

- Actor 与 Critic 互相耦合，训练不稳定时可能相互放大误差；
- 对学习率、优势估计质量、奖励尺度较敏感；
- 简单一步 TD 在长时延奖励问题上偏差可能较大。

## R08

建议前置知识：

- 马尔可夫决策过程（MDP）：状态、动作、回报、折扣因子；
- 策略梯度与 `log-derivative trick`；
- 时序差分学习（TD(0)）与价值函数逼近；
- PyTorch 自动求导与优化器基本用法。

## R09

适用场景：

- 动作策略需要直接参数化优化（离散/连续）；
- 需要在线、轻量、可解释地更新策略；
- 作为 PPO/A2C/SAC 等进阶算法的基础构件。

不太适用：

- 极端稀疏奖励且无奖励塑形；
- 高维部分可观测任务但没有记忆结构（RNN/Transformer）；
- 必须强离策略重用历史数据但又不做稳定化修正的场景。

## R10

实现正确性检查建议：

1. Actor 输出概率和应为 1（softmax 后）；
2. Critic 输出应为有限实数，无 `NaN/Inf`；
3. 优势 `A_t` 在训练中不应长期全零；
4. 训练后评估成功率应显著高于随机策略；
5. 固定随机种子可复现实验趋势。

`demo.py` 用断言检查了有限数值、评估成功率阈值与价值排序一致性。

## R11

数值稳定性要点：

- 使用 logits 构造 `Categorical`，避免手动 softmax 后再取 log；
- Actor 更新时对优势 `detach`，防止策略梯度错误回传到 Critic；
- 加入小熵正则 `entropy_coef`，降低策略过早塌缩风险；
- 控制学习率（Actor 与 Critic 可分开设置）避免互相“追逐震荡”。

## R12

关键参数与调优经验：

- `gamma`：常设 `0.95~0.99`，越大越重视长期收益；
- `actor_lr`：过大策略会抖动，过小收敛慢；
- `critic_lr`：通常可略高于 Actor，先让价值估计跟上；
- `entropy_coef`：探索-收敛平衡，过小易早熟，过大难收敛；
- 网络宽度：本例 32 隐层神经元足够，小任务不需大模型。

建议先锁定环境，再做单参数扫描，观察 `return_ma20` 与成功率曲线。

## R13

理论上（在适当步长条件与函数逼近假设下）：

- Critic 近似策略下价值函数，Actor 沿近似策略梯度上升；
- 两时间尺度更新常用于稳定收敛分析（Critic 更快、Actor 更慢）；
- 一步 TD 属于“有偏但低方差”估计，在工程上常比纯蒙特卡洛更稳。

实际中通常追求“稳定可用”而非严格全局最优证明。

## R14

常见失效模式与应对：

- 策略塌缩到单动作：
  增大 `entropy_coef`，降低 `actor_lr`。
- Critic 估计漂移严重：
  降低 `critic_lr`，或对奖励做缩放/裁剪。
- 学习过程高方差、波动大：
  增大 batch（或并行环境）、使用优势归一化/GAE。
- 奖励几乎学不到：
  增加奖励塑形或课程学习，检查环境终止逻辑是否正确。

## R15

工程落地建议：

- 将“环境逻辑”与“学习器逻辑”解耦（本实现即分离 `LineWorld` 与训练函数）；
- 固定随机种子并记录关键指标（成功率、回报滑动均值、损失）；
- 保留简洁断言，尽早暴露数值异常；
- 迁移到复杂任务时优先引入：并行采样、GAE、梯度裁剪、归一化。

## R16

与相关算法对比：

- REINFORCE：无 Critic，梯度无偏但方差高；Actor-Critic 方差更低。
- DQN：学习动作价值并隐式贪心，不直接输出随机策略；Actor-Critic 直接参数化策略。
- PPO：本质是带 clip 约束的 Actor-Critic，稳定性更强、工程更常用。
- SAC：离策略最大熵 Actor-Critic，样本效率高但实现更复杂。

## R17

`demo.py` 实验输出包含：

- 500 回合训练统计；
- 最近 20 回合平均回报与成功率；
- 200 回合确定性策略评估结果；
- 最近 10 回合的表格明细（`pandas`）；
- 断言检查通过后输出 `All checks passed.`。

运行方式：

```bash
cd Algorithms/数学-强化学习-0413-强化学习_-_Actor-Critic
uv run python demo.py
```

脚本不需要任何交互输入。

## R18

`demo.py` 源码级算法流（8 步）：

1. `LineWorld.reset/step` 生成可重复、低维的 episodic MDP 交互轨迹。  
2. `train_actor_critic` 初始化 Actor/Critic 网络与各自优化器，设置随机种子。  
3. 每步将离散状态转成 one-hot，Actor 前向得到 logits 并用 `Categorical` 采样动作。  
4. 与环境交互后，Critic 用 `V(s_t)` 与 `V(s_{t+1})` 构造一步 TD 目标 `y_t`。  
5. 计算优势 `A_t = y_t - V(s_t)`，并在 Actor 损失中使用 `A_t.detach()` 保持梯度路径正确。  
6. 分别执行 Actor 与 Critic 反向传播：策略梯度更新策略，均方 TD 误差更新价值估计。  
7. 训练期记录每回合回报、成功标记、损失；再用 `pandas` 计算滑动均值监控收敛。  
8. `evaluate_policy` 用贪心动作做独立评估，并通过成功率与价值排序断言完成闭环验证。  

实现没有调用黑盒 RL 框架训练 API，Actor-Critic 的关键更新都在源码中显式展开。

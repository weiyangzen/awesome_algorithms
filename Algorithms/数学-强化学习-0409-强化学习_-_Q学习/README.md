# 强化学习 - Q学习

- UID: `MATH-0409`
- 学科: `数学`
- 分类: `强化学习`
- 源序号: `409`
- 目标目录: `Algorithms/数学-强化学习-0409-强化学习_-_Q学习`

## R01

Q 学习（Q-learning）是经典的离策略（off-policy）时序差分强化学习算法。
它直接学习动作价值函数 `Q(s, a)`，不需要环境模型，适合“状态离散、动作离散”的中小规模控制问题。

本条目 MVP 聚焦：
- 在一个可复现的离散 GridWorld 中实现 Q-learning；
- 使用 epsilon-greedy 完成探索与利用平衡；
- 输出训练曲线摘要、最终策略与随机策略对照结果。

## R02

MVP 任务定义为有限马尔可夫决策过程（MDP）：
- 状态 `s`：网格中的位置索引；
- 动作 `a`：`{上, 右, 下, 左}`；
- 奖励 `r`：到达目标给正奖励，陷阱给负奖励，其余步给小负奖励；
- 终止条件：到达目标、落入陷阱或步数上限。

目标：最大化折扣累计回报 `G_t = Σ_k gamma^k r_{t+k+1}`，对应到实现里即让策略在有限步内高成功率到达目标。

## R03

选择该设定的原因：
- Q-learning 的核心更新可以在离散网格上清晰观察；
- 相比直接调用 Gym，本实现包含环境动力学细节，便于源码级审计；
- 训练与评估耗时低，可在 `uv run python demo.py` 下快速完成验证。

## R04

核心公式：

1. Bellman 最优方程（动作价值形式）

`Q*(s,a) = E[r + gamma * max_a' Q*(s', a') | s, a]`

2. Q-learning 单步更新

`Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))`

其中：
- `alpha` 是学习率；
- `gamma` 是折扣因子；
- 对终止状态，将 `max_a' Q(s',a')` 视为 0。

## R05

算法流程（高层）：

1. 初始化 `Q` 表为全 0（形状 `n_states x n_actions`）。
2. 每个 episode 从起点 `reset`。
3. 按 epsilon-greedy 选动作：以 `epsilon` 概率随机探索，否则选 `argmax_a Q(s,a)`。
4. 执行动作得到 `(s', r, done)`。
5. 计算 TD 目标并更新 `Q(s,a)`。
6. 状态切换到 `s'`，直到 episode 终止。
7. 多轮迭代后导出贪心策略用于评估。

## R06

正确性与实现一致性要点：
- `q_learning` 严格按 TD 误差做原位更新，不做经验回放或目标网络，确保算法主体就是标准 Q-learning；
- `select_action_epsilon_greedy` 对并列最优动作随机打破平局，避免固定偏置；
- 终止状态不再 bootstrap（`best_next=0`），与理论定义一致；
- 训练与评估使用独立随机种子，减少偶然性污染。

## R07

复杂度分析（记状态数 `S`、动作数 `A`、episode 数 `E`、每集最大步数 `T`）：
- 时间复杂度：`O(E * T)`，每步更新和取 `max_a Q(s',a)` 代价为 `O(A)`，此处 `A=4` 为常数；
- 空间复杂度：`O(S * A)`，主要由 Q 表占用。

在本 MVP 中网格规模固定，实际运行时间主要由 episode 数线性决定。

## R08

边界与异常处理：
- 环境配置检查：网格尺寸、起终点、奖励、滑移概率、最大步数都做有效性约束；
- 越界移动会被裁剪回边界，撞墙保持原地；
- 训练结束后检查 `Q` 是否全为有限值（非 NaN/Inf）；
- 所有流程无交互输入，避免阻塞式失败。

## R09

MVP 取舍说明：
- 采用表格法 Q-learning，不引入函数逼近；
- 使用手写 GridWorld，不依赖外部 RL 环境包；
- 目标是“最小可运行 + 可解释”，不追求大规模并行训练。

未覆盖内容：
- 连续状态/动作空间；
- 深度 Q 网络（DQN）及其稳定化技巧；
- 多智能体、层次强化学习等扩展。

## R10

`demo.py` 主要模块职责：
- `GridWorldConfig`：环境超参数定义；
- `GridWorld`：状态转移、奖励与终止逻辑；
- `epsilon_by_episode`：探索率调度；
- `select_action_epsilon_greedy`：行为策略；
- `q_learning`：主训练循环；
- `greedy_policy_from_q`：从 Q 表导出策略；
- `evaluate_policy` / `evaluate_random_policy`：策略评估与随机基线；
- `render_policy_map`：策略可视化；
- `main`：组织实验并输出验收指标。

## R11

运行方式：

```bash
cd Algorithms/数学-强化学习-0409-强化学习_-_Q学习
uv run python demo.py
```

脚本将自动训练、评估并打印策略地图和检查项。

## R12

输出字段说明：
- `[train] episode=...`：训练进度日志；
- `avg_return(last_100)`：最近 100 轮平均回报，反映学习趋势；
- `[eval-greedy]`：训练后贪心策略评估结果；
- `[eval-random]`：随机策略基线结果；
- `success_rate`：到达目标比例；
- `avg_steps`：终止所需平均步数；
- `improvement_vs_random`：相对随机策略平均回报提升值。

## R13

最小测试与验收项（脚本内已打印）：
- `q_table_finite=True`：Q 表数值稳定；
- `history_length_ok=True`：训练记录长度与设定 episode 一致；
- 贪心策略相较随机策略在 `avg_return` 与 `success_rate` 上应明显更优；
- 训练日志后期 `avg_return(last_100)` 通常高于前期。

## R14

关键参数与调参建议：
- `alpha`（学习率）：过大易震荡，过小收敛慢；
- `gamma`（折扣）：越接近 1 越重视长期回报；
- `eps_start/eps_end/eps_decay_ratio`：决定探索衰减节奏；
- `episodes`：训练预算；
- `slip_prob`：环境随机性，越高学习难度越大。

建议先固定环境，按“先调 `episodes`，再调 `alpha`/`epsilon`”顺序迭代。

## R15

与相关方法简对比：
- Monte Carlo：需完整回合回报，方差较大；
- SARSA：在更新中使用当前行为策略下一动作，偏保守；
- Q-learning：更新目标用 `max_a' Q(s',a')`，离策略、通常更激进；
- DQN：把 Q 表替换成神经网络，适合大状态空间但工程复杂度更高。

## R16

典型应用场景：
- 网格路径与离散导航原型；
- 简化资源分配与调度问题；
- 作为教学与实验基线，用于验证奖励设计和探索策略。

## R17

可扩展方向：
- 加入 Double Q-learning 缓解过估计；
- 引入资格迹（Eligibility Trace）形成 Q(lambda)；
- 将表格状态编码升级为特征向量，过渡到函数逼近；
- 扩展为 DQN 并增加目标网络与经验回放。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `main` 构建 `GridWorldConfig`，初始化训练环境、评估环境与随机种子。
2. `q_learning` 创建 `Q` 表并进入 episode 循环，每轮调用 `env.reset()` 从起点出发。
3. 在每个时间步用 `epsilon_by_episode` 计算当前 `epsilon`，再通过 `select_action_epsilon_greedy` 选择动作。
4. 调用 `env.step(action)` 获得 `next_state, reward, done`，环境内部处理滑移、边界、墙体和终止逻辑。
5. 用 `td_target = reward + gamma * max(Q[next_state])`（终止时去掉 bootstrap）构造目标。
6. 用 `Q[state, action] += alpha * (td_target - Q[state, action])` 完成 TD 更新，并推进状态。
7. 训练结束后由 `greedy_policy_from_q` 生成贪心策略，再用 `evaluate_policy` 与 `evaluate_random_policy` 分别评估学习策略和随机基线。
8. `main` 打印训练摘要、策略地图、性能提升与稳定性检查项，形成可直接验收的最小闭环。

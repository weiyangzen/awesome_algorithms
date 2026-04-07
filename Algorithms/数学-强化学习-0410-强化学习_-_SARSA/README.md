# 强化学习 - SARSA

- UID: `MATH-0410`
- 学科: `数学`
- 分类: `强化学习`
- 源序号: `410`
- 目标目录: `Algorithms/数学-强化学习-0410-强化学习_-_SARSA`

## R01

SARSA（State-Action-Reward-State-Action）是经典的 on-policy 时序差分控制算法：

- 它直接学习动作价值函数 `Q(s,a)`；
- 更新目标使用“下一步实际执行的动作” `a'`，而不是仅用最大价值动作；
- 因此学习到的是当前行为策略（如 `epsilon`-greedy）下的价值。

本目录的 MVP 用表格型 SARSA 在自定义 `CliffWalking` 环境中训练一个离散策略，并输出训练前后评估、学习曲线尾部和最终贪心策略图。

## R02

SARSA 主要解决的问题：

给定一个离散 MDP，希望得到使长期回报尽可能大的策略，同时在学习阶段保留探索行为。

核心更新式：

`Q(s_t,a_t) <- Q(s_t,a_t) + alpha * [r_t + gamma * Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t)]`

其中 `a_{t+1}` 是按当前行为策略实际选出的动作（本实现是 `epsilon`-greedy）。

## R03

为什么本题选 SARSA + CliffWalking：

- CliffWalking 是强化学习教材中的标准例子，能直观看出 on-policy 与 off-policy 差异；
- “掉崖重罚（`-100`）+ 每步代价（`-1`）”结构能突出风险敏感策略学习；
- 状态和动作离散，适合用最小表格实现，不需要深度网络。

## R04

on-policy 直觉：

- 如果训练时行为策略含探索（`epsilon > 0`），那么“下一步动作 `a'` 也可能是探索动作”；
- SARSA 用该真实 `a'` 更新，会把探索风险纳入价值估计；
- 在 CliffWalking 中，这常导致策略更保守，倾向离悬崖更远的路径。

这也是 SARSA 在教学场景里常被用来和 Q-learning 对照的原因。

## R05

本实现的高层伪代码：

```text
初始化 Q 表为 0
for episode in 1..N:
  epsilon 按衰减计划更新
  s = reset()
  a = epsilon-greedy(Q[s])
  while not done:
    执行动作 a，得到 s', r, done
    if done:
      td_target = r
    else:
      a' = epsilon-greedy(Q[s'])
      td_target = r + gamma * Q[s', a']
    Q[s,a] += alpha * (td_target - Q[s,a])
    s, a = s', a'
训练后用 epsilon=0 做贪心评估
```

## R06

复杂度分析（表格型场景）：

- 单步更新复杂度：`O(1)`；
- 单回合复杂度：`O(T)`，`T` 为该回合步数；
- 总训练复杂度：`O(N*T)`，`N` 为回合数；
- 空间复杂度：`O(|S|*|A|)`，即 Q 表大小。

在默认 `4x12` 网格下，`|S|=48`、`|A|=4`，计算成本很低。

## R07

`demo.py` 使用的最小技术栈：

- `numpy`：Q 表、随机数、数值运算；
- `pandas`：训练日志表与滑动均值；
- `scipy.stats`：评估平均回报的 95% 置信区间。

未使用黑盒强化学习框架，训练循环与更新公式均在源码中显式实现。

## R08

默认实验配置（`SarsaConfig`）：

- 环境：`rows=4`, `cols=12`, `max_steps=200`
- 学习率与折扣：`alpha=0.5`, `gamma=1.0`
- 探索计划：`epsilon_start=0.15`, `epsilon_end=0.01`, `epsilon_decay=0.995`
- 训练回合：`train_episodes=700`
- 评估回合：`eval_episodes=250`
- 随机种子：`seed=410`

这些参数偏向“教学可收敛”，在 CPU 下可快速完成。

## R09

环境定义摘要：

- 起点：左下角 `S=(3,0)`；
- 终点：右下角 `G=(3,11)`；
- 悬崖区：底行中间 `[(3,1)...(3,10)]`；
- 动作：`0=up, 1=right, 2=down, 3=left`。

奖励规则：

- 普通移动：`-1`；
- 掉入悬崖：`-100`，并重置到起点（回合继续）；
- 到达终点：`-1` 并终止。

## R10

`demo.py` 函数职责：

- `CliffWalkingEnv`：状态编码、边界移动、奖励与终止逻辑；
- `epsilon_by_episode`：按回合计算探索率；
- `epsilon_greedy_action`：带随机打破平局的 `epsilon`-greedy 选动作；
- `train_sarsa`：执行 SARSA 主循环并记录日志；
- `evaluate_policy`：评估平均回报、成功率、平均掉崖次数、置信区间；
- `render_greedy_policy`：把贪心策略渲染成网格箭头图；
- `main`：串联训练前评估、训练、训练后评估和断言校验。

## R11

运行方式（无交互输入）：

```bash
uv run python Algorithms/数学-强化学习-0410-强化学习_-_SARSA/demo.py
```

输出包含：

- 训练前 baseline 评估；
- 训练后贪心策略评估与带探索评估；
- 贪心策略网格图；
- 训练日志尾部表格；
- 关键断言通过信息。

## R12

关键输出指标说明：

- `avg_return`：评估回合平均总回报；
- `success_rate`：到达终点比例；
- `avg_cliff_hits`：每回合平均掉崖次数；
- `return_ci95`：平均回报 95% 置信区间；
- `return_ma30`：训练回报 30 回合滑动均值；
- `goal_rate_ma30`：最近 30 回合到达终点率。

一般期望训练后 `success_rate` 上升，且 `avg_return` 明显优于训练前 baseline。

## R13

正确性快速检查（脚本内含断言）：

- Q 表与训练日志必须是有限值（无 `NaN/Inf`）；
- 训练后贪心策略成功率应至少 `0.90`；
- 训练后平均回报应显著高于训练前（阈值差 `> 15`）。

这些检查保证“不仅能运行，而且学到了有效策略”。

## R14

常见失败模式与排查建议：

- `alpha` 过大：Q 值震荡，回报波动大；
- `epsilon` 过高且衰减慢：长期探索导致收敛慢；
- `epsilon` 过低：前期探索不足，可能陷入次优路径；
- `max_steps` 太小：尚未到达终点就超时终止；
- 随机种子不同：曲线细节会变，但整体趋势应一致。

## R15

与 Q-learning 对比：

- Q-learning（off-policy）目标是 `r + gamma * max_a Q(s',a)`；
- SARSA（on-policy）目标是 `r + gamma * Q(s',a')`，`a'` 来自当前行为策略。

在 CliffWalking 这类高风险环境里：

- Q-learning 往往学到“贴崖捷径”；
- SARSA 常学到“更保守但更稳”的路径。

## R16

适用场景：

- 离散状态/动作空间的教学与基线实验；
- 需要把探索行为本身纳入价值估计的任务；
- 从 TD 控制过渡到更复杂 RL 算法（Expected SARSA、Actor-Critic）的入门。

局限：

- 表格法难扩展到高维连续状态；
- 对探索率计划较敏感；
- 样本效率与泛化能力不如现代深度 RL 方法。

## R17

可扩展方向：

- 把表格 Q 扩展为线性函数逼近或神经网络（形成深度 SARSA）；
- 对比 Expected SARSA，降低更新目标的采样噪声；
- 引入更系统的调参与可视化（学习曲线、访问热力图）；
- 加入并行环境采样以提高统计稳定性。

## R18

`demo.py` 源码级算法流（9 步）：

1. `main()` 构建 `SarsaConfig`，先用全零 Q 表做一次 baseline 评估（`epsilon=0.10`）。
2. `train_sarsa()` 初始化 `CliffWalkingEnv` 与 `q_table[|S|,|A|]`，逐回合训练。
3. 每回合先调用 `epsilon_by_episode()` 得到探索率，再在起点用 `epsilon_greedy_action()` 选初始动作。
4. 与环境交互 `env.step(action)`，显式拿到 `next_state, reward, done` 与 `fell_cliff` 标记。
5. 若终止，执行 `td_target = reward`；否则先按当前策略在 `next_state` 选 `next_action`。
6. 用 SARSA 目标 `reward + gamma * Q[next_state, next_action]` 计算 TD 误差并原位更新 `Q[state, action]`。
7. 记录 `return/steps/cliff_hits/goal_reached` 到 `pandas` 日志，并计算 `MA30` 指标。
8. 训练后 `evaluate_policy()` 在 `epsilon=0.0` 与 `epsilon=0.05` 下评估，`scipy.stats.t.interval` 计算回报置信区间。
9. 输出贪心策略网格图与示例轨迹，最后通过断言完成“数值有效 + 策略改进”的闭环校验。

以上流程完整展开了 SARSA 的状态-动作级更新链路，没有依赖黑盒 RL 训练 API。

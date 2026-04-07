# 令牌环算法

- UID: `CS-0210`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `363`
- 目标目录: `Algorithms/计算机-计算机网络-0363-令牌环算法`

## R01

令牌环算法（Token Ring）的核心思想是“控制权随令牌循环传递”：
- 环上任意时刻只有 1 个令牌；
- 只有持有令牌的节点允许发送数据；
- 发送完毕（或达到配额）后必须把令牌交给下一个节点。

因此它天然避免多节点竞争同一介质时的冲突。

## R02

问题定义（本 MVP）：
- 给定 `N` 个节点构成的逻辑环，每个节点有待发送帧队列；
- 每个离散时刻 `tick`，当前令牌持有者最多发送 `token_quota_frames` 帧；
- 发送后令牌按固定方向传给下一节点；
- 可选地在某些 `tick` 注入新到达业务（`arrival_plan`）。

## R03

输入与输出：
- 输入：
1. `node_count`、`initial_queues`；
2. `arrival_plan[tick] -> 每节点新增帧数`；
3. `token_quota_frames`、`start_holder`、`max_ticks`。
- 输出：
1. 每个 tick 的详细记录（谁持令牌、发了多少、队列变化）；
2. 每节点总发送量 `sent_per_node`；
3. 每节点令牌访问次数 `visits_per_node`；
4. 最终剩余积压 `backlog_final`。

## R04

目标与约束：
- 目标：在无冲突前提下完成队列排空并观察公平性；
- 约束：
1. 任一 tick 只有令牌持有者可发送；
2. 单次持有发送量不超过 `token_quota_frames`；
3. 令牌严格按 `(i+1) mod N` 顺序传递；
4. 队列长度始终非负。

## R05

关键状态变量：
- `queues[i]`：节点 `i` 当前待发送帧数；
- `token_holder`：当前持令牌节点下标；
- `sent_per_node[i]`：节点 `i` 累计发送帧数；
- `visits_per_node[i]`：节点 `i` 被令牌访问次数；
- `records`：逐 tick 日志，用于校验与解释算法行为。

## R06

离散迭代流程（每个 tick）：
1. 按 `arrival_plan` 将新业务加入各节点队列；
2. 记录当前 `token_holder` 与 `queue_before`；
3. 计算可发送量 `send_now = min(token_quota_frames, queues[token_holder])`；
4. 从持有者队列扣减并累加到 `sent_per_node`；
5. 记录 `queue_after` 与总积压 `total_backlog`；
6. 令牌移动到下一节点；
7. 若“无后续到达且总积压为 0”则提前结束。

## R07

正确性直觉：
- 互斥发送：令牌是唯一发送许可，保证同一时刻不会有两个节点并发发送，冲突为 0；
- 有界服务：有积压的节点在被访问时至少可发送 1 帧（配额 > 0），不会被无限跳过；
- 完整清空：若输入总业务量有限、仿真时长足够，循环访问会持续减少总积压直到清零。

## R08

复杂度分析（`T` 为实际运行 tick 数，`N` 为节点数）：
- 时间复杂度：`O(T * N)`（每 tick 需要处理长度为 `N` 的队列/到达向量）；
- 空间复杂度：`O(T * N)`（若保存完整日志）；
- 在本 MVP 中 `N` 很小，主要开销来自可解释性日志而非计算本身。

## R09

参数影响：
- `token_quota_frames` 越大：单节点突发能力更强，但短时公平性可能下降；
- `node_count` 越大：令牌一圈耗时更长，单节点等待时间变大；
- `arrival_plan` 越突发：更容易形成局部队列堆积；
- `start_holder` 只影响起始阶段时序，不改变长期守恒结果。

## R10

MVP 设计取舍：
- 使用 Python + `numpy`（最小工具栈）实现，避免黑箱网络仿真框架；
- 使用“帧数”而非比特级时延建模，保持实现精简；
- 业务到达采用确定性 `arrival_plan`，便于复现实验与断言；
- 不扩展优先级令牌、丢令牌恢复、物理层误码，聚焦核心轮转机制。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0363-令牌环算法
uv run python demo.py
```

脚本无需交互输入，会打印摘要、前 15 个 tick 明细和断言结果。

## R12

输出指标说明：
- `demand_per_node`：每节点总需求（初始队列 + 到达总和）；
- `sent_per_node`：每节点实际发送总量；
- `visits_per_node`：每节点被令牌访问次数；
- `backlog_final`：结束时环上总积压；
- `jain_fairness(sent_per_node)`：基于发送量的 Jain 公平性指标（越接近 1 越均衡）。

## R13

结果解读：
- 在 `backlog_final=0` 且 `sent_per_node == demand_per_node` 时，说明业务被完整排空；
- `visits_per_node` 的最大最小差不超过 1，说明轮转调度近似均匀；
- 若某节点持续积压，可通过提高 `token_quota_frames` 或调整业务分布改善排队时延。

## R14

边界与异常处理：
- `node_count <= 1`、`token_quota_frames <= 0`、`max_ticks <= 0` 会抛 `ValueError`；
- `initial_queues` 长度不匹配或含负数会抛 `ValueError`；
- `arrival_plan` 中 tick 为负、向量长度错误、含负值会抛 `ValueError`；
- 若 `max_ticks` 太小导致未清空，断言会报告 `backlog_final != 0`。

## R15

与常见介质访问方式对比：
- 与 CSMA/CD：令牌环通过显式授权避免冲突；CSMA/CD 先竞争后冲突检测；
- 与纯轮询：令牌在链路层分布式传递，不依赖中心控制节点；
- 与随机接入：令牌环在高负载下时延更可预测，但空闲时可能有无效轮转开销。

## R16

可扩展方向：
- 增加“令牌丢失检测 + 令牌再生”超时机制；
- 支持优先级令牌或加权配额（不同节点不同 `quota`）；
- 引入链路传播时延和帧长度，改为更细粒度时间模型；
- 加入随机业务到达分布，对平均等待时间做统计实验。

## R17

最小验证清单：
- `README.md` 与 `demo.py` 不包含任何模板占位符；
- `uv run python demo.py` 可直接运行；
- 输出存在 `All checks passed.`；
- 断言覆盖至少以下性质：
1. 队列最终清空；
2. 发送守恒（发送量等于需求量）；
3. 令牌访问公平性（访问次数差距受限）；
4. 持令牌且有积压时不会“空转不发”。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `run_demo()` 配置环大小、初始队列、确定性 `arrival_plan`、令牌配额并创建 `TokenRingSimulator`。  
2. `TokenRingSimulator.__init__` 校验输入合法性，并把到达计划转换为定长 `numpy` 向量。  
3. `run()` 在每个 `tick` 先应用到达流量，更新 `queues`。  
4. 读取当前 `token_holder`，计算 `send_now = min(token_quota_frames, queues[token_holder])`。  
5. 扣减持有者队列并累加 `sent_per_node`，同时记录 `TickRecord`（含队列前后状态与总积压）。  
6. 将令牌移动到下一个节点 `(token_holder + 1) % node_count`，形成严格轮转。  
7. 当“总积压为 0 且不会再有新到达”时提前结束，返回 `SimulationResult`。  
8. `run_demo()` 对结果执行守恒、公平、无空转等断言，再输出摘要与前 15 个 tick 明细。  

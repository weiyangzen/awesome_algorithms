# CSMA/CD

- UID: `CS-0208`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `361`
- 目标目录: `Algorithms/计算机-计算机网络-0361-CSMA／CD`

## R01

CSMA/CD（Carrier Sense Multiple Access with Collision Detection）是经典以太网共享总线上的介质访问控制机制。核心思想：
- `CS`：发送前先监听信道，空闲才允许发送；
- `MA`：多个站点共享同一物理介质；
- `CD`：发送过程中持续检测冲突，一旦检测到冲突立即中止并执行退避重传。

它的目标是在“无中心调度器”的前提下，让多个节点在共享信道上高效且可扩展地发送数据帧。

## R02

本条目 MVP 的问题定义：
- 多个站点共享单一总线；
- 每站有待发送帧队列，且可在指定时隙到达新帧；
- 信道按离散时隙推进；
- 空闲信道下站点按 1-persistent 策略尝试发送；
- 同一时隙出现多站并发尝试即发生冲突，执行 jam + 二进制指数退避（BEB）。

输出需要给出每站发送结果、冲突统计和吞吐指标，以验证算法机制而非物理层细节。

## R03

建模假设（用于最小可运行实现）：
1. 时间离散化为“争用时隙”（可理解为标准槽时间的归一化抽象）；
2. 成功发送占用 `frame_slots` 个时隙；
3. 冲突后 jam 信号占用 `jam_slots` 个时隙；
4. 退避计数只在信道空闲时递减；
5. 每次成功发送仅处理队首 1 帧。

该抽象保留了 CSMA/CD 的关键控制逻辑：监听、冲突检测、快速中止、指数退避。

## R04

输入与输出：
- 输入：
1. `station_count`、`initial_queues`；
2. `arrival_plan[slot] -> 各站新增帧数`；
3. `frame_slots`、`jam_slots`、`max_retries`、`max_slots`、`seed`。
- 输出：
1. 每时隙日志（忙闲状态、尝试站点、是否冲突、成功站点、丢帧站点、队列快照）；
2. 每站统计：`attempts/success/collisions/drops`；
3. 全局统计：`collision_probability`、`throughput(success/slot)`、`backlog_final`、公平性指标。

## R05

关键状态变量：
- `queues[i]`：站点 `i` 当前积压帧数；
- `backoff_counter[i]`：站点 `i` 当前退避倒计时（`-1` 表示未激活）；
- `collision_level[i]`：站点 `i` 当前队首帧累计冲突次数；
- `medium_busy_remaining`：信道忙状态剩余时隙；
- `attempts/success/collisions/drops`：每站计数器。

这些变量足以完整表达 CSMA/CD MVP 的时序演化。

## R06

二进制指数退避（BEB）规则：
- 某站队首帧第 `k` 次碰撞后，随机退避槽数

`r ~ Uniform({0, 1, ..., 2^m - 1})`, 其中 `m = min(k, 10)`。

- 若 `k > max_retries`（本实现默认 16），则丢弃该帧并重置冲突级别。

直观上，碰撞越多，退避窗口越大，从而降低短期再次碰撞概率。

## R07

离散时隙主流程（每个 slot）：
1. 注入 `arrival_plan` 的新到达；
2. 为“有积压且未激活”的站点初始化发送资格（首次帧立即可尝试；冲突后帧按 BEB 随机倒计时）；
3. 若当前信道忙，则仅消耗忙时隙；
4. 若信道空闲，则递减 `backoff_counter>0` 的站点；
5. 统计 `backoff_counter==0` 的尝试集合：
- 0 个：保持空闲；
- 1 个：发送成功；
- 多个：发生冲突，执行 jam 与 BEB。
6. 更新队列与统计，记录时隙日志。

## R08

正确性不变量（demo 也会做断言）：
- 非负性：任意时刻 `queues[i] >= 0`；
- 守恒关系：`total_success + total_drops == total_demand`（仿真结束且无积压时）；
- 计数一致性：`collisions_per_station[i] <= attempts_per_station[i]`；
- 互斥性：每时隙只能是“无尝试 / 单站成功 / 多站冲突”之一。

## R09

复杂度分析（`S` 为实际运行时隙数，`N` 为站点数）：
- 时间复杂度：`O(S * N)`，每个时隙需要遍历站点更新倒计时与状态；
- 空间复杂度：`O(S + N)`，状态数组是 `O(N)`，时隙日志是 `O(S)`。

这与多数离散事件网络仿真的复杂度级别一致。

## R10

MVP 技术栈与实现取舍：
- `numpy`：状态向量与统计计算；
- `pandas`：前若干时隙日志表格化输出；
- 纯 Python 控制流显式实现 CSMA/CD，不调用黑盒网络仿真器。

取舍上优先“机制可审计”而非“协议全细节覆盖”。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0361-CSMA／CD
uv run python demo.py
```

脚本无交互输入，会打印摘要统计、前 25 个时隙明细，并在断言通过后输出 `All checks passed.`。

## R12

输出指标说明：
- `attempts_per_station`：每站尝试发送次数；
- `success_per_station`：每站成功发送帧数；
- `collisions_per_station`：每站发生冲突次数；
- `drops_per_station`：每站因超过重传上限而丢弃帧数；
- `collision_probability = total_collisions / total_attempts`；
- `throughput = total_success / total_slots`；
- `jain_fairness(success_per_station)`：发送公平性。

## R13

参数影响直觉：
- `frame_slots` 越大，单次成功占用信道更久，吞吐上限受长帧占用影响；
- `jam_slots` 越大，碰撞代价越高；
- `max_retries` 越小，系统更倾向快速丢帧而不是持续重传；
- 业务突发越强，同步尝试概率越高，冲突率通常上升。

## R14

边界与异常处理：
- `station_count <= 1`、`max_slots <= 0`、`frame_slots < 1`、`jam_slots < 1` 会抛 `ValueError`；
- `initial_queues` 长度不匹配或含负数会抛 `ValueError`；
- `arrival_plan` 的 slot 为负、向量长度错误或出现负值也会抛 `ValueError`。

这些检查防止无效输入导致“看似可运行但语义错误”的仿真结果。

## R15

与相关方法对比：
- 对比 CSMA/CA：CSMA/CD依赖“发送时检测冲突并中止”，更契合有线共享介质；
- 对比令牌环：CSMA/CD 无中心令牌，部署简洁但时延确定性较弱；
- 对比纯 ALOHA：CSMA/CD 通过载波监听与冲突后快速退避显著降低无效重传。

## R16

局限与扩展方向：
- 当前模型未显式模拟拓扑距离与传播时延差异；
- 未建模帧长分布、突发误码、优先级队列；
- 可扩展为事件驱动微秒级仿真，并引入 heterogeneous 站点速率与突发流量模型；
- 可进一步对比不同退避参数下的吞吐-公平性折中曲线。

## R17

最小验收清单：
- `README.md` 与 `demo.py` 无模板占位符；
- `uv run python demo.py` 可直接运行；
- 输出包含 `All checks passed.`；
- 断言至少覆盖：
1. 最终无积压；
2. 成功+丢弃与总需求守恒；
3. 场景中观察到冲突；
4. 本示例无丢帧（用于验证重传路径在给定参数下可收敛）。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `run_demo()` 设定站点数、初始队列、到达计划和仿真参数，实例化 `CSMACDSimulator`。  
2. `CSMACDSimulator.__init__` 校验输入并把 `arrival_plan` 归一化为 `numpy` 向量映射。  
3. `run()` 初始化 `queues/backoff_counter/collision_level` 与统计数组，进入逐时隙循环。  
4. 每个时隙先注入新到达，再为“有积压且未激活”的站点设置发送资格（首次 0 退避，冲突后按 BEB 随机退避）。  
5. 若信道忙，仅扣减 `medium_busy_remaining`；若信道空闲，则让 `backoff_counter > 0` 的站点倒计时。  
6. 汇总 `backoff_counter == 0` 的尝试站点：单站则成功发送并占用 `frame_slots`；多站则碰撞并占用 `jam_slots`。  
7. 对碰撞站点更新 `collision_level`，超过 `max_retries` 则丢弃队首帧，否则按 `2^m` 窗口重抽退避槽。  
8. 记录 `SlotRecord` 日志；`run_demo()` 对守恒与边界性质断言后，调用 `print_report()` 输出统计与前 25 时隙明细。  

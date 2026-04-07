# 进程调度 - CFS

- UID: `CS-0166`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `311`
- 目标目录: `Algorithms/计算机-操作系统-0311-进程调度_-_CFS`

## R01

问题定义：给定一组进程（`pid/arrival/burst/nice`），在单核 CPU 上模拟 CFS（Completely Fair Scheduler）调度，输出调度时间线与每个进程的完成、周转、等待、响应等指标。

## R02

输入：
- 进程列表：`[(pid, arrival, burst, nice), ...]`
- 调度参数：
  - `target_latency`：期望一个调度周期内“所有可运行任务都能运行一次”
  - `min_granularity`：最小时间片，避免切换过于频繁

输出：
- 时间线：`(start, end, pid, nice, vruntime_before)`
- 每进程指标：
  - `completion`
  - `turnaround = completion - arrival`
  - `waiting = turnaround - burst`
  - `response = first_start - arrival`
  - `cpu_share` 与 `weight_share`（公平性观测）
- 平均指标：平均周转、平均等待、平均响应

## R03

CFS 核心思想：
- 不直接按固定优先级队列轮转，而是维护“虚拟运行时间” `vruntime`。
- 调度器总是选择 `vruntime` 最小的任务运行，尽量让各任务的 `vruntime` 增速接近。
- 任务权重来自 `nice` 值，`nice` 越小（优先级越高）权重越大。
- 权重越大，单位真实时间对应的 `vruntime` 增量越小，因此该任务会更频繁地被再次选中。

## R04

关键公式：
- `weight = prio_to_weight[nice]`（本实现使用 Linux 常见 40 档映射）
- `delta_vruntime = delta_exec * NICE_0_LOAD / weight`
- `vruntime += delta_vruntime`
- `sched_period = target_latency`（当运行任务数较少）
- `sched_period = nr_running * min_granularity`（当任务数过多）
- `ideal_slice = max(min_granularity, sched_period * weight / total_weight)`

## R05

核心数据结构：
- `Process`：保存 `arrival/burst/nice/weight/remaining/vruntime` 等状态
- 最小堆 `heapq`：键为 `(vruntime, tie_seq, process)`，确保总是取到当前最“欠运行”的任务
- `timeline`：记录每段执行区间
- `next_idx`：按到达时间扫描未入队任务，降低到达处理开销

## R06

高层流程：
1. 校验输入并构建 `Process` 列表。
2. 按 `(arrival, pid)` 排序，保证模拟可复现。
3. 把 `arrival <= time` 的任务入堆。
4. 若堆为空，CPU 进入 `IDLE`，时间跳到下一个到达点。
5. 弹出最小 `vruntime` 任务作为当前运行任务。
6. 按当前 `nr_running/total_weight` 计算 `ideal_slice`。
7. 本段运行时长取 `min(remaining, ideal_slice, 到下一到达的间隔)`。
8. 运行结束后更新 `remaining/executed/vruntime`；未完成则重新入堆，完成则记录 `completion`。
9. 所有任务完成后计算指标与平均值。

## R07

正确性要点：
- 最小堆保证“每次选择最小 `vruntime`”这一 CFS 选择规则。
- `vruntime` 更新包含权重归一化，体现了 nice 差异。
- 当新任务到达时可截断当前运行段，避免长时间忽略新可运行任务。
- 指标全部由时间线和任务原始参数推导，避免二次估算误差。

## R08

复杂度分析：
- 设任务数 `n`，时间线段数 `S`。
- 排序：`O(n log n)`。
- 每个切段涉及 1 次弹堆 + 最多 1 次入堆：`O(log n)`。
- 主循环总成本约 `O(S log n)`。
- 空间复杂度：`O(n + S)`。

## R09

边界与异常处理：
- `target_latency <= 0` 或 `min_granularity <= 0`：拒绝。
- `arrival < 0`、`burst <= 0`、`nice` 超范围、`pid` 重复：拒绝。
- 所有任务都在未来到达：时间线会出现 `IDLE` 片段。
- 浮点误差：对接近 0 的 `remaining` 使用阈值收敛到 0。

## R10

MVP 范围：
- 单核、纯 CPU burst、无 I/O 阻塞。
- 不模拟上下文切换开销与多核迁移。
- 不依赖调度黑盒库，使用 Python 标准库直接实现 CFS 关键机制。

## R11

`demo.py` 内置样例：
- 参数：`target_latency=12.0`, `min_granularity=1.0`
- 任务：
  - `P1(0, 18, nice=0)`
  - `P2(0, 8, nice=5)`
  - `P3(2, 6, nice=-5)`
  - `P4(4, 12, nice=10)`
  - `P5(6, 4, nice=-10)`

## R12

预期行为：
- `nice` 更小（权重更大）的任务，其 `vruntime` 增长更慢，长期会拿到更多 CPU 份额。
- 新到达任务会较快进入竞争并切分后续时间线。
- 低权重长任务仍会前进，但速度相对慢，体现“公平而非平均”。

## R13

指标解释：
- `response`：首次获得 CPU 的等待时长，反映交互体验。
- `waiting`：总排队时间，反映就绪队列拥塞程度。
- `turnaround`：从到达到完成的端到端时延。
- `cpu_share` 与 `weight_share` 对比可用于观察实现是否接近“按权重公平”。

## R14

与其他调度策略对比：
- 相比 FCFS：CFS 不会让早到长任务长时间垄断 CPU。
- 相比 RR：CFS 不是固定量子平均分配，而是按权重动态分配。
- 相比静态优先级：CFS 用 `vruntime` 连续平衡，减少极端饥饿风险。

## R15

常见实现错误：
- 忘记按权重更新 `vruntime`，退化为近似 RR。
- 仅按 nice 选任务而不看 `vruntime`，导致公平性失真。
- 新任务到达后不及时入队，响应时间异常偏大。
- `ideal_slice` 未设置最小粒度，导致过密切换。
- 统计指标时把 `waiting` 与 `response` 混淆。

## R16

最小测试清单：
- 同时到达 + 不同 nice 的公平性测试。
- 交错到达（运行中有新任务加入）。
- 仅一个任务（应连续执行至完成）。
- 存在空闲区间（应输出 `IDLE`）。
- 非法输入校验（负 burst、重复 pid、nice 越界等）。

## R17

可扩展方向：
- 加入睡眠/唤醒与 I/O 阻塞，模拟交互应用。
- 加入上下文切换开销，评估吞吐与延迟折中。
- 扩展多核 runqueue 与任务迁移策略。
- 增加 tail latency（P95/P99 响应时间）统计。
- 对比不同 `target_latency/min_granularity` 参数的灵敏度。

## R18

源码级算法拆解（对应 `demo.py`，非黑盒）：
1. `nice_to_weight` 把 `nice` 映射到权重，形成 CFS 的公平基准。
2. `_validate_inputs` 校验参数合法性，阻止非法状态进入主循环。
3. `cfs_schedule` 构建 `Process` 对象并排序，初始化时间、堆和到达指针。
4. `enqueue_arrivals` 将 `arrival <= 当前时间` 的任务放入最小堆，键为 `vruntime`。
5. 若堆为空则记录 `IDLE` 并把时间跳到下一到达时刻。
6. 每轮弹出最小 `vruntime` 任务，按 `nr_running` 与 `total_weight` 计算 `sched_period` 和 `ideal_slice`。
7. 运行一段 `run_for`（同时受 `remaining`、`ideal_slice`、下一到达事件约束），并用 `delta_exec * 1024 / weight` 更新 `vruntime`。
8. 未完成任务重新入堆，完成任务记录 `completion`；循环结束后计算 `turnaround/waiting/response/cpu_share` 并打印结果。

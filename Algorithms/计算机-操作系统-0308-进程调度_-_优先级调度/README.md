# 进程调度 - 优先级调度

- UID: `CS-0163`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `308`
- 目标目录: `Algorithms/计算机-操作系统-0308-进程调度_-_优先级调度`

## R01

本条目实现的是**非抢占式优先级调度（Priority Scheduling）**：
- 每个进程有 `arrival(到达时间)`、`burst(运行时长)`、`priority(优先级)`。
- 在任一调度点，从“已到达且未完成”的进程中选择优先级最高者执行。
- 本实现约定：`priority` 数值越小，优先级越高。

## R02

适用场景：
- 操作系统中需要按任务重要性区别调度顺序。
- 后台批处理、控制任务、实时性要求分层的作业集。

局限：
- 低优先级任务可能长期得不到执行（饥饿问题）。
- 非抢占版本对突发高优先级任务响应慢于抢占版本。

## R03

输入（`demo.py` 中 `Process`）：
- `pid: str` 进程标识
- `arrival: int` 到达时间（>=0）
- `burst: int` 运行时长（>0）
- `priority: int` 优先级（值越小优先级越高）

输出（`ScheduleRecord` 列表）：
- `start / finish` 开始与结束时刻
- `waiting` 等待时间
- `turnaround` 周转时间
- `response` 响应时间（非抢占下与等待时间相同）

## R04

核心数据结构：
- `Process`：输入任务描述（不可变 dataclass）。
- `ScheduleRecord`：调度结果记录（不可变 dataclass）。
- `pending`：按到达时间排序的待进入就绪队列列表。
- `ready`：当前可调度进程集合。
- `done`：最终执行顺序与统计结果。

## R05

算法思路（非抢占）：
1. 维护全局时间 `time`。
2. 把 `arrival <= time` 的进程加入 `ready`。
3. 若 `ready` 为空，CPU 空闲并把 `time` 跳到下一个进程到达时刻。
4. 若 `ready` 非空，按 `(priority, arrival, pid)` 最小值选中进程执行到完成。
5. 记录各统计量，更新时间，重复直到所有进程完成。

## R06

伪代码：

```text
sort pending by (arrival, pid)
time = 0
while done_count < n:
    move all arrived processes into ready
    if ready is empty:
        time = next pending arrival
        continue

    job = argmin(ready, key=(priority, arrival, pid))
    remove job from ready

    start = time
    finish = start + burst
    waiting = start - arrival
    turnaround = finish - arrival
    response = waiting

    save record
    time = finish
```

## R07

正确性直觉：
- 在每个调度点，只在“已到达”集合中选择，满足可执行性约束。
- 选择规则固定为“最高优先级（数值最小）优先”，与算法定义一致。
- 非抢占意味着被选中的任务会连续执行到完成，保证时间推进单调。
- 所有任务最终都会从 `pending/ready` 移入 `done`，算法终止。

## R08

复杂度（设进程数为 `n`）：
- 初始排序：`O(n log n)`。
- 主循环中，每次在 `ready` 线性选最小项，最坏合计 `O(n^2)`。
- 空间复杂度：`O(n)`（存储任务与结果）。

说明：若改为优先队列（堆），可把选择过程降到 `O(log n)`，总体接近 `O(n log n)`。

## R09

边界与约束处理：
- 支持 CPU 空闲区间（最早到达时间 > 0 或中间出现空窗）。
- 支持多个任务同一时刻到达。
- 同优先级时用 `(arrival, pid)` 做稳定决策，避免结果不确定。
- 本 MVP 假设输入合法，不额外做异常恢复（例如负时间、零 burst）。

## R10

`demo.py` 覆盖内容：
- 调度主函数：`priority_non_preemptive`。
- 选择函数：`_pick_next`。
- 结果展示：表格、平均指标、甘特图。
- 无交互输入，直接运行即可复现实验。

## R11

运行方式：

```bash
uv run python demo.py
```

示例输出会包含：
- 每个进程的开始/结束/等待/周转/响应时间
- 平均等待、平均周转、平均响应
- 文本甘特图，例如 `[0-7:P1] [7-11:P2] ...`

## R12

当前示例任务集：
- `P1(arrival=0, burst=7, priority=3)`
- `P2(arrival=2, burst=4, priority=1)`
- `P3(arrival=4, burst=1, priority=4)`
- `P4(arrival=5, burst=4, priority=2)`

可直接修改 `main()` 中该列表测试不同负载分布。

## R13

工程取舍：
- 采用纯 Python 标准库实现，降低运行依赖与环境摩擦。
- 先保证“可读 + 可运行 + 可验证”的最小 MVP。
- 未引入 NumPy/Pandas/PyTorch 等库，原因是该任务核心是离散调度逻辑，不依赖数值加速。

## R14

常见错误：
- 把优先级方向写反（把大值当高优先级）。
- 忘记处理 `ready` 为空，导致时间不推进死循环。
- 在计算等待时间时错误使用 `finish-arrival`（这其实是周转时间）。
- 未定义同优先级 tie-break，导致结果在不同运行中不稳定。

## R15

可扩展方向：
- 抢占式优先级调度（高优任务到达可打断当前任务）。
- 老化（Aging）机制，缓解低优先级饥饿。
- 多核场景下的全局/局部就绪队列策略。
- 与 FCFS、SJF、RR 在同数据集上的指标对比评测。

## R16

建议测试点：
- 单进程与空输入（若要支持空输入可在主函数前置判断）。
- 全部同优先级（应退化为到达顺序 + pid 决策）。
- 全部同到达时间（应按优先级排序执行）。
- 存在较长空闲窗口（验证时间跳转逻辑）。

## R17

结论：
- 优先级调度能显式体现“重要任务先执行”的策略目标。
- 非抢占版实现简单、开销低，适合教学与基础调度原型。
- 在真实系统中通常需要结合抢占与老化机制，平衡响应性与公平性。

## R18

源码级算法拆解（对应 `demo.py`，无第三方黑箱）：
1. `main()` 构造进程列表并调用 `priority_non_preemptive(processes)`。
2. 调度函数先将 `processes` 按 `(arrival, pid)` 排序到 `pending`，初始化 `time/i/ready/done`。
3. 进入主循环后，先把所有 `arrival <= time` 的进程搬入 `ready`。
4. 若 `ready` 为空，说明当前时刻无可执行任务，直接将 `time` 跳到 `pending[i].arrival`。
5. 若 `ready` 非空，调用 `_pick_next(ready)`，按 `(priority, arrival, pid)` 选出当前应执行进程。
6. 计算该进程的 `start/finish/waiting/turnaround/response`，写入 `ScheduleRecord` 并追加到 `done`。
7. 将 `time` 更新为 `finish`，继续循环直到 `len(done) == n`。
8. 返回结果后，`_print_table/_print_summary/_print_gantt` 分别输出明细、平均指标和甘特图。

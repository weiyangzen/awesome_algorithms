# 进程调度 - FCFS

- UID: `CS-0161`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `306`
- 目标目录: `Algorithms/计算机-操作系统-0306-进程调度_-_FCFS`

## R01

问题定义：给定一组进程（包含到达时间 `arrival` 与 CPU 执行时间 `burst`），在单核环境下使用先来先服务（FCFS, First Come First Served）策略进行调度，输出执行时间线与每个进程的完成时间、周转时间、等待时间、响应时间。

## R02

输入：
- 进程集合 `P = {p1, p2, ...}`，每个进程包含：
  - `pid`：进程标识（唯一）
  - `arrival`：到达时间（非负整数）
  - `burst`：CPU 执行时间（正整数）

输出：
- 甘特图式时间线 `[(start, end, pid)]`，其中 `pid="IDLE"` 表示 CPU 空闲
- 每个进程指标：
  - `completion`：完成时刻
  - `turnaround = completion - arrival`
  - `waiting = turnaround - burst`
  - `response = first_start - arrival`
- 平均指标：平均周转、平均等待、平均响应时间

## R03

核心思想（非抢占式 FCFS）：
- 调度顺序完全由到达先后决定，先到先执行。
- 一旦某进程被选中，就连续运行直到完成（不被后续到达任务打断）。
- 当前没有可运行进程时，CPU 进入空闲并将时钟推进到下一到达时刻。
- FCFS 实现最简单、行为最直观，是很多复杂调度策略的基线。

## R04

关键数据结构：
- `Process`：保存 `pid/arrival/burst/first_start/completion`
- 已排序进程表：按 `(arrival, pid)` 排序，确保同到达时刻下结果可复现
- `timeline`：记录执行片段 `(start, end, pid)`
- `metrics`：按 `pid` 汇总统计指标

## R05

算法流程（高层）：
1. 校验输入并构造进程对象。
2. 按 `(arrival, pid)` 排序，初始化 `time=0`。
3. 依次扫描排序后的每个进程。
4. 若 `time < arrival`，记录 `IDLE` 片段并将 `time` 跳到 `arrival`。
5. 记录进程首次开始时刻 `first_start=time`。
6. 执行到完成：`end = time + burst`，把 `(time, end, pid)` 写入时间线。
7. 记录 `completion=end`，更新时间 `time=end`。
8. 全部结束后计算每进程指标与平均指标。

## R06

正确性要点：
- 顺序正确性：排序后顺序与“先来先服务”一致，同到达时刻通过 `pid` 打破平局，保证可复现。
- 非抢占性：每个进程被调度后一次运行到完成，符合 FCFS 定义。
- 时间连续性：若 CPU 空闲则显式插入 `IDLE`，时间线无断裂。
- 指标一致性：`completion` 和 `first_start` 都来自真实时间线，派生公式可直接校验。

## R07

复杂度分析：
- 设进程数为 `n`。
- 排序开销：`O(n log n)`。
- 线性扫描调度：`O(n)`。
- 总时间复杂度：`O(n log n)`。
- 额外空间复杂度：`O(n)`（进程状态、时间线、统计字典）。

## R08

边界与异常处理：
- 空输入：抛出异常（MVP 至少需要一个进程）。
- `arrival < 0`：非法。
- `burst <= 0`：非法。
- `pid` 重复：非法（会导致统计覆盖）。
- 首个进程在未来到达：时间线前缀出现 `IDLE`。
- 多进程同到达：按 `pid` 确定稳定顺序。

## R09

MVP 实现范围：
- 仅实现单核、非抢占式 FCFS。
- 不考虑 I/O 阻塞、多段 CPU burst、上下文切换开销。
- 输入为离散整数时间，输出为文本化时间线与指标表。
- 实现只依赖 Python 标准库（`dataclasses`、`typing`），便于最小化部署与审计。

## R10

`demo.py` 内置示例：
- `P1(arrival=1, burst=6)`
- `P2(arrival=2, burst=8)`
- `P3(arrival=3, burst=2)`
- `P4(arrival=5, burst=4)`

对应时间线（预期）：
- `[0,1] IDLE -> [1,7] P1 -> [7,15] P2 -> [15,17] P3 -> [17,21] P4`

## R11

预期现象：
- 先到达的进程会稳定地先完成。
- 短作业若到达较晚（如 `P3`）也必须等待前面长作业，响应可能明显变差。
- 该示例可观察到典型“护航效应”（Convoy Effect）：前部长任务让后续短任务排队。

## R12

指标解释：
- `Turnaround`：任务从提交到完成的总时长。
- `Waiting`：任务在就绪队列中排队等待 CPU 的总时长。
- `Response`：任务首次获得 CPU 的延迟。
- 对单核非抢占 FCFS，通常 `response == waiting`（每个进程仅被调度一次）。

## R13

与其他调度策略关系：
- 与 SJF 对比：FCFS 更公平直观，但平均等待时间通常高于 SJF。
- 与 RR 对比：FCFS 几乎无时间片切换成本，但交互响应能力弱于 RR。
- 与优先级调度对比：FCFS 不需要优先级信息，实现简单但难表达业务重要性。

## R14

局限与工程注意事项：
- 护航效应：前部 CPU 密集型长作业会拖慢整体响应。
- 对交互型系统不友好：短任务可能长时间等待。
- 本 MVP 未建模上下文切换成本；真实系统还需综合吞吐、公平与延迟目标。
- 工程实践中 FCFS 常作为基线或子队列策略，而非唯一调度策略。

## R15

常见实现错误：
- 错把 FCFS 实现成“就绪队列中按最短作业优先”。
- 忘记在空闲区间插入 `IDLE`，导致时间线与指标不一致。
- 未验证 `pid` 唯一，造成统计字典覆盖。
- 把 `response` 错写为 `completion - arrival`。
- 同到达进程缺少稳定 tie-break，导致结果不可复现。

## R16

最小测试清单：
- 正常交错到达 + 不同 burst。
- 首个进程非零到达（验证 `IDLE`）。
- 同时到达的多个进程（验证稳定顺序）。
- 全部 burst 相同（应仅按到达/标识顺序执行）。
- 非法输入：空数组、重复 pid、负到达、非正 burst。

## R17

可扩展方向：
- 增加上下文切换开销并重新评估平均指标。
- 引入 I/O 阻塞和多段 CPU burst，模拟更真实的进程生命周期。
- 对比 FCFS/SJF/RR/优先级 在同一数据集上的性能差异。
- 增加甘特图可视化输出用于教学演示与回归测试。

## R18

源码级算法拆解（对应 `demo.py`，非黑盒）：
1. `_validate_inputs` 检查输入合法性：非空、`pid` 唯一、`arrival>=0`、`burst>0`。
2. 将输入转为 `Process` 对象，并按 `(arrival, pid)` 排序，建立 FCFS 的确定性服务顺序。
3. 初始化全局时钟 `time=0` 和空时间线 `timeline=[]`。
4. 顺序遍历每个进程；若当前 `time` 早于该进程到达时刻，则追加 `IDLE` 并跳时钟到 `arrival`。
5. 在实际开始执行时记录 `first_start=time`，保证响应时间来源可追溯。
6. 计算 `end=time+burst`，将 `(time, end, pid)` 写入时间线，并设置 `completion=end`。
7. 更新时间 `time=end` 后继续下一个进程，直到全部进程完成。
8. 基于 `completion/arrival/burst/first_start` 计算 `turnaround/waiting/response`，再汇总平均值并打印。

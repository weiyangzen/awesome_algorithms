# TCP Reno

- UID: `CS-0201`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `348`
- 目标目录: `Algorithms/计算机-计算机网络-0348-TCP_Reno`

## R01

TCP Reno 是经典的 TCP 拥塞控制实现，核心由三部分组成：
- `慢启动 (Slow Start)`：窗口快速增长以探测可用带宽；
- `拥塞避免 (Congestion Avoidance)`：窗口线性增长（AIMD 中的加法增）；
- `快速重传 + 快速恢复 (Fast Retransmit / Fast Recovery)`：收到 3 个重复 ACK 后不等超时，先重传并进入恢复态。

本条目实现一个离散事件的教学型 MVP，用可复现实验展示 Reno 的关键状态切换。

## R02

问题定义（MVP 范围）：
- 输入：
1. `init_cwnd`：初始拥塞窗口（单位 MSS）；
2. `init_ssthresh`：慢启动阈值；
3. `events`：事件序列，仅支持 `ACK_NEW / DUP_ACK / TIMEOUT`。
- 输出：
1. 每步的 `cwnd/ssthresh/state/dupACK/retransmissions`；
2. 完整时间线表（pandas DataFrame）；
3. 断言结果，确保行为与预期规则一致。

## R03

核心状态机规则（教学化 Reno）：
- 状态集合：`slow_start`、`congestion_avoidance`、`fast_recovery`。
- 在 `slow_start` 收到 `ACK_NEW`：`cwnd *= 2`，当 `cwnd >= ssthresh` 切换到 `congestion_avoidance`。
- 在 `congestion_avoidance` 收到 `ACK_NEW`：`cwnd += 1`（线性增长）。
- 在 `slow_start / congestion_avoidance` 收到 `DUP_ACK`：
1. 重复 ACK 计数递增；
2. 当计数达到 3：`ssthresh = max(cwnd // 2, 2)`，`cwnd = ssthresh + 3`，触发快速重传并进入 `fast_recovery`。
- 在 `fast_recovery`：
1. 收到额外 `DUP_ACK`：`cwnd += 1`；
2. 收到任意 `ACK_NEW`：`cwnd = ssthresh`，退出到 `congestion_avoidance`（Reno 的关键点）。
- 任意状态收到 `TIMEOUT`：`cwnd=1`，阈值折半，回到 `slow_start`。

## R04

`demo.py` 执行流程：
1. 校验输入并标准化事件；
2. 初始化 Reno 状态变量；
3. 逐事件推进状态机；
4. 每一步写入 `StepRecord`；
5. 提取 numpy 序列做确定性断言；
6. 验证 Reno 特有性质（新 ACK 立即退出快速恢复）；
7. 以 pandas 表格打印全过程；
8. 输出 `All checks passed.`。

## R05

关键数据结构：
- `StepRecord`：单步快照，记录前后状态、窗口、阈值、dupACK 和重传计数；
- `records: List[StepRecord]`：全流程轨迹；
- `numpy.ndarray`：承载断言序列；
- `pandas.DataFrame`：用于可读展示。

## R06

正确性直觉：
- `DUP_ACK` 连续出现意味着对端在持续收到后续数据，通常代表中间有分段丢失；
- 三重重复 ACK 触发“快速重传”，避免等待超时；
- 快速恢复期间保持一定发送速率，减少吞吐骤降；
- Reno 在恢复态收到新 ACK 后立即退出，逻辑简单但在多丢包场景可能过早离开恢复态。

## R07

复杂度（设事件数为 `E`）：
- 时间复杂度：`O(E)`；
- 空间复杂度：`O(E)`（保存逐步记录）。

## R08

边界与异常处理：
- `init_cwnd < 2` 或 `init_ssthresh < 2`：抛 `ValueError`；
- 事件序列为空：抛 `ValueError`；
- 事件类型不在 `{ACK_NEW, DUP_ACK, TIMEOUT}`：抛 `ValueError`；
- 遇到未知状态：抛 `ValueError`。

## R09

MVP 取舍说明：
- 采用离散事件仿真，不追求内核 TCP 的逐字节/逐定时器细节；
- 保留 Reno 的关键机制（AIMD、三重重复 ACK、快速恢复退出逻辑）；
- 不依赖任何拥塞控制黑箱库，窗口更新规则全部在源码里显式实现；
- 仅使用 `numpy + pandas` 做断言与展示，工具栈最小且透明。

## R10

`demo.py` 函数职责：
- `validate_inputs`：输入校验与事件归一化；
- `enter_fast_recovery`：三重重复 ACK 入口计算；
- `simulate_tcp_reno`：主状态机；
- `extract_series`：提取断言序列；
- `records_to_dataframe`：格式化输出表；
- `run_demo`：固定样例 + 断言 + 打印；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0348-TCP_Reno
uv run python demo.py
```

脚本无需交互输入，直接执行并输出结果。

## R12

输出说明：
- `Step timeline`：每步事件及 `state/cwnd/ssthresh/dupACK/retrans` 变化；
- 通过断言后输出 `All checks passed.`；
- 任一规则不满足会抛出 `AssertionError` 并给出定位信息。

## R13

最小验证清单：
- `cwnd/ssthresh/state/dupACK/retransmissions` 五组序列与期望值精确一致；
- 快速恢复入口必须满足：`dupACK == 3` 且 `cwnd = ssthresh + 3`；
- 快速恢复中 `ACK_NEW` 必须立刻退出到 `congestion_avoidance`；
- 超时事件后必须满足 `cwnd=1` 且进入 `slow_start`。

## R14

固定实验参数：
- `init_cwnd = 12`；
- `init_ssthresh = 18`；
- 事件序列（20 步）：
1. `ACK_NEW`
2. `ACK_NEW`
3. `DUP_ACK`
4. `DUP_ACK`
5. `DUP_ACK`
6. `DUP_ACK`
7. `ACK_NEW`
8. `ACK_NEW`
9. `DUP_ACK`
10. `DUP_ACK`
11. `DUP_ACK`
12. `ACK_NEW`
13. `TIMEOUT`
14. `ACK_NEW`
15. `ACK_NEW`
16. `ACK_NEW`
17. `DUP_ACK`
18. `DUP_ACK`
19. `DUP_ACK`
20. `ACK_NEW`

该序列覆盖 3 次快速恢复入口、1 次超时回退与慢启动回升。

## R15

与相关算法差异：
- Tahoe：丢包后通常直接回到 `cwnd=1`，更保守；
- Reno（本条目）：引入快速恢复，避免每次都从 1 重新爬升；
- NewReno：在恢复期区分 partial/full ACK，处理多丢包更稳健；
- Vegas/BBR：控制信号与目标不同，不是 Reno 的同一机制链路。

## R16

适用场景：
- 网络课程中讲解 TCP Reno 状态机；
- 离散事件仿真中快速验证 Reno 规则；
- 作为 NewReno/CUBIC 等后续对比的基线实现。

不适用场景：
- 要求与 Linux 内核 TCP Reno 行为逐细节一致；
- 需要 SACK、乱序缓存、ACK 压缩等生产级机制。

## R17

可扩展方向：
- 将事件级模型升级为按字节 ACK 驱动；
- 增加随机丢包与抖动 RTT，统计吞吐/恢复时长；
- 扩展为多流竞争，测公平性与收敛速度；
- 与 NewReno/Vegas/BBR 在同一场景下对比。

## R18

`demo.py` 源码级算法流（8 步，非黑箱）：
1. `run_demo` 构造固定参数和 20 步事件序列，然后调用 `simulate_tcp_reno`。  
2. `simulate_tcp_reno` 先通过 `validate_inputs` 校验初值与事件集合。  
3. 初始化 `cwnd/ssthresh/state/dup_ack_count/retransmissions` 五个核心变量。  
4. 逐事件处理：若是 `TIMEOUT`，统一执行阈值折半、`cwnd=1`、回到 `slow_start`。  
5. 在 `slow_start` 和 `congestion_avoidance` 中处理 `ACK_NEW` 增窗；处理 `DUP_ACK` 计数，达到 3 时调用 `enter_fast_recovery` 并触发快速重传。  
6. 在 `fast_recovery` 中，`DUP_ACK` 令 `cwnd+1`；收到任意 `ACK_NEW` 则立刻退出恢复态，并把 `cwnd` 回落到 `ssthresh`（Reno 特征）。  
7. 每一步把前后状态写入 `StepRecord`，随后 `extract_series` 生成 numpy 序列并与预期序列逐项比较。  
8. `run_demo` 继续做性质断言（FR 入口、ACK 退出、TIMEOUT 语义），再用 `records_to_dataframe` 输出时间线并打印 `All checks passed.`。

# TCP Tahoe

- UID: `CS-0200`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `347`
- 目标目录: `Algorithms/计算机-计算机网络-0347-TCP_Tahoe`

## R01

TCP Tahoe 是早期经典 TCP 拥塞控制版本，核心特点是“保守回退”：
- 慢启动（Slow Start）阶段快速增大拥塞窗口 `cwnd`；
- 达到阈值 `ssthresh` 后进入拥塞避免（Congestion Avoidance）；
- 一旦检测到丢包（超时或三重重复 ACK），执行乘法减小并把 `cwnd` 重置为 1，重新慢启动。

本条目给出一个可运行的离散事件 MVP，强调 Tahoe 的“无快速恢复（Fast Recovery）”语义。

## R02

问题定义（MVP 范围）：
- 输入：
1. `init_cwnd`：初始拥塞窗口（MSS 单位）；
2. `init_ssthresh`：初始慢启动阈值；
3. `events`：事件序列，仅支持 `ACK_NEW / DUP_ACK / TIMEOUT`。
- 输出：
1. 每一步的 `cwnd`、`ssthresh`、状态、重复 ACK 计数、拥塞避免 ACK 计数器、重传计数；
2. 完整时间线表格；
3. 固定样例断言结果。

## R03

核心状态机规则（教学化简）：
- 状态：`slow_start`、`congestion_avoidance`。
- `ACK_NEW`：
1. 在 `slow_start`：`cwnd += 1`；
2. 若 `cwnd >= ssthresh`，切换到 `congestion_avoidance`；
3. 在 `congestion_avoidance`：累计 ACK 计数，累计到当前 `cwnd` 后才 `cwnd += 1`（近似 AIMD 的线性增长）。
- `DUP_ACK`：
1. 重复 ACK 计数加一；
2. 当达到 3（triple dup ACK）时，触发快速重传并执行 Tahoe 回退：
   `ssthresh=max(cwnd//2,2)`，`cwnd=1`，回到 `slow_start`。
- `TIMEOUT`：
1. 与丢包回退一致：`ssthresh=max(cwnd//2,2)`，`cwnd=1`，回到 `slow_start`；
2. 计一次重传。

## R04

`demo.py` 执行流程：
1. 校验参数与事件集合合法性；
2. 初始化 `cwnd/ssthresh/state/dup_ack_count/ca_ack_counter/retransmissions`；
3. 逐事件推进 Tahoe 状态机；
4. 每步记录 `StepRecord`；
5. 抽取 numpy 序列并与预设序列做确定性断言；
6. 校验 Tahoe 关键性质（triple dup ACK 与 timeout 都退回慢启动）；
7. 使用 pandas 输出时间线表格；
8. 打印 `All checks passed.`。

## R05

关键数据结构：
- `StepRecord`：单步状态转移快照；
- `records: List[StepRecord]`：仿真全过程；
- `numpy.ndarray`：`cwnd/ssthresh/state/dup/ca_counter/retrans` 序列断言；
- `pandas.DataFrame`：可读时间线输出。

## R06

正确性直觉：
- 慢启动让连接快速探测可用带宽；
- 拥塞避免通过“每 `cwnd` 个 ACK 才加 1”降低增长斜率；
- 三重重复 ACK 提供“尚未超时但可能丢包”的信号；
- Tahoe 对三重重复 ACK 和超时都回到 `cwnd=1`，体现其保守策略：优先稳定性与网络安全性，而非短期吞吐。

## R07

复杂度（设事件数为 `E`）：
- 时间复杂度：`O(E)`；
- 空间复杂度：`O(E)`（保存每步记录）。

## R08

边界与异常处理：
- `init_cwnd < 1`：抛 `ValueError`；
- `init_ssthresh < 2`：抛 `ValueError`；
- `events` 为空：抛 `ValueError`；
- 事件不在 `{ACK_NEW, DUP_ACK, TIMEOUT}`：抛 `ValueError`；
- 状态非法（防御式检查）：抛 `ValueError`。

## R09

MVP 设计取舍：
- 使用离散事件而非内核字节级实现，突出 Tahoe 机制本身；
- 仅依赖 `numpy + pandas`，避免复杂框架；
- 不调用任何黑箱拥塞控制库，窗口更新规则均在源码显式展开；
- 对拥塞避免采用“ACK 计数器近似线性增长”，兼顾可读性与算法语义。

## R10

`demo.py` 函数职责：
- `validate_inputs`：参数与事件校验；
- `multiplicative_decrease`：`ssthresh` 更新规则；
- `simulate_tcp_tahoe`：主状态机仿真；
- `extract_series`：提取 numpy 断言序列；
- `records_to_dataframe`：构建输出表；
- `run_demo`：固定样例、断言、打印；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0347-TCP_Tahoe
uv run python demo.py
```

脚本无需交互输入，直接输出时间线和断言结果。

## R12

输出说明：
- `Step timeline`：每一步事件、状态、窗口、阈值、计数器和说明；
- `*_sequence`：断言使用的确定性序列；
- `All checks passed.`：表示数值断言与性质断言均通过。

## R13

最小验证清单：
- `cwnd/ssthresh/state/dup_ack/ca_ack_counter/retransmissions` 六组序列完全匹配预期；
- 恰好一次 triple dup ACK 回退，且满足 `cwnd=1`、`state=slow_start`、`ssthresh` 按乘法减小；
- 恰好一次 timeout 回退，且满足与 Tahoe 语义一致；
- 全程 `ssthresh >= 2` 且 `cwnd >= 1`。

## R14

当前固定实验参数：
- `init_cwnd = 1`
- `init_ssthresh = 8`
- 事件序列（25 步）：
1. ACK_NEW
2. ACK_NEW
3. ACK_NEW
4. ACK_NEW
5. ACK_NEW
6. ACK_NEW
7. ACK_NEW
8. ACK_NEW
9. ACK_NEW
10. DUP_ACK
11. DUP_ACK
12. DUP_ACK
13. ACK_NEW
14. ACK_NEW
15. ACK_NEW
16. ACK_NEW
17. TIMEOUT
18. ACK_NEW
19. ACK_NEW
20. DUP_ACK
21. ACK_NEW
22. ACK_NEW
23. DUP_ACK
24. DUP_ACK
25. ACK_NEW

该序列覆盖：慢启动到拥塞避免、一次 triple dup ACK 回退、一次超时回退。

## R15

与相关算法差异：
- Tahoe：丢包（含 triple dup ACK）后直接 `cwnd=1`，没有快速恢复；
- Reno：引入快速恢复，三重重复 ACK 后不会立即回到 1；
- NewReno：在恢复期进一步区分 `partial ACK/full ACK`；
- BBR/CUBIC：控制目标与增长模型不同，不属于 Tahoe 的 AIMD 早期实现链路。

## R16

适用场景：
- 教学演示 TCP 早期拥塞控制思想；
- 用于离散仿真中验证丢包回退机制；
- 作为 Reno/NewReno 对比基线。

不适用场景：
- 需要与 Linux 内核 TCP 实现逐细节一致；
- 需要 SACK、重排序缓冲、ACK 压缩等生产级行为建模。

## R17

可扩展方向：
- 引入随机丢包与时延抖动，统计吞吐与恢复时长；
- 增加多流竞争，比较公平性；
- 对照实现 Reno/NewReno，比较同事件流下窗口演化；
- 升级到字节级 ACK 与 RTT 驱动模型。

## R18

`demo.py` 源码级算法流（9 步，非黑箱）：
1. `main` 调用 `run_demo`，构造初始参数与 25 步事件流。  
2. `run_demo` 调用 `simulate_tcp_tahoe`，先由 `validate_inputs` 校验输入。  
3. 初始化状态变量：`cwnd`、`ssthresh`、`state`、`dup_ack_count`、`ca_ack_counter`、`retransmissions`。  
4. 对每个事件先记录 `before` 快照，随后进入分支处理。  
5. 若事件是 `TIMEOUT`，调用 `multiplicative_decrease` 更新阈值，并执行 Tahoe 回退：`cwnd=1`、`state=slow_start`、计一次重传。  
6. 若事件是 `DUP_ACK`，先累加重复计数；当计数到 3 时触发快速重传并执行同样回退（Tahoe 不进入快速恢复）。  
7. 若事件是 `ACK_NEW`：在 `slow_start` 做 `cwnd += 1`；在 `congestion_avoidance` 使用 `ca_ack_counter` 累计 ACK，累计到窗口大小后才 `cwnd += 1`。  
8. 每步写入 `StepRecord`；结束后 `extract_series` 生成 numpy 序列并进行逐项精确断言。  
9. 继续执行性质断言（triple dup ACK/timeout 语义、边界约束），最后 `records_to_dataframe` 打印时间线并输出 `All checks passed.`。  

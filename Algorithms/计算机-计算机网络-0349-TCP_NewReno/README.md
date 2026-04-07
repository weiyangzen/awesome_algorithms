# TCP NewReno

- UID: `CS-0202`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `349`
- 目标目录: `Algorithms/计算机-计算机网络-0349-TCP_NewReno`

## R01

TCP NewReno 是 Reno 的改进版拥塞控制，关键改动在快速恢复阶段：
- Reno 在快速恢复里收到“任何新 ACK”通常就退出恢复；
- NewReno 区分 `partial ACK` 和 `full ACK`：
1. `partial ACK` 说明还有丢失分段未恢复，应继续留在快速恢复并重传下一个丢失分段；
2. 只有 `full ACK`（确认到恢复点）才退出快速恢复。

本条目给出一个可运行的离散事件 MVP，重点演示这条差异链路。

## R02

问题定义（MVP 范围）：
- 输入：
1. `init_cwnd`：初始拥塞窗口（单位：MSS）；
2. `init_ssthresh`：初始慢启动阈值；
3. `events`：事件序列，支持 `ACK_NEW / DUP_ACK / PARTIAL_ACK / FULL_ACK / TIMEOUT`。
- 输出：
1. 每步 `cwnd`、`ssthresh`、状态、重复 ACK 计数、重传计数；
2. 完整步骤时间线表；
3. 固定断言结果（序列与性质检查）。

## R03

核心状态机规则（教学简化版）：
- 状态集合：`slow_start`、`congestion_avoidance`、`fast_recovery`。
- 在 `congestion_avoidance`/`slow_start` 收到 `DUP_ACK`：累计重复 ACK；当达到 3 时：
1. `ssthresh = max(cwnd // 2, 2)`；
2. `cwnd = ssthresh + 3`；
3. 触发一次快速重传并进入 `fast_recovery`。
- 在 `fast_recovery`：
1. 额外 `DUP_ACK`：`cwnd += 1`；
2. `PARTIAL_ACK`：重传下一个丢失分段，保持在 `fast_recovery`，窗口轻度回调；
3. `FULL_ACK`：`cwnd = ssthresh` 并退出到 `congestion_avoidance`。
- 任意状态收到 `TIMEOUT`：`cwnd=1`，`ssthresh` 折半，回到 `slow_start`。

## R04

`demo.py` 执行流程：
1. 校验输入参数与事件合法性；
2. 初始化 `cwnd/ssthresh/state/dup_ack_count/retransmissions`；
3. 逐事件推进状态机；
4. 记录每一步前后状态到 `StepRecord`；
5. 抽取 numpy 序列并与期望序列做确定性断言；
6. 检查 NewReno 关键性质（`PARTIAL_ACK` 不退出快速恢复）；
7. 使用 pandas 打印可读时间线；
8. 输出 `All checks passed.`。

## R05

关键数据结构：
- `StepRecord`：单步转移快照，含窗口、阈值、状态、dupACK、重传计数；
- `records: List[StepRecord]`：仿真时间线；
- `numpy.ndarray`：用于断言的序列（`cwnd/ssthresh/state/dup/retrans`）；
- `pandas.DataFrame`：最终可读输出表。

## R06

正确性直觉：
- 三次重复 ACK 提供“超时前”丢包证据，触发快速重传；
- 进入快速恢复后保持一定发送能力，避免吞吐骤降；
- `partial ACK` 代表“只修复了一部分丢包”，继续留在恢复态可连续修复同窗口多重丢包；
- 仅 `full ACK` 才结束恢复，体现 NewReno 对多丢包场景更稳健。

## R07

复杂度（设事件数为 `E`）：
- 时间复杂度：`O(E)`；
- 空间复杂度：`O(E)`（保存逐步记录）。

## R08

边界与异常处理：
- `init_cwnd < 2` 或 `init_ssthresh < 2`：抛 `ValueError`；
- `events` 为空：抛 `ValueError`；
- 事件类型不在允许集合：抛 `ValueError`；
- 在非快速恢复状态收到 `PARTIAL_ACK`：抛 `ValueError`；
- 出现未知状态或非法状态-事件组合：抛 `ValueError`。

## R09

MVP 设计取舍：
- 采用“离散事件”而非真实内核逐 ACK/逐定时器实现，重点突出算法机制；
- 仅使用 `numpy + pandas` 做校验和展示，工具栈小且透明；
- 不依赖任何黑箱拥塞控制库，窗口更新逻辑全部在源码中显式给出；
- 保留 NewReno 核心差异（partial ACK 路径），其余细节（SACK、精确字节级窗口）做教学化简化。

## R10

`demo.py` 函数职责：
- `validate_inputs`：输入校验和事件标准化；
- `enter_fast_recovery`：三重重复 ACK 入口规则；
- `partial_ack_cwnd_update`：partial ACK 下的窗口回调规则；
- `simulate_tcp_newreno`：主状态机仿真；
- `extract_series`：提取 numpy 断言序列；
- `records_to_dataframe`：生成表格输出；
- `run_demo`：固定样例 + 断言 + 打印；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0349-TCP_NewReno
uv run python demo.py
```

脚本无需交互输入，会直接输出时间线和断言结果。

## R12

输出说明：
- `Step timeline`：每步事件、状态迁移、`cwnd`、`ssthresh`、dupACK、重传计数；
- `cwnd_after/ssthresh_after/state_after/dup_ack_count_after/retransmissions_after sequence`：断言用序列；
- `All checks passed.`：表示全部数值断言与性质断言通过。

## R13

最小验证清单：
- 序列断言：`cwnd`、`ssthresh`、状态、dupACK、重传计数全部精确匹配预期；
- 恰好 2 次进入快速恢复，且入口满足 `dup_ack_count=3` 与 `cwnd=ssthresh+3`；
- 所有 `PARTIAL_ACK` 都满足“进入前后状态均为 `fast_recovery`”；
- 所有 `PARTIAL_ACK` 都触发且仅触发一次重传计数递增；
- `TIMEOUT` 事件后必须 `cwnd=1` 且状态切到 `slow_start`。

## R14

当前固定实验参数：
- `init_cwnd = 14`；
- `init_ssthresh = 20`；
- 事件序列（22 步）：
1. `ACK_NEW`
2. `DUP_ACK`
3. `DUP_ACK`
4. `DUP_ACK`
5. `DUP_ACK`
6. `PARTIAL_ACK`
7. `DUP_ACK`
8. `PARTIAL_ACK`
9. `DUP_ACK`
10. `FULL_ACK`
11. `ACK_NEW`
12. `ACK_NEW`
13. `TIMEOUT`
14. `ACK_NEW`
15. `ACK_NEW`
16. `ACK_NEW`
17. `DUP_ACK`
18. `DUP_ACK`
19. `DUP_ACK`
20. `PARTIAL_ACK`
21. `FULL_ACK`
22. `ACK_NEW`

该序列覆盖两轮快速恢复、三次 partial ACK、一次超时回退。

## R15

与相关算法差异：
- Tahoe：丢包后通常直接把 `cwnd` 降到 1，更保守；
- Reno：快速恢复中对新 ACK 处理更粗，易在多丢包时过早退出恢复；
- NewReno（本条目）：用 partial/full ACK 区分恢复进度，适合单窗口多丢包；
- BBR/CUBIC：控制目标与增长函数不同，不属于 NewReno 的 AIMD + FR 改进链路。

## R16

适用场景：
- 教学演示 NewReno 与 Reno 的关键区别；
- 离散网络仿真中快速验证“partial ACK 持续恢复”机制；
- 作为后续更细粒度 TCP 模型的起点。

不适用场景：
- 需要与 Linux 内核 NewReno 逐字节、逐计时器行为完全一致；
- 需要覆盖 SACK、乱序缓存、ACK 压缩等生产级细节。

## R17

可扩展方向：
- 从事件级升级到“按 ACK 字节量”窗口更新；
- 增加 SACK 与 NewReno 对照仿真；
- 注入随机丢包/时延抖动，统计吞吐与恢复时长；
- 增加多流竞争，比较 NewReno 与 CUBIC/BBR 公平性。

## R18

`demo.py` 源码级算法流（9 步，非黑箱）：
1. `main` 调用 `run_demo`，构造固定初始窗口与 22 步事件流。  
2. `run_demo` 调用 `simulate_tcp_newreno`，先由 `validate_inputs` 校验参数和事件集合。  
3. 初始化状态变量：`cwnd`、`ssthresh`、`state`、`dup_ack_count`、`retransmissions`。  
4. 对每个事件先处理全局超时路径：`TIMEOUT` 统一令 `cwnd=1`、阈值折半并回到 `slow_start`。  
5. 若处于 `congestion_avoidance/slow_start`，处理新 ACK 增长或重复 ACK 计数；达到 3 次重复 ACK 时调用 `enter_fast_recovery`，执行 `ssthresh=max(cwnd//2,2)` 与 `cwnd=ssthresh+3` 并计一次重传。  
6. 若处于 `fast_recovery`，`DUP_ACK` 使 `cwnd+1`；`PARTIAL_ACK` 调用 `partial_ack_cwnd_update` 并重传下一个丢失分段，同时保持在 `fast_recovery`。  
7. 在 `fast_recovery` 收到 `FULL_ACK/ACK_NEW` 时执行窗口回落 `cwnd=ssthresh`，退出到 `congestion_avoidance`。  
8. 每步状态变化写入 `StepRecord`，随后 `extract_series` 生成 numpy 序列并做逐项断言。  
9. `run_demo` 继续做性质断言（FR 入口、partial ACK 语义、timeout 语义），最后通过 `records_to_dataframe` 输出时间线并打印 `All checks passed.`。

# 选择性重传

- UID: `CS-0214`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `367`
- 目标目录: `Algorithms/计算机-计算机网络-0367-选择性重传`

## R01

本条目实现一个可运行的 **Selective Repeat (SR) ARQ** 最小 MVP。
目标是用确定性仿真展示 SR 的核心机制：
- 发送端维护滑动窗口并支持并发在途分组；
- 接收端可乱序缓存并按序交付；
- ACK 丢失或数据帧丢失时，仅重传超时分组（而不是整窗回退）；
- `uv run python demo.py` 无需交互即可复现固定结果。

## R02

问题定义（MVP 范围）：
- 输入参数：
1. `total_packets`：总分组数；
2. `window_size`：发送/接收窗口大小；
3. `seq_modulus`：序号空间模数；
4. `timeout`：超时阈值（tick）；
5. `data_delay/ack_delay`：单向传输时延；
6. `data_drop_plan`：`(packet_id, attempt)->是否丢帧`；
7. `ack_drop_plan`：`(packet_id, ack_index)->是否丢 ACK`。
- 输出结果：
1. `delivered_payloads`：接收端按序交付数据；
2. `attempts_per_packet`：每个分组的发送次数；
3. `tx_records/rx_records/ack_records`：完整时间线日志；
4. `stats`：重传数、丢包数、完成时刻、goodput 等统计。

## R03

SR 核心规则与不变量：
- 约束：`seq_modulus >= 2 * window_size`（避免 ACK/序号歧义）；
- 发送端窗口 `[sender_base, sender_base + window_size)` 内可并发发送；
- 仅当单个在途分组超时才重传该分组；
- 接收端窗口 `[receiver_base, receiver_base + window_size)` 内分组可缓存；
- 接收端始终按 `receiver_base` 连续交付，不连续则等待；
- ACK 命中后对应分组置为已确认，并驱动发送窗口前移。

## R04

`demo.py` 执行流程：
1. 参数合法性检查（窗口、序号空间、超时等）；
2. 发送端按窗口上限发送新分组；
3. 信道按 `data_drop_plan` 决定是否丢弃数据帧；
4. 接收端处理到达分组：缓存/交付/判定是否回 ACK；
5. 信道按 `ack_drop_plan` 决定 ACK 是否丢失；
6. 发送端处理 ACK 并滑动窗口；
7. 扫描在途分组，超时则选择性重传；
8. 所有分组确认完成后输出统计并执行断言。

## R05

关键数据结构：
- `SenderFrame`：发送端在途帧状态（包号、序号、尝试次数、最近发送时刻）；
- `TxRecord`：每次发送行为日志（new/retransmit、是否丢帧）；
- `RxRecord`：接收行为日志（buffered_new、duplicate、outside_window）；
- `AckRecord`：ACK 发射/到达日志（是否丢 ACK、是否命中发送端）；
- `SimulationResult`：最终交付数据、尝试次数数组、日志与统计的统一封装；
- `numpy.ndarray`：`attempts_per_packet` 与断言比较。

## R06

正确性直觉：
- SR 把“确认”粒度降到单分组，因此单点丢失不会拖累整窗重发；
- 接收端缓存乱序数据，保证链路乱序下仍可恢复按序交付；
- ACK 丢失时，发送端最终通过超时重传恢复一致性；
- 只要丢失不是永久且 `max_ticks` 充足，系统会收敛到“全部分组被确认并交付”。

## R07

复杂度（设 `N=total_packets`，`R` 为平均重传次数上界）：
- 时间复杂度：`O(N + N*R)`（每次发送/重传/ACK 处理均为常数级状态更新）；
- 空间复杂度：`O(N)`（发送确认数组 + 接收缓存 + 时间线日志主量级）。

本 MVP 是离散事件仿真，不追求协议栈级性能，仅追求机制透明与可验证性。

## R08

边界与异常处理：
- `total_packets/window_size/timeout/max_ticks <= 0`：抛 `ValueError`；
- `seq_modulus < 2 * window_size`：抛 `ValueError`；
- `data_delay/ack_delay <= 0`：抛 `ValueError`；
- 仿真在 `max_ticks` 内未完成：抛 `RuntimeError`；
- 若断言检测到交付序列或发送次数与预期不一致：抛 `AssertionError`。

## R09

MVP 设计取舍：
- 使用 `numpy + dataclasses`，工具栈小、可复现；
- 不调用现成网络模拟器或黑箱协议库，核心状态机全部显式实现；
- 信道模型采用确定性丢失计划而非随机噪声，便于精确断言；
- 不实现真实 TCP 细节（拥塞控制、SACK 选项、RTT 估计），保持 SR 主线清晰。

## R10

`demo.py` 函数职责：
- `validate_parameters`：参数与 SR 约束检查；
- `payload_for_packet`：固定 payload 生成；
- `seq_bits`：序号位宽计算（用于统计）；
- `transmit_frame`：发送/重传并注入数据帧丢失；
- `simulate_selective_repeat`：SR 仿真主循环；
- `run_demo`：固定样例、断言、打印报告；
- `main`：无交互脚本入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0367-选择性重传
uv run python demo.py
```

脚本不读取交互输入，直接输出时间线与校验结果。

## R12

输出说明：
- `config`：本次仿真参数；
- `attempts_per_packet`：每个分组实际发送次数；
- `sender tx timeline`：逐时刻发送日志（新发/重传、是否丢失）；
- `stats`：
1. `total_tx`：总发送次数；
2. `retransmissions`：重传次数；
3. `data_drops`：数据帧丢失次数；
4. `ack_drops`：ACK 丢失次数；
5. `completion_time`：仿真完成时间；
6. `max_receiver_buffer`：接收端峰值缓存占用；
7. `goodput`：有效载荷比。
- `All checks passed.`：所有内置断言通过。

## R13

最小验证清单：
- `delivered_payloads == payloads_sent`（接收端按序完整恢复）；
- `attempts_per_packet == [1,2,2,1,2,2,1,1]`；
- 重传分组集合恰为 `[1,2,4,5]`（体现“选择性”）；
- `retransmissions == 4`；
- `data_drops == 2` 且 `ack_drops == 2`。

## R14

当前 demo 固定参数：
- `total_packets = 8`
- `window_size = 4`
- `seq_modulus = 8`
- `timeout = 4`
- `data_delay = ack_delay = 1`
- `data_drop_plan = {(1,1):True, (4,1):True}`
- `ack_drop_plan = {(2,1):True, (5,1):True}`

这组参数同时覆盖：
- 数据帧丢失触发的超时重传；
- ACK 丢失触发的冗余重传；
- 乱序缓存与按序交付。

## R15

与其它 ARQ 机制对比：
- Stop-and-Wait：实现简单但吞吐低；
- Go-Back-N：接收端通常不缓存乱序，丢一个可能回退重发多个；
- Selective Repeat（本实现）：缓存乱序并按分组确认，链路利用率更高，但状态管理更复杂。

## R16

适用场景：
- 课程教学中解释滑动窗口与 ARQ 机制；
- 协议原型阶段验证“按包确认 + 超时重传”逻辑；
- 研究丢帧/丢 ACK 对重传行为和 goodput 的影响。

不适用场景：
- 需要真实网络时钟、拥塞控制、流量整形的系统级仿真；
- 需要大规模随机统计置信区间的实验（应增加 Monte Carlo 框架）。

## R17

可扩展方向：
- 引入随机信道与多轮统计（loss rate vs retransmission/goodput 曲线）；
- 增加 RTT 抖动和自适应超时（RTO）估计；
- 增加接收端缓存上限与溢出策略；
- 扩展为 TCP SACK 风格的选择确认块；
- 对比 SR、GBN、Stop-and-Wait 在同一丢包配置下的指标差异。

## R18

`demo.py` 源码级算法流（8 步，非黑箱）：
1. `run_demo` 设置固定参数和丢失计划，调用 `simulate_selective_repeat`。  
2. `simulate_selective_repeat` 先执行 `validate_parameters`，确保满足 SR 约束 `seq_modulus >= 2*window_size`。  
3. 发送端在窗口允许范围内创建 `SenderFrame`，经 `transmit_frame` 发送新分组并记录 `TxRecord`。  
4. 信道根据 `data_drop_plan` 决定数据帧是否到达；到达后接收端按窗口规则写入缓存并尽可能连续交付。  
5. 接收端对可确认分组发 ACK，信道再依据 `ack_drop_plan` 决定 ACK 是否丢失，并记录 `AckRecord`。  
6. 发送端处理到达 ACK：命中则标记该包已确认，并通过 `slide_sender_base` 推进发送窗口基线。  
7. 对仍在途但超时的分组，仅对该分组重传（`kind="retransmit"`），实现选择性重传而非整窗回退。  
8. 全部分组完成后返回 `SimulationResult`，`run_demo` 对交付序列、每包尝试次数、重传与丢失统计做断言并打印 `All checks passed.`。  

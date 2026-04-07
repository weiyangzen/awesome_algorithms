# 停等协议

- UID: `CS-0212`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `365`
- 目标目录: `Algorithms/计算机-计算机网络-0365-停等协议`

## R01

本条目实现一个可运行的 **停等协议（Stop-and-Wait ARQ）** 最小 MVP。  
目标是用确定性离散事件仿真清楚展示停等协议核心机制：
- 发送端同一时刻最多在途 1 个分组；
- 接收端用 1 bit 序号（`0/1`）识别新包与重复包；
- 数据帧丢失或 ACK 丢失时，发送端依赖超时触发重传；
- `uv run python demo.py` 无需交互即可稳定复现结果。

## R02

问题定义（MVP 范围）：
- 输入参数：
1. `total_packets`：待发送分组总数；
2. `timeout`：超时阈值（tick）；
3. `data_delay / ack_delay`：数据帧与 ACK 单向时延；
4. `data_drop_plan`：`(packet_id, attempt) -> 是否丢数据帧`；
5. `ack_drop_plan`：`(packet_id, ack_index) -> 是否丢 ACK`。
- 输出结果：
1. `delivered_payloads`：接收端最终按序交付载荷；
2. `attempts_per_packet`：每个分组实际发送次数；
3. `tx_records / rx_records / ack_records`：完整时间线日志；
4. `stats`：重传数、丢包数、完成时间、goodput 等统计。

## R03

停等协议规则与不变量：
- 发送端只维护一个在途分组，收到合法 ACK 前不发下一个新分组；
- 序号位仅取 `0/1`，每成功发送一个新分组后翻转；
- 接收端维护 `expected_seq`：
1. 收到 `seq == expected_seq` 视为新分组，交付并翻转 `expected_seq`；
2. 否则视为重复/过期分组，不重复交付；
- 接收端始终回 ACK，ACK 中携带“最近一次按序成功接收分组”的序号位；
- 发送端仅在 ACK 匹配“当前等待分组 + 当前序号”时推进状态，否则忽略。

## R04

`demo.py` 执行流程：
1. 参数合法性检查；
2. 若发送端空闲，发送当前新分组；
3. 信道按 `data_drop_plan` 决定是否丢弃数据帧；
4. 接收端处理到达数据帧并判定 `accepted_new / duplicate`；
5. 接收端发 ACK，信道按 `ack_drop_plan` 决定 ACK 是否丢弃；
6. 发送端处理 ACK，命中则推进到下一个分组并翻转序号位；
7. 若等待 ACK 超时，则仅重传当前分组；
8. 全部分组确认后输出统计并做断言。

## R05

关键数据结构：
- `TxRecord`：发送端每次发送日志（新发/重传、是否丢帧）；
- `RxRecord`：接收端每次接收日志（是否交付、`expected_seq` 变化）；
- `AckRecord`：ACK 发射/到达日志（是否丢 ACK、是否被发送端接受）；
- `SimulationResult`：仿真结果封装（载荷、尝试次数、日志、统计）；
- `numpy.ndarray`：`attempts_per_packet` 存储与断言比较。

## R06

正确性直觉：
- 停等协议把状态压到最小：发送端只关心“当前包是否被确认”；
- 序号位可区分“新到达包”和“因 ACK 丢失导致的重复重传包”；
- ACK 丢失时，发送端超时重传，接收端因序号检查不会重复交付；
- 因此在非永久性丢失场景下，系统最终会收敛到“全部包成功交付”。

## R07

复杂度分析（设总分组数 `N`，每包平均尝试次数为 `A`）：
- 时间复杂度：`O(N * A)`（每次发送、接收、ACK 处理均为常数级状态更新）；
- 空间复杂度：`O(N * A)`（主要来自时间线日志）；  
  若不保留日志，仅保留协议状态则为 `O(1)`。

## R08

边界与异常处理：
- `total_packets <= 0`：抛 `ValueError`；
- `timeout <= 0`：抛 `ValueError`；
- `data_delay <= 0` 或 `ack_delay <= 0`：抛 `ValueError`；
- `max_ticks <= 0`：抛 `ValueError`；
- 在 `max_ticks` 内未完成全量发送确认：抛 `RuntimeError`；
- 若断言检测到交付序列或统计不符：抛 `AssertionError`。

## R09

MVP 设计取舍：
- 使用 `numpy + dataclasses`，工具栈小、可复现、可直接读源码；
- 不调用黑箱网络仿真器，状态机逻辑全部显式展开；
- 丢包模型采用确定性计划（不是随机信道），便于固定断言；
- 不包含拥塞控制、RTT 自适应 RTO、流水线窗口等复杂机制，聚焦停等协议本体。

## R10

`demo.py` 函数职责：
- `validate_parameters`：参数合法性检查；
- `payload_for_packet`：确定性生成测试载荷；
- `transmit_packet`：发送/重传并注入数据帧丢失；
- `simulate_stop_and_wait`：停等协议主仿真循环；
- `run_demo`：固定参数、执行断言、打印时间线；
- `main`：无交互脚本入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0365-停等协议
uv run python demo.py
```

脚本不读取交互输入，直接输出仿真日志和校验结果。

## R12

输出说明：
- `config`：本次仿真参数；
- `attempts_per_packet`：每个分组发送次数；
- `delivered_payloads`：最终交付序列；
- `Sender TX timeline`：发送端逐时刻发送/重传日志；
- `Receiver RX timeline`：接收端逐时刻判定日志；
- `ACK timeline`：ACK 发射/到达/接收状态；
- `Stats`：
1. `total_packets`：分组总数；
2. `total_tx`：总发送次数；
3. `retransmissions`：重传次数；
4. `timeout_retransmissions`：由超时触发的重传次数；
5. `data_drops`：数据帧丢失次数；
6. `ack_drops`：ACK 丢失次数；
7. `completion_time`：完成时刻；
8. `goodput`：有效发送占比；
- `All checks passed.`：所有内置断言通过。

## R13

最小验证清单（内置断言）：
- `delivered_payloads == payloads_sent`；
- `attempts_per_packet == [1,2,2,1,3,1]`；
- `retransmissions == 4`；
- `data_drops == 2`；
- `ack_drops == 2`。

## R14

当前 demo 固定参数：
- `total_packets = 6`
- `timeout = 4`
- `data_delay = 1`
- `ack_delay = 1`
- `data_drop_plan = {(1,1):True, (4,1):True}`
- `ack_drop_plan = {(2,1):True, (4,1):True}`

该参数覆盖：
- 数据帧丢失导致的超时重传；
- ACK 丢失导致的重复帧到达；
- 接收端去重（不重复交付）逻辑。

## R15

与其他 ARQ 机制对比：
- Stop-and-Wait（本实现）：实现最简单，但链路利用率低；
- Go-Back-N：可流水线发送，但丢一个包可能回退重传多个；
- Selective Repeat：按包确认与选择重传，吞吐更高但状态管理更复杂。

## R16

适用场景：
- 计算机网络课程中讲解 ARQ 入门机制；
- 协议原型验证阶段快速检查“发送-确认-重传”闭环；
- 小规模、可解释、可复现的演示实验。

不适用场景：
- 高带宽高时延链路（停等吞吐受限明显）；
- 需要拥塞控制、乱序缓存、批量统计的工程级仿真；
- 需要真实 OS 网络栈行为复现的系统测试。

## R17

可扩展方向：
- 扩展为 Go-Back-N 或 Selective Repeat 滑动窗口；
- 将固定丢包计划替换为随机信道并做 Monte Carlo 统计；
- 增加 RTT 抖动与自适应 RTO；
- 增加帧校验（CRC）与 NACK 分支；
- 同一组信道参数下对比不同 ARQ 的 goodput 与时延。

## R18

`demo.py` 源码级算法流（8 步，非黑箱）：
1. `run_demo` 设置固定参数、数据/ACK 丢失计划，并调用 `simulate_stop_and_wait`。  
2. `simulate_stop_and_wait` 先调用 `validate_parameters`，确保超时与时延参数有效。  
3. 发送端空闲时构造当前分组（`packet_id + seq + payload`），通过 `transmit_packet` 发送并记录 `TxRecord`。  
4. 信道根据 `data_drop_plan` 决定是否丢数据帧；未丢失的数据帧在 `data_delay` 后进入接收事件队列。  
5. 接收端按 `expected_seq` 判定帧是新包还是重复包：新包交付并翻转期望序号，重复包不交付；随后发送 ACK。  
6. ACK 再经 `ack_drop_plan` 判定是否丢失；未丢 ACK 在 `ack_delay` 后到达发送端并更新 `AckRecord`。  
7. 发送端仅在 ACK 与当前等待分组精确匹配时推进到下一包；否则继续等待，超时后仅重传当前包。  
8. 全部分组完成后返回 `SimulationResult`，`run_demo` 对交付序列、发送次数与丢失统计执行断言并打印 `All checks passed.`。  

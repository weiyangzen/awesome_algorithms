# Go-Back-N协议

- UID: `CS-0213`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `366`
- 目标目录: `Algorithms/计算机-计算机网络-0366-Go-Back-N协议`

## R01

Go-Back-N（GBN）是滑动窗口 ARQ 协议：发送端可连续发送最多 `N` 个未确认分组，接收端仅接收按序到达的分组；一旦发现乱序，只确认“最后按序分组”，发送端超时后从窗口基序号开始回退重传。

本条目给出一个可运行 MVP，展示：
- 发送窗口推进（`base` / `next_packet`）；
- 累计 ACK 对窗口的推进作用；
- 丢包与 ACK 丢失下的超时回退重传；
- 固定随机种子下的可复现结果与断言。

## R02

问题定义（MVP 范围）：
- 输入参数：
1. `total_packets`：待传输分组总数；
2. `window_size`：发送窗口大小 `N`；
3. `seq_mod`：序号空间模数；
4. `timeout_slots`：超时阈值（时隙）；
5. `propagation_delay`：单程传播时延（时隙）；
6. `data_loss_prob` / `ack_loss_prob`：数据包与 ACK 丢失概率；
7. `seed`：随机种子；
8. `max_slots`：仿真上限。
- 输出结果：
1. 每个时隙的状态转移记录；
2. 最终按序交付序列；
3. 发送次数、重传次数、丢弃次数、吞吐与效率统计。

## R03

协议规则（本实现）：
- 发送端窗口约束：`next_packet < base + window_size` 时可继续发送；
- 接收端仅接受“正好等于期望分组”的数据包，其他视为乱序丢弃；
- 接收端返回累计 ACK（确认最后按序分组编号 `ack_id`）；
- 发送端收到有效累计 ACK 后推进 `base = ack_id + 1`；
- 若最老未确认分组超时，则从 `base` 到 `next_packet-1` 全部重传（Go-Back-N 核心行为）。

## R04

`demo.py` 主流程：
1. 参数校验；
2. 初始化发送端/接收端状态；
3. 按时隙循环；
4. 发送端在窗口内尽可能发送新包；
5. 信道投递本时隙到达的数据包并由接收端处理；
6. 接收端发送 ACK（可能丢失）；
7. 发送端处理到达 ACK，推进窗口；
8. 若超时则触发回退重传；
9. 记录时隙状态并检查窗口不变量；
10. 结束后做结果断言并输出统计。

## R05

关键数据结构：
- `DataPacket(packet_id, seq)`：数据分组；
- `AckPacket(ack_id, ack_seq)`：累计 ACK；
- `SenderState(base, next_packet, timer_start)`：发送端状态；
- `ReceiverState(expected_packet, expected_seq, delivered)`：接收端状态；
- `SlotRecord`：单时隙快照；
- `SimulationResult`：全局结果与统计。

## R06

正确性直觉：
- 发送端始终保持“窗口内连续发送”，提高链路利用率；
- 接收端按序接收 + 累计 ACK，保证交付序列单调有序；
- 丢包导致 ACK 推进停滞，超时触发从 `base` 回退重传，确保丢失分组最终可达；
- 因此在非 100% 丢失场景中，协议会通过“发送-确认-超时重传”闭环最终完成可靠传输。

## R07

复杂度分析（`T` 为时隙数，`W` 为窗口大小）：
- 时间复杂度：
1. 正常每时隙处理为 `O(1)` 到 `O(W)`；
2. 超时重传时会遍历窗口，单次 `O(W)`；
3. 总体可记为 `O(T * W)` 上界。
- 空间复杂度：
1. 主要来自在途包和记录表；
2. 为 `O(T + W)`。

## R08

边界与异常处理：
- `total_packets <= 0`、`window_size <= 0`、`timeout_slots <= 0`、`propagation_delay <= 0`、`max_slots <= 0`：抛 `ValueError`；
- `seq_mod <= window_size`：抛 `ValueError`（避免窗口与序号空间冲突）；
- `data_loss_prob` 或 `ack_loss_prob` 不在 `[0, 1)`：抛 `ValueError`；
- 若达到 `max_slots` 仍未完成，抛 `RuntimeError`。

## R09

MVP 设计取舍：
- 使用离散时隙仿真，不接入真实 socket，便于教学与复现；
- 用 `numpy` 的随机数生成器控制随机性，其余逻辑全部显式编码，不依赖黑箱协议库；
- 为避免序号回绕歧义，数据包携带绝对 `packet_id`，序号仅用于协议语义展示；
- 关注 GBN 主线，不扩展 CRC、比特错误模型、RTT 自适应估计等细节。

## R10

`demo.py` 函数职责：
- `validate_params`：参数合法性检查；
- `enqueue_data_packet` / `enqueue_ack_packet`：信道入队与随机丢弃；
- `split_arrivals_data` / `split_arrivals_ack`：按时隙分离到达事件；
- `process_receiver`：接收端按序接收与累计 ACK 生成；
- `process_sender_acks`：发送端 ACK 处理与窗口推进；
- `simulate_go_back_n`：协议主循环；
- `verify_result`：交付正确性与行为断言；
- `run_demo`：固定配置运行并打印结果。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0366-Go-Back-N协议
uv run python demo.py
```

脚本不需要交互输入。

## R12

输出说明：
- `Configuration`：本次仿真参数；
- `Slot timeline (first 20 slots)`：前 20 个时隙的关键状态；
- 每行字段：
1. `base/next`：发送窗口边界变化；
2. `new/rtx`：新发与重传数量；
3. `deliv/disc`：本时隙接收端交付与丢弃量；
4. `ack`：发送端本时隙吸收的有效 ACK 数；
5. `timeout`：是否触发超时重传；
- `Stats`：总时隙、总发送、重传、丢包、吞吐、效率；
- `All checks passed.`：断言通过。

## R13

最小验证清单：
- 交付序列必须等于 `0..total_packets-1`；
- 在当前非零丢包配置下必须出现至少一次重传；
- 总数据发送次数应不小于原始包数；
- 总时隙数应为正；
- 每个时隙检查不变量 `next_packet - base <= window_size`。

## R14

当前 demo 固定参数：
- `total_packets = 14`
- `window_size = 4`
- `seq_mod = 8`
- `timeout_slots = 4`
- `propagation_delay = 1`
- `data_loss_prob = 0.25`
- `ack_loss_prob = 0.20`
- `seed = 7`
- `max_slots = 300`

该组合可稳定触发：窗口推进、ACK 丢失、超时与回退重传。

## R15

与近邻协议对比：
- 停等 ARQ：实现简单但链路利用率低；
- Go-Back-N：发送端可流水发送，但乱序时会回退重传整个未确认尾段；
- 选择重传（Selective Repeat）：缓存乱序并只重传丢失包，效率更高但状态更复杂。

## R16

适用场景：
- 可靠传输机制教学；
- 协议课程中的离散事件实验；
- 与 SR、停等协议做成本/效率对比。

不适用场景：
- 需要真实网络细节（RTT 抖动、自适应超时估计、拥塞控制耦合）；
- 需要内核级 TCP 实现等价行为。

## R17

可扩展方向：
- 引入可变传播时延与抖动；
- 将超时从常数改为基于采样 RTT 的自适应估计；
- 增加接收端缓冲并升级到选择重传；
- 增加多流并发与公平性统计；
- 增加图形化（窗口轨迹、重传热度）输出。

## R18

`demo.py` 源码级算法流（8 步，非黑箱）：
1. `run_demo` 组装固定配置并调用 `simulate_go_back_n`。  
2. `simulate_go_back_n` 先通过 `validate_params` 确认参数合法，再初始化发送端、接收端和两条在途信道队列。  
3. 每个时隙先按窗口条件发送新包：`next_packet < base + window_size`，发送后必要时启动计时器。  
4. 用 `split_arrivals_data` 取出本时隙到达数据包，`process_receiver` 仅接收按序包、更新 `expected_packet/expected_seq`，并为每个到达包生成累计 ACK。  
5. ACK 经 `enqueue_ack_packet` 进入 ACK 信道，按概率被保留或丢弃。  
6. 发送端用 `split_arrivals_ack` 取 ACK，再在 `process_sender_acks` 中按累计确认推进 `base=ack_id+1`，并重置或关闭计时器。  
7. 若 `slot - timer_start >= timeout_slots` 且仍有未确认包，则触发 Go-Back-N 回退：从 `base` 到 `next_packet-1` 逐个重传。  
8. 记录 `SlotRecord`，循环结束后在 `verify_result` 里用 `numpy` 显式校验交付序列与关键约束，最后打印统计信息。  

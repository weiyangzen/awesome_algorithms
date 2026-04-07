# 滑动窗口协议

- UID: `CS-0211`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `364`
- 目标目录: `Algorithms/计算机-计算机网络-0364-滑动窗口协议`

## R01

滑动窗口协议用于在不可靠链路上提升吞吐：发送方无需“发一个等一个 ACK”，而是允许在窗口内并行发送多个分组。

本条目实现的是教学化的 **选择性重传（Selective Repeat, SR-ARQ）** MVP：
- 发送窗口内每个分组独立确认；
- 丢失的分组只重传该分组，不回退整个窗口；
- 用离散时间 tick 模型显式演示窗口滑动、超时、重传与 ACK 丢失影响。

## R02

问题定义（MVP 范围）：
- 输入：
1. 协议参数 `total_packets / window_size / timeout_ticks / propagation_delay`；
2. 首发数据包丢失集合 `data_loss_on_first_tx`；
3. 首次 ACK 丢失集合 `ack_loss_on_first_emit`。
- 输出：
1. 每个 tick 的发送基序号、接收基序号、在途包数量、当 tick 新发/重传/确认信息；
2. `sender_base`、`receiver_base`、累计重传次数的确定性序列；
3. 断言结果（最终完成性与固定轨迹一致性）。

## R03

核心协议规则（SR-ARQ 简化版）：
- 发送端维护：`sender_base`、`next_seq`、每包发送时间、每包 ACK 状态；
- 接收端维护：`receiver_base`、接收缓存位图 `received[]`；
- 发送约束：仅允许 `seq in [sender_base, sender_base + window_size)` 的分组在途；
- 接收行为：窗口内分组可乱序接收并缓存，随后尝试推进 `receiver_base`；
- 确认机制：按分组独立 ACK；
- 超时机制：某个未确认分组超过 `timeout_ticks` 后，单独重传该分组。

## R04

`demo.py` 每个 tick 的执行顺序：
1. 处理到达接收端的数据包（含重复包/过期包）；
2. 接收端生成 ACK（可命中“首次 ACK 丢失”规则）；
3. 处理到达发送端的 ACK，并推进 `sender_base`；
4. 扫描发送窗口中超时分组，触发选择性重传；
5. 在窗口允许范围内发送新分组；
6. 记录 `TickRecord` 到时间线；
7. 全部分组确认且链路清空后结束。

## R05

关键数据结构：
- `ProtocolConfig`：协议参数配置；
- `TickRecord`：单个 tick 的状态快照；
- `acked: np.ndarray[bool]`：发送端确认位图；
- `received: np.ndarray[bool]`：接收端缓存位图；
- `inflight_data / inflight_acks`：在途数据和 ACK 队列（到达 tick, seq）；
- `pandas.DataFrame`：最终打印的可读时间线。

## R06

正确性直觉：
- “窗口”保证链路并行度，“独立 ACK + 独立计时器”保证局部恢复；
- 接收端缓存乱序包，避免 Go-Back-N 的整段回退；
- 当 ACK 丢失时，发送端虽会超时重传，但接收端可识别旧包并再次回 ACK，从而最终收敛；
- `sender_base == total_packets` 且 `receiver_base == total_packets` 表示端到端传输完成。

## R07

复杂度（`N=total_packets`，`T=仿真 tick 数`，`W=window_size`）：
- 时间复杂度：`O(T * W)`（每 tick 扫描一个发送窗口处理超时）；
- 空间复杂度：`O(N + W)`（确认位图、接收位图和在途队列）。

## R08

边界与异常处理：
- `total_packets <= 0`、`window_size <= 0`、`timeout_ticks <= 0` 等参数非法时抛 `ValueError`；
- 丢包配置序号越界时抛 `ValueError`；
- `max_ticks` 内未完成收敛时抛 `RuntimeError`（防止死循环）；
- 对重复到达的旧包，接收端不会重复交付，但会重发 ACK。

## R09

MVP 设计取舍：
- 采用离散 tick 仿真，放弃真实字节流/中断/内核队列细节，换取可读和可验证；
- 仅使用 `numpy + pandas` 做状态序列与输出展示，工具栈小、逻辑透明；
- 将丢包建模为“首次发送/首次 ACK 固定丢失”，保证复现实验结果可重复；
- 不引入第三方协议栈黑箱库，窗口推进与重传策略全部源码显式实现。

## R10

`demo.py` 函数职责：
- `validate_inputs`：校验配置与丢包集合；
- `_emit_ack`：按规则发 ACK（含首次 ACK 丢失）；
- `_send_data_packet`：发送/重传数据包（含首次数据包丢失）；
- `simulate_sliding_window_sr`：主仿真循环；
- `records_to_dataframe`：时间线表格化；
- `run_demo`：固定样例、断言与打印；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0364-滑动窗口协议
uv run python demo.py
```

脚本无需任何输入，直接执行固定实验并输出结果。

## R12

输出说明：
- `Tick timeline`：每个 tick 的窗口位置、在途包、新发/重传、ACK 与事件说明；
- `sender_base series`：发送端滑窗基序号轨迹；
- `receiver_base series`：接收端滑窗基序号轨迹；
- `cumulative_retransmissions series`：累计重传轨迹；
- `All checks passed.`：全部断言通过。

## R13

最小验证清单：
- 最终条件：`sender_base` 与 `receiver_base` 都到达 `total_packets`；
- 轨迹条件：`sender_base / receiver_base / retrans` 序列与预期完全一致；
- 重传条件：重传发生在固定 tick `[3, 5, 8]`；
- 次数条件：总重传次数为 `3`（2 次数据首发丢失 + 1 次 ACK 首次丢失触发的超时重传）。

## R14

固定实验参数：
- `total_packets = 8`
- `window_size = 4`
- `timeout_ticks = 3`
- `propagation_delay = 1`
- `data_loss_on_first_tx = {2, 6}`
- `ack_loss_on_first_emit = {4}`

该配置覆盖三类关键路径：
- 数据首发丢失后的选择性重传（`seq=2,6`）；
- ACK 丢失导致的“冗余重传但可恢复”（`seq=4`）；
- 乱序缓存后批量推进接收基序号。

## R15

与相关协议差异：
- 停等协议（Stop-and-Wait）：并行度最低，每次只允许一个未确认分组；
- 回退 N 帧（Go-Back-N）：单个丢包可能触发后续整段重传；
- 选择性重传（本条目）：只重传超时分组，链路利用率通常更高；
- TCP 滑窗：工程实现更复杂（字节流、拥塞控制、重排序队列等），本实现仅保留 ARQ 核心思想。

## R16

适用场景：
- 网络协议教学中演示“滑动窗口 + 选择性重传”；
- 离散事件仿真中快速验证窗口/超时策略；
- 作为后续实现 Go-Back-N、TCP 细化模型的基线。

不适用场景：
- 需要与真实 TCP 内核行为逐字节对齐；
- 需要拥塞控制、公平性、多流竞争、带宽估计等复杂机制。

## R17

可扩展方向：
- 增加随机丢包与抖动，做多轮统计（吞吐、时延、重传率）；
- 引入 ACK 压缩、乱序上限、接收缓存容量限制；
- 对比 SR 与 Go-Back-N 在不同丢包率下的效率差异；
- 将当前单链路扩展为多流共享瓶颈链路。

## R18

`demo.py` 源码级算法流（8 步，非黑箱）：
1. `main` 调用 `run_demo`，构造固定协议参数和两个丢失集合。  
2. `run_demo` 调 `simulate_sliding_window_sr`，先在 `validate_inputs` 中检查参数合法性与序号范围。  
3. 初始化发送端位图 `acked`、接收端位图 `received`、窗口指针 `sender_base/next_seq/receiver_base`、在途队列与发送时间表。  
4. 每个 tick 先消费 `inflight_data`：接收端判断是否在窗口内，写入缓存并通过 `_emit_ack` 产生 ACK（或按规则丢弃首次 ACK）。  
5. 再消费 `inflight_acks`：发送端将对应分组标记已确认，并按连续确认结果推进 `sender_base`。  
6. 扫描当前发送窗口中未确认且超时的分组，调用 `_send_data_packet` 做选择性重传并累计重传计数。  
7. 在窗口余量允许下发送新分组，更新 `next_seq` 与在途数据队列；将本 tick 状态写入 `TickRecord`。  
8. 仿真结束后返回三条 `numpy` 序列；`run_demo` 做精确断言、打印 `pandas` 时间线，最终输出 `All checks passed.`。

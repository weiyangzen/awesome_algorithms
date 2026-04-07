# 滑动窗口协议

- UID: `CS-0211`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `364`
- 目标目录: `Algorithms/计算机-计算机网络-0364-滑动窗口协议`

## R01

滑动窗口协议（Sliding Window Protocol）是一类可靠传输机制，用有限大小的“发送窗口”与“接收窗口”实现流水线传输。与停等协议相比，它允许在未收到前一帧 ACK 时继续发送后续帧，从而显著提升链路利用率。

本条目 MVP 采用 **Go-Back-N（GBN）** 变体：
- 发送方维护窗口 `[base, base + window_size - 1]`。
- 接收方只接收按序到达的数据。
- ACK 为累计确认（`ACK=k` 表示 `0..k` 均已按序收到）。
- 超时后发送方回退并重传 `[base, next_seq-1]`。

## R02

问题目标：在存在数据包与 ACK 丢失的信道上，保证数据最终按序、无缺失地送达。

最小化描述：
- 输入：`N` 个待发送帧、窗口大小 `W`、丢包概率、超时阈值。
- 输出：传输日志与统计信息，验证协议可在非理想信道中完成可靠传输。
- 约束：不依赖交互输入；应可直接 `uv run python demo.py` 运行。

## R03

核心机制分为 4 个并发逻辑：
- **发送滑动**：只要 `next_seq < base + W` 就允许继续发新帧。
- **累计确认**：收到较大 ACK 会一次性推进 `base`，窗口整体右移。
- **乱序处理**：接收方对乱序帧丢弃并回 ACK 最近连续确认点。
- **超时重传**：若最老未确认帧超时，重传窗口内所有未确认帧。

## R04

本实现使用离散时间（tick）仿真，主要状态变量如下：
- 发送方：`base`、`next_seq`、`timer_start`。
- 接收方：`receiver_expected`（下一个期望的序号）。
- 事件队列：`data_events[tick]`、`ack_events[tick]`。
- 统计量：发送总次数、重传次数、超时次数、丢包次数等。

这种事件驱动方式比“立即送达”的简化模型更能展示窗口与 RTT 的关系。

## R05

高层伪代码（GBN）：

```text
while base < total_frames:
    处理本 tick 到达的 DATA
    处理本 tick 到达的 ACK

    while next_seq < total_frames and next_seq < base + window_size:
        发送 next_seq
        next_seq += 1

    if oldest_unacked_timeout:
        for s in [base, next_seq):
            重传 s
```

接收方逻辑：

```text
if seq == expected:
    accept(seq)
    expected += 1
    send ACK(expected - 1)
else:
    drop(seq)
    send ACK(expected - 1)
```

## R06

正确性直觉：
- 任何未被确认的最小序号始终由 `base` 标识。
- `base` 只会在收到 `ACK >= base` 时单调递增，不会回退。
- 丢包不会导致永久停滞，因为超时会触发重传。
- 接收方只按序交付，保证输出流有序。

因此，当丢包概率小于 1 且超时机制可重复触发时，协议会以概率 1 最终完成传输（在该随机模型下）。

## R07

复杂度（单次仿真，设发送总尝试次数为 `S`）：
- 时间复杂度：`O(S)`，每次发送/接收/确认事件处理为 `O(1)`。
- 空间复杂度：`O(W + E)`，`W` 为窗口规模，`E` 为暂存的在途事件数量。

说明：由于重传存在，`S` 可大于原始帧数 `N`。

## R08

与停等协议（Stop-and-Wait）对比：
- 停等：任意时刻最多 1 个在途帧，吞吐易受 RTT 限制。
- 滑动窗口：允许最多 `W` 个在途帧，能更好填满“带宽-时延积”。
- 代价：状态管理、计时器与重传控制更复杂。

## R09

与选择重传（Selective Repeat, SR）对比：
- GBN（本实现）：接收方不缓存乱序帧，发送方超时后回退重传一段。
- SR：接收方可缓存乱序帧，发送方只重传真丢失帧。
- 结论：GBN 实现更简单，但在高丢包场景下可能产生更多冗余重传。

## R10

`demo.py` 的实现边界：
- 仅使用 Python 标准库（`dataclasses`, `random`, `typing`）。
- 非交互、固定随机种子，保证结果可复现。
- 采用离散 tick + 事件队列，显式模拟：发送、在途传播、接收、ACK 返回、超时重传。

## R11

默认参数（`main()` 内）：
- `total_frames=12`：要发送的帧总数。
- `window_size=4`：发送窗口大小。
- `timeout_ticks=4`：最老未确认帧超时阈值。
- `propagation_delay=1`：单向传播时延（tick）。
- `data_loss_prob=0.25`：数据帧丢失概率。
- `ack_loss_prob=0.15`：ACK 丢失概率。
- `seed=7`：随机种子。

可自行修改上述参数观察不同网络条件下的行为变化。

## R12

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0364-滑动窗口协议
uv run python demo.py
```

程序会输出：
- 分 tick 的事件日志
- 最终统计汇总
- 完成性检查（若未完成会抛异常）

## R13

输出重点解读：
- `发送方首次发送 seq=x`：新数据进入信道。
- `DATA 丢失` / `ACK 丢失`：模拟不可靠链路。
- `窗口左边界 a -> b`：累计确认推动窗口滑动。
- `超时触发：重传区间 [l, r]`：GBN 回退重传行为。
- `最终 base == total_frames`：表示全部帧可靠送达。

## R14

建议关注的关键观测指标：
- 可靠性：`最终 base` 是否等于 `total_frames`。
- 重传开销：`retransmissions / sent_total`。
- 信道质量影响：`data_dropped` 与 `ack_dropped`。
- 协议效率：`timeouts` 次数与完成总 tick。

通过调小丢包率、增大窗口，通常可观察到更高的有效吞吐。

## R15

边界与鲁棒性讨论：
- `window_size=1` 时退化为停等式行为。
- 高丢包率下可能需要更多 tick 才能完成，`max_ticks` 过小会提前终止。
- ACK 重复/过期是正常现象，发送方会忽略不推进窗口的 ACK。

本 MVP 已处理上述边界，不会因为重复 ACK 导致状态回退。

## R16

当前 MVP 的简化假设：
- 未实现序号回绕（wrap-around）。
- 单连接、单方向数据流。
- 计时器模型为“单一最老未确认帧定时器”。
- 未引入拥塞控制（如慢启动/拥塞避免）。

这些是教学模拟中的常见取舍，不影响理解滑动窗口核心思想。

## R17

工程映射（到 TCP 语境）：
- GBN 的“累计 ACK + 超时重传”可对应 TCP 的基础可靠传输思想。
- 实际 TCP 还叠加了：快速重传、快速恢复、拥塞控制、流量控制、选择确认（SACK）等。
- 因此本例适合做“可靠传输机制”的第一性演示，而非 TCP 全量行为仿真。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main()` 构造 `GoBackNSimulator`，注入窗口、超时、丢包率和随机种子。
2. `run()` 进入按 tick 推进的主循环，并记录当前 `base/next_seq/receiver_expected`。
3. `_process_data_arrivals()` 处理到达接收方的数据：按序则接收并前移 `receiver_expected`，乱序则丢弃。
4. 接收方针对每个到达数据生成累计 ACK，并按 `ack_loss_prob` 决定是否丢失；未丢失则排入 `ack_events`。
5. `_process_ack_arrivals()` 处理 ACK：若 `ack >= base`，发送方把 `base` 推进到 `ack+1`，实现窗口滑动。
6. `_send_new_data_within_window()` 在 `next_seq < base + window_size` 条件下持续发新帧，并把成功发送的数据排入未来 tick 的 `data_events`。
7. `_maybe_timeout_and_retransmit()` 检查最老未确认帧是否超时，若超时则重传区间 `[base, next_seq-1]`（Go-Back-N 回退）。
8. 当 `base == total_frames` 时主循环结束，输出日志与统计；若未完成则抛异常提示参数不合适。

该实现未把第三方库当黑盒，协议状态流转全部在源码中可追踪、可单步验证。

# 流量控制算法

- UID: `CS-0215`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `368`
- 目标目录: `Algorithms/计算机-计算机网络-0368-流量控制算法`

## R01

流量控制（Flow Control）的目标是让发送端发送速率不超过接收端处理能力，防止接收缓冲区溢出。该问题关注“端到端接收能力匹配”，而不是网络路径拥塞本身。

## R02

典型场景是传输层（如 TCP）中接收端通过通告窗口（`rwnd`）告诉发送端“还能接收多少字节”，发送端据此限制在途未确认数据量。

## R03

本 MVP 的输入参数：
- 总发送数据量 `total_data`
- 每个 tick 的链路容量 `link_capacity_per_tick`
- 接收缓冲区容量 `receiver_buffer_capacity`
- 应用读取函数 `app_read_fn(tick)`

输出：
- 每个 tick 的发送/确认/缓冲区/窗口/阻塞记录
- 总吞吐、阻塞 tick 数、是否丢包等汇总指标

## R04

约束与目标：
- 发送端始终满足 `unacked_bytes <= advertised_window`
- 接收端缓冲区占用不超过 `receiver_buffer_capacity`
- 在不溢出的前提下尽可能提高链路利用率

## R05

核心状态变量：
- `unacked_bytes`：发送端未确认字节数
- `receiver_buffer_used`：接收端缓冲区当前占用
- `advertised_window`：接收端通告窗口（剩余可写空间）
- `in_flight[tick]`：计划在未来某 tick 到达的数据量

## R06

离散时间迭代（每个 tick）流程：
1. 处理到达接收端的数据，写入缓冲区。
2. 将已接收数据确认到发送端，减少 `unacked_bytes`。
3. 接收端应用层从缓冲区读取数据。
4. 重新计算并通告窗口 `advertised_window`。
5. 发送端按 `min(链路容量, 剩余数据, 窗口允许量)` 发送新数据。

## R07

正确性直觉：
- 发送端发送上限由 `window_allowance = max(0, advertised_window - unacked_bytes)` 决定。
- 该约束确保发送后未确认数据不超过接收端可承载空间。
- 因此只要窗口计算与 ACK 更新一致，缓冲区不会因发送端超发而系统性溢出。

## R08

复杂度：
- 时间复杂度：`O(T)`，`T` 为运行 tick 数。
- 空间复杂度：`O(T_arrival)`，由在途到达映射 `in_flight` 的活跃键数量决定（本实现中常数级上界很小）。

## R09

参数建议：
- `receiver_buffer_capacity` 越小，流控触发越频繁。
- `app_read_fn` 越慢，窗口收缩越明显，发送端更易阻塞。
- `link_capacity_per_tick` 越大，越容易暴露接收端处理瓶颈。

## R10

本目录 MVP 设计选择：
- 使用 Python 标准库实现，避免外部依赖。
- 使用 1 tick 传播延迟，保留最小网络时序特征。
- 应用读取速率使用周期函数，便于稳定复现实验结果。

## R11

运行方式：

```bash
uv run python demo.py
```

无需交互输入，脚本会直接输出统计摘要与前 15 个 tick 的明细。

## R12

输出指标解释：
- `total_sent`：发送端累计发出的字节。
- `total_acked`：发送端累计收到确认的字节。
- `blocked_ticks_by_flow_control`：因窗口为 0 或窗口不足导致无法发送的 tick 数。
- `link_utilization`：发送字节 / 理论可发送字节。

## R13

结果解读重点：
- 当应用读取变慢时，`buffer_used` 升高，`advertised_window` 降低。
- 窗口下降会直接压缩 `send_now`，出现 `blocked_by_flow_control=True`。
- 当应用恢复读取后，窗口回升，发送端恢复吞吐。

## R14

边界与异常处理：
- 读取预算若为负，按 0 处理。
- 发送剩余数据为 0 时不再发送。
- 运行在 `max_ticks` 内未完成时会自然停止，便于防止无限仿真。

## R15

与拥塞控制的区别：
- 流量控制解决“接收端来不及处理”的问题。
- 拥塞控制解决“网络路径排队和丢包过多”的问题。
- 在真实 TCP 中二者共同生效，发送窗口通常受 `min(cwnd, rwnd)` 约束。

## R16

可扩展方向：
- 增加可变 RTT 与 ACK 延迟，模拟更真实时序。
- 引入拥塞窗口 `cwnd`，联合分析流控与拥塞控制。
- 将 `app_read_fn` 替换为真实应用 trace，做离线回放评估。

## R17

最小验证清单：
- 脚本可直接运行且无交互输入。
- 输出中 `total_acked` 最终接近或等于 `total_data`。
- 在低读取速率配置下可观察到窗口收缩与阻塞 tick 增加。
- `README.md` 与 `demo.py` 不包含任何模板占位符。

## R18

源码级算法流程（对应 `demo.py`）：
1. `main()` 初始化仿真参数（总数据量、链路容量、接收缓冲区、读取模式）并创建 `FlowControlSimulator`。
2. `run()` 进入 tick 循环，先从 `in_flight` 取出当前到达数据 `arriving`。
3. 使用 `free_space = buffer_capacity - buffer_used` 计算可写空间，得到 `accepted` 与 `dropped`，更新接收缓冲区占用。
4. 将 `accepted` 作为 ACK 生效，更新 `acked_total` 与发送端 `unacked_bytes`。
5. 调用 `app_read_fn(tick)` 模拟应用读取，更新 `receiver_buffer_used`，再计算新 `advertised_window`。
6. 发送端计算 `window_allowance = max(0, advertised_window - unacked_bytes)`，并以 `send_now = min(link_capacity, remaining, window_allowance)` 发送新数据。
7. 若 `send_now > 0`，将其登记到 `in_flight[tick+1]`，并记录 `TickRecord`；若 `remaining > 0` 且 `send_now == 0`，标记流控阻塞。
8. 当 `acked_total >= total_data` 且无在途数据时提前结束，返回 `SimulationResult`，由 `print_report()` 生成可读结果。

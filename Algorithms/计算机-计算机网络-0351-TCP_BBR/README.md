# TCP BBR

- UID: `CS-0204`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `351`
- 目标目录: `Algorithms/计算机-计算机网络-0351-TCP_BBR`

## R01

TCP BBR（Bottleneck Bandwidth and RTT）是模型驱动拥塞控制算法。核心思想不是“丢包后再退让”，而是持续估计两类网络模型量：
- `BtlBw`：瓶颈带宽（单位时间可交付数据上限）；
- `RTprop`：传播时延下界（近似无排队时 RTT）。

发送端根据这两个估计量计算 BDP，并用 `pacing_rate` 与 `cwnd` 控制发送速率和在途数据量。

## R02

本条目给出一个可运行、可解释的 BBR 教学 MVP，目标是：
- 用离散 RTT 轮次演示 BBR 四个关键状态：`STARTUP / DRAIN / PROBE_BW / PROBE_RTT`；
- 显式实现 `BtlBw` 最大滤波与 `RTprop` 最小滤波；
- 通过固定链路容量变化场景，观察带宽下降与恢复时的自适应行为；
- 用固定断言保证 `uv run python demo.py` 可复现并自动校验。

## R03

问题定义（MVP 范围）：
- 输入：
1. `rounds`：仿真轮数；
2. `capacities`：每轮链路瓶颈容量（packets/RTT）；
3. `base_rtt_ms`：基础传播时延。
- 输出：
1. 每轮发送、交付、排队、RTT 样本；
2. 每轮 `state_before/state_after`；
3. 每轮 `BtlBw/RTprop/BDP/cwnd`；
4. 固定断言结果（吞吐变化、状态覆盖、RTT/队列合理性）。

## R04

核心控制关系（MVP 公式）：
- `BDP = BtlBw * RTprop`；
- `pacing_rate = pacing_gain * BtlBw`；
- `send_budget ≈ pacing_rate * RTprop`（`PROBE_RTT` 时改为最小窗口预算）；
- `send_packets = min(cwnd, send_budget)`；
- 队列与 RTT：
1. `available = queue + send`；
2. `delivered = min(available, capacity)`；
3. `queue = available - delivered`；
4. `rtt = base_rtt + (queue/capacity)*base_rtt`。

## R05

状态机与增益策略：
- `STARTUP`：`pacing_gain = 2.77`，快速探测带宽；
- `DRAIN`：`pacing_gain = 1/2.77`，尝试排空队列；
- `PROBE_BW`：循环增益 `[1.25, 0.75, 1, 1, 1, 1, 1, 1]`；
- `PROBE_RTT`：短暂降到 `min_cwnd`，刷新时延下界。

转移条件（简化实现）：
- `STARTUP -> DRAIN`：连续 3 轮带宽估计增长不足 25%；
- `DRAIN -> PROBE_BW`：在途数据（这里用 `queue` 近似）低于 `BDP`；
- `PROBE_BW <-> PROBE_RTT`：按固定轮次触发短期探测。

## R06

模型估计器：
- `BtlBw`：最近 `bw_window=8` 个交付速率样本的最大值（max filter）；
- `RTprop`：最近 `rtprop_window_rounds=20` 个 RTT 样本的最小值（min filter）。

这对应 BBR 的“带宽看峰值、时延看下界”原则。

## R07

正确性直觉：
- `STARTUP` 通过高增益把发送速率快速推向瓶颈附近；
- 一旦带宽估计不再显著上升，判定“管道接近填满”，切到 `DRAIN` 降低排队；
- `PROBE_BW` 周期性小幅激进/保守发送，持续重估瓶颈能力；
- `PROBE_RTT` 周期性压低在途量，避免 `RTprop` 长期被队列时延污染。

## R08

复杂度（设轮数为 `T`）：
- 时间复杂度：`O(T)`；
- 空间复杂度：`O(T)`（保存完整记录）+ `O(1)` 滤波窗口状态。

## R09

边界与异常处理：
- `rounds <= 0`：抛 `ValueError`；
- `len(capacities) != rounds`：抛 `ValueError`；
- `base_rtt_ms <= 0`：抛 `ValueError`；
- `capacities` 含非正值：抛 `ValueError`；
- 状态字符串异常：抛 `ValueError`（控制器内部保护）。

## R10

`demo.py` 结构与职责：
- `RoundRecord` / `SimulationResult`：承载仿真全过程数据；
- `BBRController`：实现状态机、模型估计、发送预算与窗口更新；
- `build_capacity_schedule`：生成固定瓶颈变化场景；
- `simulate_bbr`：执行主循环并记录每轮状态；
- `run_checks`：做确定性断言；
- `print_report`：输出摘要与关键轮次明细；
- `run_demo` / `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0351-TCP_BBR
uv run python demo.py
```

脚本无需输入，直接运行仿真并输出校验结果。

## R12

输出说明：
- 概览指标：`avg_delivered_packets`、`max_queue_packets`、`min/max RTT`、状态集合；
- `First 18 rounds`：观察从 `STARTUP` 到后续状态的早期动态；
- `Last 8 rounds`：观察容量变化后后期稳态行为；
- `All checks passed.`：全部断言通过。

## R13

最小验证清单（`run_checks`）：
- 四个状态都至少出现一次；
- 容量下降区间平均交付量低于早期区间；
- 容量恢复后平均交付量高于下降区间；
- 所有 RTT 样本不低于基础传播时延；
- 至少出现一次正排队（说明探测行为发生）。

## R14

当前固定实验参数：
- `rounds = 48`；
- `base_rtt_ms = 50`；
- `capacity schedule`：
1. 轮 `1-18`：`120 packets/RTT`；
2. 轮 `19-30`：`80 packets/RTT`（降容）；
3. 轮 `31-48`：`140 packets/RTT`（恢复并提升）。

该场景可同时覆盖状态切换与带宽变化响应。

## R15

与 Reno/CUBIC 的差异（概念层）：
- Reno 更依赖丢包事件驱动窗口调节；
- CUBIC 使用立方函数控制窗口增长；
- BBR 用速率和时延模型直接驱动 `pacing + cwnd`。

本 MVP 仍是“教学近似”，不等价 Linux 内核 BBRv1/v2 逐 ACK 精确行为。

## R16

适用场景：
- 讲解 BBR 核心机制与状态机；
- 快速演示模型估计与发送控制耦合；
- 为更精细网络仿真提供简化原型。

不适用场景：
- 需要内核级逐包时序复现；
- 需要与真实 NIC/队列管理器严格对齐；
- 需要 BBRv2 细节（inflight_hi/loss response 等）。

## R17

可扩展方向：
- 从按 RTT 轮次升级到按 ACK 事件驱动；
- 引入随机丢包、ACK 压缩、可变 RTT 抖动；
- 支持多流竞争并统计公平性；
- 对比 BBR/Reno/CUBIC 在同一拓扑下的吞吐与排队时延。

## R18

`demo.py` 源码级算法流（9 步，非黑箱）：
1. `run_demo` 固定仿真参数，并通过 `build_capacity_schedule` 生成三段式容量轨迹。  
2. `simulate_bbr` 初始化 `BBRController` 和队列状态 `queue_packets`。  
3. 每轮开始先调用 `maybe_enter_probe_rtt`，按周期判断是否进入 `PROBE_RTT`。  
4. 控制器基于当前状态给出 `pacing_gain`、`pacing_rate`、`send_packets`（`plan_send_packets`）。  
5. 网络侧用 `available=queue+send` 与 `delivered=min(available, capacity)` 计算交付量，再更新队列。  
6. 由队列长度计算 RTT 样本：`rtt=base_rtt + queue_delay`，形成本轮观测信号。  
7. `on_round_end` 内先更新 `BtlBw` 最大滤波和 `RTprop` 最小滤波，再更新 `cwnd` 与状态转移。  
8. 记录 `RoundRecord`（前后状态、发送/交付、模型估计、窗口演化），构成完整时间线。  
9. `run_checks` 做状态覆盖与吞吐变化断言，`print_report` 输出摘要与关键轮次，最终打印 `All checks passed.`。  

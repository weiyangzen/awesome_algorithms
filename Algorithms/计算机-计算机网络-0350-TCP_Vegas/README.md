# TCP Vegas

- UID: `CS-0203`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `350`
- 目标目录: `Algorithms/计算机-计算机网络-0350-TCP_Vegas`

## R01

TCP Vegas 是典型的基于时延（delay-based）拥塞控制算法。与 Reno 等“以丢包为拥塞信号”的方案不同，Vegas 通过监控 `RTT` 相对 `BaseRTT` 的增量来估计排队程度，尽量在丢包发生前就调整发送窗口。

## R02

本条目的目标是实现一个可运行、可解释的 Vegas 教学 MVP：
- 显式实现 `Expected/Actual` 吞吐比较；
- 用 `alpha/beta/gamma` 规则驱动 `cwnd` 调整与阶段切换；
- 在固定容量变化场景中展示 Vegas 的提前减速特性；
- 通过确定性断言保证 `uv run python demo.py` 可复现。

## R03

问题定义（MVP 范围）：
- 输入：
1. `rounds`：仿真轮数（按 RTT 轮次）；
2. `capacities`：每轮瓶颈容量（packets/RTT）；
3. `propagation_rtt_ms`：传播时延基线；
4. `alpha/beta/gamma`：Vegas 控制参数。
- 输出：
1. 每轮 `cwnd`、状态、发送/交付、排队、RTT；
2. 每轮 `Expected/Actual/diff` 估计值；
3. 断言结果与摘要报表。

## R04

核心公式（源码中显式实现）：
- `BaseRTT = min(observed RTT)`；
- `Expected = cwnd / BaseRTT`；
- `Actual = cwnd / RTT`；
- `diff = (Expected - Actual) * BaseRTT`。

队列与 RTT 的网络侧模型：
1. `available = queue + send`；
2. `delivered = min(available, capacity)`；
3. `queue = available - delivered`；
4. `RTT = propagation_rtt + (queue/capacity)*propagation_rtt`。

## R05

控制规则（教学化简版）：
- `SLOW_START`：
1. 若 `diff > gamma`，退出到 `CONGESTION_AVOIDANCE`；
2. 否则 `cwnd *= 2`（按 RTT 轮次近似慢启动）。
- `CONGESTION_AVOIDANCE`：
1. `diff < alpha`：`cwnd += 1`（链路可能未填满）；
2. `diff > beta`：`cwnd -= 1`（排队过多，应减速）；
3. 否则保持窗口。

## R06

参数语义：
- `alpha`：期望的低排队阈值；
- `beta`：允许的高排队阈值；
- `gamma`：慢启动提前退出阈值。

本 MVP 默认 `alpha=1, beta=3, gamma=1`，与常见 Vegas 教学参数一致，便于观察“队列目标区间控制”。

## R07

正确性直觉：
- 若 `RTT` 接近 `BaseRTT`，说明排队轻，`Actual` 接近 `Expected`，应继续增长窗口；
- 若 `RTT` 持续高于 `BaseRTT`，`Actual` 下降，`diff` 增大，窗口应回调；
- 相比丢包驱动算法，Vegas 更早感知拥塞并尝试把队列压在较小区间。

## R08

复杂度（设轮数为 `T`）：
- 时间复杂度：`O(T)`；
- 空间复杂度：`O(T)`（保存完整轮次记录）。

## R09

边界与异常处理：
- `rounds <= 0`：抛 `ValueError`；
- `len(capacities) != rounds`：抛 `ValueError`；
- `propagation_rtt_ms <= 0`：抛 `ValueError`；
- `capacities` 含非正值：抛 `ValueError`；
- `alpha >= beta` 或阈值非正：抛 `ValueError`；
- 未知状态字符串：抛 `ValueError`。

实现未依赖任何拥塞控制黑箱库，窗口与状态转移逻辑均在 `demo.py` 中逐行可追踪。

## R10

`demo.py` 结构与职责：
- `RoundRecord / SimulationResult`：保存全量过程数据；
- `VegasController`：实现 BaseRTT 估计、`diff` 计算、窗口更新；
- `build_capacity_schedule`：构造三阶段容量轨迹；
- `simulate_tcp_vegas`：执行主仿真循环；
- `run_checks`：做确定性正确性断言；
- `records_to_dataframe / print_report`：输出可读报表；
- `run_demo / main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0350-TCP_Vegas
uv run python demo.py
```

脚本无需交互输入。

## R12

输出说明：
- 概览指标：平均交付量、最大队列、RTT 范围、平均 `diff`、状态集合；
- `First 16 rounds`：观察慢启动与首次拥塞感知；
- `Last 8 rounds`：观察后期恢复与稳定阶段；
- `All checks passed.`：表示全部断言通过。

## R13

最小验证清单（`run_checks`）：
- 至少出现 `SLOW_START` 与 `CONGESTION_AVOIDANCE` 两个状态；
- 容量下降阶段吞吐均值低于早期阶段；
- 后期吞吐均值高于下降阶段；
- RTT 不低于传播时延下界；
- 至少出现一次正队列（保证 Vegas 有时延信号可观测）；
- `cwnd` 始终处于配置边界 `[1, 256]` 内。

## R14

固定实验参数：
- `rounds = 48`；
- `propagation_rtt_ms = 50`；
- 容量轨迹（packets/RTT）：
1. 前 1/3 轮：`30`；
2. 中 1/3 轮：`20`（降容）；
3. 后 1/3 轮：`35`（恢复并略升）。

该场景可稳定触发 Vegas 的“排队感知 -> 调窗 -> 恢复”过程。

## R15

与相关算法差异：
- Reno/NewReno：主要依赖丢包或重复 ACK 触发回退；
- Vegas：依赖 RTT 偏移估计排队，倾向更早减速；
- BBR：显式建模瓶颈带宽与传播时延，控制目标不同于 Vegas 的 `alpha/beta` 排队窗口。

## R16

适用场景：
- 讲解 delay-based 拥塞控制核心思想；
- 课程/实验中快速对比 Vegas 与 loss-based 行为差异；
- 作为更复杂网络仿真前的最小原型。

不适用场景：
- 需要内核 TCP 逐 ACK、逐计时器细节复现；
- 需要包含 SACK、乱序、真实 NIC/队列管理等生产级机制。

## R17

可扩展方向：
- 从按 RTT 轮次升级到按 ACK 事件驱动；
- 引入随机丢包、ACK 压缩、抖动 RTT；
- 增加多流竞争并评估公平性与时延分位数；
- 与 Reno/CUBIC/BBR 在同一拓扑下做可视化对比实验。

## R18

`demo.py` 源码级算法流（9 步，非黑箱）：
1. `run_demo` 固定参数并调用 `build_capacity_schedule` 生成三段容量轨迹。  
2. `simulate_tcp_vegas` 初始化 `VegasController` 与网络队列 `queue_packets`。  
3. 每轮先按 `send=cwnd` 与 `capacity` 计算 `delivered` 和新的排队长度。  
4. 由排队长度计算 RTT 样本，并在控制器中更新 `BaseRTT=min(BaseRTT, RTT)`。  
5. 控制器用 `Expected=cwnd/BaseRTT` 与 `Actual=cwnd/RTT` 计算 `diff`。  
6. 若处于慢启动且 `diff>gamma`，状态切到 `CONGESTION_AVOIDANCE`；否则继续指数增窗。  
7. 在拥塞避免阶段按 `diff<alpha` 增窗、`diff>beta` 减窗、区间内保持不变。  
8. 每轮把前后状态、窗口、吞吐、排队、RTT、`diff` 写入 `RoundRecord`。  
9. `run_checks` 执行吞吐/RTT/边界断言，`print_report` 打印表格摘要并输出 `All checks passed.`。

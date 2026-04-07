# TCP拥塞控制 - 慢启动

- UID: `CS-0196`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `343`
- 目标目录: `Algorithms/计算机-计算机网络-0343-TCP拥塞控制_-_慢启动`

## R01

TCP 慢启动（Slow Start）是经典拥塞控制阶段，用于连接建立后或出现丢包恢复后快速探测可用带宽。
本条目提供一个可运行 MVP，目标是：
- 用离散 RTT 仿真 `cwnd`（拥塞窗口）和 `ssthresh`（慢启动阈值）的演化；
- 展示慢启动 `指数增长` 与阈值后的 `线性增长` 切换；
- 展示丢包后的乘法减小与重启慢启动；
- 提供固定断言，保证 `uv run python demo.py` 可复现。

## R02

问题定义（MVP 范围）：
- 输入：
1. `init_cwnd`：初始拥塞窗口（单位：MSS）；
2. `init_ssthresh`：初始慢启动阈值；
3. `total_rtts`：仿真 RTT 轮数；
4. `loss_rtts`：发生丢包的 RTT 集合（1-based）。
- 输出：
1. 每个 RTT 的 `cwnd_before/cwnd_after`；
2. 每个 RTT 的 `ssthresh_before/ssthresh_after`；
3. 每个 RTT 的 `phase_before/phase_after`（`slow_start` 或 `congestion_avoidance`）。

## R03

核心规则（本 MVP 使用 Tahoe 风格简化）：
- 慢启动阶段：每个 RTT 近似 `cwnd <- min(2*cwnd, ssthresh)`；
- 到达阈值后进入拥塞避免：每个 RTT 线性增长 `cwnd <- cwnd + 1`；
- 丢包时：
1. `ssthresh <- max(cwnd // 2, 2)`；
2. `cwnd <- 1`；
3. 阶段切换回 `slow_start`。

## R04

`demo.py` 执行流程：
1. 校验输入参数合法性；
2. 初始化 `cwnd/ssthresh/phase`；
3. 逐 RTT 迭代；
4. 若 RTT 命中 `loss_rtts`，执行丢包恢复规则；
5. 否则按当前阶段执行增长规则；
6. 记录本 RTT 前后状态；
7. 结束后提取序列并做断言；
8. 打印表格化时间线与校验通过信息。

## R05

关键数据结构：
- `RTTRecord`：保存单个 RTT 的完整状态转移；
- `records: List[RTTRecord]`：仿真主结果；
- `np.ndarray`：提取窗口、阈值、阶段序列用于断言。

## R06

正确性直觉：
- 在没有丢包的慢启动阶段，窗口按 RTT 指数扩张，快速逼近可用带宽；
- `ssthresh` 把“激进探测”与“保守增长”分段，避免持续指数膨胀；
- 丢包被视为拥塞信号后，窗口重置为 1 并下调阈值，形成负反馈闭环；
- 因此系统会在“增长-拥塞-回退”之间稳定振荡而不是无限增大。

## R07

复杂度（设 RTT 轮数为 `T`）：
- 时间复杂度：`O(T)`；
- 空间复杂度：`O(T)`（保存全部 RTT 记录）。

## R08

边界与异常处理：
- `init_cwnd <= 0`：抛 `ValueError`；
- `init_ssthresh < 2`：抛 `ValueError`；
- `total_rtts <= 0`：抛 `ValueError`；
- `loss_rtts` 出现越界值（不在 `1..total_rtts`）：抛 `ValueError`；
- 阶段字符串非法：抛 `ValueError`。

## R09

MVP 设计取舍：
- 仅依赖 `numpy`，不依赖网络协议栈或抓包工具，保证最小可运行；
- 使用离散 RTT 模型而非逐 ACK 事件仿真，减少样板代码；
- 使用可解释规则而非调用外部黑箱拥塞控制库；
- 覆盖慢启动主线，不扩展 SACK、RTO 细节和现代 CUBIC 内核参数。

## R10

`demo.py` 函数职责：
- `validate_inputs`：参数校验；
- `next_window_without_loss`：无丢包时窗口演化；
- `apply_loss_reaction`：丢包时窗口/阈值回退；
- `simulate_tcp_slow_start`：主仿真循环；
- `extract_series`：从记录提取 numpy 序列；
- `theoretical_slow_start_curve`：纯慢启动理论曲线；
- `run_demo`：固定样例、断言、打印；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0343-TCP拥塞控制_-_慢启动
uv run python demo.py
```

脚本不需要任何输入，直接输出仿真时间线和校验结果。

## R12

输出说明：
- `RTT timeline`：逐行显示每轮事件、阶段、窗口和阈值变化；
- `cwnd_after sequence`：每轮结束后的拥塞窗口序列；
- `ssthresh_after sequence`：每轮结束后的阈值序列；
- `phase_after sequence`：每轮结束后的阶段序列；
- `All checks passed.`：表示断言全部通过。

## R13

最小验证清单：
- 固定样例断言 `cwnd_after` 为预期序列；
- 固定样例断言 `ssthresh_after` 为预期序列；
- 固定样例断言阶段切换序列正确；
- 对每个丢包 RTT 验证 `cwnd` 重置为 `1` 且 `ssthresh=max(cwnd_before//2,2)`；
- 无丢包场景下验证理论慢启动曲线。

## R14

当前 demo 参数：
- `init_cwnd = 1`；
- `init_ssthresh = 16`；
- `total_rtts = 12`；
- `loss_rtts = {6, 11}`。

该参数能覆盖：
- 一次完整慢启动到阈值；
- 拥塞避免线性增长；
- 两次丢包回退与再次慢启动。

## R15

与相关拥塞控制机制对比：
- TCP Tahoe：丢包后 `cwnd` 直接回到 1（本 MVP 采用该风格）；
- TCP Reno：快重传/快恢复后不一定回到 1；
- CUBIC/BBR：增长函数和带宽估计机制更复杂，目标不只是“慢启动 + AIMD”。

## R16

适用场景：
- 教学展示 TCP 基础拥塞控制机制；
- 在离散事件仿真中作为更复杂协议的子模块；
- 解释链路突发拥塞后吞吐抖动的来源。

不适用场景：
- 需要内核级精确行为复现（RTO、ACK 压缩、SACK 等）；
- 需要与真实抓包时序逐包对齐的性能诊断。

## R17

可扩展方向：
- 从按 RTT 仿真升级到按 ACK 仿真；
- 增加 Reno 快恢复状态；
- 增加随机丢包模型并统计稳态吞吐；
- 增加与 BDP（带宽时延积）关系的可视化；
- 增加多流竞争与公平性分析。

## R18

`demo.py` 源码级算法流（8 步，非黑箱）：
1. `main` 调用 `run_demo`，构造固定参数与丢包 RTT 集合。  
2. `run_demo` 调用 `simulate_tcp_slow_start`，先由 `validate_inputs` 检查参数域。  
3. 初始化状态变量 `cwnd/ssthresh/phase`，其中 `phase` 初始为 `slow_start`。  
4. 对每个 RTT：若命中 `loss_rtts`，调用 `apply_loss_reaction` 执行 `ssthresh` 折半和 `cwnd=1` 重置。  
5. 若当前 RTT 无丢包，调用 `next_window_without_loss`：慢启动阶段用 `min(2*cwnd, ssthresh)`，拥塞避免阶段用 `cwnd+1`。  
6. 把 RTT 前后状态写入 `RTTRecord`，最终形成完整时间线 `records`。  
7. `extract_series` 抽取窗口/阈值/阶段序列，与硬编码期望序列逐项断言。  
8. 再用 `theoretical_slow_start_curve` 校验无丢包理论曲线，并打印结果，成功时输出 `All checks passed.`。  

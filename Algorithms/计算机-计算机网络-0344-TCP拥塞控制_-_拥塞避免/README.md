# TCP拥塞控制 - 拥塞避免

- UID: `CS-0197`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `344`
- 目标目录: `Algorithms/计算机-计算机网络-0344-TCP拥塞控制_-_拥塞避免`

## R01

问题定义：在 TCP 连接已经进入拥塞避免（Congestion Avoidance）阶段后，发送方需要根据网络反馈（ACK 与丢包）动态调整拥塞窗口 `cwnd`，在不触发持续拥塞的前提下尽可能提高吞吐量。

## R02

核心思想：使用 AIMD（Additive Increase, Multiplicative Decrease）。
1. 加性增大：网络看起来通畅时，`cwnd` 线性增长，避免激进探测造成突发拥塞。
2. 乘性减小：检测到拥塞后，`cwnd` 快速减半，立即降低发送速率。
3. 二者结合形成“探测-回退-再探测”的锯齿形动态平衡。

## R03

最小建模变量：
1. `cwnd`：拥塞窗口，单位 MSS。
2. `ssthresh`：慢启动阈值，单位 MSS。
3. `RTT`：往返时延，作为按轮次推进的时间尺度。
4. `loss event`：该 RTT 触发丢包事件（简化为离散轮次）。
5. `ACK count`：该 RTT 预计确认的分组数量，近似为 `floor(cwnd)`。

## R04

拥塞避免阶段状态转移（本 MVP）：
1. 正常 ACK：保持在拥塞避免阶段，执行加性增大。
2. 发生丢包：执行乘性减小，更新 `ssthresh`，并将 `cwnd` 回落到新阈值附近后继续拥塞避免。
3. 本示例不模拟超时回到慢启动，仅聚焦“拥塞避免”机制本体。

## R05

关键更新公式：
1. 每个 ACK 到来时：`cwnd <- cwnd + 1/cwnd`
2. 一个 RTT 内大约收到 `cwnd` 个 ACK，因此一轮 RTT 近似：`cwnd <- cwnd + 1`
3. 发生拥塞时：`ssthresh <- max(cwnd/2, 2)`，`cwnd <- ssthresh`

这就是经典 Reno 风格拥塞避免中的 AIMD 更新规则。

## R06

伪代码（按 RTT 粒度）：

```text
initialize cwnd, ssthresh
for rtt in 1..T:
    if rtt is loss_event:
        ssthresh = max(cwnd / 2, 2)
        cwnd = ssthresh
    else:
        ack_count = max(1, floor(cwnd))
        repeat ack_count times:
            cwnd = cwnd + 1 / cwnd
    record metrics
```

## R07

正确性直觉：
1. 当网络未拥塞时，线性增窗可以持续试探可用带宽。
2. 当出现拥塞时，立刻减半可显著降低队列压力，避免拥塞塌陷。
3. 线性增长与减半回退形成稳定振荡区间，平均吞吐与稳定性之间取得工程平衡。

## R08

复杂度分析：
1. 设模拟总 RTT 数为 `T`，平均窗口为 `W`。
2. 每 RTT 进行 `O(W)` 次 ACK 级更新，因此总时间复杂度 `O(T * W)`。
3. 记录每轮统计，空间复杂度 `O(T)`。

注：若采用纯 RTT 级近似（每 RTT 直接 `cwnd += 1`），时间可降为 `O(T)`。

## R09

与慢启动的区别：
1. 慢启动：指数增长（每 RTT 约翻倍），用于快速爬升。
2. 拥塞避免：线性增长（每 RTT 约 +1 MSS），用于稳定探测。
3. 实际 TCP 会在 `cwnd >= ssthresh` 后从慢启动切换到拥塞避免。

## R10

参数与实现取舍（本目录 MVP）：
1. 初始 `cwnd=10 MSS`，符合现代 TCP 常见初始窗口量级。
2. 固定 `RTT=100ms` 便于将 `cwnd` 映射为近似吞吐率。
3. 使用预设离散丢包轮次，确保输出可复现，便于自动验证。

## R11

`demo.py` 的实现结构：
1. `RoundRecord`：保存每个 RTT 的前后窗口、阈值、事件和吞吐估计。
2. `CongestionAvoidanceSimulator`：封装 ACK 增长、丢包回退与主循环。
3. `render_table`：以文本表格输出主要轨迹。
4. `summarize`：给出均值、峰值、丢包次数等汇总指标。

## R12

预期输出特征：
1. 无丢包 RTT：`cwnd` 逐轮上升，吞吐稳步增加。
2. 丢包 RTT：`cwnd` 明显回落到约一半，并同步更新 `ssthresh`。
3. 全局曲线呈锯齿形，这是拥塞避免工作正常的典型信号。

## R13

边界条件处理：
1. `ssthresh` 下界设置为 `2 MSS`，避免窗口降得过低。
2. ACK 数最少按 `1` 处理，防止 `floor(cwnd)=0` 引发空循环。
3. 吞吐计算依赖 `RTT > 0`，代码中固定为正值常量。

## R14

局限与风险：
1. 未模拟真实链路排队、重排序、ACK 压缩等现象。
2. 未区分三次重复 ACK 与超时恢复路径。
3. RTT 固定不变，忽略了时延抖动对拥塞判定的影响。

因此该实现是教学级机制验证，不是协议栈级仿真器。

## R15

与现实协议的关系：
1. 与 TCP Reno 的拥塞避免核心规则一致（AIMD）。
2. 未覆盖 CUBIC/BBR 等现代算法的窗口或速率模型。
3. 若扩展到生产评估，建议加入随机丢包模型、可变 RTT 和多流公平性对比。

## R16

最小测试清单：
1. 基线运行：脚本可直接启动，无交互输入。
2. 行为检查：输出中存在 ACK 和 LOSS 两类事件。
3. 数值检查：LOSS 轮次后 `cwnd` 明显下降且 `ssthresh` 更新。
4. 健壮性检查：全程 `cwnd >= 2`（本模型下界）。

## R17

运行方式：

```bash
uv run python Algorithms/计算机-计算机网络-0344-TCP拥塞控制_-_拥塞避免/demo.py
```

或在目标目录内直接运行：

```bash
uv run python demo.py
```

## R18

源码级算法流（`demo.py`，8 步）：
1. 初始化模拟器参数：`cwnd`、`ssthresh`、`RTT`、总轮次、丢包轮次集合。
2. 进入主循环，逐个 RTT 执行一次状态更新。
3. 记录本轮 `cwnd_before` 作为对比基线。
4. 若当前 RTT 命中丢包集合，执行 `_on_loss()`：`ssthresh=max(cwnd/2,2)`，`cwnd=ssthresh`。
5. 若未丢包，执行 `_on_ack_round()`：按 `floor(cwnd)` 次 ACK 循环，每次 `cwnd += 1/cwnd`。
6. 计算本轮 `cwnd_after` 与估计吞吐 `throughput_mbps`，写入 `RoundRecord`。
7. 循环结束后输出明细表格，展示 RTT 级窗口轨迹与事件标签。
8. 计算并打印汇总统计（平均窗口、峰值窗口、丢包次数、平均吞吐），用于验证拥塞避免效果。

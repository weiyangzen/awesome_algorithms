# TCP拥塞控制 - 快速恢复

- UID: `CS-0199`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `346`
- 目标目录: `Algorithms/计算机-计算机网络-0346-TCP拥塞控制_-_快速恢复`

## R01

TCP 快速恢复（Fast Recovery）用于处理“非超时丢包”场景：当发送端连续收到 3 个重复 ACK 时，不必像 Tahoe 那样把拥塞窗口直接降到 1，而是进入恢复态，边重传边维持一定发送速率。
本条目提供可运行 MVP，目标是：
- 用离散事件流（`ACK_NEW` / `DUP_ACK`）模拟 Reno 风格快速恢复；
- 展示“三次重复 ACK 触发恢复、额外重复 ACK 膨胀窗口、新 ACK 退出恢复”的完整链路；
- 给出固定断言，保证 `uv run python demo.py` 可复现。

## R02

问题定义（MVP 范围）：
- 输入：
1. `init_cwnd`：初始拥塞窗口（单位：MSS）；
2. `init_ssthresh`：初始慢启动阈值；
3. `events`：ACK 事件序列，每步取值为 `ACK_NEW` 或 `DUP_ACK`。
- 输出：
1. 每步的 `cwnd_before/cwnd_after`；
2. 每步的 `ssthresh_before/ssthresh_after`；
3. 每步的 `state_before/state_after`（`congestion_avoidance` 或 `fast_recovery`）；
4. 每步的 `dup_ack_count_before/dup_ack_count_after`。

## R03

核心规则（Reno 风格简化）：
- 在拥塞避免（CA）中：
1. `ACK_NEW` 到来时，执行加性增长（本 MVP 简化为 `cwnd <- cwnd + 1`）；
2. `DUP_ACK` 到来时，`dup_ack_count += 1`。
- 当 `dup_ack_count == 3`：
1. `ssthresh <- max(cwnd // 2, 2)`；
2. `cwnd <- ssthresh + 3`（窗口膨胀，进入快速恢复）；
3. 状态切换到 `fast_recovery`。
- 在快速恢复（FR）中：
1. 额外 `DUP_ACK`：`cwnd <- cwnd + 1`；
2. 第一个 `ACK_NEW`（视作恢复确认）：`cwnd <- ssthresh`，退出 FR 回到 CA，`dup_ack_count <- 0`。

## R04

`demo.py` 执行流程：
1. 校验初始窗口、阈值和事件序列合法性；
2. 初始化 `cwnd/ssthresh/state/dup_ack_count`；
3. 逐事件迭代状态机；
4. 在 CA 中累计重复 ACK，达到 3 次时触发 FR 入口规则；
5. 在 FR 中处理“额外重复 ACK 膨胀”或“新 ACK 退出恢复”；
6. 记录每步前后状态到结构化记录；
7. 抽取 numpy 序列并与期望序列断言对比；
8. 打印时间线并输出 `All checks passed.`。

## R05

关键数据结构：
- `StepRecord`：单步事件的完整状态转移快照；
- `records: List[StepRecord]`：完整仿真轨迹；
- `np.ndarray`：提取 `cwnd/ssthresh/state/dup_ack_count` 序列用于校验。

## R06

正确性直觉：
- 三次重复 ACK 能在“链路还在工作”时快速推断丢包，无需等待超时；
- `ssthresh` 折半体现拥塞反馈，`cwnd = ssthresh + 3` 让连接在重传期间仍保持一定吞吐；
- 额外重复 ACK 对应接收端持续报告“后续分段已到达”，发送端可进一步注入新数据；
- 新 ACK 到来说明缺失分段已被修复，窗口回落到 `ssthresh`，系统回归拥塞避免。

## R07

复杂度（设事件数为 `E`）：
- 时间复杂度：`O(E)`；
- 空间复杂度：`O(E)`（保存每步记录）。

## R08

边界与异常处理：
- `init_cwnd < 2`：抛 `ValueError`；
- `init_ssthresh < 2`：抛 `ValueError`；
- `events` 为空：抛 `ValueError`；
- 事件不在 `{ACK_NEW, DUP_ACK}`：抛 `ValueError`；
- 状态机若出现未知状态：抛 `ValueError`。

## R09

MVP 设计取舍：
- 仅依赖 `numpy`，不依赖抓包和内核协议栈，保持最小工具栈；
- 采用离散事件流而非逐包真实时间戳，重点放在快速恢复机制本身；
- 不调用黑箱第三方拥塞控制库，所有窗口与阈值更新均在源码显式实现；
- 采用 Reno 的“新 ACK 直接退出恢复”简化，不扩展 NewReno 的 partial ACK 细节。

## R10

`demo.py` 函数职责：
- `validate_inputs`：输入校验与事件标准化；
- `additive_increase`：拥塞避免加性增长；
- `enter_fast_recovery`：三次重复 ACK 的入口计算；
- `simulate_tcp_fast_recovery`：主状态机仿真；
- `extract_series`：从记录提取 numpy 序列；
- `run_demo`：固定样例、断言、打印；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0346-TCP拥塞控制_-_快速恢复
uv run python demo.py
```

脚本不需要任何输入，直接输出仿真时间线和校验结果。

## R12

输出说明：
- `Step timeline`：逐步显示事件、状态、重复 ACK 计数、`cwnd` 和 `ssthresh` 的前后变化；
- `cwnd_after sequence`：每步结束后的拥塞窗口序列；
- `ssthresh_after sequence`：每步结束后的阈值序列；
- `state_after sequence`：状态序列；
- `dup_ack_count_after sequence`：重复 ACK 计数序列；
- `All checks passed.`：表示全部断言通过。

## R13

最小验证清单：
- 固定样例断言 `cwnd_after`、`ssthresh_after`、`state_after`、`dup_ack_count_after`；
- 验证恰好出现 2 次快速恢复入口；
- 验证每次入口满足 `dup_ack_count=3` 且 `cwnd=ssthresh+3`；
- 验证快速恢复中的额外重复 ACK 都满足 `cwnd_after = cwnd_before + 1`；
- 验证恢复确认 ACK 都满足 `cwnd_after = ssthresh_before` 且退出到 CA。

## R14

当前 demo 参数：
- `init_cwnd = 12`；
- `init_ssthresh = 20`；
- 事件序列（14 步）：
1. `ACK_NEW`
2. `DUP_ACK`
3. `DUP_ACK`
4. `DUP_ACK`
5. `DUP_ACK`
6. `DUP_ACK`
7. `ACK_NEW`
8. `ACK_NEW`
9. `ACK_NEW`
10. `DUP_ACK`
11. `DUP_ACK`
12. `DUP_ACK`
13. `ACK_NEW`
14. `ACK_NEW`

该序列覆盖两次“进入快速恢复 -> 退出快速恢复”的完整闭环。

## R15

与相关机制对比：
- TCP Tahoe：检测丢包后通常把 `cwnd` 降到 `1`，恢复更保守；
- TCP Reno（本 MVP）：通过窗口膨胀和快速恢复减少吞吐骤降；
- TCP NewReno：对 partial ACK 处理更细，可能在 FR 停留更久；
- CUBIC/BBR：拥塞控制目标与增长函数更复杂，不仅仅是 Reno 的 AIMD + FR。

## R16

适用场景：
- 课堂或自学中演示 Reno 快速恢复核心流程；
- 在离散事件网络仿真中作为拥塞控制教学模块；
- 解释“重复 ACK 密集出现时窗口为何先膨胀后回落”的行为。

不适用场景：
- 需要与 Linux 内核 TCP 实现逐包逐定时器完全一致；
- 需要覆盖 RTO、SACK、ACK 压缩、乱序缓存等完整生产细节。

## R17

可扩展方向：
- 将 CA 增长从“每事件 +1”细化到“每 RTT +1 / 每 ACK +1/cwnd”；
- 增加 NewReno partial ACK 路径；
- 增加超时事件与慢启动重启逻辑；
- 注入随机丢包模型并统计吞吐/收敛行为；
- 增加图形化曲线展示 `cwnd` 与 `ssthresh` 的动态。

## R18

`demo.py` 源码级算法流（8 步，非黑箱）：
1. `main` 调用 `run_demo`，构造固定初始窗口和 ACK 事件流。  
2. `run_demo` 调用 `simulate_tcp_fast_recovery`，先由 `validate_inputs` 校验参数域和事件合法性。  
3. 初始化 `cwnd/ssthresh/state/dup_ack_count`，其中状态从 `congestion_avoidance` 开始。  
4. 逐事件执行状态机：在 CA 中遇到 `ACK_NEW` 做 `additive_increase`，遇到 `DUP_ACK` 累计重复 ACK 计数。  
5. 当重复 ACK 计数达到 3，调用 `enter_fast_recovery` 计算 `ssthresh=max(cwnd//2,2)` 与 `cwnd=ssthresh+3`，并切换到 FR。  
6. 在 FR 中，额外 `DUP_ACK` 触发 `cwnd+1` 膨胀；首次 `ACK_NEW` 触发 `cwnd=ssthresh`、状态回到 CA、重复 ACK 计数清零。  
7. 每步状态变化都保存为 `StepRecord`，最终 `extract_series` 抽取 numpy 序列并与期望序列逐项断言。  
8. 对 FR 入口/出口与膨胀规则做额外性质断言，全部通过后打印时间线和 `All checks passed.`。  

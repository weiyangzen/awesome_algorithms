# TCP拥塞控制 - 快速重传

- UID: `CS-0198`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `345`
- 目标目录: `Algorithms/计算机-计算机网络-0345-TCP拥塞控制_-_快速重传`

## R01

TCP 快速重传（Fast Retransmit）的核心目标是避免“等超时再重传”的高延迟恢复：
当发送端连续收到同一个累计 ACK 的 3 个重复 ACK 时，立即推断存在单段丢失并重传该段。

本条目实现一个最小可运行 MVP，重点演示：
1. 累计 ACK 前沿 `highest_ack` 的推进；
2. 重复 ACK 计数到阈值（3）触发快速重传；
3. 触发时的窗口/阈值更新与重传记录。

## R02

问题定义（MVP 范围）：
- 输入：
1. `init_cwnd`：初始拥塞窗口（单位：MSS）；
2. `init_ssthresh`：初始慢启动阈值；
3. `initial_highest_ack`：初始累计 ACK 前沿；
4. `ack_stream`：按时间顺序到达的 ACK 编号流（整数）。
- 输出：
1. 每步 ACK 分类（`NEW_ACK / DUP_ACK / STALE_ACK`）；
2. 每步 `highest_ack`、`dup_ack_count`、`cwnd`、`ssthresh`、重传状态变化；
3. 快速重传触发步骤与被重传分段号列表。

## R03

核心规则（教学化 Reno 风格）：
1. `ack_no > highest_ack`：视为新累计 ACK，`highest_ack <- ack_no`，`dup_ack_count <- 0`；
2. `ack_no == highest_ack`：视为重复 ACK，`dup_ack_count += 1`；
3. 当 `dup_ack_count == 3` 时触发快速重传：
   - 立刻重传 `seq = highest_ack`；
   - `ssthresh <- max(cwnd // 2, 2)`；
   - `cwnd <- ssthresh + 3`；
   - 进入“快速重传中”状态；
4. 快速重传中若继续收到重复 ACK，`cwnd += 1`（保持管道不空）；
5. 快速重传中收到新累计 ACK 时，`cwnd <- ssthresh`，退出快速重传状态；
6. `ack_no < highest_ack` 为陈旧 ACK，不触发重传。

## R04

`demo.py` 执行流程：
1. 校验参数和 ACK 序列合法性；
2. 初始化 `highest_ack/dup_ack_count/cwnd/ssthresh`；
3. 逐个 ACK 做 `NEW/DUP/STALE` 分类；
4. 在 `DUP_ACK` 路径累计重复 ACK 计数；
5. 第 3 个重复 ACK 触发快速重传并记录 `retransmitted_seq`；
6. 将每步前后状态写入 `StepRecord`；
7. 提取 numpy 序列并与期望序列做严格断言；
8. 输出 pandas 时间线和 `All checks passed.`。

## R05

关键数据结构：
- `StepRecord`：单步状态快照，覆盖 ACK 分类、计数、窗口、阈值、重传动作；
- `records: List[StepRecord]`：完整仿真轨迹；
- `numpy.ndarray`：确定性校验序列；
- `pandas.DataFrame`：可读时间线表格。

## R06

正确性直觉：
- 重复 ACK 表示接收端持续“期待同一个缺失分段”，通常意味着该分段丢失但后续分段已到达；
- 连续 3 个重复 ACK 可在无超时条件下较高置信触发重传；
- 这样能显著缩短恢复时间，减少吞吐塌陷；
- `ssthresh` 折半体现拥塞反馈，`cwnd` 调整体现“减速但不中断发送”。

## R07

复杂度（设 ACK 事件数为 `E`）：
- 时间复杂度：`O(E)`（单次线性扫描）；
- 空间复杂度：`O(E)`（保存完整步骤记录）。

## R08

边界与异常处理：
- `init_cwnd < 2` 抛 `ValueError`；
- `init_ssthresh < 2` 抛 `ValueError`；
- `initial_highest_ack <= 0` 抛 `ValueError`；
- `ack_stream` 为空抛 `ValueError`；
- ACK 编号出现非正整数抛 `ValueError`。

## R09

MVP 设计取舍：
- 选择 ACK 编号流模型，而非真实抓包/内核栈，便于聚焦快速重传触发机制；
- 仅用 `numpy + pandas` 做断言和展示，工具栈小且透明；
- 不依赖黑箱拥塞控制库，触发条件与状态更新在源码中显式可审计；
- 保留最必要窗口更新逻辑，但不扩展到完整 Linux TCP 实现细节。

## R10

`demo.py` 函数职责：
- `validate_inputs`：输入约束检查；
- `classify_ack`：按当前累计 ACK 前沿分类 ACK；
- `on_triple_duplicate_ack`：计算快速重传入口的 `cwnd/ssthresh`；
- `simulate_tcp_fast_retransmit`：主仿真循环；
- `extract_series`：提取断言序列；
- `records_to_dataframe`：生成输出表；
- `run_demo`：固定样例 + 断言 + 打印；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0345-TCP拥塞控制_-_快速重传
uv run python demo.py
```

脚本无需任何输入，运行后直接打印时间线与验证结果。

## R12

输出说明：
- `Step timeline`：每步 ACK 类型和状态变化；
- `highest_ack_after sequence`：累计 ACK 前沿推进序列；
- `dup_ack_count_after sequence`：重复 ACK 计数序列；
- `cwnd_after / ssthresh_after sequence`：窗口与阈值序列；
- `fast_retransmit trigger steps`：触发快速重传的步骤编号；
- `fast_retransmit retransmitted_seq`：被重传的分段号；
- `All checks passed.`：全部断言通过。

## R13

最小验证清单：
1. `highest_ack/dup_ack_count/cwnd/ssthresh/in_fast_retransmit/retransmissions` 序列断言全部匹配；
2. 仅在第 3 个重复 ACK 时触发重传；
3. 触发步骤固定为 `[5, 11]`；
4. 被重传分段固定为 `[3, 8]`；
5. 快速重传状态下收到新 ACK 时满足 `cwnd_after == ssthresh_before`；
6. 陈旧 ACK 不触发重传。

## R14

当前 demo 固定参数：
- `init_cwnd = 10`
- `init_ssthresh = 16`
- `initial_highest_ack = 1`
- `ack_stream`（15 步）：
1. `2`
2. `3`
3. `3`
4. `3`
5. `3`   -> 第一次快速重传（重传 `seq=3`）
6. `3`
7. `7`
8. `8`
9. `8`
10. `8`
11. `8`  -> 第二次快速重传（重传 `seq=8`）
12. `8`
13. `12`
14. `11`（陈旧 ACK）
15. `13`

该序列覆盖两次快速重传触发、重传中额外重复 ACK、重传后新 ACK 收敛、以及陈旧 ACK 场景。

## R15

与相关算法关系：
- 慢启动（CS-0196）：关注连接初期指数增长；
- 拥塞避免（CS-0197）：关注线性增窗；
- 快速重传（本条目）：关注“无需超时”的丢包检测与立即重传；
- 快速恢复（CS-0199）：关注快速重传触发后的恢复阶段窗口管理。

## R16

适用场景：
- 讲解“为何三次重复 ACK 足以触发重传”；
- 在离散事件教学仿真中快速验证触发逻辑；
- 对比“超时重传 vs 快速重传”的延迟差异机制。

不适用场景：
- 要求与 Linux 内核 TCP 每个分支行为完全一致；
- 需要真实 RTT、重排序、SACK、ACK 压缩等生产级协议细节。

## R17

可扩展方向：
1. 引入 RTT 与 RTO 计时器，对比超时路径和快速重传路径；
2. 将 ACK 流替换为可配置丢包/重排序信道模型；
3. 加入 NewReno `partial ACK` 与 SACK 逻辑；
4. 增加多流公平性指标（吞吐、恢复时延、重传次数）。

## R18

`demo.py` 源码级算法流（8 步，非黑箱）：
1. `main` 调用 `run_demo`，构造固定 `ack_stream` 与初始窗口参数。  
2. `run_demo` 调用 `simulate_tcp_fast_retransmit`，先经 `validate_inputs` 检查参数域。  
3. 主循环对每个 ACK 调用 `classify_ack`，将其判定为 `NEW_ACK`、`DUP_ACK` 或 `STALE_ACK`。  
4. `NEW_ACK` 分支推进 `highest_ack` 并清零重复计数；若当前在快速重传状态则执行 `cwnd <- ssthresh` 并退出该状态。  
5. `DUP_ACK` 分支增加重复计数，计数到 3 时调用 `on_triple_duplicate_ack` 计算 `ssthresh=max(cwnd//2,2)` 与 `cwnd=ssthresh+3`，同时记录本次 `retransmitted_seq`。  
6. 若处于快速重传状态且继续收到重复 ACK，则执行 `cwnd+1` 以维持发送管道。  
7. 每步结果写入 `StepRecord`，`extract_series` 将关键字段转为 numpy 序列并与期望序列逐项比对。  
8. `records_to_dataframe` 输出可读时间线，最后打印触发步骤、重传分段号和 `All checks passed.`。  

第三方库角色说明：
- `numpy` 仅用于序列化断言，不参与协议决策；
- `pandas` 仅用于展示表格，不参与算法逻辑。

# 三阶段提交 (3PC)

- UID: `CS-0243`
- 学科: `计算机`
- 分类: `数据库`
- 源序号: `399`
- 目标目录: `Algorithms/计算机-数据库-0399-三阶段提交_(3PC)`

## R01

三阶段提交（Three-Phase Commit, 3PC）是对 2PC 的扩展型分布式事务提交协议。
它把“提交决议”拆成三步：
- 阶段一 `CanCommit`：协调者询问参与者是否可提交；
- 阶段二 `PreCommit`：若全体同意，协调者先广播“预提交”；
- 阶段三 `DoCommit`：确认预提交完成后，再广播最终提交。

本目录给出一个可运行 MVP，用确定性脚本演示 3PC 的状态转换、故障点和超时分支。

## R02

MVP 要解决的问题：

在跨分片转账事务中（`shard_A` 扣款、`shard_B` 加款），如何在以下场景保持一致：
- 正常情况下三阶段完整走通并提交；
- 任一参与者阶段一否决时全局中止；
- 协调者在阶段一后崩溃时，参与者超时自动中止；
- 协调者在阶段二后崩溃时，参与者超时自动提交并最终与协调者日志收敛。

## R03

选择 3PC 作为该条目的原因：
- 它是经典分布式提交协议里“减少阻塞”的代表；
- 对比 2PC，能更清晰地展示“中间态（PRECOMMIT）”的工程意义；
- 可以在纯 Python、无网络框架条件下完整表达核心语义。

## R04

本实现的核心状态与数据结构：

1. 投票 `Vote`：`YES / NO / TIMEOUT`。
2. 参与者状态 `ParticipantState`：`INIT / WAIT / PRECOMMIT / COMMITTED / ABORTED`。
3. 全局决议 `Decision`：`COMMIT / ABORT`。
4. 事务结构 `Transaction`：`txn_id + operations_by_participant`。
5. 参与者缓存 `pending_txn`：阶段一通过后缓存待提交操作，最终在 `COMMIT` 时落地。

## R05

协议流程（对应 `demo.py`）：

1. 协调者 `execute_3pc` 进入阶段一，向所有参与者发送 `can_commit`。
2. 若存在 `NO/TIMEOUT`，直接 `ABORT` 并广播中止。
3. 若全体 `YES`，进入阶段二，广播 `PRECOMMIT` 并收 ACK。
4. 若阶段二 ACK 不全，协调者也走 `ABORT`。
5. 阶段二 ACK 全部成功后，进入阶段三广播 `DoCommit`。
6. 参与者收到 `DoCommit` 后应用本地操作并进入 `COMMITTED`。
7. 若协调者崩溃，参与者依据超时规则做本地决策（`WAIT->ABORT`, `PRECOMMIT->COMMIT`）。

## R06

一致性与容错语义（本 MVP 的假设边界）：

- 原子性：全局提交仅在“阶段一全 `YES` 且阶段二 ACK 完整”时发生。
- 收敛性：参与者超时策略使故障后能走向终态，而非长期阻塞。
- 幂等性：重复 `COMMIT/ABORT` 消息不会破坏终态。
- 关键假设：
  - 网络满足有界延迟（bounded delay）；
  - 不考虑长时间网络分区；
  - 不实现拜占庭行为。

因此本实现强调“协议行为教学”，不是生产级强容错事务管理器。

## R07

复杂度分析（设参与者数为 `P`，单参与者操作数均值为 `K`）：

- 阶段一（试算投票）时间：`O(P*K)`。
- 阶段二（预提交 ACK）时间：`O(P)`。
- 阶段三（最终提交）时间：`O(P*K)`。
- 总体时间复杂度：`O(P*K)`。
- 空间复杂度：
  - 参与者待提交缓存 `pending_txn` 约 `O(P*K)`；
  - 协调者日志 `decision_log/event_log` 约 `O(T)`（`T` 为事务数）。

## R08

边界与异常处理：

- 余额不足：`can_commit` 返回 `NO`，该参与者立即本地 `ABORTED`。
- 参与者宕机：阶段调用返回失败，协调者可判定中止。
- 协调者阶段一后崩溃：参与者停在 `WAIT`，超时后转 `ABORTED`。
- 协调者阶段二后崩溃：参与者处于 `PRECOMMIT`，超时后可执行 `COMMIT`。
- 重放消息：`receive_do_commit / receive_abort` 都做了终态幂等判断。

## R09

MVP 取舍说明：

- 使用 Python 标准库实现，不依赖外部数据库、中间件或 RPC 框架；
- 不实现真实 WAL、选主、时钟同步与重传风暴控制；
- 不引入异步线程，改为“显式故障点 + 显式 timeout 调用”的可复现实验法；
- 优先保证协议状态机清晰、可读、可运行。

## R10

`demo.py` 关键函数职责：

- `Participant.can_commit`：阶段一本地可提交性检查与投票。
- `Participant.receive_precommit`：进入 `PRECOMMIT` 中间态。
- `Participant.receive_do_commit`：应用待提交操作并进入 `COMMITTED`。
- `Participant.receive_abort`：中止并清理待提交缓存。
- `Participant.on_timeout`：实现 3PC 超时规则。
- `Coordinator.execute_3pc`：执行三阶段主流程并支持崩溃注入。
- `Coordinator.observe_and_record_decision`：恢复时根据集群终态补齐决议。
- `Coordinator.replay_decision`：重放已记录决议到参与者。
- `run_case_*`：四个固定场景测试。

## R11

运行方式（无交互输入）：

```bash
cd Algorithms/计算机-数据库-0399-三阶段提交_(3PC)
uv run python demo.py
```

脚本会自动执行 4 个场景并打印日志，最后输出 `Global checks pass: True/False`。

## R12

输出字段说明：

- `decision` / `decision_before_timeout`：当前阶段返回的全局结果或不确定态。
- `recovered_decision`：协调者恢复后依据参与者终态推断出的决议。
- `total_balance(before, after)`：全局守恒检查。
- `state[Tx]`：参与者在事务 `Tx` 的状态。
- `blocked`：参与者仍在 `WAIT/PRECOMMIT` 的事务集合。
- `case_pass`：场景级断言是否通过。
- `Event log example entries`：一次完整成功路径的关键协议日志。
- `Global checks pass`：全部场景总通过状态。

## R13

最小测试集（脚本内置）：

1. `CASE 1` 正常路径：`YES -> PRECOMMIT -> COMMIT`。
2. `CASE 2` 阶段一否决：余额不足触发 `NO`，全局 `ABORT`。
3. `CASE 3` 阶段一后崩溃：协调者故障，参与者超时 `ABORT`。
4. `CASE 4` 阶段二后崩溃：参与者超时 `COMMIT`，协调者恢复并补录决议。

每个场景都验证余额结果、状态终态与布尔断言。

## R14

关键参数与调节建议：

- `amount`：转账金额，控制成功提交或余额不足否决。
- `crash_point`：
  - `after_cancommit` 模拟阶段一后崩溃；
  - `after_precommit` 模拟阶段二后崩溃。
- `Participant.set_crashed(True/False)`：可扩展为参与者故障测试。

建议：
- 看成功链路：小额转账 + `crash_point=None`。
- 看中止链路：超额转账触发 `NO`。
- 看 3PC 超时差异：分别试 `after_cancommit` 与 `after_precommit`。

## R15

与相关方案对比：

- 对比 2PC：
  - 2PC 在协调者故障下更易出现阻塞；
  - 3PC 增加 `PRECOMMIT` 以便在超时条件下让参与者做确定性推进。
- 对比共识型提交（Paxos/Raft 上层事务）：
  - 共识型通常容错更强；
  - 3PC 实现更直观，但对网络时序假设更敏感。

## R16

典型应用语境：

- 课程与面试中的分布式事务协议教学；
- 事务协调器原型验证（先做协议层模拟，再接真实 RPC）；
- 故障注入与状态机单元测试的最小实验底座。

## R17

可扩展方向：

- 引入真实持久化日志与崩溃重放；
- 增加事务并发、锁管理和冲突检测；
- 模拟消息乱序、重复、丢包与网络分区；
- 把 `on_timeout` 改为真实计时器驱动；
- 增加指标输出（提交率、超时率、恢复时延）。

## R18

`demo.py` 的源码级算法流（8 步）：

1. `main` 顺序执行四个 `run_case_*`，确保实验可复现且无需交互。
2. 每个场景先构建 `participants + coordinator + transaction`，记录 `before` 总余额。
3. `Coordinator.execute_3pc` 先做阶段一，逐个调用 `Participant.can_commit` 收集投票。
4. 若阶段一不全是 `YES`，协调者立刻写 `ABORT` 并广播 `receive_abort`。
5. 若阶段一全 `YES`，进入阶段二广播 `receive_precommit` 收集 ACK。
6. 阶段二 ACK 全部成功时进入阶段三，广播 `receive_do_commit`，参与者应用待提交操作。
7. 若在阶段边界注入协调者崩溃，参与者通过 `on_timeout` 执行 3PC 本地规则：`WAIT->ABORT`、`PRECOMMIT->COMMIT`。
8. 恢复后协调者用 `observe_and_record_decision` 补齐日志并 `replay_decision`，最终由场景断言输出 `case_pass` 与总结果。

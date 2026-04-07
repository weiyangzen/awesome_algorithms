# 两阶段提交 (2PC)

- UID: `CS-0242`
- 学科: `计算机`
- 分类: `数据库`
- 源序号: `398`
- 目标目录: `Algorithms/计算机-数据库-0398-两阶段提交_(2PC)`

## R01

两阶段提交（Two-Phase Commit, 2PC）是分布式事务里最经典的一致性提交协议：
- 阶段一（Prepare/Vote）：协调者询问所有参与者“是否可以提交”；
- 阶段二（Commit/Abort）：若全体 `YES` 则全局 `COMMIT`，否则全局 `ABORT`。

本目录实现一个可运行的最小模拟器，覆盖正常路径和故障路径，重点展示协议行为而非网络工程细节。

## R02

MVP 要解决的问题：

在一个跨分片转账事务中（`shard_A` 扣款、`shard_B` 加款），如何在以下场景下保持一致性：
- 全部参与者可提交时完成原子提交；
- 任一参与者拒绝/超时时全局回滚；
- 协调者在阶段一后崩溃时，展示参与者阻塞与恢复后的决议收敛。

输出内容包括：
- 每个场景的全局决议；
- 每个参与者余额、事务状态、阻塞状态；
- 全局余额守恒检查；
- 事件日志样例。

## R03

选择该问题的原因：
- 2PC 是数据库分布式事务的基础协议，教学价值高；
- 协议有明确状态机，适合用小规模代码完整表达；
- 可以在不引入真实网络和存储系统的前提下，复现关键语义（阻塞、超时、恢复、重放）。

## R04

本实现的核心状态模型：

1. 投票结果 `Vote`：`YES / NO / TIMEOUT`。
2. 参与者事务状态 `TxnState`：`INIT / READY / COMMITTED / ABORTED`。
3. 全局决议 `Decision`：`COMMIT / ABORT`。
4. 事务结构 `Transaction`：按参与者拆分本地操作（账户增减）。

参与者在 `prepare` 时做“试算校验”（不能出现负余额）；校验通过进入 `READY`，否则立即本地 `ABORT` 并返回 `NO`。

## R05

协议流程（对应 `demo.py`）：

1. 协调者 `execute_2pc` 发起事务并记录 `BEGIN_2PC`。
2. 向所有参与者发送 `prepare` 并收集投票。
3. 若配置了 `crash_after_prepare=True`，协调者在投票后立刻崩溃，事务进入 `IN_DOUBT`。
4. 非崩溃情况下：
   - 全部 `YES` -> 决议 `COMMIT`；
   - 否则（`NO/TIMEOUT` 任一出现）-> 决议 `ABORT`。
5. 协调者持久化决议（`decision_log`）并广播。
6. 参与者执行 `apply_commit` 或 `apply_abort`，从 `READY` 进入终态。
7. 恢复阶段可 `recover_and_finish`（协调者恢复）或 `replay_decision`（参与者恢复后重放）。

## R06

正确性与一致性要点：

- 原子性：
  - 仅当全部参与者 `YES` 才会 `COMMIT`；
  - 出现任一 `NO/TIMEOUT` 即全局 `ABORT`。
- 一致决议：
  - 协调者为单一决议源，`decision_log` 记录最终结果；
  - 通过 `replay_decision` 支持幂等重放，确保迟到节点收敛到同一终态。
- 故障语义：
  - 协调者在阶段一后崩溃时，参与者可能停留在 `READY`（阻塞）；
  - 恢复后若无已记录决议，本实现采用“缺省中止（presumed abort）”完成收敛。

## R07

复杂度分析（设参与者数为 `P`，每个参与者本地操作数均值为 `K`）：

- 阶段一投票：每个参与者做本地试算，时间约 `O(P*K)`。
- 阶段二广播：每个参与者执行本地提交/中止，时间约 `O(P*K)`。
- 单事务总时间复杂度：`O(P*K)`（常数因子约 2 个阶段）。
- 空间复杂度：
  - 参与者快照与状态约 `O(P*K)`；
  - 协调者决议与事件日志约 `O(T)`（`T` 为事务数）。

## R08

边界与异常处理：

- 参与者崩溃：`prepare/commit/abort` 返回失败或 `TIMEOUT`，由协调者走 `ABORT` 或后续重放。
- 余额不足：`prepare` 直接投 `NO` 并本地标记 `ABORTED`。
- 重复消息：
  - 已 `COMMITTED` 的事务重复 `commit` 可安全幂等；
  - 已 `ABORTED` 的事务重复 `abort` 可安全幂等。
- 不合法转换：
  - 已 `COMMITTED` 后拒绝 `abort`；
  - 非 `READY` 状态拒绝直接 `commit`。

## R09

MVP 取舍说明：

- 使用纯 Python 标准库实现，不引入外部数据库或消息系统；
- 不实现真实网络、WAL 落盘、超时重传线程、选主与共识；
- 聚焦协议本身：状态机、决议条件、崩溃恢复与阻塞现象；
- 通过固定场景和断言给出“可运行且可审计”的最小闭环。

## R10

`demo.py` 中关键函数职责：

- `Participant.prepare`：本地预检查并投票。
- `Participant.apply_commit`：执行本地提交。
- `Participant.apply_abort`：执行本地中止并回滚快照。
- `Participant.blocked_transactions`：识别处于 `READY` 且未决议事务。
- `Coordinator.execute_2pc`：执行 2PC 两阶段主流程。
- `Coordinator.recover_and_finish`：协调者恢复后决议收敛。
- `Coordinator.replay_decision`：对恢复参与者重放既有决议。
- `run_case_*`：四个内置测试场景。
- `main`：串行运行全部场景并汇总全局通过状态。

## R11

运行方式（无交互输入）：

```bash
cd Algorithms/计算机-数据库-0398-两阶段提交_(2PC)
uv run python demo.py
```

脚本会自动执行 4 个场景并打印日志，最后输出 `Global checks pass: True/False`。

## R12

输出字段说明：

- `decision`：当前事务全局决议或中间状态（如 `IN_DOUBT`）。
- `total_balance(before, after)`：事务前后系统总余额，用于守恒校验。
- `state[Tx]`：参与者在事务 `Tx` 的状态（`INIT/READY/COMMITTED/ABORTED`）。
- `blocked`：参与者当前阻塞事务列表。
- `crashed`：参与者是否处于崩溃状态。
- `case_pass`：该场景断言是否通过。
- `Event log`：协调者记录的协议关键步骤。
- `Global checks pass`：所有场景总通过结果。

## R13

最小测试集（已在脚本中内置）：

1. `CASE 1` 正常路径：全部 `YES`，全局 `COMMIT`。
2. `CASE 2` 否决路径：余额不足导致 `NO`，全局 `ABORT`。
3. `CASE 3` 协调者崩溃：Prepare 后崩溃，参与者短暂阻塞，恢复后 `ABORT` 收敛。
4. `CASE 4` 超时路径：某参与者崩溃导致 `TIMEOUT`，先 `ABORT`，恢复后重放决议。

每个用例都校验余额正确性与状态收敛。

## R14

关键参数与可调项：

- `amount`：转账金额，决定是否触发 `NO`（余额不足）。
- `crash_after_prepare`：是否模拟协调者在阶段一后崩溃。
- `Participant.set_crashed(True/False)`：模拟参与者故障与恢复。

调参建议：
- 想看成功提交：设置较小 `amount` 且无崩溃。
- 想看拒绝与回滚：将 `amount` 调到超过扣款账户余额。
- 想看阻塞：打开 `crash_after_prepare=True`。

## R15

与相关方案对比：

- 对比本地事务：本地事务无需分布式投票，开销更小但不能跨节点原子提交。
- 对比三阶段提交（3PC）：3PC 试图降低阻塞，但协议更复杂、对网络假设更强。
- 对比基于共识（如 Paxos/Raft 的事务提交）：容错更强但工程复杂度更高。

2PC 的定位通常是：在“协调者可恢复、故障可控”的环境中提供实现简单的原子提交。

## R16

典型应用场景：

- 跨分片数据库事务（分库分表转账、库存冻结与扣减）；
- 数据库与消息系统的原子协调（例如事务消息）；
- 需要严格“要么都成功，要么都失败”的多资源更新。

## R17

可扩展方向：

- 引入真实持久化日志（WAL）与崩溃后重放；
- 增加超时重试、消息去重与重传窗口；
- 支持并发多事务与锁冲突检测；
- 添加监控指标（阻塞时长、超时率、重放次数）；
- 对接真实 RPC 和外部存储以做集成测试。

## R18

`demo.py` 的源码级算法流（8 步）：

1. `main` 顺序调用 4 个场景函数，确保无交互、可复现。
2. 场景中先构造 `participants + coordinator + transaction`，并记录事务前总余额。
3. `Coordinator.execute_2pc` 进入阶段一：遍历所有参与者执行 `prepare`，收集 `Vote`。
4. 若命中崩溃注入点（`crash_after_prepare`），协调者返回 `IN_DOUBT`，参与者可能停在 `READY`。
5. 非崩溃路径下，协调者按“全 `YES` 才 `COMMIT`，否则 `ABORT`”写入 `decision_log`。
6. 协调者广播决议：参与者执行 `apply_commit`（应用本地增减）或 `apply_abort`（回滚快照/标记中止）。
7. 若存在故障恢复：
   - 协调者用 `recover_and_finish` 依据日志补发决议；
   - 参与者恢复后用 `replay_decision` 接收并收敛状态。
8. 每个场景最后校验余额与状态，输出 `case_pass`；`main` 汇总为 `Global checks pass`。

# WAL日志

- UID: `CS-0232`
- 学科: `计算机`
- 分类: `数据库`
- 源序号: `386`
- 目标目录: `Algorithms/计算机-数据库-0386-WAL日志`

## R01

WAL（Write-Ahead Logging，预写式日志）是一种数据库持久化与崩溃恢复机制：**在修改数据页之前，先把对应的变更记录落到日志并刷盘**。这样即使系统在任意时刻宕机，也能通过日志把“已经承诺提交”的事务恢复出来，保证持久性与一致性。

## R02

要解决的问题是“崩溃窗口”中的数据丢失与状态撕裂：

- 如果先写数据文件再记日志，崩溃时可能无法判断哪些改动属于已提交事务。
- 如果事务提交响应给客户端后，日志尚未落盘，宕机会导致“客户端以为成功、系统却丢数据”。
- WAL 通过“日志先行 + 提交标记”把提交语义绑定到稳定存储，缩小不确定区间。

## R03

核心思想是把事务拆成两条时间线：

- 逻辑时间线：`BEGIN -> 若干操作 -> COMMIT`。
- 物理时间线：每条日志追加到 `wal.log`，并在关键点 `fsync`。

恢复时只重放“看到 COMMIT 的事务”，忽略未提交事务，从而实现“redo 已提交、丢弃未提交”。

## R04

本目录 MVP 使用的数据结构：

- `wal.log`：JSON Lines 日志，每条记录含 `lsn/txid/op/...`。
- `data.json`：快照文件，含两部分：
  - `state`：当前 KV 状态
  - `last_applied_lsn`：快照已应用到的最大日志序号
- 内存结构：
  - `active_ops[txid]`：事务暂存操作列表
  - `state`：运行时 KV 映射
  - `next_lsn`：下一条日志序号

## R05

提交路径（正常运行）：

1. `BEGIN` 记日志并刷盘。
2. `SET/DELETE` 操作逐条写 WAL 并刷盘。
3. 写 `COMMIT` 并刷盘，此时事务进入“可恢复的已提交状态”。
4. 把事务操作应用到内存状态。
5. 写快照 `data.json`（原子替换）并更新 `last_applied_lsn`。

## R06

正确性直觉：

- 持久性：只要 `COMMIT` 已刷盘，崩溃后重放 WAL 必能恢复该事务。
- 原子性（简化版）：无 `COMMIT` 的事务一律不生效。
- 幂等恢复：通过 `last_applied_lsn` 只重放“快照之后”的已提交事务，避免重复应用。

## R07

复杂度（设日志条数为 `N`，单事务操作数为 `K`）：

- 单事务提交：时间 `O(K)`（日志写入 + 应用 + 快照写出），空间 `O(K)`（暂存操作）。
- 启动恢复：最坏时间 `O(N)`，空间 `O(T*K)`（`T` 为未决事务数量上界）。

## R08

边界与失败场景：

- 崩溃在 `COMMIT` 之前：事务被忽略。
- 崩溃在 `COMMIT` 之后、快照之前：恢复阶段重放该事务。
- WAL 行损坏：本 MVP 会跳过无效 JSON 行（工程上可升级为校验和 + 截断）。
- WAL 不截断：长期会膨胀；生产系统应定期 checkpoint + truncate。

## R09

伪代码：

```text
load_snapshot()
recover_from_wal(last_applied_lsn)

begin(tx): append(BEGIN)
set(tx,k,v): append(SET)
delete(tx,k): append(DELETE)

commit(tx):
  commit_lsn = append(COMMIT)
  apply(tx.ops to state)
  persist_snapshot(state, last_applied_lsn=commit_lsn)

recover():
  for record in wal ordered by lsn:
    collect ops by txid
    if COMMIT and commit_lsn > last_applied_lsn:
      apply ops
      last_applied_lsn = commit_lsn
  persist_snapshot_if_changed()
```

## R10

`demo.py` 的 MVP 特性：

- 纯 Python 标准库实现，零交互运行。
- 支持 `BEGIN/SET/DELETE/COMMIT`。
- 使用 `lsn` 与 `last_applied_lsn` 控制重放边界。
- 内置“注入未提交事务”用于演示宕机恢复。

## R11

运行方式：

```bash
cd Algorithms/计算机-数据库-0386-WAL日志
uv run python demo.py
```

预期结果：

- 第一阶段完成两次提交。
- 注入一笔未提交事务（模拟崩溃前中断）。
- 重启恢复后仅保留已提交状态，未提交写入不会出现在最终快照中。

## R12

与常见替代方案对比：

- 仅快照无 WAL：崩溃窗口更大，提交语义弱。
- 仅操作日志无快照：恢复速度随日志增长变慢。
- WAL + 快照（本方案）：提交安全性与恢复速度在 MVP 层面取得平衡。

## R13

工程实现注意点：

- 日志必须“先于数据页”持久化，这是 WAL 的定义性约束。
- 写文件建议原子替换（临时文件 + `os.replace`）。
- 真实数据库通常会引入页缓存、group commit、校验和、归档与主从复制。

## R14

最小测试清单：

- 正常提交后重启，数据保持不变。
- `COMMIT` 前故障，事务不生效。
- `COMMIT` 后故障，事务可恢复。
- 重复重启不应重复应用同一事务（依赖 `last_applied_lsn`）。

## R15

可扩展方向：

- 增加 `ABORT` 与回滚段，支持更完整事务语义。
- 引入 checkpoint 截断 WAL，降低恢复扫描开销。
- 加入日志校验和、页 LSN、并发控制（锁或 MVCC）。

## R16

术语速记：

- WAL：写前日志。
- LSN：日志序列号（Log Sequence Number）。
- REDO：重做已提交事务。
- Checkpoint：把内存/脏页状态与日志位置对齐的持久化点。

## R17

本条目给出一个“够小但诚实”的 WAL 教学实现：

- 不追求数据库全功能；
- 明确展示日志先行、提交标记、崩溃恢复三个关键机制；
- 可直接运行并复现实验现象，便于后续扩展到更复杂存储引擎。

## R18

下面按 `demo.py` 的源码级流程拆成 8 步（非黑盒）：

1. `SimpleWALKV.__init__` 加载 `data.json`，得到 `state` 与 `last_applied_lsn`，并扫描 WAL 计算 `next_lsn`。
2. `recover()` 线性读取 `wal.log`，把 `SET/DELETE` 暂存到 `pending_ops[txid]`，并记录每个事务的 `COMMIT lsn`。
3. 只对 `commit_lsn > last_applied_lsn` 的事务执行重放，按日志顺序调用 `_apply_op` 修改 `state`。
4. 若恢复阶段有变更，调用 `_persist_snapshot()` 原子写入 `{state, last_applied_lsn}`。
5. 运行期 `begin()/set()/delete()` 每次都走 `_append_log()`：生成 `lsn`、写 JSON 行、`flush + fsync`。
6. `commit(txid)` 先写并刷 `COMMIT`，再把该事务内存操作应用到 `state`，最后持久化快照。
7. `inject_uncommitted_for_demo()` 专门写 `BEGIN/SET` 而不写 `COMMIT`，制造“崩溃中断事务”样本。
8. 主程序重启实例后再次 `recover()`，验证未提交事务被丢弃，且已提交事务保持可见。

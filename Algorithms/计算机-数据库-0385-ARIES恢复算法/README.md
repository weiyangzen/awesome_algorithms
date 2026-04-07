# ARIES恢复算法

- UID: `CS-0231`
- 学科: `计算机`
- 分类: `数据库`
- 源序号: `385`
- 目标目录: `Algorithms/计算机-数据库-0385-ARIES恢复算法`

## R01

ARIES（Algorithms for Recovery and Isolation Exploiting Semantics）是经典数据库恢复算法。它在 **WAL（日记先行）** 前提下，通过 `Analysis -> Redo -> Undo` 三阶段恢复，在“steal + no-force”缓冲策略下同时保证：

- 已提交事务最终可恢复（持久性）
- 未提交事务最终被回滚（原子性）
- 发生崩溃后可从日志重建一致状态（可恢复性）

## R02

ARIES解决的是“崩溃时磁盘状态可能混杂已提交与未提交写入”的问题：

- `steal` 允许脏页提前刷盘，磁盘可能含未提交事务影响。
- `no-force` 允许提交时不强制刷所有数据页，磁盘可能缺少已提交事务影响。
- 仅靠“提交标记”不足以还原完整页面状态，必须同时做 REDO 与 UNDO。

## R03

ARIES的设计前提与关键约束：

- 日志记录按 LSN 单调递增。
- 每个页面维护 `pageLSN`，表示该页已反映到哪个日志点。
- 每条更新日志有 `prevLSN`，把同一事务串成反向链表。
- WAL 约束：对应日志先落盘，再允许页面刷盘。

## R04

核心元数据结构：

- `TT`（Transaction Table）：记录活跃事务状态与 `lastLSN`。
- `DPT`（Dirty Page Table）：记录脏页首次变脏时的 `recLSN`。
- `Master Record`：保存最近一次 checkpoint 起点（`BEGIN_CKPT` LSN）。
- `CLR`（Compensation Log Record）：撤销动作本身也记日志，确保恢复过程可重入。

## R05

正常执行路径（运行期）：

1. `BEGIN` 生成事务起点日志。
2. `UPDATE` 记录页面 `before/after`，并更新 TT、DPT。
3. `COMMIT` 先写日志，表示事务逻辑完成。
4. `END` 表示事务生命周期结束。
5. 可选 `CHECKPOINT` 把 TT/DPT 快照写入日志，并更新 master 记录。

## R06

Analysis阶段的目标是“重建崩溃时刻上下文”：

- 从 master 记录指向的 checkpoint 开始扫描日志。
- 合并 `END_CKPT` 中的 TT/DPT 快照。
- 遇到 `UPDATE/CLR` 更新事务 `lastLSN` 与 DPT。
- 扫描结束后：
  - `RUNNING` 事务写 `ABORT`，转 `ABORTING`（loser）。
  - `COMMITTING` 事务补写 `END`（winner 收尾）。

## R07

Redo阶段采用“repeating history”（重复历史）：

- 从 `min(recLSN)` 起顺序扫描 `UPDATE/CLR`。
- 对每条候选记录，只有在以下条件满足时才重做：
  1. 页面在 DPT 中。
  2. 记录 `LSN >= recLSN(page)`。
  3. 磁盘页 `pageLSN < LSN`（尚未反映该更新）。
- 这样可安全重复执行而不造成重复写错（幂等）。

## R08

Undo阶段只针对 loser 事务：

- 使用“按 LSN 最大优先”的回退队列。
- 撤销 `UPDATE` 时写入 `CLR`，并应用补偿后的页面值。
- 若遇到 `CLR`，沿 `undoNextLSN` 继续回退。
- 某事务回退到头（无下一条）后写 `END`。

## R09

伪代码：

```text
analysis():
  scan from master.last_checkpoint_lsn
  rebuild TT, DPT
  for tx in TT:
    if RUNNING: append ABORT, mark ABORTING
    if COMMITTING: append END, remove tx

redo(DPT):
  start = min(DPT.recLSN)
  for log in [start..end]:
    if log is UPDATE/CLR and need_redo(log, DPT, pageLSN):
      apply_after_image(log)

undo(losers):
  heap = max(lastLSN of each loser)
  while heap not empty:
    lsn = pop_max(heap)
    rec = fetch(lsn)
    if rec is UPDATE:
      append CLR(rec)
      apply_undo(rec)
      push rec.prevLSN
    elif rec is CLR:
      push rec.undoNextLSN
    else:
      push rec.prevLSN
    if no next LSN: append END
```

## R10

复杂度（设日志长度为 `N`，loser 事务更新总数为 `U`，脏页数为 `P`）：

- Analysis：`O(N)`
- Redo：最坏 `O(N)`，实际受 `DPT` 与 `pageLSN` 筛选降低
- Undo：`O(U log T)`，`T` 为 loser 数（优先队列）
- 额外空间：`O(P + T)`（DPT + TT）

## R11

本目录 `demo.py` 的实现策略：

- 使用纯 Python 标准库实现最小可运行 MVP。
- 持久化文件：
  - `wal.jsonl`：顺序 WAL 记录
  - `data_pages.json`：磁盘页值 + `page_lsn`
  - `master.json`：最近 checkpoint 起点
- 通过“故意崩溃场景”演示 ARIES 三阶段完整链路。

## R12

运行方式：

```bash
cd Algorithms/计算机-数据库-0385-ARIES恢复算法
uv run python demo.py
```

脚本会自动：

- 构造一组已提交/未提交混合事务
- 模拟崩溃重启
- 打印 Analysis/Redo/Undo 结果
- 用断言验证最终页状态正确

## R13

MVP中的关键正确性检查：

- 未提交事务 T2 的影响会被撤销（含已偷刷到磁盘的页）。
- 已提交事务 T3 即使未刷页，仍通过 Redo 恢复。
- Undo 使用 CLR，保证“恢复中再次崩溃也可继续”。
- 通过 `pageLSN` 防止 Redo 重复改写。

## R14

边界场景与处理：

- `COMMIT` 已写但 `END` 未写：Analysis 补 `END`。
- `RUNNING` 事务崩溃：Analysis 自动补 `ABORT` 并进入 Undo。
- Checkpoint 缺失或无效：回退为从日志开头扫描。
- DPT 过估计：允许出现额外 Redo 尝试，但 `pageLSN` 会拦截无效重放。

## R15

与完整工业 ARIES 的差异（本实现有意简化）：

- 只演示整数页值更新，不含 B+Tree 等复杂页格式。
- 未实现并发锁/MVCC，仅聚焦恢复流程。
- 未实现 fuzzy checkpoint 的全部细节与日志归档。
- 未实现介质故障、校验和、归档恢复等高级能力。

## R16

可扩展方向：

- 引入页类型与生理日志（physiological logging）细化。
- 加入并发控制，与恢复协同验证隔离级别。
- 支持 checkpoint 截断与归档 WAL，降低恢复扫描成本。
- 增加故障注入测试：在 Analysis/Redo/Undo 中途再次崩溃。

## R17

与“仅 WAL 重放”相比，ARIES的增益在于：

- 不只 REDO winners，还能精确 UNDO losers。
- 使用 DPT 缩小 Redo 起点，而非总从头扫描。
- 使用 CLR 让 Undo 本身具备可恢复性，形成闭环。
- 与 steal/no-force 策略兼容，工程上更现实。

## R18

下面按 `demo.py` 的源码级流程拆成 8 步（非黑盒）：

1. `main()` 先调用 `reset_demo_files()` 清空旧的 WAL/页文件，构造可重复实验环境。
2. 运行期通过 `begin()/update()/commit()/flush_page()` 生成 `BEGIN/UPDATE/COMMIT/END` 日志，`update()` 同步维护 TT 与 DPT。
3. `create_checkpoint()` 写 `BEGIN_CKPT + END_CKPT`，并把 checkpoint 起点落到 `master.json`。
4. 构造崩溃前状态：T2 未提交且部分页面已 `flush`（steal），T3 已 `COMMIT` 但未 `END`（模拟宕机窗口）。
5. 重启后 `recover()` 先执行 `_analysis_phase()`：从 master 指向位置扫描日志，重建 DPT/TT；为 RUNNING 事务补 `ABORT`，为 COMMITTING 事务补 `END`。
6. `_redo_phase()` 从 `min(recLSN)` 开始执行 repeating history，对每条 `UPDATE/CLR` 按 `DPT + recLSN + pageLSN` 三重条件决定是否真正重做。
7. `_undo_phase()` 对 loser 事务按最大 LSN 逆序回退：遇 `UPDATE` 生成 `CLR` 并应用补偿值，沿 `prevLSN`/`undoNextLSN` 继续直到写 `END`。
8. 恢复结束后持久化 `data_pages.json`，`main()` 断言最终页面满足“winners 保留、losers 清除”，验证 ARIES 闭环成立。

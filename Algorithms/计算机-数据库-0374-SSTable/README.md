# SSTable

- UID: `CS-0220`
- 学科: `计算机`
- 分类: `数据库`
- 源序号: `374`
- 目标目录: `Algorithms/计算机-数据库-0374-SSTable`

## R01

SSTable（Sorted String Table）是“按 key 有序、不可变、顺序写入”的磁盘表结构。它通常作为 LSM-Tree 的持久化组件：内存中的 MemTable 达到阈值后被一次性刷盘为 SSTable，后续不再原地修改。

## R02

要解决的核心问题是：在高写入吞吐下，避免随机磁盘写带来的放大和抖动。SSTable 把大量随机更新转化为顺序追加和批量合并，用空间换时间，获得更稳定的写性能。

## R03

算法思想：
1. 写入先进入 MemTable（内存字典）。
2. 达到阈值后按 key 排序并刷盘成不可变文件（SSTable）。
3. 查找时先查 MemTable，再按“新表到旧表”查 SSTable。
4. 删除不立刻物理删除，而是写入 tombstone（删除标记）。
5. 周期性 compaction 合并多张 SSTable，丢弃被覆盖版本和 tombstone。

## R04

本目录 MVP 使用的数据结构：
- `memtable: Dict[str, ValueType]`：可变写缓冲。
- `table_XXXXXX.data`：按 key 有序的 JSON Line 记录文件。
- `table_XXXXXX.index`：稀疏索引，保存每隔 `sparse_step` 条记录的 key 与字节偏移。
- `table_ids`：按新到旧维护 SSTable 访问顺序，保证“最新版本优先”。

## R05

单表查询（point lookup）流程：
1. 在索引中用二分找到 `<= target_key` 的最大锚点偏移。
2. 从该偏移开始顺序扫描 `.data`。
3. 若扫描到目标 key，返回值或 tombstone。
4. 若当前 key 已大于目标 key，可提前终止（因文件全局有序）。

## R06

正确性直觉：
- 单个 SSTable 内 key 全序，故可用“二分定位 + 局部顺扫”完成查找。
- 多表按新到旧查询，最先命中的记录一定是逻辑最新版本。
- tombstone 在查询链路优先级等同普通值，故能正确屏蔽旧值。
- compaction 按新到旧保留首个版本，可维持与查询一致的语义。

## R07

复杂度（设单表记录数为 `n`，稀疏间隔为 `k`，表数量为 `L`）：
- Flush：排序 `O(n log n)`，顺序写 `O(n)`。
- 单表查找：索引二分 `O(log(n/k))` + 局部顺扫（平均与 `k` 相关）。
- 全局查找：最坏 `O(L * (log(n/k) + scan))`，命中新表时通常更快。
- Compaction：近似 `O(total_records)` 顺序读写。

## R08

适用场景：
- 写多读少或写入突发明显的 KV/时序/日志索引场景。
- 允许后台异步压实，追求吞吐稳定而非最低单次读延迟。
- 可以接受“删除延迟回收”（依赖 compaction 清理）的系统。

## R09

边界与约束：
- 空 MemTable flush 应直接返回，避免空文件。
- 多次更新同一 key 必须以“更年轻层”覆盖旧层。
- 删除后在 compaction 前仍会占据空间。
- 稀疏索引过稀会增大顺扫长度；过密会增大索引文件体积。

## R10

常见误区：
- 误把 SSTable 当作可原地更新文件；实际是 immutable。
- 忽略 tombstone，导致删除后旧值“复活”。
- compaction 合并时未按新旧优先级去重，造成版本回退。
- 仅靠 Bloom Filter 就认为不需索引；二者职责不同。

## R11

与 B-Tree 的取舍：
- B-Tree：读路径短、点查稳定，但写入会触发随机 IO 与页分裂。
- SSTable/LSM：写优化明显（顺序写），但读可能跨多层，需索引、布隆过滤器、压实策略协同。
- 工程上常按工作负载选择：写密集偏 LSM，读延迟敏感偏 B-Tree。

## R12

MVP 与工业实现差距：
- 本实现未加 WAL（崩溃恢复能力有限）。
- 未实现 Bloom Filter、分层/分级 compaction 策略。
- 未处理并发控制、压缩编码、块缓存与校验和。
- 但核心算法链路（排序刷盘、稀疏索引、tombstone、压实）已完整可验证。

## R13

`demo.py` 的最小实现组件：
- `SSTableWriter`：把排序后的 `(key, value)` 列表落盘，并生成稀疏索引。
- `SSTableReader`：基于索引偏移执行局部扫描查找。
- `LSMTreeMini`：管理 MemTable、多张 SSTable、查询优先级与 `compact()`。

## R14

运行方式：

```bash
uv run python demo.py
```

程序会在当前目录生成 `_demo_data/`，自动构造两轮写入、更新、删除与压实，并输出压实前后 SSTable 数量及样例查询结果。

## R15

预期行为：
- `banana` 被更新后读取新值 `green`。
- `cherry` 被 tombstone 删除后读取 `None`。
- compaction 后多张 SSTable 收敛为 1 张（本数据集下）。
- 所有断言通过后打印 `SSTable MVP demo completed.`。

## R16

可扩展方向：
- 增加 WAL 与重放，补齐崩溃恢复。
- 在每个 SSTable 增加 Bloom Filter，减少不存在 key 的磁盘扫描。
- 引入 leveled compaction，控制读放大与写放大平衡。
- 把 JSON Line 换为块编码（二进制）以提升存储与扫描效率。

## R17

最小测试清单：
1. 插入后读取命中。
2. 同 key 更新后返回最新值。
3. 删除后返回空。
4. 多表并存时新表覆盖旧表。
5. 压实后语义不变且表数量下降。
6. 未命中 key 返回空而非异常。

## R18

`demo.py` 的源码级算法流（8 步）：
1. `LSMTreeMini.put/delete` 写入 `memtable`，达到阈值触发 `flush()`。
2. `flush()` 调用 `SSTableWriter.write`，先按 key 排序，再顺序写 `table_xxxxxx.data`。
3. 写 `.data` 时每隔 `sparse_step` 记录一次 `{key, offset}` 到索引数组。
4. 索引数组落盘为 `table_xxxxxx.index`，形成“稀疏锚点”。
5. `get(key)` 先查 `memtable`，再按 `table_ids`（新到旧）逐表读取。
6. `SSTableReader.get` 在索引 key 上二分，seek 到起始偏移后顺序扫描到命中或提前终止。
7. 若命中 tombstone，立即返回“已删除”；若命中普通值，立即返回该值。
8. `compact()` 先 flush，再按新到旧归并首版本、丢弃 tombstone，重写为新 SSTable 并删除旧文件。

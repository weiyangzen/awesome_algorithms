# MemTable

- UID: `CS-0221`
- 学科: `计算机`
- 分类: `数据库`
- 源序号: `375`
- 目标目录: `Algorithms/计算机-数据库-0375-MemTable`

## R01

MemTable 是 LSM-Tree 写路径中的核心内存结构，职责是先承接写入，再按序批量刷盘为 SSTable。  
它本质上是“有序可变映射 + 多版本记录”，常见能力包括：
- `put(key, value)` 写入新版本；
- `delete(key)` 写入墓碑（tombstone）；
- `get(key, snapshot)` 读某个快照下的可见值；
- `range_scan(start, end, snapshot)` 有序范围读。

本条目给出一个可运行、可追踪源码流程的最小 MVP。

## R02

在典型 LSM 架构中，写入流程为：
1. 请求先进入 WAL（本 MVP 不实现持久化，仅说明语义）；
2. 再写入 MemTable；
3. MemTable 达到阈值后冻结为 Immutable MemTable；
4. 后台刷盘生成 SSTable。

因此 MemTable 要同时满足：低延迟写、按 key 有序遍历、删除可表达（墓碑）、可按序列号做快照读。

## R03

本 MVP 的问题定义：
- 输入：一组离线写操作序列（PUT/DEL）；
- 操作：点查、快照点查、范围扫描、导出刷盘候选 run；
- 输出：
1. MemTable 实现（非黑箱）；
2. 与参考模型的一致性断言；
3. flush 触发状态；
4. 可读表格（操作日志、immutable run、可见范围结果）。

实现聚焦“算法透明”，不追求完整数据库工程组件。

## R04

`demo.py` 的核心实现策略：
1. 用 `dict[key] -> list[VersionedRecord]` 存每个 key 的版本链；
2. 用 `bisect` 维护全局有序 `sorted_keys`，支持范围扫描；
3. 每次写入分配递增 `seq`，形成单调版本序；
4. 删除写入 `DEL` 记录而非物理删除；
5. 读取时按快照 `snapshot_seq` 逆序找“最新 <= snapshot”的版本；
6. flush 时导出每个 key 的最新可见记录（含墓碑），形成 immutable run。

## R05

关键数据结构：
- `VersionedRecord(seq, op, value)`：单条版本记录；
- `MemTable._versions: dict[str, list[VersionedRecord]]`：每个 key 的历史版本；
- `MemTable._sorted_keys: list[str]`：维护 key 有序性；
- `MemTable._seq`：全局递增序列号；
- `MemTable._entry_count / _approx_bytes`：flush 判定统计；
- `OperationRecord`：用于参考模型与输出展示。

## R06

正确性直觉：
- 最新读：对某 key 取“序号不超过快照的最后一条记录”；
- 墓碑语义：若最新记录为 `DEL`，则该 key 在该快照不可见；
- 范围扫描：先在有序 key 列表中截取区间，再逐 key 做快照可见性判断；
- flush 导出：每个 key 只导出一条“当前最新记录”，保持 run 的 key 有序。

这与真实 LSM 中“上层覆盖下层、墓碑屏蔽旧值”的语义一致。

## R07

设：
- `N` 为 MemTable 总记录数（版本总数）；
- `K` 为不同 key 数；
- `v_k` 为某 key 的版本数。

复杂度：
- `put/delete`：`O(log K)`（新 key 插入有序列表）+ `O(1)` 追加版本；
- `get(key)`：`O(v_k)`（逆序扫描该 key 版本链）；
- `range_scan`：`O(log K + S * avg(v_k))`，`S` 为区间内 key 数；
- 空间：`O(N)`。

该实现偏教学可读性，未做版本链索引优化。

## R08

边界与异常处理：
- `max_entries <= 0` 或 `max_bytes <= 0`：构造时抛 `ValueError`；
- `start_key > end_key`：范围扫描抛 `ValueError`；
- 对未出现 key 执行 `delete` 允许写墓碑（LSM 常见策略）；
- `snapshot_seq <= 0` 视为无可见版本并返回空结果；
- `get` 对不存在 key 返回 `None`。

## R09

MVP 设计取舍：
- 不实现跳表，改用“有序 key 列表 + 版本字典”以降低代码复杂度；
- 不实现 WAL、并发控制、刷盘文件格式；
- 保留对快照读/墓碑/flush 触发这些核心语义的最小支撑；
- 使用 `numpy` 做统计、`pandas` 做结果表展示，保持实验可读。

## R10

`demo.py` 主要函数职责：
- `MemTable.put/delete`：写入新版本并分配序号；
- `MemTable.get`：点查（支持快照）；
- `MemTable.range_scan`：按 key 区间返回可见值；
- `MemTable.freeze_to_immutable_run`：导出刷盘候选记录；
- `MemTable.should_flush`：按条数/字节阈值判断是否触发 flush；
- `reference_get/reference_range`：参考模型，用于一致性断言；
- `apply_workload_and_validate`：边执行边校验 MemTable 行为；
- `main`：运行固定工作负载并输出统计。

## R11

运行方式：

```bash
cd Algorithms/计算机-数据库-0375-MemTable
uv run python demo.py
```

脚本无交互输入，执行后会打印操作日志、状态表和校验结果。

## R12

输出解释：
- `current_seq`：最终写入序号；
- `entry_count/key_count/approx_bytes`：MemTable 规模；
- `should_flush`：是否达到 flush 阈值；
- `avg_versions_per_key`：平均版本链长度；
- `Operation log`：按序列号的写操作；
- `Immutable run`：可刷盘记录（含墓碑）；
- `Visible range`：当前快照下可见键值。

## R13

最小验证清单：
1. 每步写入后，MemTable 点查结果与参考模型一致；
2. 历史快照读（`seq-1`）与参考模型一致；
3. 删除后读不到值，重写后能读到新值；
4. 范围扫描结果与参考模型一致且按 key 有序；
5. immutable run 的 key 全局有序；
6. `should_flush()` 在设定阈值下被触发。

## R14

本实验固定工作负载（9 条）：
1. `PUT user:001=alice`
2. `PUT user:002=bob`
3. `PUT user:003=carol`
4. `PUT user:002=bobby`（更新）
5. `DEL user:003`
6. `PUT user:004=dave`
7. `PUT user:010=zoe`
8. `DEL user:999`（不存在 key 的墓碑）
9. `PUT user:003=carol_v2`（删除后重生）

阈值配置：`max_entries=9`，`max_bytes=4096`。

## R15

与相关结构对比：
- 跳表 MemTable：并发友好、查找更快，但代码复杂度更高；
- 红黑树 MemTable：有序性强、实现成熟；
- 哈希表：点查快但范围扫描弱；
- 本 MVP：强调“语义完整 + 可读可验证”，适合教学和原型验证。

## R16

适用场景：
- 需要解释 LSM 写路径语义；
- 需要演示 tombstone 与 snapshot 的可见性规则；
- 需要小规模离线验证 MemTable 行为。

不适用场景：
- 生产级高并发数据库写路径；
- 需要真实持久化（WAL/SSTable）；
- 需要百万级写入性能压测。

## R17

可扩展方向：
- 把 `_sorted_keys` 替换为跳表实现；
- 增加 WAL 持久化与崩溃恢复重放；
- 增加 mutable/immutable 双缓冲与后台 flush 线程；
- 对接 Bloom Filter 与多层 SSTable 读路径；
- 支持按时间戳或事务版本的 MVCC 可见性规则。

## R18

`demo.py` 源码级算法流（8 步）：
1. `main` 构造 `MemTable(max_entries=9, max_bytes=4096)` 与固定工作负载。  
2. `apply_workload_and_validate` 逐条执行写操作：`PUT` 调 `put`，`DEL` 调 `delete`，每次写入都会生成递增 `seq`。  
3. 写入时 `_append_record` 负责：新 key 用 `bisect.insort` 进入有序键列表；版本记录追加到该 key 的版本链；更新条数与估算字节。  
4. 每一步写入后，对所有已见 key 执行 `get(snapshot=current_seq)`，并与 `reference_get` 的结果逐项断言一致。  
5. 同时对“上一快照 `seq-1`”再次断言，验证历史可见性（快照读）语义。  
6. 写入完成后，`range_scan(start, end)` 通过二分截取 key 区间，再逐 key 取快照可见版本，过滤墓碑，得到有序可见结果。  
7. `freeze_to_immutable_run` 导出每个 key 的最新记录（包括 DEL 墓碑），用于模拟 flush 前的 immutable 视图，并校验 key 有序。  
8. `main` 汇总 `stats`、flush 触发状态与三张 `pandas` 表（操作日志/immutable run/可见范围），最终输出 `All checks passed.`。

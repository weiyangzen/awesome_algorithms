# CRDT

- UID: `CS-0318`
- 学科: `计算机`
- 分类: `并行与分布式`
- 源序号: `480`
- 目标目录: `Algorithms/计算机-并行与分布式-0480-CRDT`

## R01

问题定义：
在多副本分布式系统中，节点可能离线、网络分区、消息乱序/重复，仍希望各副本在无需全局锁或中心协调的前提下最终收敛到一致状态。

本题实现的算法是状态型 CRDT（Conflict-free Replicated Data Type）中的 `OR-Set`（Observed-Remove Set，观察删除集合），并通过可运行脚本验证其收敛与合并代数性质。

## R02

输入与输出（本 MVP）：

- 输入：`demo.py` 内置确定性事件序列 + 固定随机种子（无交互输入）。
- 输出：
  - 确定性场景事件表（每步副本值、活跃 tag 数、墓碑数）；
  - 随机 gossip 场景后 15 行轨迹与操作统计；
  - 断言结果（全部通过时输出 `All assertions passed.`）。

运行方式：

```bash
cd Algorithms/计算机-并行与分布式-0480-CRDT
uv run python demo.py
```

## R03

CRDT 核心思想：

- 每个副本独立执行本地更新；
- 副本交换状态后只做单调合并（不回滚历史）；
- 只要消息最终可达，经过有限次反熵同步后所有副本收敛。

对状态型 CRDT，关键是 `merge` 满足：

- 交换律（commutative）
- 结合律（associative）
- 幂等律（idempotent）

## R04

`OR-Set` 状态表示：

- `adds: Dict[element, Set[tag]]`：记录每次 `add(element)` 产生的唯一 tag；
- `tombstones: Set[tag]`：记录已观察到并被删除的 tag；
- tag 结构为 `(replica_id, local_counter)`，由本地单调计数器生成。

语义：

- `add(e)`：新增一个 tag 到 `adds[e]`；
- `remove(e)`：只移除当前副本“已观察到”的 `e` 的活跃 tag（加入墓碑）；
- `contains(e)`：存在至少一个 tag 属于 `adds[e]` 且不在墓碑中；
- `merge`：对 `adds` 和 `tombstones` 做并集。

## R05

主流程（`demo.py`）可拆为：

1. 定义 `ORSet` 数据结构与 `add/remove/contains/value/merge`；
2. `prove_merge_laws()` 构造 3 个副本状态并验证交换律/结合律/幂等律；
3. `run_deterministic_scenario()` 执行固定事件序列（含并发 add/remove 语义）；
4. `_sync_all()` 做全互连反熵同步，验证三副本收敛；
5. `run_random_gossip()` 用固定随机种子生成 `add/remove/merge` 混合操作；
6. 随机场景结束后再次全量同步，验证最终一致；
7. 用 `pandas` 打印事件轨迹和操作统计，给出可审计输出。

## R06

正确性要点：

- 安全性（无冲突合并）：
  - `merge` 只做集合并，不依赖消息顺序；
  - 重复合并不会改变结果（幂等）。
- 语义正确性（OR-Set）：
  - 删除仅作用于“已观察 tag”；
  - 并发新增 tag 不会被旧删除误杀。
- 收敛性：
  - 全量反熵后，各副本状态相同；
  - `demo.py` 用断言检查 `A/B/C` 最终 value 集合完全相等。

## R07

复杂度分析（设）：

- `k = |adds[e]|`（元素 `e` 的 tag 数）
- `T = 总 tag 数`

单操作复杂度：

- `add(e)`：均摊 `O(1)`；
- `remove(e)`：`O(k)`（扫描并墓碑化活跃 tag）；
- `contains(e)`：`O(k)`；
- `merge(other)`：`O(T_other)` 级别（对对方全部 add/tag 与墓碑做并集）。

空间复杂度约 `O(T + |tombstones|)`。

## R08

边界与异常语义：

- 删除不存在元素：合法，等价空操作；
- 同一元素重复 add：合法，生成多个 tag；
- 重复 merge 同一状态：结果不变；
- 消息乱序/重复/延迟：只要最终可达，收敛不受影响；
- `demo.py` 使用固定随机种子，保证实验可复现。

## R09

MVP 取舍：

- 选用单一代表性 CRDT：状态型 `OR-Set`；
- 强调“算法透明实现”，不调用第三方 CRDT 黑盒库；
- 仅用 `numpy`（随机操作生成）与 `pandas`（表格输出）；
- 不实现生产级压缩策略（如 tombstone GC、版本向量压缩、增量 Delta-CRDT 传输）。

## R10

`demo.py` 关键函数职责：

- `ORSet.add/remove/contains/value/merge`：CRDT 核心逻辑。
- `ORSet.canonical_state`：把状态转换为可比较规范表示（用于代数断言）。
- `_sync_all`：模拟 all-to-all 反熵同步。
- `prove_merge_laws`：验证交换律、结合律、幂等律。
- `run_deterministic_scenario`：固定案例（并发 add/remove）并输出步骤表。
- `run_random_gossip`：固定种子随机混合操作，最后强制同步并检查收敛。
- `main`：串联执行并打印结果。

## R11

确定性场景（脚本内置）重点：

- `A` 先看到 `B` 的 `banana(tag=B:1)` 后执行 remove；
- `B` 随后并发再 add `banana(tag=B:2)`；
- 全量同步后：
  - `B:1` 被墓碑化；
  - `B:2` 存活（并发新增）；
  - 最终三副本集合一致，且包含 `banana`。

该场景直观体现 OR-Set 的“观察删除”语义。

## R12

输出指标解释：

- `value`：该步副本当前可见集合；
- `live_tag_count`：仍存活的 tag 总数；
- `tombstone_count`：墓碑条目数；
- 随机轨迹表字段：`op/replica/peer/element/note`；
- `operation summary`：`add/remove/merge/final_sync` 各操作出现次数。

## R13

与相关一致性方法对比：

- 与共识协议（Raft/Paxos）：
  - 共识保证全序与线性化；
  - CRDT 更关注可用性与最终一致，适合弱同步场景。
- 与 LWW-Set：
  - LWW 依赖时间戳，易受时钟偏差影响；
  - OR-Set 通过 tag 与观察删除表达因果，不依赖全局时钟。
- 与 G-Set/2P-Set：
  - G-Set 不支持删除；
  - 2P-Set 删除后不可再加；
  - OR-Set 可重复 add/remove，表达力更强。

## R14

工程化注意事项：

- tombstone 会增长，生产系统需垃圾回收策略；
- tag 需要全局唯一（常见做法：副本 ID + 本地单调序号）；
- 反熵可用周期同步、对等 gossip 或增量传播；
- 输出可审计轨迹很重要，便于定位“为什么某元素仍存在/被删除”。

## R15

常见实现错误：

- 把 `remove` 写成“删除所有历史 tag”（会误杀并发新增）；
- `merge` 用覆盖而非并集（破坏幂等/交换性质）；
- tag 非唯一（不同副本生成同 tag）；
- 忽略重复消息，导致二次应用产生副作用；
- 只检查值是否相同，不检查 merge 代数性质。

## R16

最小测试清单（本脚本已覆盖）：

- 交换律：`X merge Y == Y merge X`；
- 结合律：`(X merge Y) merge Z == X merge (Y merge Z)`；
- 幂等律：`X merge X == X`；
- 并发 add/remove 场景语义断言；
- 随机 gossip 后全副本收敛断言。

## R17

目录交付内容：

- `README.md`：R01-R18 完整说明；
- `demo.py`：可运行 CRDT MVP（OR-Set + 收敛断言 + 轨迹输出）；
- `meta.json`：任务元数据与本任务一致。

运行期望：

- 输出确定性与随机两段轨迹；
- 最后打印 `All assertions passed.`。

## R18

源码级算法拆解（对应 `demo.py`，非黑盒，8 步）：

1. `main()` 先调用 `prove_merge_laws()`，构造三个 OR-Set 状态并逐条断言 merge 的交换律、结合律、幂等律。
2. `run_deterministic_scenario()` 建立 `A/B/C` 三副本，按固定顺序执行 `add/merge/remove`，其中 `remove` 只删除已观察到的 live tag。
3. `ORSet.add()` 用 `(replica_id, local_counter)` 生成唯一 tag，并写入 `adds[element]`。
4. `ORSet.remove()` 通过 `live_tags(element)` 找到当前可见 tag，放入 `tombstones`，保证删除只影响可观察历史。
5. `ORSet.merge()` 对 `adds` 与 `tombstones` 执行并集，不依赖消息先后，形成单调状态演化。
6. `_sync_all()` 执行 all-to-all 反熵传播，随后检查三副本 `value()` 完全一致，验证最终收敛。
7. `run_random_gossip()` 使用 `numpy.random.default_rng(seed)` 生成可复现的随机 `add/remove/merge` 序列，再次全量同步并断言收敛。
8. 全部轨迹由 `pandas.DataFrame` 表格化输出；第三方库只用于“随机样本生成+打印统计”，CRDT 核心决策（tag、观察删除、状态合并）完全在源码中手写实现。

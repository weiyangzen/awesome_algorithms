# slab分配器

- UID: `CS-0185`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `332`
- 目标目录: `Algorithms/计算机-操作系统-0332-slab分配器`

## R01

`slab分配器` 是操作系统内核中面向“小对象、固定尺寸、频繁分配释放”场景的内存分配策略。  
本条目目标是给出一个可运行 Python MVP，复刻其核心机制：按对象大小建立缓存、按 slab 管理对象槽位、维护 `empty/partial/full` 三态并支持回收。

## R02

核心思想是“对象缓存化 + 分层管理”：
- 按对象大小建立 `SlabCache`，避免每次都走通用分配器。
- 每个 cache 由多个 slab 组成，每个 slab 含固定数量同尺寸对象。
- 分配优先从 `partial` slab 取空槽，次选 `empty`，都没有再新建 slab。
- 释放后更新 slab 状态；空 slab 超过阈值时回收，平衡吞吐与内存占用。

## R03

MVP 输入输出约定（无交互）：
- 输入：`demo.py` 内固定工作负载参数（对象大小、slab 大小、操作次数、随机种子、分配概率）。
- 处理：执行一轮分配/释放混合 workload + 一轮确定性 sanity checks。
- 输出：分配器统计指标（操作数、峰值并发、slab 三态数量、碎片率、校验结论）。

## R04

典型应用场景：
- 内核中的 inode、dentry、task 控制块等固定结构体分配。
- 高频对象池管理（网络包描述符、文件系统元数据节点）。
- 需要降低外部碎片并减少分配路径锁竞争的系统组件。

## R05

本实现的数据结构：
- `Slab`：维护 `free_list`、`in_use` 位图语义（布尔数组）与 `payloads` 缓冲区。
- `SlabCache`：维护 `slabs` 字典和 `empty_ids/partial_ids/full_ids` 三组集合。
- `Handle`：用 `(slab_id, object_index)` 作为“对象引用”，模拟内核指针返回。

## R06

分配路径（allocate）：
1. 若存在 `partial` slab，优先选其空槽（提高局部性，减少 slab 扩张）。
2. 否则复用 `empty` slab。
3. 若仍无可用 slab，则创建新 slab 并注册到 cache。
4. 从 slab 的 `free_list` 弹出一个对象槽位并写入 payload。
5. 按槽位占用数重分类该 slab 到 `partial/full`。

## R07

释放路径（free）：
1. 通过 handle 定位 slab 与对象索引，检查越界和重复释放。
2. 标记槽位未占用并写入毒化字节（`0xDD`）辅助调试。
3. 槽位归还 `free_list`。
4. 重新计算 slab 状态并放入 `empty/partial/full` 对应集合。
5. 若空 slab 数量超过 `max_empty_slabs`，执行回收（reap）。

## R08

时间复杂度（均摊）：
- `allocate`：`O(1)`（集合选 slab + free_list 弹栈）。
- `free`：`O(1)`（定位 + 回收槽位 + 状态重分类）。
- `validate`：`O(S + C)`，`S` 为 slab 数，`C` 为总槽位数（用于审计，不在热路径）。

## R09

空间复杂度：
- 对象数据区：`O(活跃slab数 × slab_size)`。
- 元数据区：`O(活跃slab数 + 总槽位数)`，主要来自 `free_list`、状态集合和布尔标记。
- 内部碎片：按 `reserved_bytes - inuse_bytes` 度量，是 slab 方案的主要代价之一。

## R10

MVP 技术栈：
- Python 3
- `numpy`（随机工作负载与分位数统计）
- 标准库：`dataclasses`、`time`、`typing`

实现刻意保持小依赖，不依赖大型框架。

## R11

运行方式（仓库根目录）：

```bash
uv run python Algorithms/计算机-操作系统-0332-slab分配器/demo.py
```

脚本无需参数，不会请求任何交互输入。

## R12

输出字段说明：
- `Ops`：分配/释放操作计数。
- `Peak in-use objects`：峰值并发对象数。
- `Active slab states`：当前 `empty/partial/full` slab 数量。
- `Slab lifecycle`：累计创建、回收、当前存活 slab 数。
- `Memory`：预留字节、在用字节、内部碎片字节与比例。
- `Validation`：不变式检查是否通过。

## R13

边界与鲁棒性处理：
- `obj_size > slab_size` 时直接拒绝初始化（容量为 0 无意义）。
- `free` 时检测未知 slab、索引越界、double free。
- 通过 `validate()` 检查状态集合互斥、并集完整、计数一致性。
- 通过固定随机种子确保实验可复现。

## R14

当前实现局限：
- 单线程模拟，未覆盖并发锁与 CPU 本地缓存（per-CPU cache）。
- 仅实现固定对象大小 cache，未实现多 cache 大小级（kmalloc-size classes）。
- 回收策略简化为“空 slab 超阈值即回收”，未建模 NUMA/冷热分层。

## R15

可扩展方向：
- 增加多 size-class 管理器，模拟 `kmalloc-32/64/128/...`。
- 引入 per-CPU freelist，降低争用并改善局部性。
- 增加对象构造/析构回调，贴近内核 slab cache 生命周期。
- 记录延迟统计（每次 alloc/free 纳秒级），分析抖动。

## R16

最小测试建议：
- 功能性：填满一个 slab、释放再复用，验证状态迁移正确。
- 异常性：非法 handle 和 double free 必须抛异常。
- 一致性：批量 workload 后执行 `validate()` 必须无错误。
- 指标性：观察不同 `alloc_prob` 下内部碎片变化是否符合预期。

## R17

方案对比：
- 通用 `malloc`：灵活但对固定小对象不够高效，容易产生外部碎片。
- 纯 buddy 分配：擅长页级块管理，但小对象粒度开销较高。
- 简单对象池：实现容易，但缺少 slab 三态和回收策略，扩展性差。
- slab：在固定尺寸对象场景下吞吐与碎片控制更均衡，工程上常用。

## R18

`demo.py` 源码级流程（8 步）：
1. `main()` 先调用 `run_sanity_checks()`，用确定性流程校验填满/释放/复用/double free 的基础行为。
2. 创建 `SlabCache`，在 `__post_init__` 中计算 `capacity_per_slab = slab_size // obj_size` 并做参数合法性检查。
3. `run_workload()` 按随机种子逐步执行操作：若分配则走 `cache.allocate()`，否则从存活 handle 中随机选一个 `cache.free()`。
4. `allocate()` 通过 `_pick_alloc_slab()` 按 `partial -> empty -> create` 顺序选 slab，调用 `Slab.allocate()` 从 `free_list` 取槽位并写 payload。
5. 每次分配/释放后调用 `_reclassify()`，把 slab 精确放入 `empty/partial/full` 三态集合之一。
6. 释放路径中 `Slab.free()` 检查非法索引和重复释放，随后 `_reap_extra_empty_slabs()` 在空 slab 过多时执行回收。
7. 负载循环期间与结束后调用 `validate()`，审计集合互斥、并集完整、在用对象计数一致等不变式。
8. 最后 `snapshot()` 汇总内存指标并打印 `Validation: PASS/FAIL`，形成可直接验收的最小可运行实验。

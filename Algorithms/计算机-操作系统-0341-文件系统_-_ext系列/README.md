# 文件系统 - ext系列

- UID: `CS-0194`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `341`
- 目标目录: `Algorithms/计算机-操作系统-0341-文件系统_-_ext系列`

## R01

`ext` 系列是 Linux 上最重要的一支通用磁盘文件系统家族：`ext2 -> ext3 -> ext4`。  
它们共享“块组 + inode + 位图”的基本设计，但在一致性与大文件性能上逐代增强。

本条目给出一个可运行 MVP：用同一套简化模型对比三代在两件事上的行为差异：
- 文件块映射元数据开销（block map vs extent）。
- 崩溃恢复能力（无日志 vs 日志回放）。

## R02

三代核心差异可概括为：
- `ext2`：无日志，结构简单，崩溃后依赖离线检查修复（真实系统常见 `fsck`）。
- `ext3`：在 ext2 基础上引入日志（JBD），重点提升崩溃后一致性恢复速度。
- `ext4`：继续使用日志体系（JBD2），并引入 extent、延迟分配、多块分配等机制，降低碎片与元数据成本。

本 MVP 保留“演示必需差异”，不追求完整内核语义。

## R03

ext 系列的共同磁盘抽象（教学简化版）：
- `Superblock`：全局参数（块大小、总块数、状态标志等）。
- `Block Group`：把磁盘分为局部管理单元，缩短寻址路径。
- `Block Bitmap`：标记数据块占用状态。
- `Inode`：保存文件元数据与数据定位信息。
- `Data Blocks`：实际文件内容。
- `Journal(ext3/ext4)`：事务日志，崩溃后按提交边界回放。

`demo.py` 用内存结构分别对应这些对象，并保留事务提交概念。

## R04

文件数据定位策略是本题关键：

1. 块指针映射（ext2/ext3 代表性模型）
- inode 中记录大量块号。
- 元数据条目数近似等于“数据块数”。
- 逻辑简单，但大文件元数据开销高。

2. extent 映射（ext4 代表性模型）
- inode 记录“连续块区间” `(start, length)`。
- 若文件块连续，少量 extent 即可描述大量数据。
- 大幅降低映射条目数，通常也有利于顺序 I/O。

## R05

日志恢复思想（ext3/ext4）：
- 事务开始：`BEGIN(txid)`。
- 记录操作：如 `CREATE(name, data)`。
- 提交完成：`COMMIT(txid)`。
- 崩溃恢复时：仅回放“有 COMMIT 的事务”。

因此“已写入内存但未提交”的变更会被丢弃，这正是 MVP 演示的恢复差异。

## R06

本 MVP 的实验目标：
- 用相同工作负载分别运行 `ext2/ext3/ext4`。
- 比较三组指标：
  - `mapping_entries_per_data_block`：每个数据块平均需要多少映射条目。
  - `avg_contiguous_run`：连续块平均长度（越大通常越好）。
  - `volatile_visible_after_recovery`：崩溃恢复后未提交文件是否仍可见。

预期：
- ext4 的映射开销明显低于 ext2/ext3。
- ext3/ext4 能在恢复后移除未提交文件。
- ext2 在本模型中无日志回放，未提交文件仍“可见”。

## R07

形式化简化定义：

- 块大小：`B`（字节）
- 文件长度：`S`（字节）
- 需要块数：
  `N = ceil(S / B)`

映射条目开销：
- block map：`entries ~= N`
- extent：`entries = K`（`K` 为连续区间数，常有 `K << N`）

报告指标：
- `mapping_entries_per_data_block = entries / N_total`
- `avg_contiguous_run = mean(run_length_i)`

## R08

`demo.py` 主流程伪代码：

```text
for fs in [ext2, ext3, ext4]:
  初始化 MiniExtFS(fs)
  创建两个 committed 文件
  创建一个未提交 volatile 文件（只写脏态，不写 COMMIT）
  触发 recover_from_journal()
  校验 committed 文件数据完整
  统计映射开销、连续区间长度、volatile 可见性
输出文件表与汇总表
```

其中 `ext4` 使用 `_allocate_extents`，`ext2/ext3` 使用 `_allocate_block_map`。

## R09

复杂度（设磁盘块数为 `M`，单次写文件块数为 `N`）：
- `_allocate_block_map`：顺序扫描位图，`O(M)`，分配阶段 `O(N)`。
- `_allocate_extents`：先提取空闲区间 `O(M)`，再按长度排序 `O(R log R)`（`R` 为空闲区间数），分配 `O(K)`。
- 日志回放：设日志条目数 `L`，回放 `O(L + replay_cost)`。

本实验规模很小，运行时间通常在秒级内。

## R10

运行方式（无交互输入）：

```bash
uv run python Algorithms/计算机-操作系统-0341-文件系统_-_ext系列/demo.py
```

程序会打印：
- 每种文件系统的文件映射表。
- 三种系统的对比汇总表。
- 末尾一致性断言通过信息（`Checks passed`）。

## R11

如何解读输出：
- `mapping_entries`：越少表示元数据映射更紧凑。
- `mapping_entries_per_data_block`：用于跨系统横向比较。
- `run_count / max_run_len`：反映文件块连续性。
- `volatile_visible_after_recovery`：
  - `ext3/ext4` 应为 `0.0`（未提交事务被丢弃）。
  - `ext2` 在本模型中为 `1.0`（无日志回放）。

## R12

代码中的正确性检查：
- 恢复后逐文件 `read_file` 与原始 payload 做字节级比对。
- 断言 ext3/ext4 恢复后 volatile 文件不可见。
- 断言 ext4 的 `mapping_entries_per_data_block` 小于 ext2。
- 若任一条件失败，程序显式抛出 `RuntimeError`。

## R13

建模假设与边界：
- 仅模拟常规文件，不含目录层级、权限、时间戳、链接计数。
- 不实现真实 ext 的间接块树、HTree、预分配、写回缓存细节。
- 崩溃模型是“进程内重建 + 日志回放”，不是块设备级故障注入。
- 日志语义是教学化的 committed replay，不等同内核全部模式。

## R14

关键参数（见 `MiniExtFS.__init__` 与 `run_single`）：
- `total_blocks=160`：总块数。
- `block_size=64`：单块字节数。
- `reserved_blocks`：模拟元数据与历史占用，制造碎片背景。
- `alpha.bin=860B`、`beta.bin=1240B`：已提交文件。
- `volatile.tmp=780B`：故意不提交，用于恢复差异演示。

这些参数保证实验可重复且统计差异稳定。

## R15

与真实 Linux ext 实现相比，本 MVP 省略了：
- ext2/3 的直接块、单/双/三重间接块层次。
- ext4 extent B+tree 深层节点与合并/拆分策略。
- JBD/JBD2 的检查点、屏障、校验和、事务批处理。
- delayed allocation 与多队列 I/O 调度联动。

因此它是“机制解释模型”，不是“内核等价实现”。

## R16

可扩展方向：
- 加入目录 inode 与路径解析，模拟 `create/unlink/rename`。
- 增加写放大、碎片率、平均寻道距离等指标。
- 把日志细分为 metadata/data 两类并模拟 `data=ordered/writeback/journal`。
- 引入随机故障注入（断电点、部分写）做 Monte Carlo 可靠性评估。

## R17

与工程实践的连接：
- 该模型适合解释“为什么 ext4 在大文件与碎片场景下常优于 ext2/ext3”。
- 也可作为教学入口，帮助理解日志文件系统的 commit 边界语义。
- 若迁移到真实系统分析，可继续对接：
  - `dumpe2fs/debugfs` 观察超级块与组描述符。
  - `filefrag` 对比 extent 分布。
  - `fsck`/挂载日志观察崩溃恢复行为。

## R18

本条目没有把 ext 行为交给第三方黑盒库；核心流程都在 `demo.py` 源码内显式实现。  
以 `create_file + recover_from_journal` 为主线，算法级流程可拆为 8 步：

1. `create_file` 先写入 `BEGIN` 与 `CREATE` 日志项，建立事务上下文。  
2. 调用 `_apply_create_file` 计算 `block_count = ceil(size / block_size)`。  
3. 若是 `ext2/ext3`，走 `_allocate_block_map`：按位图顺序扫描并逐块占用。  
4. 若是 `ext4`，走 `_allocate_extents`：提取空闲连续区间并优先分配长区间。  
5. 把文件字节流切块写入 `self.blocks`，同时建立 inode（块列表或 extent 列表）。  
6. 若事务 `commit=True` 且启用日志，追加 `COMMIT`；否则留下“未提交脏态”。  
7. 发生“崩溃”后执行 `recover_from_journal`：重置内存状态，仅按 `BEGIN...COMMIT` 成对事务回放。  
8. 回放完成后再统计 `mapping_entries_per_data_block` 与 `volatile` 可见性，得到 ext2/ext3/ext4 差异。

这 8 步对应代码中的关键函数：`create_file`、`_allocate_block_map`、`_allocate_extents`、`recover_from_journal`。

# 文件系统 - NTFS

- UID: `CS-0195`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `342`
- 目标目录: `Algorithms/计算机-操作系统-0342-文件系统_-_NTFS`

## R01

NTFS（New Technology File System）是 Windows 系统的主力文件系统之一。  
从“算法视角”看，NTFS 的核心并不是单一排序或图算法，而是以下机制的组合：

- 用 **MFT（Master File Table）** 统一管理文件元数据；
- 用 **位图（$Bitmap）** 跟踪簇（cluster）分配状态；
- 用 **runlist（数据区间映射）** 描述非驻留数据在磁盘上的簇范围；
- 用 **日志（$LogFile）** 在崩溃后重放已提交事务，保证元数据一致性。

本目录 MVP 聚焦这些关键流程，构建一个最小可运行的 NTFS 简化实现。

## R02

NTFS 中与本题最相关的几个结构：

1. MFT 记录：每个文件对应一条记录，包含文件名、大小、属性；
2. Resident / Non-Resident 数据：
   - 小文件可驻留在 MFT 记录内（resident）；
   - 大文件以 runlist 指向磁盘簇（non-resident）；
3. 簇位图：1 表示已占用，0 表示空闲；
4. 日志条目（LSN/事务）：BEGIN -> 操作 -> COMMIT；
5. 崩溃恢复：仅重放“已 COMMIT”事务，忽略未提交事务。

这些机制共同实现“空间管理 + 元数据管理 + 一致性恢复”。

## R03

可将本问题抽象为以下状态机约束：

- 状态：
  - `bitmap[i] in {0,1}`：簇 `i` 是否占用；
  - `records[name]`：文件的 MFT 记录；
  - `journal`：按 LSN 追加的事务日志。
- 安全性约束：
  - 任何两个文件的簇集合不能重叠；
  - `resident=true` 的文件必须无 runlist；
  - `resident=false` 的文件 runlist 对应簇数应满足文件大小需求。
- 恢复约束：
  - 仅当事务存在 `COMMIT` 时，其操作才可落地。

MVP 的所有断言都围绕这些约束展开。

## R04

本实现采用“教学友好”的最小 NTFS 路线：

- 使用 `numpy` 位数组模拟簇分配位图；
- 使用 `dataclass` 表示 MFT 记录与日志项；
- 使用 **delta runlist 编码**（长度 + 相对偏移）模拟 NTFS 的区间映射思路；
- 使用 **WAL 风格**事务日志（BEGIN/CREATE_FILE/COMMIT）；
- 提供 `recover_from_journal()` 从日志重建状态，模拟崩溃恢复。

它不是完整 NTFS，但覆盖了核心算法流程且可直接运行验证。

## R05

记文件总数为 `F`，簇总数为 `C`，单文件涉及的 run 数为 `r`：

- 创建文件：
  - 分配簇在最坏情况下需扫描位图，复杂度 `O(C)`；
  - runlist 编解码为 `O(r)`。
- 读取文件：
  - resident 文件 `O(1)`；
  - non-resident 文件按 run 访问，`O(r + k)`（`k` 为读取簇数）。
- 崩溃恢复：
  - 对日志线性扫描，复杂度 `O(|journal| + 恢复阶段分配成本)`。

空间复杂度：
- 位图 `O(C)`；
- MFT 与日志约 `O(F + |journal|)`。

## R06

一次简化流程示例：

1. 事务 `T1` BEGIN，创建小文件 `tiny.txt`，写 COMMIT；
2. 小文件长度低于阈值，被写入 resident 区（无需簇分配）；
3. 事务 `T2` BEGIN，创建大文件 `manual.bin`，位图分配多个簇并写 runlist，COMMIT；
4. 事务 `T3` BEGIN，创建 `temp.log`，但崩溃前未 COMMIT；
5. 重启后执行恢复：回放 `T1/T2`，忽略 `T3`；
6. 最终可读 `tiny.txt`、`manual.bin`，`temp.log` 不存在。

这体现了 NTFS 事务恢复“只认已提交日志”的关键语义。

## R07

优点：

- 清楚展示 MFT / runlist / 位图 / 日志之间的数据流；
- 代码规模小，可快速理解 NTFS 元数据路径；
- 运行时断言可直接验证一致性与恢复语义。

局限：

- 未实现真实 NTFS 的 ACL、安全描述符、目录 B+ 树索引等高级特性；
- 未实现撤销（UNDO）与检查点裁剪；
- 簇分配策略为简单首适应，不等价于真实系统优化器。

## R08

前置知识：

- 文件系统基本概念（块/簇、inode/MFT、日志）；
- 事务与 WAL 思想；
- Python 基础数据结构。

运行环境：

- Python `>=3.10`
- `numpy`（位图与统计）
- `pandas`（MFT 表格展示）

运行：

```bash
cd Algorithms/计算机-操作系统-0342-文件系统_-_NTFS
uv run python demo.py
```

## R09

适用场景：

- 教学中解释 NTFS 的最小工作机制；
- 需要演示“日志提交与崩溃恢复”基本模型；
- 需要快速构建文件系统原型的测试骨架。

不适用场景：

- 生产级文件系统实现；
- 需要真实磁盘 I/O、并发写入、权限模型；
- 需要与 Windows NTFS 二进制格式完全兼容。

## R10

本 MVP 的关键正确性检查：

1. runlist 编码后再解码必须与原 runs 一致；
2. 非驻留文件读取数据需与写入原始字节完全一致；
3. 分配后 `bitmap` 中已占簇数必须覆盖文件 runlist 需求；
4. 恢复后仅保留已 COMMIT 事务中的文件；
5. 日志恢复前后，已提交文件内容保持一致。

`demo.py` 中通过断言自动执行这些检查。

## R11

崩溃一致性分析（简化版）：

- 若崩溃发生在 COMMIT 前：
  - 事务不应对最终状态产生影响；
- 若崩溃发生在 COMMIT 后：
  - 恢复阶段应重放该事务并恢复目标文件。

本实现通过“恢复时从空状态重放日志”保证幂等结果。  
这相当于一个偏 REDO-only 的教学模型：提交后可重做，未提交丢弃。

## R12

`demo.py` 可调参数：

- `total_clusters`：磁盘簇总数；
- `cluster_size`：单簇字节数；
- `resident_threshold`：驻留数据阈值；
- `reserved_clusters`：预占用簇（模拟系统区/坏块）。

调参建议：

- 想观察 runlist 多段化：增大文件体积并设置稀疏可用簇；
- 想观察空间不足异常：减小 `total_clusters`；
- 想让更多文件成为 resident：增大 `resident_threshold`。

## R13

理论属性说明：

- 近似比保证：N/A（非优化近似问题）；
- 概率成功保证：N/A（算法流程为确定性逻辑）；
- 可验证性质：
  - 提交可见性（commit visibility）；
  - 簇分配互斥；
  - runlist 可逆编解码。

这些性质均可在单次运行中被程序性断言检查。

## R14

常见失效模式与防护：

1. runlist 解码偏移逻辑写错 -> 读出错误簇；
2. 未检查空闲簇数量 -> 隐式越界或覆盖；
3. resident 与 non-resident 混用字段 -> 元数据不一致；
4. 恢复时误回放未提交事务 -> 幽灵文件出现。

防护措施：

- 统一 `encode_runlist_delta / decode_runlist_delta`；
- 分配前先检查可用簇数量；
- 用断言约束 resident/runlist 互斥；
- 恢复严格以 `COMMIT` 为准。

## R15

工程落地建议：

- 真实系统中应把日志持久化到独立介质区域并带校验；
- 需要加入 checkpoint 与日志截断，避免无限增长；
- 可将簇分配策略升级为 best-fit / 分层空闲链以降低碎片；
- 若扩展目录能力，可在 MFT 上增量加入 B+ 树索引结构。

本 MVP 可作为进一步实现日志文件系统实验的起点。

## R16

相关主题：

- FAT / exFAT：结构更简单，但事务一致性机制较弱；
- ext4 + journaling：同样依赖日志恢复元数据一致性；
- Copy-on-Write 文件系统（如 Btrfs/ZFS）：以写时复制替代部分日志语义；
- 数据库 WAL：与 NTFS 日志恢复思想高度同构。

学习 NTFS 有助于理解“存储系统中的事务化状态机”。

## R17

本目录交付物：

- `README.md`：R01-R18 完整说明；
- `demo.py`：可运行的简化 NTFS MVP（MFT + runlist + 位图 + 日志恢复）；
- `meta.json`：任务元信息（UID、分类、源序号）保持一致。

执行后会输出：

- 恢复后的文件列表；
- MFT 表格（含 resident/runlist 信息）；
- 位图占用统计与 runlist 片段统计；
- 全部断言通过提示。

## R18

`demo.py` 源码级流程可分为 8 步：

1. 初始化 `MiniNTFS`：建立 `numpy` 位图、日志区、MFT 映射及保留簇。  
2. `create_file()` 启动事务，写入 `BEGIN` 与 `CREATE_FILE` 日志；若提交则追加 `COMMIT`。  
3. `_apply_create_file()` 根据大小决定 resident 或 non-resident 路径。  
4. non-resident 路径调用 `_allocate_runs_first_fit()` 扫描位图分配簇，并写入 `clusters` 数据页。  
5. 分配得到的 `(start, length)` runs 通过 `encode_runlist_delta()` 压缩为 delta runlist 存入 MFT。  
6. 读取时 `read_file()` 对 runlist 调用 `decode_runlist_delta()`，按簇拼接并截断到文件原长度。  
7. `recover_from_journal()` 在“崩溃后”重置内存状态，仅回放带 `COMMIT` 的事务并重建 MFT/位图。  
8. `main()` 构造“已提交 + 未提交”事务场景，恢复后做内容一致性与可见性断言，并打印统计结果。  

该实现没有把第三方库当黑盒：核心 NTFS 机制（分配、编码、恢复）均在源码中显式展开并可逐步追踪。

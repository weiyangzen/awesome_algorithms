# 文件系统 - FAT

- UID: `CS-0193`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `340`
- 目标目录: `Algorithms/计算机-操作系统-0340-文件系统_-_FAT`

## R01

FAT（File Allocation Table）是经典文件系统家族（FAT12/FAT16/FAT32）的核心思想：
- 用一个“文件分配表”记录每个簇（cluster）的后继簇；
- 每个文件只需目录项中的 `start_cluster` + `size`，即可沿 FAT 表串出整条簇链。

本条目实现一个可运行 MVP，聚焦 FAT 的三件核心机制：
- 簇链分配（first-fit）；
- 按链读写与删除；
- 基础一致性检查（fsck 风格）。

## R02

本 MVP 的数据结构对应关系：

1. `fat: np.ndarray[int32]`
- `FREE=-1`：空闲簇；
- `EOF=-2`：链尾；
- `RESERVED=-3`：保留/坏簇；
- `>=0`：下一簇编号。

2. `directory: dict[str, DirectoryEntry]`
- 保存文件名、字节大小、起始簇。

3. `cluster_data: dict[int, bytes]`
- 保存簇编号到数据块（固定 `cluster_size`）的映射。

这与真实 FAT 的“目录项 + FAT 表 + 数据区”结构同构，但省略了 BPB、长文件名目录项等细节。

## R03

可形式化为状态机：

- 状态：
  - `fat[c]`：簇 `c` 的状态或后继；
  - `directory[name]`：文件目录项；
  - `cluster_data[c]`：簇数据。

- 关键不变量：
  - 任一文件链必须以 `EOF` 结束；
  - 文件链中不能出现环；
  - 不同文件不能共享同一数据簇（避免 cross-link）；
  - 目录项所需簇数 `ceil(size/cluster_size)` 不得大于实际可达链长。

`fsck()` 就是围绕这些不变量做自动检查。

## R04

核心算法路径：

1. `create_file(name, data)`
- 计算所需簇数 `needed = ceil(len(data)/cluster_size)`；
- 调用 `_allocate_chain_first_fit(needed)` 在 FAT 中建立链；
- 把数据按簇写入 `cluster_data`；
- 在 `directory` 写入目录项。

2. `read_file(name)`
- 从目录项取 `start_cluster`；
- 调用 `_chain_from_start()` 沿 FAT 追踪完整链；
- 拼接簇数据并截断到原始 `size`。

3. `delete_file(name)`
- 追踪文件链；
- 把链上 FAT 项重置为 `FREE`，删除对应 `cluster_data`；
- 移除目录项。

## R05

设簇总数为 `C`，单文件簇链长度为 `k`。

- `_allocate_chain_first_fit`：扫描空闲簇并取前 `k` 个，复杂度约 `O(C)`；
- `_chain_from_start`：沿链遍历，复杂度 `O(k)`；
- `read_file/delete_file`：同样由链遍历主导，为 `O(k)`；
- `fsck`：遍历所有文件链，复杂度约 `O(sum(k_i))`。

空间复杂度：
- FAT 表 `O(C)`；
- 目录项与数据区约 `O(file_count + sum(k_i))`。

## R06

`demo.py` 构造了一个“先占用、后释放、再重分配”的可重复场景：

1. 创建 `kernel.sys`、`video.raw`、`notes.txt`；
2. 删除 `video.raw`，在中间留下空洞；
3. 创建更大的 `dataset.bin`，first-fit 会优先填洞并跨区延伸；
4. 最终 `dataset.bin` 被断成多段连续区间（可见碎片化）。

这正是 FAT 在长期写删后常见的链式碎片行为。

## R07

FAT 机制优点：
- 结构直观，易实现；
- 目录项小，仅需起始簇与大小；
- 适合教学和简单设备场景。

典型局限：
- 大文件随机访问需走簇链，定位成本较高；
- 易碎片化，链段增多后顺序读写性能下降；
- 缺少日志事务语义，崩溃一致性能力较弱。

## R08

运行环境：
- Python `>=3.10`
- `numpy`
- `pandas`

运行方式（无交互）：

```bash
uv run python Algorithms/计算机-操作系统-0340-文件系统_-_FAT/demo.py
```

输出包括：
- 文件目录快照表；
- 碎片与空间利用统计；
- `fsck` 检查结果与断言状态。

## R09

适用场景：
- 操作系统/文件系统课程中的 FAT 机制演示；
- 嵌入式存储原型的最小可运行参考；
- 簇链碎片化现象的可视化教学。

不适用场景：
- 生产级 FAT 实现（含完整 on-disk 兼容）；
- 需要崩溃恢复事务保障的系统；
- 并发高负载随机 I/O 场景。

## R10

本 MVP 自动执行的正确性检查：

1. 创建后读取：`kernel.sys/notes.txt/dataset.bin` 必须字节级一致；
2. 删除后可见性：`video.raw` 必须不可读；
3. `fsck()`：`broken_files == 0` 且 `cross_link_count == 0`；
4. 场景完整性：`dataset.bin` 的 `run_count >= 2`，证明确实出现链式碎片。

任一失败都会触发 `assert` 或 `RuntimeError`。

## R11

一致性与故障语义（简化）：

- 本实现不含日志回放，等价于“非事务 FAT 语义”；
- `create_file` 与 `delete_file` 是进程内原子函数，不模拟断电中断写；
- `fsck()` 作为事后检查，能发现：
  - 环链；
  - 断链（链上指向 FREE/RESERVED）；
  - 交叉链接（多个文件共享簇）。

因此它更像“机制演示 + 离线校验”，而非完整崩溃恢复系统。

## R12

关键参数（见 `main()`）：

- `total_clusters=96`
- `cluster_size=32`
- `reserved_clusters={0,1}`
- `bad_clusters={11,12,26,27,43,60}`

以及工作负载：
- `kernel.sys=950B`
- `video.raw=640B`（随后删除）
- `notes.txt=210B`
- `dataset.bin=820B`

这些参数被刻意设置为可重复触发碎片化。

## R13

理论保证与非保证：

- 保证：
  - first-fit 分配是确定性的（同输入同输出）；
  - 链遍历有环检测，不会无界死循环；
  - `fsck` 可程序化验证核心链一致性。

- 不保证：
  - 最优碎片率（first-fit 不是全局最优）；
  - 掉电后的可恢复性（无 journal/WAL）；
  - 与 FAT32 on-disk 二进制布局完全兼容。

## R14

典型失效模式与防护：

1. 目录项起始簇越界 -> `_assert_cluster_range` 立即报错；
2. FAT 链形成环 -> `_chain_from_start` 通过 `visited` 检测；
3. FAT 指向 FREE/RESERVED -> 读链时抛错，视为损坏；
4. 两文件共享簇 -> `fsck` 的 `owned_by` 冲突计数捕获。

这四类错误覆盖了课堂上最常见的 FAT 破坏模式。

## R15

与真实 FAT12/16/32 的差异：

- 未实现 BPB/FSInfo、短名+长名目录项、时间戳与属性位；
- 未实现簇号宽度差异（12/16/28-bit）及保留值编码细节；
- 未实现 FAT 镜像副本、多扇区更新顺序与电源故障语义；
- 未实现子目录层级与路径解析。

因此该实现是“算法级骨架”，不是“磁盘格式兼容实现”。

## R16

可扩展方向：

- 增加目录树与路径解析，支持 `mkdir`/`rename`；
- 增加简单 defrag（重写链为更长连续区间）；
- 增加坏簇动态标记与跳过策略；
- 加入故障注入点（写 FAT 半途断电）并增强 `fsck` 修复策略；
- 对比 FAT 与 ext/NTFS 在碎片指标与恢复语义上的差异。

## R17

本目录交付内容：

- `README.md`：R01-R18 完整说明；
- `demo.py`：可直接运行的 FAT 最小 MVP；
- `meta.json`：元信息与本任务保持一致。

执行 `demo.py` 后可得到可解释的、可断言验证的实验输出，满足自动化验证要求。

## R18

本条目没有把核心流程交给第三方黑盒库；`numpy/pandas` 仅用于数组与表格展示，FAT 算法本身在 `demo.py` 明确实现。源码流程可拆为 8 步：

1. `MiniFAT.__init__` 初始化 FAT 表，把保留簇/坏簇标记为 `RESERVED`。  
2. `create_file` 计算 `needed_clusters = ceil(size / cluster_size)`。  
3. `_allocate_chain_first_fit` 线性扫描空闲簇并取前 `k` 个，逐项写 FAT 后继指针，尾簇写 `EOF`。  
4. `create_file` 按链顺序把字节流切块写入 `cluster_data`，并登记目录项 `start_cluster`。  
5. `read_file` 调用 `_chain_from_start`，沿 FAT 指针逐簇追踪并拼接，截断到原文件大小。  
6. `delete_file` 同样追链，把链上 FAT 项全部回收为 `FREE` 并清除数据块。  
7. `fsck` 遍历所有目录项，检查断链、环链、长度不足和 cross-link 冲突。  
8. `main` 构造“写入-删除-再写入”负载，验证读写正确性并输出碎片统计（`run_count`、`fragmented_file_ratio`）。

这 8 步就是 FAT 簇链机制在该 MVP 中的完整可追踪算法闭环。

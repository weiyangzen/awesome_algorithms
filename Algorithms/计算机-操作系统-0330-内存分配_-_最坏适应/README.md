# 内存分配 - 最坏适应

- UID: `CS-0183`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `330`
- 目标目录: `Algorithms/计算机-操作系统-0330-内存分配_-_最坏适应`

## R01

问题对象是操作系统中的连续内存动态分区分配。  
“最坏适应（Worst-Fit）”策略指: 每次为进程分配内存时，在所有可用空闲分区中选择“能够容纳且最大的那一块”。

## R02

形式化定义:
- 输入: 当前空闲/已分配分区序列 `blocks`，申请大小 `request_size`
- 约束: 仅允许从单个连续空闲块中分配，不跨块拼接
- 目标: 在满足 `block.size >= request_size` 的候选中，使 `block.size` 最大

## R03

本题 MVP 支持两类操作:
- `allocate(pid, size)`: 按最坏适应策略分配，成功返回起始地址，失败返回 `None`
- `free(pid)`: 释放该进程占用块，并与相邻空闲块合并

## R04

核心思想:
- 扫描所有空闲块，记录“当前最大块”候选
- 命中后执行“整块占用”或“拆分成已分配块 + 剩余空闲块”
- 释放时进行邻接合并，降低外部碎片对后续大申请的影响

## R05

数据结构设计（见 `demo.py`）:
- `MemoryBlock(start, size, free, pid)` 表示一个连续区间
- `WorstFitAllocator.blocks` 是按地址递增的分区列表
- 由 `start` 与 `size` 可推出 `end = start + size`

## R06

分配伪代码:

```text
worst_fit_allocate(pid, size):
    worst_index = None
    worst_size = -1
    for i, block in blocks:
        if block.free and block.size >= size:
            if block.size > worst_size:
                worst_size = block.size
                worst_index = i
    if worst_index is None:
        return FAIL
    chosen = blocks[worst_index]
    if chosen.size == size:
        mark chosen as allocated(pid)
    else:
        split chosen into [allocated(size), free(remain)]
    return allocated.start
```

## R07

释放与合并伪代码:

```text
free(pid):
    find block where block.pid == pid
    if not found: return false
    mark block as free
    while exists adjacent free neighbors:
        merge them
    return true
```

## R08

正确性直觉:
- 分配时遍历了所有可行空闲块，因此不会漏掉更大的候选
- 以“块大小最大”为判据更新候选，最终选择当前最坏适应目标块
- 释放后仅合并地址连续且均为空闲的相邻块，保持分区表示一致性

## R09

复杂度:
- 单次 `allocate`: `O(n)`，`n` 为当前分区数量
- 单次 `free`: 查找 `O(n)`，局部合并最坏 `O(n)`，总体 `O(n)`
- 空间开销: `O(n)`，用于维护分区表

## R10

碎片说明:
- 本实现关注“外部碎片”指标: `external = total_free - largest_free_block`
- 最坏适应常见动机是尽量保留较大的剩余块，避免生成太多极小碎片
- 但在某些负载下它也可能快速切碎大块，实际效果依赖请求分布

## R11

`demo.py` 的实验流程:
- 总内存 `1024`
- 固定操作序列: `alloc A/B/C/D -> free B/D -> alloc E/F/G`
- 关键观察点: 在同时存在 `300` 与 `424` 的空闲块时，最坏适应会优先选择 `424`

## R12

运行方式:

```bash
uv run python Algorithms/计算机-操作系统-0330-内存分配_-_最坏适应/demo.py
```

脚本无需交互输入，会自动打印每一步分区状态表和碎片统计。

## R13

输出解读重点:
- 每轮输出 `start/end/size/state`
- `state=FREE` 表示空闲分区，`PID=...` 表示被某进程占用
- 底部统计 `free_total / largest_free / external_frag` 用于量化碎片演化

## R14

边界与错误处理:
- 总内存、申请大小必须为正
- `pid` 不允许为空，也不允许重复分配同一 `pid`
- 释放不存在的 `pid` 返回 `False`，不抛异常
- 无可用连续块时分配失败（返回 `None`）

## R15

与最佳适应（Best-Fit）对比:
- Best-Fit: 选择最贴合请求的空闲块，追求“单次剩余最小”
- Worst-Fit: 选择最大空闲块，期望把剩余空间保持在较大尺度
- 代价: Worst-Fit 可能反复消耗最大块，导致后续大请求反而更早失败

## R16

MVP 局限:
- 仅模拟“连续分区”，不含分页/分段机制
- 未实现压缩（compaction），因此外部碎片会持续累积
- 未考虑并发与锁，不适用于真实内核态并发场景

## R17

可扩展方向:
- 增加 First-Fit/Best-Fit/Next-Fit 同场景对照
- 增加随机工作负载生成器，输出长期失败率与碎片曲线
- 引入平衡树或分桶结构，将线性扫描优化为更高效的候选查询

## R18

源码级算法流（对应 `demo.py`）:
1. `main()` 创建 `WorstFitAllocator(1024)`，初始只有一个空闲块 `[0,1024)`。
2. 每次 `allocate(pid, size)` 先做输入合法性检查（`pid` 非空、`size` 正数、`pid` 不重复）。
3. 分配阶段线性扫描 `self.blocks`，对满足 `free && block.size >= size` 的块维护 `worst_index/worst_size`，始终保留“当前最大候选块”。
4. 若无候选则返回 `None`；若命中块等长则直接标记为已分配，否则分裂成 `[allocated, residue_free]` 替换原块。
5. `free(pid)` 将目标块设为空闲后调用 `_coalesce_around()`，循环合并地址连续的相邻空闲块，减少碎片。
6. `print_state()` 在每步后输出分区表与 `external_fragmentation = total_free - largest_free`，用于观察最坏适应策略对碎片和可分配性的影响。

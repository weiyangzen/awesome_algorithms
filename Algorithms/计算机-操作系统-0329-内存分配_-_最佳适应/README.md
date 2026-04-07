# 内存分配 - 最佳适应

- UID: `CS-0182`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `329`
- 目标目录: `Algorithms/计算机-操作系统-0329-内存分配_-_最佳适应`

## R01

问题对象是操作系统中的连续内存动态分区分配。  
“最佳适应（Best-Fit）”策略指: 每次为进程分配内存时，在所有可用空闲分区中选择“能够容纳且剩余最小”的那一块。

## R02

形式化定义:
- 输入: 当前空闲/已分配分区序列 `blocks`，申请大小 `request_size`
- 约束: 仅允许从单个连续空闲块中分配，不跨块拼接
- 目标: 在满足 `block.size >= request_size` 的候选中，使 `block.size - request_size` 最小

## R03

本题 MVP 支持两类操作:
- `allocate(pid, size)`: 按最佳适应策略分配，成功返回起始地址，失败返回 `None`
- `free(pid)`: 释放该进程占用块，并与相邻空闲块合并

## R04

核心思想:
- 扫描所有空闲块，记录“最小浪费”候选
- 命中后执行“整块占用”或“拆分成已分配块 + 剩余空闲块”
- 释放时进行邻接合并，减少外部碎片的进一步恶化

## R05

数据结构设计（见 `demo.py`）:
- `MemoryBlock(start, size, free, pid)` 表示一个连续区间
- `BestFitAllocator.blocks` 是按地址递增的分区列表
- 由 `start` 与 `size` 可推出 `end = start + size`

## R06

分配伪代码:

```text
best_fit_allocate(pid, size):
    best_index = None
    best_waste = +inf
    for i, block in blocks:
        if block.free and block.size >= size:
            waste = block.size - size
            if waste < best_waste:
                best_waste = waste
                best_index = i
    if best_index is None:
        return FAIL
    chosen = blocks[best_index]
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
- 分配时遍历了所有可行空闲块，因此不会漏掉更优候选
- 以“最小剩余”为判据更新候选，最终得到局部最优匹配块
- 释放后仅合并地址连续且均为空闲的相邻块，保持分区表示一致性

## R09

复杂度:
- 单次 `allocate`: `O(n)`，`n` 为当前分区数量
- 单次 `free`: 查找 `O(n)`，局部合并最坏 `O(n)`，总体 `O(n)`
- 空间开销: `O(n)`，用于维护分区表

## R10

碎片说明:
- 本实现关注“外部碎片”指标: `external = total_free - largest_free_block`
- 该值越大，说明总空闲容量虽然足够，但被切碎得越严重
- 连续分配模型下这类碎片通常是主要失败原因

## R11

`demo.py` 的实验流程:
- 总内存 `1024`
- 依次执行固定操作序列: `alloc A/B/C/D -> free B/D -> alloc E/F/G`
- 特意构造“释放后再分配”场景，观察最佳适应如何优先吃掉更贴合的小空闲块

## R12

运行方式:

```bash
uv run python Algorithms/计算机-操作系统-0329-内存分配_-_最佳适应/demo.py
```

脚本无需交互输入，会自动打印每一步分区状态表和碎片统计。

## R13

输出解读重点:
- 每轮输出 `start/end/size/state`
- `state=FREE` 表示空闲分区，`PID=...` 表示被某进程占用
- 底部统计 `free_total / largest_free / external_frag` 便于量化碎片变化

## R14

边界与错误处理:
- 总内存、申请大小必须为正
- `pid` 不允许为空，也不允许重复分配同一 `pid`
- 释放不存在的 `pid` 返回 `False`，不抛异常
- 无可用连续块时分配失败（返回 `None`）

## R15

与首次适应（First-Fit）对比:
- First-Fit: 从低地址开始找到第一块可用即分配，速度实现更直接
- Best-Fit: 全表扫描后选最贴合块，通常减少单次浪费
- 代价: Best-Fit 常引入更多小碎片，长期表现依赖负载模式

## R16

MVP 局限:
- 仅模拟“连续分区”，不含分页/分段机制
- 未实现压缩（compaction），因此外部碎片会长期累积
- 未考虑并发与锁，不适用于真实内核态并发场景

## R17

可扩展方向:
- 增加 First-Fit/Worst-Fit/Next-Fit 并做对照实验
- 增加随机工作负载生成器，输出长期碎片曲线
- 引入平衡树或分桶结构，将分配查询从线性扫描优化到对数级/近似常数级

## R18

源码级算法流（对应 `demo.py`）:
1. `main()` 创建 `BestFitAllocator(1024)`，初始化为单个空闲块 `[0,1024)`。
2. 每次 `allocate(pid, size)` 先校验参数，再线性扫描 `self.blocks`，对所有满足 `free && block.size >= size` 的块计算 `waste`。
3. 扫描中维护 `best_index/best_waste`，始终保存“当前最小浪费”候选。
4. 若无候选则返回 `None`；若命中块刚好等长则直接标记为已分配，否则执行分裂替换: `[allocated, residue_free]`。
5. `free(pid)` 找到目标块后标记为空闲，并调用 `_coalesce_around()`，循环合并地址连续的相邻空闲块。
6. `print_state()` 在每步后输出分区表与 `external_fragmentation = total_free - largest_free`，用于观察策略效果与碎片演化。

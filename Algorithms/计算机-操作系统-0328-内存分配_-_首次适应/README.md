# 内存分配 - 首次适应

- UID: `CS-0181`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `328`
- 目标目录: `Algorithms/计算机-操作系统-0328-内存分配_-_首次适应`

## R01

问题对象是操作系统中的连续内存动态分区分配。  
“首次适应（First-Fit）”策略指: 每次为进程分配内存时，从低地址到高地址扫描空闲分区，找到第一块能够容纳请求的分区并立即分配。

## R02

形式化定义:
- 输入: 当前空闲/已分配分区序列 `blocks`，申请大小 `request_size`
- 约束: 仅允许从单个连续空闲块中分配，不跨块拼接
- 目标: 在地址顺序扫描下，选择第一个满足 `block.free && block.size >= request_size` 的分区

## R03

本题 MVP 支持两类操作:
- `allocate(pid, size)`: 按首次适应策略分配，成功返回起始地址，失败返回 `None`
- `free(pid)`: 释放该进程占用块，并与相邻空闲块合并

## R04

核心思想:
- 维护按地址递增的分区列表
- 分配时做“一次线性前向扫描”，命中第一块即可停止扫描
- 释放时进行邻接合并，避免空闲分区被永久切碎

## R05

数据结构设计（见 `demo.py`）:
- `MemoryBlock(start, size, free, pid)` 表示一个连续区间
- `FirstFitAllocator.blocks` 是按地址递增的分区列表
- 由 `start` 与 `size` 可推出 `end = start + size`

## R06

分配伪代码:

```text
first_fit_allocate(pid, size):
    for i, block in blocks by address order:
        if block.free and block.size >= size:
            if block.size == size:
                mark block as allocated(pid)
            else:
                split block into [allocated(size), free(remain)]
            return allocated.start
    return FAIL
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
- 线性扫描按地址顺序进行，第一块可行分区即为首次适应定义下的唯一命中目标
- 分裂逻辑保持地址覆盖不重叠且无空洞，维持分区表一致性
- 释放时仅合并地址连续且同为空闲的块，保证内存区间表示始终合法

## R09

复杂度:
- 单次 `allocate`: `O(n)`，`n` 为当前分区数量；最优情况下可提前命中提前结束
- 单次 `free`: 查找 `O(n)`，局部合并最坏 `O(n)`，总体 `O(n)`
- 空间开销: `O(n)`，用于维护分区表

## R10

碎片说明:
- 本实现关注“外部碎片”指标: `external = total_free - largest_free_block`
- First-Fit 常见现象是低地址区间容易被频繁切分，形成“前部碎片带”
- 即使总空闲量足够，若最大连续空闲块不足，也会导致大请求失败

## R11

`demo.py` 的实验流程:
- 总内存 `1024`
- 固定操作序列: `alloc A/B/C/D -> free B/D -> alloc E/F/G`
- 关键观察点: 当同时存在 `300` 与 `424` 的空闲块时，首次适应会先使用地址更低的 `300`

## R12

运行方式:

```bash
uv run python Algorithms/计算机-操作系统-0328-内存分配_-_首次适应/demo.py
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

与最佳适应/最坏适应对比:
- First-Fit: 找到第一块就分配，策略简单，常见实现开销较低
- Best-Fit: 选最贴合块，单次浪费小，但可能制造大量小碎片
- Worst-Fit: 选最大块，试图保持较大剩余块，但也可能反复消耗大块

## R16

MVP 局限:
- 仅模拟“连续分区”，不含分页/分段机制
- 未实现压缩（compaction），外部碎片会持续累积
- 未考虑并发与锁，不适用于真实内核态并发场景

## R17

可扩展方向:
- 增加 Next-Fit 与伙伴系统（Buddy System）作对照
- 增加随机工作负载生成器，输出长期碎片与失败率曲线
- 引入平衡树或分桶结构优化候选查询，降低大规模下的扫描代价

## R18

源码级算法流（对应 `demo.py`）:
1. `main()` 创建 `FirstFitAllocator(1024)`，初始化为单个空闲块 `[0,1024)`，然后按预设计划依次执行分配/释放。
2. 每次 `allocate(pid, size)` 先做输入校验（`pid` 非空、`size` 正数、`pid` 不重复），再从 `self.blocks` 的低地址开始线性扫描。
3. 扫描命中第一块满足 `free && block.size >= size` 的分区后立即停止；若无命中则返回 `None`。
4. 命中块若等长则直接标记为已分配，否则将其替换为 `[allocated, residue_free]` 两块，完成连续分区切分。
5. `free(pid)` 找到对应已分配块后置为空闲，并调用 `_coalesce_around()` 循环合并相邻空闲块，恢复更大连续空间。
6. `print_state()` 在每步后输出分区表与 `external_fragmentation = total_free - largest_free`，用于观察首次适应的碎片演化特征。

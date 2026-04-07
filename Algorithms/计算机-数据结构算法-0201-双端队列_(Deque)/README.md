# 双端队列 (Deque)

- UID: `CS-0090`
- 学科: `计算机`
- 分类: `数据结构算法`
- 源序号: `201`
- 目标目录: `Algorithms/计算机-数据结构算法-0201-双端队列_(Deque)`

## R01

双端队列（Deque, Double-Ended Queue）是一种允许在两端都进行插入与删除的线性数据结构。

- 队首（left/front）可 `append_left` 与 `pop_left`；
- 队尾（right/back）可 `append_right` 与 `pop_right`；
- 目标是让四类操作在均摊意义上保持 `O(1)`。

## R02

Deque 是栈（LIFO）和队列（FIFO）的统一抽象。

- 只用右端操作时，Deque 退化为栈；
- 左入右出（或右入左出）时，Deque 退化为队列；
- 因此它常用于需要“窗口两端都变动”的场景。

## R03

本题 MVP 采用“环形缓冲区 + 动态扩容”来实现 Deque。

- 存储：定长数组 `buffer`；
- 指针：`head` 指向当前队首元素；
- 规模：`size` 表示有效元素个数；
- 访问下标统一通过模运算映射到真实数组位置。

## R04

核心操作定义如下：

1. `append_right(x)`：在尾部追加元素。
2. `append_left(x)`：在头部追加元素。
3. `pop_right()`：删除并返回尾部元素。
4. `pop_left()`：删除并返回头部元素。
5. `to_list()`：按逻辑顺序导出当前序列（用于验证）。

## R05

复杂度分析（环形数组实现）：

- `append_left/append_right`：均摊 `O(1)`；
- `pop_left/pop_right`：`O(1)`；
- 空间复杂度：`O(n)`；
- 扩容触发时会发生 `O(n)` 搬移，但摊到多次插入后仍是均摊 `O(1)`。

## R06

`demo.py` 中的实现对象 `RingDeque` 包含以下状态：

- `_buffer: list[Optional[int]]`：底层数组；
- `_head: int`：逻辑首元素索引；
- `_size: int`：当前元素个数；
- `_grow()`：容量翻倍并重排为从 0 开始的连续布局。

这是一种最小但完整的 Deque 教学实现。

## R07

正确性依赖三个不变量：

1. `0 <= size <= capacity`；
2. 非空时，逻辑第 `k` 个元素位于 `(head + k) % capacity`；
3. 每次扩容后，逻辑顺序保持不变。

`demo.py` 在运行中会通过与 `collections.deque` 对照来持续校验这些不变量的外部行为。

## R08

动态扩容策略：

- 当 `size == capacity` 时触发扩容；
- 新容量取 `2 * old_capacity`；
- 按逻辑顺序将旧数据复制到新数组 `[0, size)`；
- 将 `head` 复位为 `0`。

该策略避免了频繁重分配，且使后续索引计算保持简单。

## R09

为了避免“只写实现不验行为”，MVP 采用双轨执行：

- 轨道 A：自实现 `RingDeque`；
- 轨道 B：标准库 `collections.deque` 作为参考语义；
- 每次 `pop` 返回值必须一致；
- 运行结束后最终序列必须完全一致。

## R10

实验数据由 `numpy` 生成，保证可复现：

- 使用固定随机种子 `seed=2026`；
- 生成操作序列（左入/右入/左出/右出）；
- 生成插入值序列；
- 对空结构上的 `pop` 进行保护性改写（改为 `append`），避免无效异常污染统计。

## R11

运行方式（无交互）：

```bash
cd Algorithms/计算机-数据结构算法-0201-双端队列_(Deque)
uv run python demo.py
```

脚本会打印操作计数、耗时对比、最终长度与序列头尾样本。

## R12

输出指标说明：

- `implementation`：实现名称（`RingDeque` / `collections.deque`）；
- `steps`：总操作步数；
- `elapsed_ms`：执行耗时（毫秒）；
- `final_size`：最终元素数；
- `size_delta`：`append` 次数减去 `pop` 次数，应等于 `final_size`。

此外会附带操作分布统计，帮助检查工作负载是否均衡。

## R13

`demo.py` 内置的门禁断言：

1. 两种实现最终 `size` 一致；
2. 两种实现最终内容完全一致；
3. `size_delta == final_size`；
4. 在执行过程中，任意 `pop` 的返回值必须一致。

这些断言可防止“看似能跑但语义已偏移”的情况。

## R14

常见边界与处理方式：

- 空 Deque 执行 `pop`：应抛出 `IndexError`（本演示通过 workload 生成避免空弹出）；
- 连续大量左插或右插：依赖环形下标与扩容机制维持正确顺序；
- 扩容后立即两端弹出：可检验搬移后的索引映射是否正确。

## R15

Deque 的典型应用：

- 滑动窗口最大值/最小值（单调队列）；
- BFS 层序遍历与 0-1 BFS；
- 任务调度中的双端优先策略；
- 文本编辑器、撤销/重做缓冲。

## R16

本 MVP 的限制：

- 仅演示整数元素（类型可扩展为泛型）；
- 未实现线程安全；
- 未实现自动缩容（避免抖动，保持代码简洁）。

这些限制不影响 Deque 核心算法与复杂度结论。

## R17

可扩展方向：

1. 增加泛型与迭代器协议；
2. 加入可选缩容策略（如低于 1/4 容量时减半）；
3. 增加批量操作与切片视图；
4. 引入基准测试（不同负载比例、不同容量增长因子）。

## R18

源码级算法链路（8 步，覆盖本实现与标准库参考实现）：

1. `main()` 调用 `generate_workload()`，用 `numpy.random.default_rng` 生成可复现操作流与数值流。  
2. `run_ring_deque()` 初始化 `RingDeque`，按操作流驱动 `append_left/right` 或 `pop_left/right`。  
3. 每次插入前 `RingDeque._ensure_capacity_for_push()` 检查是否满载；若满则调用 `_grow()`。  
4. `_grow()` 按逻辑顺序复制旧数组到新数组并复位 `head=0`，确保后续模索引不改变语义顺序。  
5. `pop_left/right` 基于 `(head + offset) % capacity` 定位元素并更新 `head/size`，形成 `O(1)` 两端删除。  
6. `run_reference_deque()` 对同一 workload 调用 `collections.deque` 的同名语义操作，作为行为基线。  
7. 在 CPython 源码中，`collections.deque` 由 `Modules/_collectionsmodule.c` 实现，底层是由固定大小 block 组成的双向链式块结构；两端插删通过移动边界指针并按需申请/释放 block 完成。  
8. `main()` 汇总 `pandas.DataFrame` 指标并执行断言：两实现返回值轨迹与最终序列一致，从而闭环验证 Deque 语义与实现正确性。

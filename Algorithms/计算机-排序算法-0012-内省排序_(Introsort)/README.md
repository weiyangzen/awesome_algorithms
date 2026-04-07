# 内省排序 (Introsort)

- UID: `CS-0012`
- 学科: `计算机`
- 分类: `排序算法`
- 源序号: `12`
- 目标目录: `Algorithms/计算机-排序算法-0012-内省排序_(Introsort)`

## R01

内省排序（Introsort）是“混合比较排序”算法：
- 先用快速排序获得平均情况下的高性能；
- 当递归深度过深时，切换到堆排序，避免最坏 `O(n^2)` 退化；
- 对小区间使用插入排序，降低常数开销。

本目录 MVP 目标：
- 用纯 Python 手写 Introsort 主流程，不依赖内置排序黑箱；
- 输出比较次数、交换次数、分区次数与回退次数；
- 用 `numpy` / `pandas` 仅做结果校验和可读展示。

## R02

历史与工程定位：
- Introsort 由 David Musser 提出，核心思想是“对快排做深度监控并自救”；
- C++ 标准库 `std::sort` 的经典实现路线就是 Introsort 家族；
- 它结合了快排平均快、堆排最坏有上界、插排处理小数组高效三者优势。

因此 Introsort 常被视为工程级默认比较排序方案之一。

## R03

问题定义：
- 输入：长度为 `n` 的可比较序列 `A`（本实现限定为 1D 有限数值序列）。
- 输出：非降序序列 `A'`，满足：
  - 有序性：`A'[0] <= A'[1] <= ... <= A'[n-1]`
  - 排列保持：`A'` 与 `A` 元素多重集一致。

Introsort 通过“分治 + 回退 + 小区间优化”完成该目标。

## R04

核心思想（本实现版本）：
1. 深度上限设为 `2 * floor(log2(n))`。
2. 当前区间长度大于阈值（默认 `16`）时，执行一次快排分区。
3. 分区时使用三数取中（首、中、尾）选枢轴，降低劣质枢轴概率。
4. 每深入一层分区，深度预算减 1。
5. 若深度预算耗尽，立即对该区间改用堆排序（回退保护）。
6. 当区间长度不大于阈值，改用插入排序收尾。

## R05

实现中的状态量与数据结构：
- 主数组：`arr`（列表，原地排序）。
- 区间边界：`lo/hi` 表示半开区间 `[lo, hi)`。
- 深度预算：`depth_limit`。
- 小区间阈值：`insertion_threshold=16`。
- 统计字段（`IntrosortStats`）：
  - `comparisons`、`swaps`、`partitions`
  - `heapsort_fallbacks`、`insertion_calls`
  - `trace`（记录 partition / fallback 事件）。

## R06

正确性说明（分治不变式）：
- 分区正确性：`_partition` 返回位置 `p` 后，满足：
  - `[lo, p)` 中元素都 `< pivot`
  - `arr[p] == pivot`
  - `(p, hi)` 中元素都 `>= pivot`
- 递归/迭代推进：分别对左右子区间继续执行同样过程，直到区间足够小。
- 回退正确性：若触发堆排，`_heapsort_range` 可保证该区间完全有序。
- 收尾正确性：插入排序能对小区间得到稳定有序结果（算法整体仍不稳定）。

因此每个子区间最终有序，进而全局有序。

## R07

时间复杂度：
- 平均：`O(n log n)`（快排主导）；
- 最坏：`O(n log n)`（深度耗尽时回退堆排，避免快排最坏退化）；
- 小区间收尾：插排在长度阈值内，整体只影响常数项。

## R08

空间复杂度：
- 额外工作空间为 `O(log n)` 级（分治调用栈，且采用“先处理小区间”策略压低栈深）；
- 堆排与插排都在原数组上进行，不创建同规模辅助数组；
- `trace` 仅用于演示，属于可选调试开销。

## R09

算法性质：
- 比较排序：是。
- 原地排序：是（除统计/轨迹外）。
- 稳定性：否。
- 自适应性：有限（主要收益来自阈值插排与枢轴策略，不如 TimSort 对 runs 的自适应强）。
- 工程特性：在“平均速度 + 最坏上界”之间平衡较好。

## R10

边界与输入约束：
- 空数组、单元素：直接返回；
- 重复值、负数、浮点数：支持；
- `NaN/Inf`：拒绝（通过 `validate_numeric_sequence` 校验）；
- 非 1D 数据或字符串输入：拒绝；
- 保留输入不变：`introsort` 会复制输入后再原地排序副本。

## R11

伪代码：

```text
introsort(A):
  depth_limit <- 2 * floor(log2(len(A)))
  introsort_loop(A, 0, n, depth_limit)

introsort_loop(A, lo, hi, depth):
  while hi - lo > threshold:
    if depth == 0:
      heapsort(A[lo:hi])
      return
    depth <- depth - 1
    p <- partition_with_median_of_three(A, lo, hi)
    sort smaller side first (recursive)
    continue with larger side (iterative)
  insertion_sort(A[lo:hi])
```

## R12

本目录 MVP 的实现策略：
- 手写 `_median_of_three` + `_partition` + `_heapsort_range` + `_insertion_sort_range`；
- `introsort` 返回 `sorted_values + IntrosortStats`，便于审计；
- 通过 `numpy.sort` 与 `sorted` 双重交叉验证正确性；
- 用 `pandas.DataFrame` 打印关键事件轨迹（非必须，仅用于展示）。

## R13

`demo.py` 输出字段：
- `Input`：原始输入；
- `Sorted by introsort`：手写 Introsort 结果；
- `Expected sorted`：Python 内置参考结果；
- `Stats`：
  - `n`、`depth_limit`
  - `comparisons`、`swaps`、`partitions`
  - `heapsort_fallbacks`、`insertion_calls`
  - `trace_events`
- `Trace table`：最近若干条分区/回退事件。

## R14

内置测试样例：
- Case 1：固定混合数组（重复值、负数、浮点）；
- Case 2：固定随机种子整数数组；
- Case 3：逆序数组；
- Case 4：高重复值数组；
- Case 5：强制 `max_depth_override=0`，显式覆盖“堆排回退路径”。

每个样例都断言：
- 结果等于 `sorted(values)`；
- 结果等于 `np.sort(values)`；
- 输出满足非降序。

## R15

与常见排序对比：
- 对比快速排序：
  - 平均复杂度同为 `O(n log n)`；
  - Introsort 通过深度上限规避快排最坏 `O(n^2)`。
- 对比堆排序：
  - 堆排最坏有上界，但平均常数通常偏大；
  - Introsort 仅在必要时回退堆排，平均性能更接近快排。
- 对比 TimSort：
  - TimSort 稳定且对已排序 runs 自适应更强；
  - Introsort 不稳定，但实现路径更贴近传统比较排序三件套。

## R16

适用场景：
- 需要通用比较排序，且希望有最坏复杂度保障；
- 希望在工程上兼顾速度与稳健性；
- 教学中演示“算法混合 + 退化保护 + 参数阈值”设计思想。

不适合：
- 明确要求稳定排序的业务（应考虑归并/TimSort 路线）。

## R17

运行方式（无交互）：

```bash
cd "Algorithms/计算机-排序算法-0012-内省排序_(Introsort)"
uv run python demo.py
```

运行结果会打印 5 组样例的排序结果、统计信息和部分轨迹，并最终输出 `All checks passed.`。

## R18

`demo.py` 源码级流程（8 步，非黑箱）：
1. `main()` 构造 5 组确定性样例，其中一组强制深度为 0 用于验证回退路径。  
2. `run_case()` 调用 `validate_numeric_sequence()`，将输入转成有限 1D 浮点列表。  
3. `introsort()` 计算默认深度上限 `2*floor(log2(n))`，初始化计数器与 `trace`。  
4. `_introsort_loop()` 在区间大于阈值时循环执行：若深度耗尽则记录事件并调用 `_heapsort_range()`。  
5. 未耗尽时，`_partition()` 先用 `_median_of_three()` 选枢轴，再用 Lomuto 分区并记录 `pivot` 位置与子区间规模。  
6. `_introsort_loop()` 始终先递归较小子区间、迭代较大子区间，控制调用栈增长。  
7. 当区间长度不超过阈值，调用 `_insertion_sort_range()` 完成小区间收尾排序。  
8. 返回后仅把 `numpy.sort` / `sorted` 作为校验器；`pandas` 仅用于把 `trace` 渲染成表格，核心排序完全由源码中的三类子过程实现。

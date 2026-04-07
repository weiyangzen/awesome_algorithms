# TimSort

- UID: `CS-0011`
- 学科: `计算机`
- 分类: `排序算法`
- 源序号: `11`
- 目标目录: `Algorithms/计算机-排序算法-0011-TimSort`

## R01

TimSort 是一种工程化的稳定比较排序，核心思想是“利用已有有序片段（runs）再合并”。

本目录 MVP 目标：
- 用纯 Python 手写 TimSort 主流程，而不是直接调用内置排序黑箱；
- 真实包含 run 检测、短 run 插入扩展、run 栈合并策略；
- 输出可审计的统计信息与轨迹（run_trace / merge_trace）；
- 用 `numpy` 与 `pandas` 做校验和可读展示。

## R02

工程背景：
- TimSort 由 Tim Peters 为 Python 列表排序设计；
- 它将“插入排序 + 归并排序 + 自适应 run 检测”组合到同一框架；
- 现实数据常有局部有序性，TimSort 能显著减少不必要的比较与移动；
- 算法保持稳定性，因此在需要“按主键排序后保留次序”的业务里非常常见。

## R03

问题定义：
- 输入：长度为 `n` 的 1D 有限数值序列 `A`。
- 输出：`A'`，满足：
  - 非降序：`A'[0] <= A'[1] <= ... <= A'[n-1]`
  - 多重集不变：`A'` 与 `A` 元素一致
  - 稳定性：若值相等，原相对顺序保持不变（本实现在 merge 和插入处都按稳定规则处理）。

## R04

本实现的 TimSort 风格主线：
1. 计算 `minrun`（`_calc_minrun`）。
2. 从左到右扫描，检测自然 runs（升序 / 严格降序）。
3. 若检测到严格降序 run，则原地反转为升序 run。
4. 若 run 长度小于 `minrun`，用二分插入排序扩展到目标长度。
5. 将 run 压入 run 栈。
6. 每次压栈后执行 `merge_collapse`，维持 run 长度不变量。
7. 扫描结束后执行 `merge_force_collapse` 合并剩余 runs。
8. 最终栈中只剩一个 run，即全数组有序。

## R05

实现中的关键状态与结构：
- 主数组：`arr`（原地排序）。
- run 栈：`stack: list[(start, length)]`。
- 统计字段（`TimSortStats`）：
  - `comparisons`、`moves`、`swaps`
  - `run_detections`、`reversals`、`insertion_extensions`
  - `merges`、`max_stack_size`
  - `run_trace`、`merge_trace`
- 轨迹作用：
  - `run_trace` 记录 run 的检测方向、扩展情况、压栈后栈高度；
  - `merge_trace` 记录每次 merge 的左右 run 长度与合并后长度。

## R06

正确性要点：
- run 归一化正确：
  - 升序 run 原样保留；
  - 严格降序 run 反转后变为升序。
- 插入扩展正确：
  - `_binary_insertion_sort` 保证扩展后的区间升序。
- 合并正确：
  - `_merge_at` 只合并相邻连续 runs；
  - 合并后 run 覆盖原两段，且保持升序。
- 全局正确：
  - 所有 runs 被持续合并到只剩一个 run `(0, n)`，因此全局有序。

稳定性说明：
- merge 时使用 `<=` 优先取左 run 元素；
- 插入排序二分定位采用“等值向右插入”策略；
- 因此等值元素原相对次序不被破坏。

## R07

时间复杂度（TimSort 一般结论 + 本实现行为）：
- 最好情况：接近 `O(n)`（输入已由长 runs 组成）；
- 平均情况：`O(n log n)`；
- 最坏情况：`O(n log n)`（归并主导）。

## R08

空间复杂度：
- 额外 run 栈：`O(log n)` 到 `O(n/minrun)` 级，远小于 `n`；
- merge 临时缓冲：每次复制左 run，额外 `O(k)`（`k` 为当前左 run 长度）；
- 整体峰值额外空间在归并阶段体现，不是严格原地排序。

## R09

算法性质：
- 比较排序：是。
- 稳定性：是。
- 原地性：否（merge 使用临时缓冲）。
- 自适应性：强（直接利用已有 runs）。
- 工程定位：通用、稳定、对真实数据友好。

## R10

边界与输入约束：
- 空数组、单元素：直接返回。
- 负数、浮点数、重复值：支持。
- 非有限值（`NaN` / `Inf`）：拒绝。
- 非 1D 输入：拒绝。
- 字符串/字节串：拒绝。
- `minrun_override < 2`：抛错。

## R11

伪代码：

```text
timsort(A):
  minrun <- calc_minrun(len(A))
  stack <- []
  i <- 0

  while i < n:
    run_len <- count_run_and_make_ascending(A, i)
    force <- min(minrun, n - i)
    if run_len < force:
      binary_insertion_sort(A, i, i + force, i + run_len)
      run_len <- force

    push(stack, (i, run_len))
    merge_collapse(stack)
    i <- i + run_len

  merge_force_collapse(stack)
  return A
```

## R12

本目录 MVP 的实现策略：
- 不调用 `list.sort()` / `sorted()` 执行主排序；
- 手写：
  - `_count_run_and_make_ascending`
  - `_binary_insertion_sort`
  - `_merge_collapse` / `_merge_force_collapse`
  - `_merge_at`
- `sorted` 与 `np.sort` 仅作为结果校验器；
- `pandas.DataFrame` 仅用于打印轨迹表格。

说明：
- 该 MVP 实现了 TimSort 的核心框架与不变量合并策略；
- 为保持代码紧凑，未实现 galloping 模式优化。

## R13

`demo.py` 的输出信息：
- 每个 case 的输入、TimSort 输出、参考输出；
- 统计字段：
  - `n`, `minrun`
  - `comparisons`, `moves`, `swaps`
  - `run_detections`, `reversals`, `insertion_extensions`
  - `merges`, `max_stack_size`
- 轨迹表：
  - `Run trace (last 10 rows)`
  - `Merge trace (last 10 rows)`

## R14

内置测试样例（全部确定性）：
- Case 1：固定混合数组；
- Case 2：近乎有序但局部乱序；
- Case 3：完全逆序（触发降序 run 反转）；
- Case 4：高重复值数组（验证稳定比较路径）；
- Case 5：固定随机种子数组（`minrun_override=16`）。

每个 case 都校验：
- 等于 `sorted(values)`；
- 等于 `np.sort(values)`；
- 满足非降序断言。

## R15

与常见排序对比：
- 对比归并排序：
  - 都是 `O(n log n)` 上界且可稳定；
  - TimSort 利用自然 runs，更适合真实“部分有序”数据。
- 对比快速排序：
  - 快排平均快但不稳定，最坏可退化到 `O(n^2)`；
  - TimSort 稳定且最坏 `O(n log n)`。
- 对比插入排序：
  - 插入排序在小规模/近乎有序时好；
  - TimSort 将其作为“局部扩展器”，而非全局主算法。

## R16

适用场景：
- 需要稳定排序（如按键排序后保留同键原顺序）；
- 数据常含局部有序段；
- 希望算法在工程数据上更稳健。

不适用场景：
- 极端内存受限且不能接受 merge 缓冲开销；
- 只关心简单教学示例且不需要工程级策略。

## R17

运行方式（无交互）：

```bash
cd "Algorithms/计算机-排序算法-0011-TimSort"
uv run python demo.py
```

运行成功时会输出 5 组样例结果，最后打印：`All checks passed.`。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `main()` 构造 5 组确定性输入并逐个调用 `run_case()`。  
2. `run_case()` 先做输入校验，再调用 `timsort(values, minrun_override)`。  
3. `timsort()` 计算 `minrun`，初始化计数器、`run_trace`、`merge_trace` 与 run 栈。  
4. 主循环中 `_count_run_and_make_ascending()` 检测自然 run；若为严格降序则 `_reverse_slice()` 原地反转。  
5. 若 run 太短，`_binary_insertion_sort()` 把该 run 扩展到 `minrun`（或尾段长度），并记录扩展事件。  
6. 把 run 压入栈后，`_merge_collapse()` 按 TimSort 不变量检查 run 长度关系，必要时调用 `_merge_at()` 立即合并。  
7. `_merge_at()` 复制左 run 到临时缓冲，再与右 run 稳定归并回主数组，更新 `merge_trace`。  
8. 扫描完所有 runs 后，`_merge_force_collapse()` 合并剩余 runs，直到栈只剩 `(0, n)`。  
9. `run_case()` 用 `sorted` 与 `np.sort` 交叉验证结果，并用 `pandas` 打印 run/merge 轨迹，最终由 `main()` 输出 `All checks passed.`。

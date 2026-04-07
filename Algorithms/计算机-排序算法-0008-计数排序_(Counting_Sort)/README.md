# 计数排序 (Counting Sort)

- UID: `CS-0008`
- 学科: `计算机`
- 分类: `排序算法`
- 源序号: `8`
- 目标目录: `Algorithms/计算机-排序算法-0008-计数排序_(Counting_Sort)`

## R01

计数排序（Counting Sort）是一种非比较排序算法。它通过“统计每个取值出现次数”，再把这些次数还原为有序序列，从而避免了元素两两比较。

本目录 MVP 的目标是：
- 给出可审计的手写计数排序实现；
- 支持负整数（通过偏移量映射到非负索引）；
- 输出频次表与统计信息，便于验证算法过程；
- 仅把 `numpy.sort` / `sorted` 作为结果对照，而不是主排序逻辑。

## R02

算法定位：
- 当输入是整数且值域范围 `k` 不大时，计数排序可达线性级时间 `O(n + k)`；
- 它不受比较排序 `O(n log n)` 下界约束；
- 常作为基数排序（Radix Sort）单轮稳定分配的基础模块。

工程价值在于：在“数据量大、值域适中”的场景下，计数排序常比通用比较排序更快且更稳定。

## R03

问题定义（本条目版本）：
- 输入：一维整数序列 `A`（允许重复值、负数、0）；
- 输出：非降序序列 `B`，满足：
  - 有序性：`B[0] <= B[1] <= ... <= B[n-1]`；
  - 排列保持：`B` 与 `A` 的元素多重集一致。

附加约束：
- 本实现只接受一维有限整数；
- 对过大值域设置上限保护，避免在演示脚本中申请不可控内存。

## R04

核心思想（稳定版本）：
1. 扫描数组得到 `min_value`、`max_value`，值域大小 `range_size = max - min + 1`；
2. 创建 `counts[range_size]`，其中下标 `i` 对应值 `i + min_value`；
3. 统计频次：`counts[x - min_value] += 1`；
4. 将频次数组做前缀和，得到每个值在输出数组中的结束位置；
5. 从右到左扫描原数组，把元素放到输出数组对应位置并回退前缀指针；
6. 最终得到稳定且有序的结果。

“从右到左放置”是稳定性的关键。

## R05

数据结构与状态量：
- `arr`：输入的整数列表副本；
- `counts`：每个值出现次数；
- `prefix`：`counts` 的前缀和，用于定位元素输出下标；
- `output`：排序结果缓冲区；
- `offset`：`-min_value` 的等价映射思想（本实现用 `x - min_value` 直接索引）。

统计字段（见 `demo.py`）：
- `n`、`min_value`、`max_value`、`range_size`；
- `count_ops`（计数次数）与 `placement_ops`（回填次数）；
- `counts` 与 `prefix_counts`（用于审计）。

## R06

正确性要点（简述）：
- 频次统计确保每个值的出现次数正确；
- 前缀和将“值”映射到输出区间边界；
- 回填时每放置一个值都会占用一个唯一位置，不会覆盖冲突；
- 对任意值 `v1 < v2`，`v1` 的输出位置区间必在 `v2` 前面，因此整体有序；
- 从右向左回填使得相等元素保持原相对顺序，因此稳定。

## R07

时间复杂度：
- 统计频次：`O(n)`；
- 构建前缀和：`O(k)`，其中 `k = range_size`；
- 回填输出：`O(n)`；
- 总体：`O(n + k)`。

当 `k` 远小于 `n log n` 时，计数排序通常具有明显优势。

## R08

空间复杂度：
- `counts` 与 `prefix` 大小均为 `O(k)`；
- `output` 大小为 `O(n)`；
- 总额外空间：`O(n + k)`。

这也是计数排序的典型代价：以空间换时间，对超大值域不友好。

## R09

算法性质：
- 非比较排序：是；
- 稳定性：是（本实现采用从右向左回填）；
- 原地性：否（需要输出缓冲区）；
- 自适应：一般不明显；
- 适配键类型：离散整数键（或可映射到小整数范围的键）。

## R10

边界与输入约束：
- 空数组：直接返回空结果；
- 单元素：直接返回；
- 包含负数：支持；
- 包含浮点但非整数值：拒绝；
- 包含 `NaN/Inf`：拒绝；
- 非一维输入：拒绝；
- 值域超出 `max_range_size`：抛出异常，避免演示程序内存失控。

## R11

伪代码（稳定计数排序）：

```text
counting_sort(A):
  validate A as 1D finite integer sequence
  if len(A) <= 1: return A

  min_v <- min(A), max_v <- max(A)
  k <- max_v - min_v + 1
  counts <- [0] * k

  for x in A:
    counts[x - min_v] += 1

  prefix <- cumulative_sum(counts)
  output <- [0] * len(A)

  for i from len(A)-1 downto 0:
    idx <- A[i] - min_v
    prefix[idx] -= 1
    pos <- prefix[idx]
    output[pos] <- A[i]

  return output
```

## R12

本目录 MVP 实现策略：
- `validate_integer_sequence` 统一输入校验；
- `counting_sort` 实现完整手写流程（计数、前缀、稳定回填）；
- `build_frequency_table` 用 `pandas` 生成人类可读频次表；
- `run_case` 做自动化断言，对照 `sorted` 与 `numpy.sort`；
- `main` 组织固定样例、随机样例与边界样例，无需交互输入。

第三方库角色：
- `numpy`：输入校验与结果对照；
- `pandas`：频次表展示；
- 不承担主排序决策。

## R13

输出字段说明（控制台）：
- `Input`：原始输入；
- `Sorted by counting sort`：手写计数排序结果；
- `Expected sorted`：`sorted` 基准结果；
- `Stats`：`n/min/max/range_size/count_ops/placement_ops`；
- `Frequency table`：值、出现次数、前缀结束位置（仅显示非零计数）。

## R14

内置测试样例（`demo.py`）：
- 案例 1：固定基础样例（重复值、非负）；
- 案例 2：固定样例（含负数与重复值）；
- 案例 3：固定随机种子生成整数序列；
- 案例 4：空数组；
- 案例 5：单元素。

每个样例都会断言：
- 结果等于 `sorted(input)`；
- 结果等于 `numpy.sort(input)`；
- 统计中的 `count_ops` 与 `placement_ops` 等于输入长度。

## R15

与常见排序对比：
- 对比快速排序：快排通用性强，但平均 `O(n log n)`；计数排序在小值域整数上更有优势；
- 对比归并排序：归并稳定且不依赖值域；计数排序在满足值域条件时可降为 `O(n + k)`；
- 对比基数排序：基数排序可处理更大整数范围，但每位通常依赖计数分配，常数更大、流程更长。

## R16

适用场景：
- 学生成绩分布、年龄分布、离散评分等小整数域排序；
- 大批量日志等级、离散桶编号排序；
- 作为基数排序的稳定子过程。

不适用场景：
- 值域极大且稀疏；
- 通用对象排序且无可控整数映射；
- 内存极度受限，无法接受 `O(k)` 计数数组。

## R17

运行方式：

```bash
cd "Algorithms/计算机-排序算法-0008-计数排序_(Counting_Sort)"
uv run python demo.py
```

运行特性：
- 全程无交互输入；
- 自动执行全部样例并打印统计；
- 任一校验失败会抛出异常并非零退出。

## R18

`demo.py` 源码级算法流程（8 步，非黑箱）：
1. `main` 构造固定和随机样例，并追加空数组与单元素边界案例。
2. `run_case` 对每个案例调用 `counting_sort`，然后与 `sorted`、`numpy.sort` 做结果对照。
3. `counting_sort` 先通过 `validate_integer_sequence` 确认输入是一维、有限、整数序列。
4. 计算 `min_value/max_value` 和 `range_size`，并检查值域是否超过 `max_range_size`。
5. 线性扫描输入，执行 `counts[x - min_value] += 1` 完成频次统计。
6. 对 `counts` 做前缀和，得到每个值对应的输出位置边界数组 `prefix`。
7. 从右向左遍历原数组：先递减 `prefix[idx]`，再把元素写入 `output[prefix[idx]]`，从而保证稳定性。
8. 回填完成后返回有序数组和统计信息，`build_frequency_table` 将频次与前缀信息表格化输出。

补充：本 MVP 没有调用任何第三方“现成计数排序”函数，核心流程（计数、前缀、稳定回填）全部在源码中逐行可追踪。

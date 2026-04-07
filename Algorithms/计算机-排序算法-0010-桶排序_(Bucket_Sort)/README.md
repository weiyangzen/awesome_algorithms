# 桶排序 (Bucket Sort)

- UID: `CS-0010`
- 学科: `计算机`
- 分类: `排序算法`
- 源序号: `10`
- 目标目录: `Algorithms/计算机-排序算法-0010-桶排序_(Bucket_Sort)`

## R01

桶排序（Bucket Sort）是一种“分配 + 局部排序 + 合并”的分布式排序思想：
- 先按数值区间把元素分配到多个桶；
- 再在每个桶内做排序；
- 最后按桶序拼接得到全局有序结果。

本目录 MVP 的目标：
- 给出可运行、可审计的 Python 桶排序实现；
- 不把第三方排序调用当黑箱核心逻辑；
- 输出桶分布统计，并用 `numpy`/`pandas` 做结果校验与可视化打印。

## R02

问题定义（升序）：
- 输入：长度为 `n` 的一维有限实数序列 `A`；
- 输出：序列 `B`，满足
  - 有序性：`B[0] <= B[1] <= ... <= B[n-1]`
  - 重排性：`B` 与 `A` 含相同元素多重集。

本实现支持负数、浮点数、重复值，并对 `NaN/Inf` 明确拒绝。

## R03

算法直觉：
- 当输入近似均匀分布时，元素会较平均地落入各桶；
- 单桶规模变小后，桶内排序代价下降；
- 总体可逼近线性时间。

因此桶排序非常依赖“分桶映射质量”和“数据分布特征”，不是对所有数据都稳定高效的通用最优解。

## R04

核心流程（本目录版本）：
1. 校验输入是一维有限数值序列。
2. 设桶数 `k`（默认 `k = floor(sqrt(n))`，至少为 1）。
3. 用线性归一化把值映射到桶下标：
   `idx = int((x - min) / (max - min) * k)`（并做边界夹紧）。
4. 把每个元素追加到对应桶。
5. 对每个桶执行手写插入排序。
6. 按桶编号从小到大依次拼接，得到最终结果。

## R05

关键数据结构与状态：
- `buckets: list[list[float]]`：桶数组。
- `bucket_histogram`：每个桶的元素个数。
- `distribution_trace`：桶区间、桶内最小/最大值、计数。
- `insertion_comparisons`：桶内插入排序累计比较次数。

这些统计用于解释“排序正确性 + 分布效果”，而不仅仅是输出最终有序数组。

## R06

正确性要点：
1. 映射函数保证：若 `x <= y`，则 `bucket(x)` 不会大于 `bucket(y)` 的可达顺序区间（允许同桶）。
2. 每个桶内排序后，桶内部有序。
3. 按桶编号拼接时，前桶元素值域不大于后桶元素值域上界。
4. 因此整体输出有序且元素总数守恒，满足重排性。

注：边界值 `x = max` 通过夹紧进入最后一个桶，避免越界。

## R07

时间复杂度（`n` 为样本数，`k` 为桶数）：
- 分配到桶：`O(n)`；
- 桶内排序：`sum O(m_i^2)`（本实现使用插入排序，`m_i` 是第 `i` 桶大小）；
- 拼接输出：`O(n)`。

综合：
- 均匀分布且 `k` 合理时，期望可接近 `O(n)`；
- 极端退化（所有元素进同一桶）时，退化为 `O(n^2)`。

## R08

空间复杂度：
- 桶存储需要 `O(n + k)` 额外空间；
- 桶内插入排序原地进行，不再额外申请与桶规模同量级的临时结构；
- 因此总体额外空间复杂度为 `O(n + k)`，通常写作 `O(n)`。

## R09

算法性质：
- 稳定性：本实现中，同桶内插入排序使用“严格大于才右移”，因此同值相对次序保持；跨桶不会打乱顺序，整体可视作稳定。
- 就地性：否（使用桶结构，非原地）。
- 自适应性：弱，主要取决于桶映射和数据分布，不直接利用“近乎有序”特性。

## R10

边界与输入约束：
- 空序列、单元素序列：直接返回；
- 全部元素相同：映射到同一桶，结果与输入一致；
- 含负数或大范围浮点：通过 `min/max` 归一化支持；
- 输入非一维、含非有限值：抛出异常；
- `bucket_count <= 0`：抛出异常，避免非法配置。

## R11

伪代码：

```text
bucket_sort(A, k):
  validate A
  if len(A) <= 1: return A
  if k is None: k <- floor(sqrt(len(A)))
  create k empty buckets
  lo <- min(A), hi <- max(A)
  for x in A:
    idx <- map_to_bucket(x, lo, hi, k)
    buckets[idx].append(x)
  for b in buckets:
    insertion_sort(b)
  return concatenate(buckets[0], buckets[1], ..., buckets[k-1])
```

## R12

MVP 实现策略（`demo.py`）：
- 主算法手写：`value_to_bucket_index` + `insertion_sort_inplace` + `bucket_sort`。
- 第三方库用途：
  - `numpy`：输入校验与基准排序对照（`np.sort`）；
  - `pandas`：打印桶分布表，便于人工审计。
- 不依赖 `scipy/scikit-learn/torch` 完成核心排序，保持最小实现闭环。

## R13

输出字段说明：
- `sorted_values`：桶排序结果；
- `bucket_count`：本次使用的桶数；
- `bucket_histogram`：桶容量分布；
- `non_empty_buckets`：非空桶数量；
- `max_bucket_size`：最大桶负载；
- `insertion_comparisons`：桶内排序比较计数；
- `distribution_trace`：每个桶的值域与统计明细。

## R14

内置测试样例（无交互）：
- Case 1：经典 `[0,1)` 风格浮点数据；
- Case 2：固定随机种子的混合范围浮点数据（含负数）；
- Case 3：全相等值数据（退化边界）。

每个样例都执行：
- `bucket_sort(values)`；
- 与 `sorted(values)` 对照；
- 与 `np.sort(values)` 对照；
- 不一致即抛错并非零退出。

## R15

与其他排序对比：
- 对比选择/冒泡：平均情况下桶排序可显著更快，但依赖分布假设。
- 对比快速排序：快速排序更通用稳健，桶排序在均匀分布数值数据上可能更有优势。
- 对比计数排序：计数排序要求离散且值域可控，桶排序可处理连续实数但精度/映射设计更关键。

## R16

适用场景：
- 数据近似均匀分布；
- 键是可度量的连续数值（如评分、概率、归一化特征）；
- 需要解释“分布式排序”思路的教学或原型验证。

不建议场景：
- 分布高度偏斜且难以预估；
- 对最坏情况性能有严格保证要求的生产关键路径。

## R17

运行方式：

```bash
cd "Algorithms/计算机-排序算法-0010-桶排序_(Bucket_Sort)"
uv run python demo.py
```

运行特征：
- 无需交互输入；
- 打印每个测试用例的排序结果与桶分布表；
- 全部通过后输出 `All checks passed.`。

## R18

`demo.py` 源码级流程拆解（8 步，非黑箱）：
1. `main` 构造三个确定性测试样例，并逐个调用 `run_case`。
2. `run_case` 调用 `validate_numeric_sequence`，把输入转成一维有限 `float` 列表。
3. `bucket_sort` 根据 `n` 计算桶数（默认 `floor(sqrt(n))`），读取 `min/max` 作为映射边界。
4. 对每个值执行 `value_to_bucket_index`，将其放入对应桶；该函数负责归一化与边界夹紧。
5. 分桶完成后，逐桶调用 `insertion_sort_inplace`，显式执行“比较 + 右移 + 插入”。
6. 按桶序拼接所有已排序桶，形成全局 `sorted_values`。
7. 同时统计 `bucket_histogram`、`non_empty_buckets`、`max_bucket_size`、`insertion_comparisons`，并生成 `distribution_trace`。
8. `run_case` 使用 `sorted` 与 `np.sort` 仅做结果校验，若一致则打印 `pandas` 分布表；全部样例通过后程序正常结束。

补充：第三方库未参与核心排序路径，主流程可在源码中逐行追踪和验证。

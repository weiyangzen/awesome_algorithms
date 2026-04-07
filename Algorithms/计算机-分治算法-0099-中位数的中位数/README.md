# 中位数的中位数

- UID: `CS-0079`
- 学科: `计算机`
- 分类: `分治算法`
- 源序号: `99`
- 目标目录: `Algorithms/计算机-分治算法-0099-中位数的中位数`

## R01

中位数的中位数（Median of Medians, BFPRT）是一个用于“选择第 k 小元素”的确定性线性时间算法。

本目录 MVP 目标：
- 用手写源码实现 BFPRT，而不是调用现成黑箱选择器；
- 给出可审计的分区轨迹（pivot 与左右规模）；
- 用 `numpy`/`sorted` 仅做结果校验，保证正确性。

## R02

历史背景：
- 1973 年由 Manuel Blum、Robert W. Floyd、Vaughan Pratt、Ronald L. Rivest、Robert E. Tarjan 提出；
- 常被称作 BFPRT 算法；
- 核心贡献是把“选择问题”在最坏情况下也做到 `O(n)`，弥补了随机快速选择最坏 `O(n^2)` 的不稳定性。

## R03

问题定义：
- 输入：长度为 `n` 的可比较序列 `A`，以及秩 `k`（`0 <= k < n`）。
- 输出：`A` 中按非降序排列后下标为 `k` 的元素。

目标不是把整个数组排序，而是只定位一个秩位置，因此在很多任务中比完整排序更省计算。

## R04

核心思想（分治）：
1. 把数组按 5 个一组分块。
2. 每组内部排序，取组中位数，得到“中位数数组”。
3. 递归求中位数数组的中位数，作为全局 pivot。
4. 按 pivot 做三路划分：`< pivot`、`== pivot`、`> pivot`。
5. 根据 `k` 落在哪个区间，只递归进入一个子问题。

关键点在第 3 步：该 pivot 不是随机值，而是有“足够好分割比例”保证，从而给出最坏线性时间。

## R05

实现中的主要状态：
- `arr`：当前递归层处理的子数组；
- `k`：当前子数组内的目标秩（0-based）；
- `groups`：按 5 元分组后的分块；
- `medians`：每组中位数组成的新数组；
- `pivot`：递归选出的中位数的中位数；
- `lows/equals/highs`：三路划分结果；
- `BFPRTStats`：统计递归调用数、分区轮数、最大深度与分区轨迹。

## R06

正确性要点：
- 组内中位数都来自原数组，递归得到的 pivot 也来自原数组；
- 三路划分后，`lows` 中元素全部 `< pivot`，`highs` 中元素全部 `> pivot`；
- 若 `k < len(lows)`，目标一定在 `lows`；
- 若 `k` 落在 `len(lows)` 到 `len(lows)+len(equals)-1`，答案就是 `pivot`；
- 否则目标在 `highs`，新秩要减去前两段长度。

每次递归都保持“答案秩不变式”，直到子数组足够小直接排序返回。

## R07

时间复杂度：
- 设 `T(n)` 为 BFPRT 复杂度，满足递推：
  `T(n) <= T(n/5) + T(7n/10) + O(n)`；
- 解得最坏时间复杂度为 `O(n)`；
- 这是该算法区别于随机 Quickselect 的核心优势。

直观解释：
- `T(n/5)` 来自递归求 pivot；
- `T(7n/10)` 来自最坏一侧子问题规模上界；
- 其余是线性分组/分区成本。

## R08

空间复杂度：
- 理论上若用原地划分可做到接近 `O(1)` 额外空间（不含递归栈）；
- 本目录 Python MVP 为可读性使用列表切分与三路列表，峰值额外空间近似 `O(n)`；
- 递归深度在良好分割下为 `O(log n)`，因此栈空间约 `O(log n)`。

## R09

算法性质：
- 是否确定性：是；
- 是否精确：是（非近似、非概率）；
- 是否稳定：N/A（选择问题不要求全排序稳定性）；
- 是否需要全排序：否；
- 是否有最坏线性保证：是。

## R10

输入约束与边界：
- 输入必须是一维有限数值序列；
- `k` 必须满足 `0 <= k < n`；
- 空数组非法；
- 对重复值要正确处理（依赖三路划分）；
- 本实现限制 `group_size` 为奇数且 `>= 5`（默认 5）。

## R11

伪代码：

```text
select(A, k):
  if len(A) <= 5:
    sort(A)
    return A[k]

  groups <- split A into chunks of 5
  medians <- [median(sort(g)) for g in groups]
  pivot <- select(medians, len(medians)//2)

  lows, equals, highs <- partition A by pivot

  if k < len(lows):
    return select(lows, k)
  else if k < len(lows) + len(equals):
    return pivot
  else:
    return select(highs, k - len(lows) - len(equals))
```

## R12

本目录 MVP 实现策略：
- 核心函数：
  - `deterministic_select(values, k, group_size=5)`
  - `_select_bfprt(...)`（递归核心）
- 工程辅助：
  - `validate_numeric_sequence` 做输入校验；
  - `BFPRTStats` 记录递归与分区轨迹；
  - `run_case` 同时对照 `sorted` 与 `np.partition`。

第三方库只用于校验和展示，不参与 pivot 选择与递归决策。

## R13

`demo.py` 输出字段说明：
- `n`：样例规模；
- `k`：目标秩；
- `BFPRT result`：手写算法结果；
- `Python sorted[k]`：全排序对照结果；
- `NumPy part[k]`：`np.partition` 对照结果；
- `recursive_calls/partition_rounds/max_depth`：复杂度观测指标；
- `Partition trace`：每轮递归的 `pivot` 与三路大小。

## R14

内置测试样例：
- Case 1：固定混合整数（含负数）；
- Case 2：固定随机种子生成的 101 个整数，取中位秩；
- Case 3：大量重复值，验证三路划分正确性。

断言规则：
- `BFPRT result == sorted(values)[k]`；
- `BFPRT result == np.partition(values, k)[k]`。

任一不一致即抛异常并非零退出。

## R15

与相关算法对比：
- 对比随机 Quickselect：
  - Quickselect 平均 `O(n)`，最坏 `O(n^2)`；
  - BFPRT 最坏 `O(n)`，但常数更大。
- 对比完整排序：
  - 排序通常 `O(n log n)`；
  - 只求一个秩时，选择算法更合适。
- 对比堆法：
  - 小顶/大顶堆取第 `k` 个常见为 `O(n log k)` 或 `O(n + k log n)`。

## R16

典型应用：
- 需要强最坏复杂度保证的选择任务；
- 作为内省式选择/排序（introselect/introsort）的“退化保护”思想来源；
- 鲁棒统计中的中位数、分位数计算；
- 流程中只关心阈值元素，不需要全量排序的场景。

## R17

运行方式：

```bash
cd "Algorithms/计算机-分治算法-0099-中位数的中位数"
uv run python demo.py
```

脚本特性：
- 无交互输入；
- 自动执行 3 组测试；
- 打印分区轨迹摘要；
- 校验失败时直接抛异常。

## R18

`demo.py` 源码级算法流程（9 步，非黑箱）：
1. `main` 构造固定样例、随机样例、重复值样例，并分别调用 `run_case`。
2. `run_case` 对输入做校验后，调用 `deterministic_select(values, k)`。
3. `deterministic_select` 检查 `group_size` 与 `k` 合法性，初始化 `BFPRTStats`。
4. 进入 `_select_bfprt(arr, k, ...)`；若 `len(arr) <= group_size`，直接排序并返回第 `k` 个。
5. 否则把 `arr` 每 5 个分组，对每组排序后取组中位数，形成 `medians`。
6. 递归在 `medians` 上求其中位数，得到本层 `pivot`（即“中位数的中位数”）。
7. 用单次线性扫描把当前数组分成 `lows/equals/highs` 三段，并记录到 `stats.trace`。
8. 按 `k` 所在区间决定递归分支：进 `lows`、直接返回 `pivot`、或进 `highs` 并换算新 `k`。
9. 递归回收后返回最终值；`run_case` 再用 `sorted` 和 `np.partition` 做外部对照校验并打印统计。

补充：`numpy` 与 `pandas` 在本实现中仅用于对照与展示，核心 BFPRT 逻辑完全由源码中的分组、递归与三路划分实现。

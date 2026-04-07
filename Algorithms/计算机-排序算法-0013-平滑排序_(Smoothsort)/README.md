# 平滑排序 (Smoothsort)

- UID: `CS-0013`
- 学科: `计算机`
- 分类: `排序算法`
- 源序号: `13`
- 目标目录: `Algorithms/计算机-排序算法-0013-平滑排序_(Smoothsort)`

## R01

平滑排序（Smoothsort）由 Edsger W. Dijkstra 提出，是一种基于堆思想的自适应原地排序。  
它使用 Leonardo 堆森林替代单一二叉堆，在“数据接近有序”时可减少调整成本。

本目录给出的 MVP 目标：
- 手写 Smoothsort 核心流程（不是 `sorted()` 黑箱实现）；
- 通过固定样例自动验证正确性；
- 输出比较次数与操作次数，直观看到算法行为。

## R02

本实现的任务定义：
- 输入：可比较元素序列（`Sequence[T]`，示例使用整数）。
- 输出：
  - 排序后数组；
  - 统计信息 `comparisons/sifts/trinkles`。

`demo.py` 中 `main()` 固定构造测试集并自动校验，不需要交互输入。

## R03

核心数学/结构基础：
- Leonardo 数定义：
  - `L(0)=1, L(1)=1`
  - `L(k)=L(k-1)+L(k-2)+1`
- Smoothsort 把数组前缀维护为若干 Leonardo 树组成的“堆森林”。
- 每棵树满足最大堆性质；森林中每个根也按结构顺序满足可提取最大值的约束。

## R04

算法高层流程：
1. 生成足够覆盖输入规模的 Leonardo 数表。
2. 线性扫描数组，逐步构建 Leonardo 堆森林。
3. 构建阶段根据位模式选择 `sift` 或 `trinkle` 修复局部有序性。
4. 构建完成后，最大元素位于森林最右根位置。
5. 进入提取阶段：不断缩小森林，把当前最大值“放到最终位置”。
6. 每次拆分树后对受影响子树执行 `trinkle` 恢复堆序。
7. 循环直到森林收缩为单节点，排序完成。

## R05

实现中的核心数据结构：
- `SmoothsortStats`：记录比较次数、`sift` 调用次数、`trinkle` 调用次数。
- `CaseResult`：单测试用例统计输出。
- `p` 与 `pshift`：位编码状态，描述当前 Leonardo 堆森林形态。
- `LP`：预置 Leonardo 数表（`LP[k] = Leonardo(k)`）。
- 原地数组 `arr`：排序对象，额外空间仅常数级状态变量。

## R06

正确性要点：
- `sift` 保证单棵 Leonardo 树内部恢复最大堆性质。
- `trinkle` 先在森林根链中上浮/回退，再回到树内 `sift`，保证跨树与树内都合法。
- 构建阶段结束后，右端根对应当前未排序前缀中的最大值。
- 提取阶段每轮把该最大值固定到后缀位置，未固定前缀继续保持 Smoothsort 不变式。
- 通过与 `sorted()`、`numpy.sort()` 双重对照验证最终排列正确。

## R07

复杂度分析：
- 最坏时间复杂度：`O(n log n)`。
- 平均复杂度：通常接近 `O(n log n)`。
- 近乎有序输入：可逼近 `O(n)`（Smoothsort 的自适应特性）。
- 额外空间复杂度：`O(1)`（原地排序，不计输入存储）。
- 稳定性：不稳定排序（与堆排序同类特征）。

## R08

边界与异常场景：
- 空数组或单元素：直接返回，无调整。
- 负数、重复值、已排序、逆序：均在内置样例覆盖。
- 可比较性前提：元素之间必须支持 `>`/`>=` 比较。
- `demo.py` 中若与参考排序不一致，会抛 `RuntimeError` 直接失败。

## R09

MVP 取舍说明：
- 保留 Smoothsort 关键机制（Leonardo 森林 + `sift/trinkle`）。
- 不引入复杂工程封装（CLI 参数、文件 IO、可视化）以保持最小可验证实现。
- 使用 `numpy` 仅用于构造确定性测试数据和做一次交叉校验，不替代核心排序逻辑。

## R10

`demo.py` 主要函数职责：
- `LP`：Leonardo 数常量表。
- `_ctz`：计算尾零位数，用于 `p` 状态迁移。
- `_sift`：修复单树堆序。
- `_trinkle`：在森林结构变化后恢复全局堆序。
- `smoothsort_inplace`：Smoothsort 主过程（构建 + 提取）。
- `smoothsort`：对外包装，返回排序结果和统计。
- `_run_case`：执行单个测试并校验。
- `main`：组织全部样例、打印汇总并做最终断言。

## R11

运行方式（无交互）：

```bash
cd Algorithms/计算机-排序算法-0013-平滑排序_(Smoothsort)
uv run python demo.py
```

## R12

输出字段说明：
- `n`：样例规模。
- `comparisons`：元素比较次数（`>`/`>=` 计数）。
- `sifts`：`_sift` 调用次数。
- `trinkles`：`_trinkle` 调用次数。
- `adaptiveness_hint`：对比“已排序 vs 随机”比较次数，用于观察自适应倾向。
- `All validation checks passed.`：全部测试通过标志。

## R13

内置最小测试集：
1. `already_sorted`：已排序数组。
2. `reverse_sorted`：逆序数组。
3. `nearly_sorted`：少量交换后的近乎有序数组。
4. `random`：固定随机种子的均匀整数数组。
5. `many_duplicates`：高重复值数组。
6. `numpy_cross_check`：和 `numpy.sort` 的固定样例交叉验证。

## R14

关键参数与可调项：
- `n=64`：主样例规模。
- `seed=20260407`：随机样例种子，保证可复现。
- `swaps=4`：近乎有序样例扰动强度。

调参建议：
- 若想更明显观察自适应性，可增大 `n` 并减少 `swaps`。
- 若要压力测试正确性，可在 `main()` 增加更多随机轮次。

## R15

与常见排序对比：
- 对比堆排序：
  - 都是原地、最坏 `O(n log n)`、不稳定；
  - Smoothsort 在近乎有序输入上通常更省比较。
- 对比 TimSort：
  - TimSort 稳定且工程上更常用；
  - Smoothsort 理论与结构更“堆化”，实现复杂度更高。
- 对比快速排序：
  - 快速排序平均性能优秀，但最坏可退化；
  - Smoothsort 提供确定的最坏复杂度上界。

## R16

适用场景：
- 需要原地排序且希望利用“近乎有序”输入特性的场景。
- 教学/研究用途：用于理解 Dijkstra 的自适应堆思想。
- 对稳定性无硬性要求、但重视空间开销可控的场景。

## R17

可扩展方向：
- 增加 `key`/`reverse` 接口，接近 Python `sort` 体验。
- 将比较器抽象为回调，支持自定义对象排序。
- 增加随机回归测试（多轮随机数组 + 长度扫描）。
- 补充基准模块，对比 `heapq`/`sorted` 在不同输入分布下的开销。

## R18

`demo.py` 源码级流程（8 步）：
1. `main()` 构造 5 组确定性测试数据，再准备一个 `numpy_cross_check` 固定数组。  
2. 每组数据通过 `_run_case()` 调用 `smoothsort()`；`smoothsort()` 复制输入并委托 `smoothsort_inplace()`。  
3. `smoothsort_inplace()` 使用预置 Leonardo 表 `LP`，并初始化森林状态变量 `p/pshift/head`。  
4. 进入构建阶段（`while head < n-1`）：按 `p` 的低位模式决定调用 `_sift()` 或 `_trinkle()`，把新元素并入堆森林。  
5. 构建完成后再次 `_trinkle()`，确保当前森林右端根满足最大堆提取前提。  
6. 进入提取阶段（第二个 `while`）：用 `_ctz(p & ~1)` 更新位状态；当树被拆分时，对左右子树根分别调用 `_trinkle(..., trusty=True)` 修复堆序。  
7. 排序完成返回 `arr + stats`；`_run_case()` 把结果与 `sorted()` 对照，不一致立即抛错。  
8. `main()` 额外执行 `numpy.sort` 交叉验证，打印比较/操作统计与自适应提示，最终输出 `All validation checks passed.`。

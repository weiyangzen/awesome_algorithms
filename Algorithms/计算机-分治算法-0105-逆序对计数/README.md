# 逆序对计数

- UID: `CS-0084`
- 学科: `计算机`
- 分类: `分治算法`
- 源序号: `105`
- 目标目录: `Algorithms/计算机-分治算法-0105-逆序对计数`

## R01

逆序对计数问题要求统计数组中满足 `i < j` 且 `a[i] > a[j]` 的元素对数量。它是分治算法中的经典题目，通常通过“归并排序 + 跨区间计数”在 `O(n log n)` 时间内完成，比朴素 `O(n^2)` 双循环快得多。

## R02

形式化定义：给定长度为 `n` 的序列 `a`，逆序对集合为

`Inv(a) = {(i, j) | 0 <= i < j < n, a[i] > a[j]}`。

目标是计算 `|Inv(a)|`。

例如：`[2, 4, 1, 3, 5]` 的逆序对为 `(2,1)`、`(4,1)`、`(4,3)`，总数为 `3`。

## R03

朴素方法：双重循环枚举所有 `(i, j)` 组合。

- 时间复杂度：`O(n^2)`
- 空间复杂度：`O(1)`（不计输入）

当 `n` 较大（如 `10^5`）时，`O(n^2)` 通常不可接受，因此需要分治优化。

## R04

分治核心思想：

1. 把数组拆成左右两半。
2. 递归统计左半逆序对数 `left_count`。
3. 递归统计右半逆序对数 `right_count`。
4. 在“归并”两个有序子数组时统计跨半区逆序对 `cross_count`。

最终答案：`left_count + right_count + cross_count`。

## R05

为什么归并时可以高效统计跨区逆序对：

- 归并阶段左右子数组都已各自有序。
- 若当前 `left[i] <= right[j]`，则 `left[i]` 不会与 `right[j]` 及其后续构成逆序对。
- 若当前 `left[i] > right[j]`，由于 `left[i..end]` 都 `>= left[i]`，因此它们都大于 `right[j]`，一次性新增 `mid - i + 1` 个逆序对。

这避免了逐对比较，把跨区统计从 `O(n^2)` 降到 `O(n)`（每层）。

## R06

伪代码：

```text
sort_count(a, l, r):
    if l >= r:
        return 0
    mid = (l + r) // 2
    cnt = sort_count(a, l, mid) + sort_count(a, mid+1, r)

    i = l, j = mid+1
    tmp = []
    while i <= mid and j <= r:
        if a[i] <= a[j]:
            tmp.append(a[i]); i += 1
        else:
            tmp.append(a[j]); j += 1
            cnt += (mid - i + 1)

    append remaining elements from left/right to tmp
    write tmp back to a[l:r+1]
    return cnt
```

## R07

正确性直觉：

- 左、右区间内部逆序对由递归保证正确。
- 任一跨区间逆序对必然形如 `i in left, j in right`，只会在当前层归并时被统计一次。
- 归并使用有序性与指针单调前进，既不漏计也不重计。
- 递归基（区间长度 `<=1`）逆序对数为 0，成立。

## R08

复杂度分析：

- 递推式：`T(n) = 2T(n/2) + O(n)`。
- 由主定理得：`T(n) = O(n log n)`。
- 额外空间：一个临时数组 `tmp`（或等价缓冲）+ 递归栈，总体 `O(n)`（实现口径下常记 `O(n)`）。

## R09

与重复元素相关的细节：

- 逆序对定义使用严格大于 `>`。
- 因此在归并比较中应写 `if left[i] <= right[j]` 走左侧分支。
- 若误写成 `<`，会把相等元素当成逆序对，导致计数偏大。

## R10

边界与输入约束：

- 空数组、单元素数组：逆序对为 `0`。
- 支持负数、重复值、已排序、逆序等情况。
- 计数结果可能较大，Python `int` 无溢出风险（自动大整数）。

## R11

本目录 MVP 的实现约定：

- API：`count_inversions_divide_conquer(nums)` 返回逆序对整数。
- 输入可为 Python 列表或一维 `numpy.ndarray`。
- 实现会拷贝输入后再排序计数，不修改调用者原数据。
- 同时提供 `count_inversions_naive(nums)` 作为基线校验。

## R12

常见错误清单：

- 归并时跨区计数公式写错（应为“左侧剩余元素个数”）。
- 忘记把归并结果回写原区间，导致上层数据无序。
- 比较符号处理不当，把重复值算成逆序对。
- 只做计数不验证，隐藏逻辑错误。

## R13

`demo.py` 结构：

- `_merge_count`：在 `arr[left:right]` 上执行归并并累计跨区逆序对。
- `_sort_count`：递归分治主过程。
- `count_inversions_divide_conquer`：对外接口，返回 `O(n log n)` 结果。
- `count_inversions_naive`：`O(n^2)` 双循环基线。
- `main`：固定样例演示 + 随机对拍测试。

## R14

运行方式：

```bash
uv run python demo.py
```

脚本不需要命令行参数，也不读取交互输入。

## R15

预期输出特征：

- 打印固定样例数组与逆序对数量。
- 打印若干随机测试摘要。
- 若所有断言通过，输出 `All checks passed.`。
- 若实现有误，会抛出 `AssertionError` 指示失败用例。

## R16

可扩展方向：

- 使用树状数组（BIT）+ 离散化实现同问题。
- 使用线段树支持动态插入场景下的逆序统计。
- 推广到“重要翻转对”（如 `a[i] > 2*a[j]`）等变体。
- 在外部排序场景下处理超大数据逆序统计。

## R17

交付文件说明：

- `README.md`：完整说明逆序对计数的分治思路与实现注意点。
- `demo.py`：可直接运行的最小 Python MVP（含自动校验）。
- `meta.json`：保存任务元数据（UID、目录、学科、分类等）。

## R18

`demo.py` 的源码级算法流（8 步）：

1. `main` 准备固定样例与随机样例，调用 `count_inversions_divide_conquer`。  
2. `count_inversions_divide_conquer` 将输入转成列表副本，创建同长度缓冲区 `buffer`。  
3. 进入 `_sort_count(arr, buffer, left, right)`；若区间长度 `<=1`，直接返回 `0`。  
4. 递归处理左右子区间，分别得到 `left_count` 与 `right_count`。  
5. 调用 `_merge_count` 归并两段有序区间：维护 `i`（左指针）、`j`（右指针）、`k`（写指针）。  
6. 当出现 `arr[i] > arr[j]` 时，把 `arr[j]` 写入缓冲，并累加 `mid - i + 1` 个跨区逆序对。  
7. 归并结束后把 `buffer[left:right+1]` 回写 `arr[left:right+1]`，返回 `left + right + cross`。  
8. `main` 用 `count_inversions_naive` 对拍固定样例和随机样例；全部一致则打印 `All checks passed.`。  

这 8 步覆盖了从递归拆分、跨区计数到结果验证的完整执行路径，不依赖黑箱计数函数。

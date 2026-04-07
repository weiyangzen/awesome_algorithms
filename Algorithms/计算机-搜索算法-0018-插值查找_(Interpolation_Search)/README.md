# 插值查找 (Interpolation Search)

- UID: `CS-0018`
- 学科: `计算机`
- 分类: `搜索算法`
- 源序号: `18`
- 目标目录: `Algorithms/计算机-搜索算法-0018-插值查找_(Interpolation_Search)`

## R01

插值查找（Interpolation Search）用于在**有序数组**中查找目标值。与二分查找总是取中点不同，插值查找会根据目标值在区间端点值之间的相对位置，估计更可能命中的下标。

## R02

问题形式化：
- 输入：升序数组 `arr`（长度 `n`）与目标值 `target`
- 输出：若存在下标 `i` 使 `arr[i] == target`，返回某个合法下标；否则返回 `-1`

## R03

适用前提：
- 输入序列必须按非降序排列
- 元素支持减法与比较操作（常见为整数/浮点数）
- 当 `arr[high] - arr[low]` 很小或为 0 时，需要显式防止除零
- 该算法在“值分布相对均匀”时效果较好

## R04

核心思想：
- 维护区间 `[low, high]`
- 通过插值公式预测位置 `pos`：
  - `pos = low + (target - arr[low]) * (high - low) / (arr[high] - arr[low])`
- 比较 `arr[pos]` 与 `target` 后缩小区间：
  - 命中则返回
  - `arr[pos] < target`，移动 `low = pos + 1`
  - `arr[pos] > target`，移动 `high = pos - 1`

## R05

简化伪代码：

```text
low = 0, high = n - 1
while low <= high and arr[low] <= target <= arr[high]:
    if arr[low] == arr[high]:
        return low if arr[low] == target else -1

    pos = low + ((target - arr[low]) * (high - low)) // (arr[high] - arr[low])
    if arr[pos] == target:
        return pos
    if arr[pos] < target:
        low = pos + 1
    else:
        high = pos - 1
return -1
```

## R06

正确性要点（不变式直观）：
- 循环中始终保持：若目标存在，则它只能位于当前 `[low, high]` 内
- 插值位置 `pos` 在边界和值域约束下不会越过合法搜索区间
- 每轮比较后都丢弃一段不可能含目标的区间，搜索区间严格收缩
- 当区间失效或目标不在端点值域内时，说明目标不存在

## R07

复杂度：
- 平均时间复杂度：`O(log log n)`（值分布较均匀时）
- 最坏时间复杂度：`O(n)`（分布极不均匀或退化情况）
- 空间复杂度：`O(1)`（迭代实现）

## R08

边界与特殊情况：
- 空数组
- 单元素数组
- 区间端点值相同（`arr[low] == arr[high]`）时的除零保护
- 目标小于最小值或大于最大值
- 重复值存在时，本实现返回任意一个合法命中位置

## R09

示例：
- `arr = [10, 20, 30, 40, 50, 60, 70]`, `target = 50`
  - 首轮估计 `pos` 接近 `4`，快速命中
- `arr = [1, 2, 4, 8, 16, 32, 64]`, `target = 8`
  - 估计位置偏左，经过数轮收缩后命中 `3`
- `arr = [5, 5, 5, 5]`, `target = 7`
  - 端点值相同且不等于目标，直接返回 `-1`

## R10

本目录 MVP 实现策略：
- `demo.py` 提供 `interpolation_search` 主函数
- 使用 `_ensure_non_decreasing` 校验输入有序性，避免误用
- 在主循环中加入 `arr[low] == arr[high]` 特判，防止除零
- 以 `bisect_left` 做“存在性对照”，验证实现可靠性

## R11

运行方式：

```bash
uv run python demo.py
```

预期输出包含：
- 多组样例的数组、目标、返回下标、存在性对照
- 无序输入触发 `ValueError` 的检查结果
- 末行 `All checks passed.`

## R12

典型应用：
- 大规模有序静态数组的值查询
- 键值大致均匀分布的索引定位（如编号、时间戳桶）
- 作为二分查找的替代方案，用于减少某些场景下比较次数

## R13

常见错误：
- 忽略有序性前提，在无序数据上直接调用
- 未处理 `arr[low] == arr[high]`，导致除零异常
- 未限制循环条件 `arr[low] <= target <= arr[high]`，造成越界估计
- 对重复值场景未定义返回语义

## R14

常见变体：
- 返回最左/最右命中位置（结合边界收缩）
- 仅判断是否存在（布尔接口）
- 在整数数组上使用纯整数公式，避免浮点误差

## R15

与二分查找对比：
- 二分查找：固定按中点分割，时间复杂度稳定 `O(log n)`
- 插值查找：按值分布估计位置，均匀分布时常优于二分
- 当数据分布极不均匀时，插值查找可能退化到 `O(n)`
- 实践中可按数据分布特征选择两者之一

## R16

工程实践建议：
- 对外 API 明确要求“输入为非降序序列”
- 保留除零保护与边界值域判断，避免异常与越界
- 将“算法主逻辑”和“校验/测试”解耦，便于单测
- 用覆盖边界条件的固定样例确保实现可回归验证

## R17

最小测试清单：
- 空数组、单元素命中/未命中
- 目标命中首元素、中间元素、末元素
- 目标不存在且位于值域外/值域内缺口
- 重复值数组与端点值相同数组
- 非有序输入（应抛出 `ValueError`）

## R18

源码级算法流程拆解（以本目录 `demo.py` 的 `interpolation_search` 为准）：
1. 接收 `arr` 与 `target`，先调用 `_ensure_non_decreasing` 验证数组为非降序，不满足则抛出 `ValueError`。
2. 初始化 `low = 0`、`high = len(arr) - 1`，并进入主循环；循环前提要求目标仍在当前端点值域内。
3. 若当前区间端点值相同（`arr[low] == arr[high]`），说明本区间所有值一致：等于目标则返回 `low`，否则返回 `-1`。
4. 用插值公式计算预测下标 `pos = low + ((target - arr[low]) * (high - low)) // (arr[high] - arr[low])`。
5. 读取 `arr[pos]` 并与目标比较；若相等则立即返回 `pos`。
6. 若 `arr[pos] < target`，把 `low` 更新为 `pos + 1`，丢弃左侧不可能区间。
7. 若 `arr[pos] > target`，把 `high` 更新为 `pos - 1`，丢弃右侧不可能区间。
8. 当循环结束仍未命中时返回 `-1`；`main()` 使用 `bisect_left` 仅做存在性交叉校验，验证实现行为。

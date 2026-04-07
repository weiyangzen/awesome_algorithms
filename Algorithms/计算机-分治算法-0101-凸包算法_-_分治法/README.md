# 凸包算法 - 分治法

- UID: `CS-0081`
- 学科: `计算机`
- 分类: `分治算法`
- 源序号: `101`
- 目标目录: `Algorithms/计算机-分治算法-0101-凸包算法_-_分治法`

## R01

**问题定义**：给定二维平面点集，求包含全部点的最小凸多边形边界（凸包）。  
**本题算法**：QuickHull（分治式凸包算法）。  
**输出形式**：按逆时针顺序返回凸包顶点，不重复首尾点。  
**目标**：实现一个可运行、可验证的分治凸包 MVP，而不是只调用黑箱几何库。

## R02

QuickHull 的核心思想与快速排序类似：  
- 先用极值点把问题切分为上下两部分；
- 在每部分中找“距离分割线最远”的点作为新的分治支点；
- 递归地继续切分，最终只保留外边界上的点。

它是计算几何里“分治 + 几何判定”非常典型的代表。

## R03

动机来自朴素方法的不可扩展性：  
- 若枚举所有边并验证“其余点是否在同侧”，时间代价可达 `O(n^3)`；  
- 分治方法通过递归过滤掉大量内点，只在可能成为边界的子集中继续工作。  

因此 QuickHull 在随机数据上通常接近 `O(n log n)`，适合中大规模点集的教学与工程原型。

## R04

本目录同时实现了一个**基线对照算法**：Andrew 单调链（`O(n log n)`）。  

用途：
- 作为 QuickHull 的正确性对拍基线；
- 在随机回归测试中比较两者输出的凸包点集合是否一致；
- 不依赖第三方凸包黑箱，保证验证过程可解释。

## R05

QuickHull 主流程：
1. 去重并找最左点 `L`、最右点 `R`。
2. 按有向线段 `L->R` 把点分为上侧集合 `upper` 与下侧集合 `lower`。
3. 对 `upper` 递归构造上凸链：在当前边 `A->B` 上找最远点 `P`，再分成 `A->P` 与 `P->B` 两个子问题。
4. 对 `lower` 类似处理（方向使用 `R->L` 保证最终顺序一致）。
5. 拼接得到逆时针凸包顶点序列。

## R06

伪代码（简化）：

```text
quickhull(points):
    pts = deduplicate(points)
    if |pts| <= 2: return pts
    L = leftmost(pts), R = rightmost(pts)
    upper = {p | cross(L, R, p) > 0}
    lower = {p | cross(L, R, p) < 0}
    return [L] + find_hull(upper, L, R) + [R] + find_hull(lower, R, L)

find_hull(S, A, B):
    if S empty: return []
    P = argmax_{p in S} |cross(A, B, p)|
    S1 = {p in S | cross(A, P, p) > 0}
    S2 = {p in S | cross(P, B, p) > 0}
    return find_hull(S1, A, P) + [P] + find_hull(S2, P, B)
```

## R07

正确性要点（归纳）：
- 对任意子问题 `find_hull(S, A, B)`，返回的是边 `A->B` 外侧那一段凸包链；
- 最远点 `P` 必为该链的顶点，否则无法包住外侧点；
- 递归划分后的 `S1/S2` 覆盖所有仍可能在外侧边界上的点，且互不遗漏；
- 当 `S` 为空时返回空链，表示 `A->B` 直接成为凸包边。

因此，顶层拼接 `L + upper_chain + R + lower_chain` 得到完整凸包。

## R08

复杂度分析：
- 平均时间复杂度：通常接近 `O(n log n)`（随机分布点集常见）；
- 最坏时间复杂度：`O(n^2)`（例如大量点分布在近圆周且分割持续不均衡）；
- 空间复杂度：主要来自递归栈和子集切分，约为 `O(n)`（实现相关）。

## R09

边界与退化场景：
- 空集 / 单点：凸包即自身；
- 两点：凸包为这两个点；
- 全部点共线：凸包退化为两端点；
- 含重复点：先去重，避免递归与比较时重复干扰。

本实现均已覆盖上述情况。

## R10

几何判定约定：
- 使用叉积 `cross(o,a,b)` 判断方向：
  - `> 0`：`b` 在有向边 `o->a` 左侧；
  - `< 0`：右侧；
  - `= 0`：共线。
- “点到线距离”比较使用未归一化有向面积 `|cross(a,b,p)|`，避免开方。
- 最终输出顶点顺序为逆时针，便于后续计算面积或可视化。

## R11

与常见凸包算法对比：
- Graham Scan：同为 `O(n log n)`，依赖极角排序，流程更偏“排序+栈”。
- Andrew 单调链：实现简洁、工程上常用，本目录将其作为验证基线。
- Jarvis March（Gift Wrapping）：`O(nh)`，当壳点 `h` 很小时表现好，但最坏可到 `O(n^2)`。
- QuickHull：分治结构清晰，平均表现好，但最坏同样可能退化到 `O(n^2)`。

## R12

实现细节与踩坑点：
- 必须先去重，否则可能出现重复壳点；
- 子集划分条件应使用严格不等号（`> 0` / `< 0`），避免把共线内点反复递归；
- 递归返回后做一次顺序去重，增强极端输入下的鲁棒性；
- 浮点比较在对拍时使用统一舍入（本 demo 用 `round(..., 12)`）防止噪声误报。

## R13

`demo.py` 的 MVP 结构：
- `convex_hull_quickhull(points)`: 分治主算法；
- `_find_hull(points, a, b)`: 递归构造外侧凸链；
- `convex_hull_monotonic_chain(points)`: Andrew 基线；
- `_run_fixed_case()`: 已知样例验证；
- `_run_random_regression()`: 随机对拍；
- `_run_perf_snapshot()`: 小规模耗时快照。

依赖最小化：仅使用 `numpy` 生成随机测试数据与稳定复现实验。

## R14

运行方式（仓库根目录）：

```bash
uv run python Algorithms/计算机-分治算法-0101-凸包算法_-_分治法/demo.py
```

或在当前目录直接运行：

```bash
uv run python demo.py
```

## R15

预期输出包含：
1. 固定样例的凸包顶点列表；
2. 随机对拍通过统计（case 数与 seed）；
3. QuickHull 与基线算法的耗时对比及凸包点数；
4. `All checks passed.` 作为最终通过标志。

## R16

验收标准：
- `README.md` 与 `demo.py` 无未填占位符；
- `uv run python demo.py` 可一次运行完成，无交互输入；
- QuickHull 与 Andrew 基线在随机测试中输出同一凸包点集合；
- 固定样例满足预期凸包顶点；
- 输出结果顺序为逆时针（面积符号为正）。

## R17

可扩展方向：
- 引入 `scipy.spatial.ConvexHull` 作为第三重校验（仅验证，不替代主算法）；
- 增加可视化（matplotlib）展示点云与凸包边；
- 对高维凸包问题扩展到 `3D`（需改用更通用结构）；
- 在超大数据场景下加入并行分区与分块外存策略；
- 输出更多几何量，如周长、面积、壳点占比。

## R18

`demo.py` 中 QuickHull 的源码级流程可拆为 8 步：

1. `main()` 依次调用固定样例、随机对拍、性能快照三个阶段。  
2. `convex_hull_quickhull()` 先做去重与排序，处理 `<=2` 点的直接返回分支。  
3. 在全部点中确定 `leftmost/rightmost`，并按 `cross(leftmost,rightmost,p)` 划分为 `upper/lower` 两个子集。  
4. 对上凸链调用 `_find_hull(upper, leftmost, rightmost)`：若子集为空直接返回空链。  
5. `_find_hull` 在当前边 `a->b` 上选取距离最大的 `farthest`，该点必定位于当前外凸边界。  
6. 再按 `a->farthest` 与 `farthest->b` 重新筛出两组外侧候选，递归生成左右子链并按拓扑顺序拼接。  
7. 顶层把 `[leftmost] + upper_chain + [rightmost] + lower_chain` 合并为逆时针壳点序列，再做一次顺序去重。  
8. 测试阶段用 `convex_hull_monotonic_chain()` 作为基线，对随机样本比较壳点集合并输出耗时，确认实现正确可运行。  

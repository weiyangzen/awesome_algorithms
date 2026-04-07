# 中点圆算法

- UID: `CS-0246`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `403`
- 目标目录: `Algorithms/计算机-计算机图形学-0403-中点圆算法`

## R01

中点圆算法（Midpoint Circle Algorithm）用于在整数像素网格上绘制圆周。它通过判定“下一步取东侧像素还是东南侧像素”来逼近理想圆弧，核心计算只依赖整数加减与比较，避免了每步开方或三角函数。

## R02

目标问题：给定圆心 `(cx, cy)` 与半径 `r`（整数，`r >= 0`），输出圆周像素点集合 `[(x, y), ...]`，要求：

- 点集在离散网格上尽量贴近 `x^2 + y^2 = r^2`；
- 覆盖完整圆周（8 对称）；
- 无交互输入，程序可直接运行并自检。

## R03

算法基于隐式圆方程 `F(x, y) = x^2 + y^2 - r^2`。

在第一八分圆中，从 `(x, y) = (0, r)` 出发，每轮考虑两种候选点：

- `E = (x+1, y)`
- `SE = (x+1, y-1)`

使用中点 `M = (x+1, y-1/2)` 的符号来决策：

- 若 `F(M) < 0`，中点在圆内，优先取 `E`；
- 否则取 `SE`。

离散化后可写成整数递推变量 `decision`：

- 初值：`decision = 1 - r`
- 选 `E`：`decision += 2*x + 1`（其中 `x` 已先加 1）
- 选 `SE`：`y -= 1`，再 `decision += 2*(x - y) + 1`

## R04

利用圆的 8 对称性：第一八分圆的每个点 `(x, y)` 可映射为 8 个像素：

- `(±x, ±y)`
- `(±y, ±x)`

因此只需迭代 `x <= y` 的一小段轨迹，即可重建整圆。

## R05

本目录 MVP 约定：

- `midpoint_circle(cx, cy, radius)` 输入均为整数；
- `radius < 0` 抛出 `ValueError`；
- `radius == 0` 返回单点 `[center]`；
- 输出点去重后返回，保证每个像素唯一；
- 使用 `numpy` 仅做 ASCII 渲染和随机测试，不依赖图形库黑箱。

## R06

伪代码：

```text
MIDPOINT_CIRCLE(cx, cy, r):
    if r == 0:
        return [(cx, cy)]

    x = 0
    y = r
    d = 1 - r
    points = empty set/list

    while x <= y:
        add 8 symmetric points of (x, y) around center

        x = x + 1
        if d < 0:
            d = d + 2*x + 1
        else:
            y = y - 1
            d = d + 2*(x - y) + 1

    return points
```

## R07

正确性直觉：

- 单调性：`x` 每轮递增 1，`y` 仅在需要时递减 1，所以循环必然结束；
- 决策合理性：`decision` 近似跟踪候选中点相对圆边界的位置，保证每步选取更接近真实圆弧的像素；
- 完整性：8 对称映射确保整圆无象限缺失；
- 去重策略避免 `(x=0)`、`(x=y)` 等对称重合时重复输出。

## R08

复杂度（半径为 `r`）：

- 时间复杂度：`O(r)`（第一八分圆迭代约 `r / sqrt(2)` 轮）
- 空间复杂度：`O(r)`（保存圆周点）

## R09

工程意义：

- 在软件光栅化、像素风渲染、嵌入式显示中常用；
- 运算稳定且仅整数运算，适合无 FPU 或低成本算力环境；
- 可作为椭圆、圆弧、圆形笔刷等离散几何构造的基础模块。

## R10

与其他画圆方法对比：

- 参数方程法 `x=r*cos(t), y=r*sin(t)`：实现直观，但依赖三角函数和浮点；
- 逐点距离最小搜索：准确但代价高；
- 中点圆算法：以非常低的计算成本得到高质量离散圆周，是经典“效率/质量”平衡方案。

## R11

边界条件：

- `r = 0`：输出圆心单点；
- `r = 1`：通常仅 4 个轴向点（由离散决策与去重决定）；
- 任意整数圆心（包含负坐标）都可直接处理；
- `r < 0` 明确拒绝，避免无意义输入悄悄传播。

## R12

常见错误：

- 忘记 8 对称，只画出一个八分圆；
- `decision` 更新顺序写错（先减 `y` 或先加 `x` 搞混）；
- 对称点不去重，导致点数异常；
- 用浮点开方逐点判圆，失去中点算法的本质优势。

## R13

`demo.py` 结构：

- `_eight_symmetric_points`：生成 8 对称像素；
- `midpoint_circle`：主算法（整数递推 + 去重）；
- `midpoint_circle_octant_trace`：输出第一八分圆轨迹与决策值；
- `validate_circle_properties`：验证半径误差、对称重建一致性、重复点等性质；
- `run_fixed_regression_cases`：固定半径 `0/1/2/3` 回归；
- `run_random_property_tests`：随机圆心和半径压力测试；
- `render_ascii`：字符化可视化；
- `main`：串联执行并打印结果。

## R14

运行方式（无交互）：

```bash
uv run python demo.py
```

## R15

预期输出特征：

- 打印展示用圆的圆心、半径、像素点数量；
- 打印第一八分圆轨迹 `(x, y, decision)`；
- 打印 ASCII 预览（`#` 为圆周、`C` 为圆心）；
- 所有回归与随机测试通过后输出 `All checks passed.`。

若实现有误，会抛出 `AssertionError` / `TypeError` / `ValueError`。

## R16

可扩展方向：

- 输出圆弧（限定角度区间）而非整圆；
- 填充圆盘（扫描线或种子填充结合）；
- 扩展到中点椭圆算法；
- 输出抗锯齿权重（与 Wu 风格线段/圆弧结合）。

## R17

交付文件（本目录）：

- `README.md`：R01-R18 说明完整；
- `demo.py`：可运行、可自检的 Python MVP；
- `meta.json`：UID/学科/分类/源序号/目录信息与任务元数据一致。

## R18

`demo.py` 的源码级算法流（8 步）：

1. `main` 先执行 `run_fixed_regression_cases()` 与 `run_random_property_tests()`，保证实现先通过回归和性质验证。  
2. `run_fixed_regression_cases()` 调用 `midpoint_circle(cx, cy, r)` 生成点集，并与半径 `0/1/2/3` 的可枚举期望集合逐一比对。  
3. `midpoint_circle` 初始化 `x=0, y=r, decision=1-r`，进入 `while x <= y` 的第一八分圆循环。  
4. 每轮调用 `_eight_symmetric_points` 把 `(x, y)` 映射成 8 个对称像素，使用 `seen` 集去重后写入输出列表。  
5. 然后执行中点判定递推：`x += 1`；若 `decision < 0` 选 `E` 并更新 `decision += 2*x + 1`，否则选 `SE`（先 `y -= 1`）再 `decision += 2*(x-y)+1`。  
6. `validate_circle_properties()` 对结果做结构化验证：重复点检查、边界盒检查、`x^2+y^2` 与 `r^2` 的误差上界检查。  
7. 同一验证函数再调用 `midpoint_circle_octant_trace()` 重建第一八分圆轨迹，并通过 `_expected_from_trace()` 还原整圆，要求与主算法点集完全一致。  
8. 全部检查通过后，`main` 打印展示圆的轨迹与 ASCII 图，最终输出 `All checks passed.`。  

以上流程完全由源码内自实现函数完成，不依赖第三方图形算法黑箱。

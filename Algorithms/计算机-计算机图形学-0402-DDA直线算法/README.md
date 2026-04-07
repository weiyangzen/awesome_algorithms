# DDA直线算法

- UID: `CS-0245`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `402`
- 目标目录: `Algorithms/计算机-计算机图形学-0402-DDA直线算法`

## R01

DDA（Digital Differential Analyzer）直线算法用于把连续平面中的线段，离散成屏幕像素网格中的点序列。它通过“每步做固定增量”的方式前进，属于最经典的直线光栅化入门算法之一。

## R02

目标问题：给定整数端点 `(x0, y0)` 与 `(x1, y1)`，输出一条离散像素路径 `[(x, y), ...]`，要求：

- 首点为起点，尾点为终点；
- 路径尽量贴近理想直线；
- 在网格上保持连续（相邻点步长不超过 1）。

## R03

DDA 的核心是“按主轴步数均匀插值”：

- `dx = x1 - x0`，`dy = y1 - y0`
- `steps = max(|dx|, |dy|)`
- `x_inc = dx / steps`，`y_inc = dy / steps`
- 从 `(x0, y0)` 出发，连续累加 `x_inc`、`y_inc`
- 每一步把浮点坐标映射到整数像素（本实现使用“0.5 远离 0”舍入）

这样可以在任意象限统一处理斜率与方向。

## R04

本目录实现的离散化约定：

- 端点输入为整数；
- 中间点由浮点累计值经 `_round_half_away_from_zero` 转为整数；
- 输出点数量固定为 `steps + 1`；
- 不使用图形库，仅返回像素坐标列表并提供 ASCII 预览。

## R05

MVP 保证：

- `dda_line` 可处理水平线、竖线、正负斜率、反向绘制、退化单点；
- `validate_path_properties` 会校验端点、长度、步长合法性与单调方向；
- `main` 内置固定回归样例与随机属性测试，运行无需交互输入。

## R06

伪代码：

```text
DDA_LINE(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))

    if steps == 0:
        return [(x0, y0)]

    x_inc = dx / steps
    y_inc = dy / steps
    x = float(x0)
    y = float(y0)
    points = []

    repeat steps + 1 times:
        points.append((round(x), round(y)))
        x = x + x_inc
        y = y + y_inc

    return points
```

## R07

正确性直觉：

- 当 `steps = max(|dx|, |dy|)` 时，至少有一个轴每步变化幅度为 `1`，保证逐步推进；
- 另一个轴按比例渐进，逼近理想直线；
- 共执行 `steps + 1` 次采样，因此必含起点与终点；
- 每次都由同一增量公式更新，路径不会“跳回去”。

## R08

复杂度：

- 时间复杂度：`O(max(|dx|, |dy|))`
- 空间复杂度：`O(max(|dx|, |dy|))`（保存输出点列表）

若直接写入帧缓冲而非存列表，额外空间可降到 `O(1)`。

## R09

DDA 在工程中的位置：

- 逻辑简单，适合教学、原型验证和 CPU 端快速实现；
- 与整数算法相比，它依赖浮点计算和舍入策略；
- 在现代图形 API 中直线光栅化通常由 GPU 完成，但软件渲染、离散几何预处理仍会使用 DDA 思路。

## R10

与 Bresenham 直线算法对比：

- DDA：每步进行浮点增量 + 舍入，代码直观；
- Bresenham：用整数误差项替代浮点，通常更高效、可预测；
- 在像素结果上两者常接近，但某些斜率与 tie-break 情况会出现路径差异。

## R11

边界条件：

- 起点等于终点：返回单点列表；
- `dx = 0`（竖线）或 `dy = 0`（横线）：自然覆盖；
- 负坐标、反向端点：由 `dx/dy` 符号和增量自动支持；
- 斜率绝对值 `> 1` 或 `< 1`：统一由 `steps` 选择主轴步数处理。

## R12

常见错误：

- 把 `steps` 写成 `min(|dx|, |dy|)`，导致漏点；
- 累加顺序写错（先加再采样）会丢起点；
- 使用默认 `round` 却不了解 tie-break（银行家舍入）造成和预期不一致；
- 忽略输入类型，混入非整数端点后行为不可控。

## R13

`demo.py` 代码结构：

- `_round_half_away_from_zero`：统一舍入规则，避免平台差异；
- `dda_line`：DDA 主算法；
- `render_ascii`：将点集渲染为字符网格；
- `validate_path_properties`：通用性质校验；
- `run_fixed_regression_cases`：固定样例回归；
- `run_random_property_tests`：随机端点属性测试；
- `main`：串联执行并打印示例结果。

## R14

运行命令（无交互）：

```bash
uv run python demo.py
```

## R15

预期输出特征：

- 打印若干固定样例的端点和离散像素点；
- 打印一条示例线段的 ASCII 预览（`S` 起点、`E` 终点、`#` 中间点）；
- 所有断言通过后输出 `All checks passed.`；
- 若实现有误，会抛出 `AssertionError` 或 `TypeError`。

## R16

可扩展方向：

- 抗锯齿版本（例如按覆盖率输出权重）；
- 与线段裁剪（Cohen–Sutherland / Liang–Barsky）组合；
- 扩展到 3D 栅格体素线遍历；
- 把当前“返回点列表”模式改为“像素写回调”模式，直接对接图像缓冲。

## R17

交付清单（本目录）：

- `README.md`：R01-R18 完整说明；
- `demo.py`：可运行 Python MVP（无需输入）；
- `meta.json`：元数据与任务 UID、学科、分类、源序号保持一致。

## R18

`demo.py` 的源码级算法流（8 步）：

1. `main` 首先调用 `run_fixed_regression_cases()`，逐条读取固定端点与期望像素序列。  
2. 每个样例进入 `dda_line(x0, y0, x1, y1)`，先做整数输入检查并计算 `dx`、`dy`、`steps`。  
3. 若 `steps == 0` 直接返回单点；否则计算 `x_inc = dx / steps`、`y_inc = dy / steps`。  
4. 初始化浮点游标 `x`、`y`，循环执行 `steps + 1` 次采样。  
5. 每轮调用 `_round_half_away_from_zero` 将浮点位置映射到整数像素并写入结果列表，再做 `x += x_inc`、`y += y_inc`。  
6. 回到 `run_fixed_regression_cases()`，将输出与期望列表逐点比较，再用 `validate_path_properties` 校验长度、步长和端点。  
7. `run_random_property_tests()` 使用随机端点重复执行主算法和性质断言，覆盖不同象限与退化情况。  
8. `main` 输出一条示例线的 ASCII 渲染；全部检查通过后打印 `All checks passed.`。  

以上 8 步都可在源码中逐函数追踪，不依赖第三方图形库黑箱实现。

# Z-Buffer算法

- UID: `CS-0259`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `417`
- 目标目录: `Algorithms/计算机-计算机图形学-0417-Z-Buffer算法`

## R01

Z-Buffer（深度缓冲）算法是光栅化渲染中最经典的隐藏面消除方法。  
它的核心思想是：为每个像素维护当前“最近可见片元”的深度值，只在新片元更靠近相机时才更新像素颜色。

本条目采用约定：`z` 越小越靠近相机（近处优先）。

## R02

本题的计算目标：
- 输入：若干带颜色的三角形（顶点给出 `x,y,z`），屏幕分辨率，绘制顺序。
- 输出：
  - `color_buffer (H,W,3)`：最终颜色缓冲；
  - `depth_buffer (H,W)`：每像素最终可见深度；
  - `owner_buffer (H,W)`：每像素由哪个三角形最终贡献；
  - 统计表：片元总量、深度通过/失败量、覆盖率等指标。

`demo.py` 固定样例，无交互输入。

## R03

数学判定可写成逐像素规则：

- 初始化：`depth[x,y] = +inf`
- 对每个落入像素的片元 `f`：
  - 计算插值深度 `z_f`
  - 若 `z_f < depth[x,y]`，则更新：
    - `depth[x,y] = z_f`
    - `color[x,y] = color_f`
  - 否则丢弃该片元。

其中片元是否在三角形内由重心坐标/边函数判定，深度由顶点深度线性插值获得。

## R04

MVP 的流程分为 8 个阶段：
1. 校验三角形顶点范围与颜色合法性；
2. 将 NDC 坐标 `[-1,1]` 映射到屏幕像素坐标；
3. 对每个三角形求屏幕包围盒；
4. 在包围盒内逐像素采样像素中心 `(x+0.5, y+0.5)`；
5. 用边函数做 inside-test；
6. 对 inside 片元计算重心权重并插值深度；
7. 执行深度测试并更新缓冲；
8. 汇总统计并输出顺序敏感性对比结果。

## R05

关键数据结构：
- `Triangle(name, vertices_ndc, color)`：三角形输入；
- `depth_buffer: float[H,W]`：深度缓冲，初始 `+inf`；
- `color_buffer: uint8[H,W,3]`：颜色缓冲；
- `owner_buffer: int32[H,W]`：像素来源三角形 ID（背景为 `-1`）；
- `tri_stats/global_metrics`：`pandas.DataFrame` 统计表。

## R06

`demo.py` 中核心函数职责：
- `_validate_triangle`：校验顶点形状、范围和颜色范围；
- `ndc_to_screen`：NDC 到屏幕坐标映射；
- `edge_function`：二维有向面积（inside-test 基础）；
- `rasterize_triangle`：逐像素光栅化 + 深度测试；
- `render_scene`：按给定绘制顺序渲染整个场景；
- `summarize_owner_counts`：统计最终可见像素归属；
- `main`：构造场景、执行 Z-Buffer 与 Painter 对照并断言。

## R07

复杂度分析（`T` 个三角形，`A_i` 为第 `i` 个三角形包围盒像素面积）：
- 时间复杂度：`O(sum(A_i))`，最坏可达 `O(T * W * H)`；
- 空间复杂度：`O(W*H)`（颜色、深度、归属三张缓冲）。

深度测试本身是常数操作，瓶颈主要来自片元遍历量。

## R08

正确性依赖以下不变量：
1. 深度缓冲初始为 `+inf`，保证首个片元可通过；
2. inside-test 对三角形绕序（顺/逆时针）都能正确判定；
3. 深度插值只在 triangle 内部执行；
4. 深度比较严格使用 `<`（避免同深度重复覆盖抖动）；
5. 每次通过测试时同步更新 `depth/color/owner` 三个缓冲。

## R09

数值与实现注意点：
- 边函数判定使用 `eps` 容差，避免边界浮点误差导致漏填；
- 三角形面积接近 0 视为退化并跳过；
- 片元深度应保持在约定区间（本例构造于 `[0,1]`）；
- 相同深度片元的 tie-break 在真实引擎中需更严格规则（如多关键字比较），本 MVP 采用简化策略。

## R10

实验场景设计：
- 分辨率：`96 x 72`；
- 三角形数量：3 个（`far_green`、`near_red`、`slanted_blue`）；
- 两组绘制顺序：`[0,1,2]` 与 `[2,0,1]`；
- 同时运行：
  - 启用 Z-Buffer 的渲染；
  - 关闭深度测试的 Painter 风格渲染。

这样可同时验证“Z-Buffer 顺序不敏感”和“无深度测试顺序敏感”。

## R11

运行方式（无交互）：

```bash
cd Algorithms/计算机-计算机图形学-0417-Z-Buffer算法
uv run python demo.py
```

运行后会打印三角形级统计、全局指标、像素归属统计与对比结论。

## R12

关键输出指标含义：
- `inside_fragments`：落在三角形内部的候选片元数；
- `depth_pass`：通过深度测试并写入缓冲的片元数；
- `depth_fail`：被更近片元遮挡而丢弃的片元数；
- `overwritten`：写入时覆盖了已有像素的次数；
- `coverage_ratio`：最终可见像素占总像素比例；
- `painter_order_diff_pixels`：无深度测试时，不同绘制顺序导致差异的像素数量。

## R13

`demo.py` 的自动断言包括：
1. Z-Buffer 两种绘制顺序得到完全一致的 `owner_buffer`；
2. Z-Buffer 两种绘制顺序得到一致的 `depth_buffer`；
3. Painter 两种绘制顺序结果必须存在像素差异；
4. Z-Buffer 结果与至少一种 Painter 结果存在差异（说明遮挡确实被处理）。

这些断言确保脚本不仅“能跑”，还“能证明机制生效”。

## R14

边界与异常处理：
- 非法分辨率（`width<=0` 或 `height<=0`）会抛 `ValueError`；
- 顶点维度不是 `(3,3)` 或含非有限值会抛 `ValueError`；
- 顶点 `xy` 不在 `[-1,1]`、`z` 不在 `[0,1]` 会抛 `ValueError`；
- 颜色不在 `0..255` 会抛 `ValueError`；
- 退化三角形（面积约 0）会被安全跳过。

## R15

Z-Buffer 在工程中的典型应用：
- 实时渲染器的隐藏面消除；
- 游戏/可视化中的网格场景遮挡计算；
- 软件光栅器教学实现；
- 与模板缓冲、阴影映射、延迟渲染等管线模块联合工作。

## R16

本 MVP 的范围限制：
- 仅处理三角形，不含线段/点精细光栅规则；
- 未实现近远裁剪与透视正确插值（这里是简化示例）；
- 未包含抗锯齿、多重采样、早深度优化（Early-Z）等高级机制；
- 未输出图像文件，仅输出统计结果用于验证算法。

## R17

可扩展方向：
1. 增加透视投影与透视正确插值（`1/w` 插值）；
2. 引入背面剔除、裁剪与视口变换完整管线；
3. 加入 MSAA 与深度偏移（depth bias）；
4. 增加层次化 Z（Hi-Z）和分块并行优化；
5. 输出 PNG/PPM 结果以便可视化回归测试。

## R18

`demo.py` 源码级算法链路（9 步）：
1. `main()` 定义 3 个重叠三角形（含不同深度）和两组绘制顺序。  
2. `render_scene(..., use_zbuffer=True)` 初始化 `depth=+inf`、`color=0`、`owner=-1` 三张缓冲。  
3. 每个三角形先经 `ndc_to_screen` 把顶点从 NDC 映射到像素坐标，再交给 `rasterize_triangle`。  
4. `rasterize_triangle` 用 `edge_function` 计算有向面积并构造包围盒，在包围盒内遍历像素中心。  
5. 对每个像素计算 `w0/w1/w2` 做 inside-test；通过后归一化成重心权重 `alpha/beta/gamma`。  
6. 用 `z = alpha*z0 + beta*z1 + gamma*z2` 插值得到片元深度，并执行 `z < depth[y,x]` 深度测试。  
7. 若通过则同步更新 `depth_buffer`、`color_buffer`、`owner_buffer`，并累积 `depth_pass/overwritten`；否则累积 `depth_fail`。  
8. `main()` 分别渲染两种顺序，断言 Z-Buffer 结果一致；再关闭深度测试执行 Painter 渲染，验证顺序差异显著。  
9. 最后输出三角形级统计、全局指标和像素归属统计，完成“机制正确 + 行为可解释”的闭环。

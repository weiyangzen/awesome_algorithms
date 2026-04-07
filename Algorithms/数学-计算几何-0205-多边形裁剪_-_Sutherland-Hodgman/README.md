# 多边形裁剪 - Sutherland-Hodgman

- UID: `MATH-0205`
- 学科: `数学`
- 分类: `计算几何`
- 源序号: `205`
- 目标目录: `Algorithms/数学-计算几何-0205-多边形裁剪_-_Sutherland-Hodgman`

## R01

Sutherland-Hodgman 是经典二维多边形裁剪算法：
- 输入一个待裁剪多边形（subject）和一个裁剪窗口多边形（clipper）。
- 按裁剪多边形的每条边依次做“半平面裁剪”。
- 每一步输出新的顶点序列，最终得到交集多边形。

本目录给出一个最小可运行 MVP：`demo.py` 仅依赖 `numpy`，完整手写算法流程，不调用黑盒几何 API。

## R02

本实现的问题定义：
- 输入：
  - `subject_polygon`: 任意简单多边形顶点序列（可凹）。
  - `clip_polygon`: 凸多边形顶点序列（可顺时针或逆时针）。
- 输出：`ClipResult`
  - `subject`: 清洗后的输入多边形；
  - `clipper`: 清洗后的裁剪多边形；
  - `clipped`: 裁剪结果顶点序列；
  - `subject_area` / `clipped_area`: 面积；
  - `is_empty`: 是否为空结果。

约定：输入点为 `shape=(n,2)` 的有限浮点数。

## R03

算法前提与约束：
- Sutherland-Hodgman 对凸裁剪窗口有严格正确性保证，因此 `clip_polygon` 必须凸。
- `subject_polygon` 可凸可凹，但默认是非自交简单多边形。
- 若裁剪窗口退化（面积接近 0）或非凸，代码直接报错。

这保证了 MVP 的行为边界清晰、可验证。

## R04

核心几何判定：
1. 叉积
   - `cross((b-a), (p-a))` 判定点 `p` 在有向边 `a->b` 的哪一侧。
2. 内外判定
   - 若裁剪多边形为逆时针，则“内侧”为边左侧；
   - 若为顺时针，则“内侧”为边右侧；
   - 统一写法：`orientation * cross >= -eps`。
3. 线段与裁剪边求交
   - 对 segment `s->e` 和 clip line `a->b` 做参数方程求交；
   - 接近平行时使用稳定回退值（中点）避免数值爆炸。

## R05

高层流程（每条裁剪边循环一次）：
1. 输入转为 `numpy` 数组并校验维度/有限性。
2. 去掉相邻重复点与重复闭合点。
3. 检查裁剪多边形凸性与方向。
4. 令 `output = subject`。
5. 对 clipper 每条边 `E`：
   - 用 4 种状态转移规则（in->in / in->out / out->in / out->out）扫描 `output`；
   - 生成新的 `output`。
6. 若某步后 `output` 为空，提前终止。
7. 清理输出重复点，计算面积并返回。

## R06

正确性直觉：
- 凸裁剪窗口可看成多个半平面的交集。
- 每次按一条边裁剪，相当于和一个半平面求交。
- 多次迭代后得到的是“subject 与所有半平面的交”，即 `subject ∩ clipper`。
- 顶点更新规则完整覆盖边与半平面的过渡情形，因此不会漏掉交点。

## R07

复杂度分析（`n` 为 subject 顶点数，`m` 为 clipper 顶点数）：
- 每一轮边裁剪线性扫描当前顶点序列，整体上界 `O(n*m)`。
- 空间复杂度 `O(n + k)`，`k` 为中间生成顶点数（通常同量级）。

Sutherland-Hodgman 的优势是实现简单、工程可控，适合小到中规模几何任务。

## R08

数值稳定性策略：
- 统一使用 `eps`（默认 `1e-9`）处理“在边上/共线/近平行”情况。
- 输入含 `NaN/Inf` 直接拒绝。
- 去除相邻重复点，减少零长度边导致的异常。
- 求交时若分母绝对值过小，采用平滑回退，避免除零和极端放大误差。

## R09

边界情况处理：
- `subject` 为空：直接返回空输出。
- `subject` 完全在窗口外：中途变空并提前终止。
- `subject` 完全在窗口内：输出与输入等价（仅有轻微去重/顺序规范化）。
- 裁剪窗口顺时针输入：通过方向符号自动兼容。
- 裁剪窗口非凸：抛出 `ValueError`，避免产生伪结果。

## R10

`demo.py` 主要函数：
- `as_xy_array`：输入校验。
- `remove_consecutive_duplicates`：去重清洗。
- `cross2` / `cross_points` / `polygon_area2`：几何基础算子。
- `is_convex_polygon` / `orientation_sign`：裁剪窗口合法性检查。
- `inside_half_plane`：点在半平面内判定。
- `segment_line_intersection`：线段与裁剪边求交。
- `clip_against_one_edge`：单边裁剪。
- `sutherland_hodgman`：主算法。
- `summarize_case` / `main`：样例运行与断言输出。

## R11

运行方式：

```bash
cd Algorithms/数学-计算几何-0205-多边形裁剪_-_Sutherland-Hodgman
python3 demo.py
```

脚本无交互输入，会自动运行三组固定样例并打印结果。

## R12

输出解读（每个 case）：
- `subject vertices`：清洗后的待裁剪多边形顶点数。
- `clipper vertices`：裁剪窗口顶点数。
- `output vertices`：裁剪结果顶点数。
- `subject area`：原多边形面积。
- `clipped area`：裁剪后面积。
- `is empty`：结果是否为空。
- `output preview`：结果顶点坐标预览。

## R13

内置测试样例：
1. `Concave clipped by rectangle`
   - 凹多边形与矩形裁剪，验证一般工作流。
2. `Convex clipped by CW triangle`
   - 裁剪窗口使用顺时针输入，验证方向鲁棒性。
3. `Disjoint polygons`
   - 两多边形不相交，验证空结果分支。

样例均附带运行时断言（顶点在窗口内、面积不增）。

## R14

可调参数：
- `eps`（默认 `1e-9`）
  - 更小：更严格，可能受浮点噪声影响；
  - 更大：更稳定，但可能吞掉细小几何细节。

实践建议：坐标尺度越大，通常需要适当增大 `eps` 或先做归一化。

## R15

与其他裁剪方法对比：
- Sutherland-Hodgman：
  - 优点：实现短、流程直观、适合凸裁剪窗口；
  - 局限：对非凸裁剪窗口不直接适用。
- Weiler-Atherton：
  - 可处理更一般的多边形布尔运算，但实现更复杂。
- Liang-Barsky / Cohen-Sutherland：
  - 主要针对线段裁剪，不直接替代多边形裁剪。

## R16

典型应用：
- 图形学渲染管线中的裁剪阶段。
- GIS 空间对象与视窗求交。
- 路径规划中的可行区域裁剪。
- 几何预处理（模型切割、边界清洗）。

## R17

本目录 MVP 特性：
- 最小工具栈：仅 `numpy`。
- 主算法、求交与几何判定全部源码可追踪。
- 对方向、空交、重复点、非凸 clipper 有明确处理。
- 提供自检断言，运行后可快速判断正确性。

## R18

`demo.py` 源码级流程拆解（8 步）：
1. `main` 构造三组固定测试多边形（凹+矩形、凸+顺时针三角形、不相交）。
2. `summarize_case` 调用 `sutherland_hodgman` 执行裁剪并汇总指标。
3. `sutherland_hodgman` 先执行 `as_xy_array` 与 `remove_consecutive_duplicates` 清洗输入。
4. 对裁剪窗口执行 `is_convex_polygon` 与 `orientation_sign`，确定合法性和内侧方向。
5. 进入主循环：按 clipper 每条边调用 `clip_against_one_edge`。
6. `clip_against_one_edge` 对 subject 的每条边做 4 状态转移，必要时通过 `segment_line_intersection` 插入交点。
7. 所有裁剪边处理完后，对结果做去重、方向规范化和面积计算，封装为 `ClipResult`。
8. `summarize_case` 再做断言：输出顶点必须位于 clipper 内/边界上，且裁剪面积不大于原面积，然后打印摘要。

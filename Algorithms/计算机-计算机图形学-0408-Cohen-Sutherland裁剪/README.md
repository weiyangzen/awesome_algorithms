# Cohen-Sutherland裁剪

- UID: `CS-0251`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `408`
- 目标目录: `Algorithms/计算机-计算机图形学-0408-Cohen-Sutherland裁剪`

## R01

Cohen-Sutherland 裁剪用于将二维线段裁剪到轴对齐矩形窗口中。  
它通过为线段端点分配 4 位区域编码（左/右/下/上）来快速判定：

1. 两端都在窗口内：直接接受；
2. 两端共享某个外侧位：直接拒绝；
3. 其余情况：与边界求交并迭代更新端点。

该方法是经典视口裁剪算法，逻辑直观且工程上稳定。

## R02

本题 MVP 目标：

1. 手写实现 Cohen-Sutherland 主算法（非黑盒）；
2. 输入固定窗口与一批线段，输出逐线段裁剪结果；
3. 记录接受/拒绝原因、迭代次数、裁剪后长度；
4. 使用手写 Liang-Barsky 作为参考路径进行一致性校验；
5. 脚本无交互，`uv run python demo.py` 直接复现实验。

## R03

区域编码定义（4 bit）：

- `LEFT = 1`：`x < xmin`
- `RIGHT = 2`：`x > xmax`
- `BOTTOM = 4`：`y < ymin`
- `TOP = 8`：`y > ymax`

对点 `P(x,y)`，`region_code()` 根据以上规则组合位码。  
编码为 `0` 表示点在窗口内（含边界）。

## R04

Cohen-Sutherland 判定规则：

1. `code0 | code1 == 0`：两端都在窗口内，线段直接接受；
2. `code0 & code1 != 0`：两端在同一外侧半平面，线段直接拒绝；
3. 其他情况：取一个外点，与对应边界求交，替换外点后继续迭代。

这等价于不断缩短线段直到“全内”或“可证全外”。

## R05

与边界求交公式（`P0(x0,y0), P1(x1,y1), dx=x1-x0, dy=y1-y0`）：

- 与上边界 `y=ymax`：`x = x0 + dx*(ymax-y0)/dy`
- 与下边界 `y=ymin`：`x = x0 + dx*(ymin-y0)/dy`
- 与右边界 `x=xmax`：`y = y0 + dy*(xmax-x0)/dx`
- 与左边界 `x=xmin`：`y = y0 + dy*(xmin-x0)/dx`

`demo.py` 在除法前做 `|dx|`/`|dy|` 近零保护，避免数值异常。

## R06

`demo.py` 的核心结构：

- `ClipWindow`：窗口参数与 `contains()` 判定；
- `ClipResult`：裁剪结果（接受标志、端点、原因、迭代次数）；
- `Segment`：测试线段样本；
- `region_code()`：计算端点外码；
- `cohen_sutherland_clip()`：主算法；
- `liang_barsky_reference()`：手写参考算法用于交叉验证；
- `run_experiment()`：批量实验与门禁断言。

## R07

主流程伪代码：

```text
for each segment (p0, p1):
    loop:
        code0 = region_code(p0)
        code1 = region_code(p1)
        if (code0 | code1) == 0: accept
        if (code0 & code1) != 0: reject
        out_code = code0 if code0 != 0 else code1
        inter = intersect_with_corresponding_boundary(out_code)
        replace outside endpoint with inter
```

MVP 还设置 `max_iters` 保护，防止退化输入触发无限循环。

## R08

正确性直觉：

- 区域编码把“点在窗口外哪一侧”离散化；
- `AND` 规则给出可证相离（同侧外部）；
- 否则每次都把一个外点推进到边界上，线段外部部分单调减少；
- 在有限步内达到“全内接受”或“全外拒绝”。

因此算法对轴对齐矩形裁剪是完备的。

## R09

复杂度分析（单条线段）：

- 时间复杂度：`O(k)`，`k` 为迭代次数，通常很小（常数级）；
- 空间复杂度：`O(1)`。

批量 `N` 条线段时，总体约为 `O(N)`，适合实时渲染中的视口裁剪前处理。

## R10

数值稳定与边界策略：

1. 边界按“闭区间”处理，边界点算 inside；
2. 使用 `EPS=1e-12` 做近零判断；
3. 若分母近零且仍需对应方向求交，返回数值保护原因；
4. `contains()` 用容差检查裁剪端点是否落在窗口内；
5. 本数据集中要求不触发数值保护分支，否则视为异常。

## R11

实验设计（固定、可复现）：

- 窗口：`[0,10] x [0,8]`；
- 样本：`10` 条手工边界线段 + `44` 条随机线段；
- 随机种子：`2026`；
- 输出：逐线段预览表 + 汇总指标表。

手工样例覆盖了：全内、全外、水平/竖直穿越、边界重合、点线段等场景。

## R12

运行方式：

```bash
cd Algorithms/计算机-计算机图形学-0408-Cohen-Sutherland裁剪
uv run python demo.py
```

脚本无交互输入，运行后直接打印结果并执行断言校验。

## R13

输出字段说明（Preview）：

- `code0, code1`：输入端点区域编码；
- `accepted`：是否被窗口接受；
- `reason`：`accepted` 或拒绝原因（如 `trivial_reject`）；
- `iterations`：Cohen-Sutherland 循环迭代步数；
- `input_len`：原始线段长度；
- `clipped_len`：裁剪后长度（拒绝时为 `0`）。

## R14

输出字段说明（Summary）：

1. `total_segments`：总线段数；
2. `accepted_segments / rejected_segments`：接受与拒绝数量；
3. `accept_rate`：接受率；
4. `avg_iterations`：平均迭代次数；
5. `avg_input_len`：平均输入长度；
6. `avg_clipped_len_accept_only`：仅在接受线段上统计的平均裁剪长度；
7. `max_endpoint_delta_vs_lb`：与 Liang-Barsky 端点偏差上界；
8. `trivial_reject_count`：平凡拒绝次数。

## R15

内置质量门禁：

1. Cohen-Sutherland 与 Liang-Barsky 的接受/拒绝判定必须一致；
2. 接受样本的裁剪端点偏差必须 `< 1e-8`；
3. 接受端点必须落在窗口内；
4. 数据集必须同时包含接受和拒绝样本；
5. 不允许出现 `max_iter_guard` 或数值并行保护原因。

## R16

与 Liang-Barsky 的对比：

- Cohen-Sutherland：位运算判定直观，适合教学和实现清晰性；
- Liang-Barsky：参数区间更新步数固定，常数开销通常更小；
- 本条目把 Liang-Barsky 作为“校验算法”而非主流程，保证主算法透明可追踪。

## R17

当前 MVP 的限制：

1. 仅支持二维轴对齐矩形窗口；
2. 不处理任意旋转窗口或一般多边形裁剪；
3. 未做可视化绘图（仅文本表格输出）；
4. 数值保护分支以保守失败处理，未实现更复杂退化细分；
5. 面向教学可读性，未做批量 SIMD/GPU 加速。

## R18

`demo.py` 源码级算法链路（8 步）：

1. `main()` 构造窗口 `ClipWindow(0,10,0,8)`，调用 `build_segments()` 生成确定性测试集。  
2. `run_experiment()` 逐条读取线段，先调用 `cohen_sutherland_clip()` 执行主裁剪流程。  
3. 在 `cohen_sutherland_clip()` 中，`region_code()` 为两个端点计算 4 位外码。  
4. 通过 `(code0 | code1)` 做平凡接受、通过 `(code0 & code1)` 做平凡拒绝。  
5. 对需要继续处理的线段，选取一个外点，根据外码选择上下左右边界并计算交点，替换该外点后迭代。  
6. 算法在“接受/拒绝/最大迭代保护”三种终止条件之一返回 `ClipResult`。  
7. `run_experiment()` 同时调用 `liang_barsky_reference()` 对同一线段做参考裁剪，并检查接受判定与端点偏差。  
8. 汇总 `detail_df` 与 `summary_df` 后执行门禁断言，最后在 `main()` 打印表格并输出 `All checks passed.`。

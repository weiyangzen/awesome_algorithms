# Liang-Barsky裁剪

- UID: `CS-0252`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `409`
- 目标目录: `Algorithms/计算机-计算机图形学-0409-Liang-Barsky裁剪`

## R01

Liang-Barsky 裁剪用于将二维线段裁剪到一个轴对齐矩形窗口内。

相比逐边试探的做法，它把线段参数化为 `P(t)=P0+t(P1-P0), t∈[0,1]`，然后通过 4 个半平面不等式直接更新参数区间 `[u_enter, u_leave]`。当区间非空时得到裁剪后线段，区间为空则拒绝。

## R02

问题定义（本 MVP）：

1. 输入：一个矩形窗口 `xmin,xmax,ymin,ymax`，以及若干二维线段端点 `P0(x0,y0), P1(x1,y1)`；
2. 输出：每条线段是否被接受、裁剪后端点、进入参数 `u_enter`、离开参数 `u_leave`；
3. 额外输出：批量统计表（接受率、平均长度等）与自动校验结果。

## R03

参数方程：

- `x(t)=x0+t*dx, dx=x1-x0`
- `y(t)=y0+t*dy, dy=y1-y0`

窗口约束可写为 4 组 `p_i * t <= q_i`：

1. 左边界 `x>=xmin`: `p1=-dx, q1=x0-xmin`
2. 右边界 `x<=xmax`: `p2= dx, q2=xmax-x0`
3. 下边界 `y>=ymin`: `p3=-dy, q3=y0-ymin`
4. 上边界 `y<=ymax`: `p4= dy, q4=ymax-y0`

## R04

Liang-Barsky 的区间更新规则：

1. 初始化 `u_enter=0, u_leave=1`；
2. 对每个 `(p_i,q_i)`：
   - 若 `p_i==0` 且 `q_i<0`，线段平行且在该边界外侧，直接拒绝；
   - 否则 `r=q_i/p_i`：
     - `p_i<0` 表示潜在进入边界，`u_enter=max(u_enter,r)`；
     - `p_i>0` 表示潜在离开边界，`u_leave=min(u_leave,r)`；
3. 若任意时刻 `u_enter>u_leave`，区间为空，拒绝；
4. 接受时端点为 `P(u_enter), P(u_leave)`。

## R05

`demo.py` 的核心数据结构：

- `ClipWindow`：矩形窗口与 `contains()` 判定；
- `ClipResult`：是否接受、裁剪端点、参数区间、拒绝原因；
- `Segment`：测试线段（索引、端点、标签）；
- `detail_df`：逐线段结果表；
- `summary_df`：汇总指标表。

## R06

正确性直觉：

- 每条边界都对应对参数 `t` 的一个上界或下界；
- 4 条边界约束交集即有效 `t` 区间；
- 区间非空意味着线段与窗口有交，且区间端点就是裁剪点；
- 该思路本质是一次线性不等式区间收缩，避免了重复几何求交。

## R07

复杂度（单条线段）：

- 时间复杂度：`O(1)`，固定处理 4 条边界；
- 空间复杂度：`O(1)`。

批量 `N` 条线段时：

- 时间复杂度：`O(N)`；
- 空间复杂度：若保存结果表为 `O(N)`。

## R08

数值与边界处理：

- 使用 `EPS=1e-12` 判断“近似平行”；
- 当 `|p_i|<=EPS` 且 `q_i<0`，按平行外侧拒绝；
- 接受后用 `window.contains()` 断言裁剪端点在窗口内（带容差）；
- 对退化线段（端点重合）同样适用：若点在窗口内则接受，否则拒绝。

## R09

MVP 设计取舍：

- 主算法严格手写 Liang-Barsky，不调用图形库黑盒裁剪；
- 额外实现 Cohen-Sutherland 仅用于交叉验证（不是主实现路径）；
- 使用 `numpy` 处理向量/数值，`pandas` 组织实验输出；
- 不引入绘图依赖，保证 `uv run python demo.py` 即可复现。

## R10

`demo.py` 函数职责：

- `liang_barsky_clip`：主算法实现；
- `_region_code` / `cohen_sutherland_clip`：参考算法校验链路；
- `build_segments`：构造手工边界样例 + 随机样例；
- `run_experiment`：逐条运行、汇总指标、执行一致性断言；
- `main`：组织流程并打印预览表与汇总表。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机图形学-0409-Liang-Barsky裁剪
uv run python demo.py
```

脚本无交互输入，运行后直接打印结果与校验结论。

## R12

输出说明：

- `Preview` 表（前 12 条）：
1. `accepted`：是否接受；
2. `reason`：`accepted / parallel_outside / empty_interval`；
3. `u_enter, u_leave`：参数区间端点；
4. `input_len, clipped_len`：原始与裁剪后长度。

- `Summary` 表：
1. `total_segments` / `accepted_segments` / `rejected_segments`；
2. `accept_rate`；
3. `avg_input_len`；
4. `avg_clipped_len_accept_only`；
5. `max_endpoint_delta_vs_cs`（与参考算法端点偏差最大值）。

## R13

内置验证门禁：

1. Liang-Barsky 与 Cohen-Sutherland 的接受/拒绝判定必须一致；
2. 对接受线段，Liang-Barsky 与参考算法的端点偏差需 `< 1e-8`；
3. 接受端点必须落在窗口内（容差检查）；
4. 数据集必须同时包含接受与拒绝样例；
5. 全部通过后打印 `All checks passed.`。

## R14

实验配置（固定）：

- 窗口：`[0,10] x [0,8]`；
- 样本数量：`8` 条手工边界样例 + `40` 条随机样例；
- 随机种子：`2026`；
- 数值容差：`EPS=1e-12`。

该配置可以稳定覆盖：完全内部、完全外部、跨边穿越、平行外侧、边界触碰、退化点等典型情况。

## R15

与相关方法差异：

- Cohen-Sutherland：通过区域编码反复与边界求交，逻辑直观但可能多次迭代；
- Liang-Barsky（本条目）：直接在参数空间做上下界收缩，步骤固定、常数小；
- Cyrus-Beck：可扩展到凸多边形，但实现和法向量管理更复杂。

## R16

适用与限制：

- 适用：二维轴对齐裁剪窗口（如视口裁剪、基础图形教学）；
- 不适用：任意旋转矩形或非凸裁剪区域（需要更通用算法）；
- 受限于浮点误差：边界极近点可能受容差参数影响。

## R17

可扩展方向：

1. 扩展到批量向量化裁剪（一次处理 `N` 条线段）；
2. 扩展到三维线段与包围盒裁剪；
3. 支持任意凸多边形（Cyrus-Beck）；
4. 增加可视化输出（matplotlib）展示裁剪前后效果；
5. 在 GPU 张量框架中做大规模裁剪性能评估。

## R18

`demo.py` 的源码级算法流程（8 步，非黑箱）：

1. `main()` 固定窗口 `[0,10]x[0,8]`，调用 `build_segments()` 生成手工+随机线段集合。  
2. `run_experiment()` 逐条读取线段，调用 `liang_barsky_clip(window,p0,p1)` 进入主算法。  
3. 在 `liang_barsky_clip` 中计算 `dx,dy`，并构造 4 组 `(p_i,q_i)` 对应左右上下边界约束。  
4. 初始化 `u_enter=0,u_leave=1`，逐条边界执行：平行外侧立即拒绝；否则计算 `r=q/p` 并更新进入/离开参数。  
5. 若更新后 `u_enter>u_leave`，返回 `empty_interval`；否则用 `P(u_enter),P(u_leave)` 生成裁剪端点并接受。  
6. `run_experiment()` 同时调用 `cohen_sutherland_clip()` 作为参考实现，检查接受判定与端点一致性。  
7. 通过 `window.contains()`、端点偏差阈值、样本覆盖性断言，构建 `detail_df` 与 `summary_df`。  
8. `main()` 打印前 12 条结果与汇总指标，全部断言通过后输出 `All checks passed.`。

# 贝塞尔曲面

- UID: `CS-0279`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `438`
- 目标目录: `Algorithms/计算机-计算机图形学-0438-贝塞尔曲面`

## R01

贝塞尔曲面（Bezier Surface）通常指张量积（tensor-product）贝塞尔曲面补丁。  
给定控制网格 `P_{i,j}`，其中 `i=0..m`、`j=0..n`，曲面定义为：

`S(u,v) = sum_{i=0..m} sum_{j=0..n} B_i^m(u) B_j^n(v) P_{i,j},  (u,v) in [0,1]^2`

这里 `B_i^m(u)=C(m,i)u^i(1-u)^(m-i)` 是 Bernstein 基函数。  
曲面经过四个角点控制点 `P_{0,0}, P_{0,n}, P_{m,0}, P_{m,n}`，整体位于控制网格凸包内。

## R02

在计算机图形学、CAD/CAM、工业造型中，贝塞尔曲面是最经典的参数曲面之一：

- 它把复杂曲面拆成可编辑的控制网格补丁；
- 边界退化为贝塞尔曲线，便于与曲线系统统一；
- 通过局部控制点调整可直观地改变形状；
- 是理解 B-spline / NURBS 的基础构件。

在教学与原型开发中，Bezier patch 也常用于演示“曲线 -> 曲面”的张量积推广。

## R03

本条目 MVP 聚焦一个可运行、可验证、非黑盒的最小实现：

- 实现 Bernstein 形式的贝塞尔曲面网格采样；
- 实现 de Casteljau 形式的曲面点求值，并与 Bernstein 逐点对比；
- 实现偏导控制网格与法向量计算；
- 做边界一致性、角点插值、偏导数有限差分对照、凸包弱校验等自动断言。

## R04

本实现用到的核心公式：

1. 曲面定义：`S(u,v)=sum_i sum_j B_i^m(u) B_j^n(v) P_{i,j}`。  
2. `u` 向偏导：
   `S_u(u,v)=m * sum_{i=0..m-1} sum_{j=0..n} B_i^{m-1}(u) B_j^n(v) (P_{i+1,j}-P_{i,j})`。  
3. `v` 向偏导：
   `S_v(u,v)=n * sum_{i=0..m} sum_{j=0..n-1} B_i^m(u) B_j^{n-1}(v) (P_{i,j+1}-P_{i,j})`。  
4. 法向量：`N(u,v)=S_u(u,v) x S_v(u,v)`（三维情形）。  
5. 边界退化性质：`S(0,v), S(1,v), S(u,0), S(u,1)` 分别是控制网格四条边对应的贝塞尔曲线。

## R05

设控制网格大小为 `(m+1) x (n+1)`，采样网格大小为 `Nu x Nv`：

- Bernstein 张量积采样：时间复杂度约 `O(Nu*Nv*m*n)`；
- de Casteljau 单点求值：先沿一方向做 `m+1` 条曲线递推，再做一次曲线递推，整体高于 Bernstein；
- de Casteljau 全网格采样通常显著慢于 Bernstein，更适合做“正确性对照”；
- 空间复杂度主要来自采样结果，约 `O(Nu*Nv*dim)`。

因此 MVP 中以 Bernstein 作为主计算路径，以 de Casteljau 作为验证路径。

## R06

`demo.py` 的内置示例使用 `4x4` 三维控制网格（双三次补丁）并输出：

- Bernstein 与 de Casteljau 的最大点位差；
- Bernstein 基函数分片和误差（partition of unity）；
- 曲面四条边与对应贝塞尔边界曲线的一致性误差；
- 四个角点插值误差；
- 偏导数解析表达与有限差分的误差；
- 中心法向量模长、包围盒检查、近似曲面面积。

脚本带 `assert`，可作为自动验收。

## R07

优点：

- 核心公式全部源码实现，便于追踪与教学；
- 同时包含两条独立求值路径，可互相校验；
- 包含几何与数值双重断言，结果可复现。

局限：

- 仅覆盖单个 Bezier patch，不含多补丁拼接；
- 未处理 `G1/C1/C2` 跨补丁连续性约束；
- 未做渲染输出，仅做数值级验证。

## R08

前置知识：

- Bernstein 多项式与二项式系数；
- de Casteljau 递推（曲线）；
- 张量积曲面与偏导/法向量基础；
- 浮点误差与有限差分近似。

运行环境：

- Python `>=3.10`
- `numpy`

## R09

适用场景：

- 图形学课程中演示参数曲面定义与求值；
- 曲面建模前端中的算法原型验证；
- B-spline / NURBS 前的基础模块测试。

不适用场景：

- 需要复杂工业曲面拼接与高阶连续性控制；
- 需要实时 GPU 渲染管线；
- 需要鲁棒 CAD 几何内核（精确算术、拓扑修复）。

## R10

正确性直觉：

1. Bernstein 与 de Casteljau 在数学上等价，逐点误差应接近机器精度；
2. 边界参数固定后，曲面应退化为边界贝塞尔曲线；
3. 四个角点必须精确插值；
4. 偏导控制网格公式应与有限差分一致；
5. 曲面采样点应落在控制网格凸包内（MVP 以轴对齐包围盒作弱校验）。

`demo.py` 中上述性质均被量化并自动断言。

## R11

数值稳定策略：

- 使用 de Casteljau 路径对 Bernstein 路径做交叉验证；
- 偏导同时做解析计算和中心差分比较，防止单实现误差隐藏；
- 浮点阈值分级设置：严格项 `1e-12`，有限差分项适当放宽到 `1e-6` 量级；
- 控制网格使用固定常量，避免随机输入导致验收波动。

## R12

关键参数：

- `control_net`：形状 `(m+1, n+1, dim)` 的控制网格；
- `u_values`, `v_values`：参数采样网格；
- `eps`：有限差分步长（脚本中为 `1e-6`）；
- 断言阈值：控制“严格性 vs 浮点容忍度”。

调参建议：

- 提高 `Nu/Nv` 可提升几何检查分辨率；
- 提高曲面次数可观察 Bernstein 与 de Casteljau 的数值差异；
- 若坐标尺度很大，可按量纲调整断言阈值。

## R13

- 近似比保证：N/A（该问题是几何构造与求值，不是近似优化）。
- 随机成功率保证：N/A（实现为确定性计算，不依赖随机采样）。

可验证保证（在当前输入与数值阈值下）：

- 双实现求值误差低于阈值；
- 边界曲线、角点插值性质成立；
- 偏导解析式与有限差分匹配；
- 曲面面积估计为正且包围盒检查通过。

## R14

常见失效模式：

1. 控制网格维度不合法（例如少于二维网格结构）；
2. 参数超出 `[0,1]` 导致“补丁内”语义失效；
3. 高次数或大尺度坐标下，直接多项式计算误差放大；
4. 控制网格退化（共线/共面特殊形状）导致法向量接近零。

诊断方式：

- 检查 `ValueError` 的输入形状提示；
- 观察 `Max(Bernstein - de Casteljau)` 是否异常增大；
- 观察 `Center normal magnitude` 与偏导有限差分误差。

## R15

可扩展方向：

- 多补丁拼接与连续性约束（`C1/G1`）；
- 使用三角化或自适应细分生成网格并导出到渲染管线；
- 引入 B-spline / NURBS 统一建模；
- 增加二阶偏导、主曲率、Gaussian/Mean curvature 估计；
- 增加与 `scipy` 或 CAD 内核的数值对照测试。

## R16

相关算法/主题：

- 贝塞尔曲线与 de Casteljau 算法；
- de Boor 算法（B-spline）；
- NURBS 曲线与曲面；
- Coons Patch、Gregory Patch 等曲面拼接方法；
- 曲面细分（Subdivision Surfaces）。

## R17

`demo.py` 的 MVP 功能清单：

- `bernstein_basis`、`bezier_surface_bernstein`：张量积 Bernstein 求值；
- `de_casteljau_surface_point`、`bezier_surface_de_casteljau`：递推求值与全网格对照；
- `derivative_control_nets`、`surface_partials`：偏导控制网格与法向量计算；
- 边界/角点/偏导/包围盒/面积等自动断言与输出指标。

运行方式：

```bash
cd Algorithms/计算机-计算机图形学-0438-贝塞尔曲面
uv run python demo.py
```

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `build_demo_control_net` 构造一个固定 `4x4x3` 控制网格，保证脚本无交互、可复现。  
2. `bernstein_basis` 分别生成 `u`、`v` 两个方向的 Bernstein 基函数矩阵。  
3. `bezier_surface_bernstein` 用 `einsum` 实现 `sum_i sum_j B_i(u)B_j(v)P_{i,j}`，得到主采样曲面。  
4. `bezier_surface_de_casteljau` 对每个 `(u,v)` 调用 `de_casteljau_surface_point`，形成独立的递推求值结果。  
5. `main` 计算两条路径最大误差，并验证基函数分片和误差、边界曲线退化、角点插值性质。  
6. `derivative_control_nets` 构造 `u/v` 偏导控制网格，`surface_partials` 求 `S_u,S_v` 并通过叉积得到法向量。  
7. 在固定点用中心差分近似偏导，与解析偏导对比，验证导数公式实现正确性。  
8. 最后执行包围盒与面积正值检查，全部断言通过后输出 `All checks passed.`。  

`numpy` 仅承担数组与线性代数运算；Bezier 曲面的核心求值、递推、偏导与验证流程都在源码中显式实现，没有把第三方库当黑盒。

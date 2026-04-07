# 克尔度规 (Kerr Metric)

- UID: `PHYS-0375`
- 学科: `物理`
- 分类: `广义相对论`
- 源序号: `394`
- 目标目录: `Algorithms/物理-广义相对论-0394-克尔度规_(Kerr_Metric)`

## R01

问题定义：实现一个可复现、可验证的 Kerr 度规最小 MVP。

本条目不做完整天体吸积盘模拟，而是聚焦“度规本体 + 基本几何量 + 数值一致性验证”三件事：
- 计算 Boyer-Lindquist 坐标下的 `g_{mu nu}` 与 `g^{mu nu}`；
- 计算事件视界、静止极限面（ergosphere 边界）和 ISCO 半径；
- 输出可机器检查的阈值结果，保证不仅可运行，而且物理关系自洽。

## R02

Kerr 度规描述无电荷旋转黑洞，是 Schwarzschild 解的旋转推广。其核心现象包括：
- 外/内事件视界 `r_+`、`r_-`；
- 静止极限面（ergosphere）导致拖曳效应（frame dragging）；
- 顺行/逆行轨道不对称（例如 ISCO 半径明显不同）。

本 MVP 使用几何单位 `G=c=M=1`，并提供 SI 量纲换算（`GM/c^2`）用于读数解释。

## R03

本实现的计算任务：
1. 对给定自旋 `a` 与采样点 `(r, theta)` 计算协变度规 `g_{mu nu}`；
2. 显式写出逆度规 `g^{mu nu}` 并检验 `g @ g_inv ≈ I`；
3. 验证行列式公式 `det(g) = -Sigma^2 sin^2(theta)`；
4. 用解析公式计算 `r_+`、`r_-`、静止极限半径、ISCO 顺/逆行半径；
5. 用 `brentq` 对 `g_tt=0` 做数值求根，并和静止极限解析式比对；
6. 检验 `a=0` 时退化为 Schwarzschild 度规。

## R04

建模假设（MVP 级别）：
- 时空为真空 Kerr（无电荷，Boyer-Lindquist 坐标）；
- 仅讨论 `|a| <= 1` 的物理解（避免裸奇点）；
- 数值检查点全部取在外视界外（`r > r_+`）；
- 不积分完整测地线，不含辐射传输、MHD、吸积盘微观物理。

这些假设适用于“几何结构与算法正确性验证”，不直接等价于观测拟合全链路。

## R05

核心公式（`demo.py` 已显式实现）：

记
- `Sigma = r^2 + a^2 cos^2(theta)`
- `Delta = r^2 - 2r + a^2`
- `sin2 = sin^2(theta)`

协变度规关键分量：
- `g_tt = -(1 - 2r/Sigma)`
- `g_rr = Sigma/Delta`
- `g_thetatheta = Sigma`
- `g_tphi = -2ar sin2 / Sigma`
- `g_phiphi = (r^2 + a^2 + 2a^2 r sin2 / Sigma) sin2`

逆度规关键分量：
- `g^tt = -A/(Sigma*Delta)`，其中 `A = (r^2+a^2)^2 - a^2 Delta sin2`
- `g^tphi = -2ar/(Sigma*Delta)`
- `g^phiphi = (Delta - a^2 sin2)/(Sigma*Delta*sin2)`
- `g^rr = Delta/Sigma`，`g^thetatheta = 1/Sigma`

几何边界：
- 视界：`r_± = 1 ± sqrt(1-a^2)`
- 静止极限面：`r_static(theta) = 1 + sqrt(1 - a^2 cos^2(theta))`

## R06

数值算法流程：
1. 读取配置（质量、`a` 网格、`r/theta` 网格、阈值）；
2. 计算质量几何长度 `GM/c^2`（用于 SI 标尺）；
3. 逐 `a` 计算 `r_±`、`r_static`、ISCO 与 `Omega_ZAMO`；
4. 在采样网格上计算 `g_{mu nu}` 与 `g^{mu nu}`；
5. 统计 `max|g g^{-1} - I|`；
6. 统计 `det(g)` 与解析行列式相对误差；
7. 用 `brentq` 求解 `g_tt(r,theta)=0` 的根；
8. 将数值根与解析 `r_static` 做误差比对；
9. 汇总阈值检查并输出 PASS/FAIL。

## R07

复杂度分析：
- 设自旋个数 `N_a`，采样半径 `N_r`，采样角度 `N_t`。
- 度规一致性检查复杂度约 `O(N_a * N_r * N_t)`；
- 静止极限求根复杂度约 `O(N_a * N_theta * N_iter)`，`N_iter` 为 Brent 迭代步数；
- 空间复杂度约 `O(N_a * N_r * N_t)`（结果表存储）。

当前默认规模很小，运行在秒级以内。

## R08

数值稳定性策略：
- 所有检查点强制 `r > r_+`，避免 `Delta <= 0` 造成奇异；
- `g^{phiphi}` 显式形式避开极点 `sin(theta)=0`（检查网格不取极点）；
- `g_tt=0` 求根使用 `brentq`（有括号区间的稳健一维根求解）；
- 同时使用三类诊断：
  - 逆矩阵恒等关系误差；
  - 行列式解析恒等式误差；
  - 静止极限解析/数值根误差。

## R09

适用场景：
- 广义相对论课程中 Kerr 几何结构演示；
- 新实现的度规张量代码做 sanity check；
- 为后续 Kerr 测地线积分提供基础组件。

不适用场景：
- 吸积盘辐射转移和观测光变拟合；
- Kerr-Newman（带电）或修正引力理论；
- 近极端自旋下高精度波形建模与数据同化。

## R10

正确性检查点（脚本已实现阈值）：
1. `r_plus >= r_minus >= 0`；
2. `r_isco_pro <= r_isco_retro`（顺行轨道更靠内）；
3. `max|g g^{-1} - I| < 1e-10`；
4. `max det(g) 相对误差 < 1e-10`；
5. `max |r_static_numeric - r_static_analytic| < 1e-11`；
6. `a=0` 时 Kerr->Schwarzschild 误差 `< 1e-12`。

全部通过才输出 `Validation: PASS`。

## R11

默认参数（`KerrConfig`）：
- `mass_solar = 10.0`
- `spins = (0.0, 0.5, 0.9, 0.99)`
- `radii_for_checks = (3.0, 6.0, 10.0)`
- `theta_for_checks = (pi/6, pi/2)`
- `static_limit_thetas = (pi/4, pi/2)`
- 容差：
  - `tolerance_identity = 1e-10`
  - `tolerance_det_rel = 1e-10`
  - `tolerance_static_root = 1e-11`
  - `tolerance_schwarzschild = 1e-12`

参数意图：覆盖从无旋到高自旋、从近场到中远场的典型检查点。

## R12

一次实测输出（`uv run python demo.py`，2026-04-07）：
- 质量标尺：`GM/c^2 = 14766.696910 m`（`10 M_sun`）
- 自旋摘要：
  - `a=0.0`: `r_plus=2.0`, `r_isco_pro=6.0`
  - `a=0.99`: `r_plus=1.141067`, `r_isco_pro=1.454498`, `r_isco_retro=8.971861`
- 误差量级：
  - 最大 `identity_max_abs_error = 4.441e-16`
  - 最大 `det_relative_error = 1.228e-15`
  - 最大静止极限求根误差 `2.121e-12`
  - Schwarzschild 极限误差 `0.0`
- 最终结果：`Validation: PASS`

## R13

理论与验证边界：
- 这是“度规与几何恒等式的数值验证”，不是完整轨道动力学证明；
- ISCO 使用标准解析公式（Bardeen-Press-Teukolsky），未做数值变分推导；
- 若面向观测数据，仍需引入辐射模型、噪声模型与参数反演流程。

## R14

常见失败模式与修复：
- 失败：`Point must satisfy r > r_+`
  - 修复：提高采样半径，确保在外视界外。
- 失败：`theta must avoid poles`
  - 修复：避免在 `theta=0, pi` 直接使用显式 `g^{phiphi}` 检查。
- 失败：`Failed to bracket static-limit root`
  - 修复：扩大上界或先检查 `|a|<=1` 与 `theta` 取值。
- 失败：阈值越界但未报异常
  - 修复：检查数值单位、公式符号、矩阵索引顺序是否一致。

## R15

工程实践建议：
- 先做几何恒等式（逆矩阵/行列式）校验，再做高层物理量；
- 坐标顺序 `(t,r,theta,phi)` 固定写入注释，避免索引误用；
- 保留解析表达与数值根的“双路径验证”，便于定位误差来源。

## R16

扩展方向：
1. 在本度规模块上加入测地线积分（timelike/null）；
2. 计算轨道频率、进动频率与 Penrose 过程指标；
3. 接入 ray-tracing，可视化黑洞阴影与拖曳效应；
4. 扩展到 Kerr-Newman 或数值相对论背景度规。

## R17

本目录交付内容：
- `demo.py`：`numpy + scipy + pandas` 的可运行 MVP；
- `README.md`：R01-R18 完整说明、公式、阈值和实测结果；
- `meta.json`：与任务元数据一致。

运行方式：

```bash
cd Algorithms/物理-广义相对论-0394-克尔度规_(Kerr_Metric)
uv run python demo.py
```

脚本无交互输入，直接打印结果表与阈值检查。

## R18

`demo.py` 源码级算法流拆解（9 步，非黑盒）：
1. `horizon_radii` 对每个 `a` 计算 `r_+`/`r_-`，并先约束 `|a|<=1`。  
2. `kerr_metric_covariant` 显式构造 `g_tt,g_rr,g_thetatheta,g_tphi,g_phiphi`，不调用外部 GR 黑盒库。  
3. `kerr_metric_contravariant` 用解析逆度规公式逐项实现 `g^{mu nu}`。  
4. `metric_identity_error` 计算 `g @ g_inv - I` 的最大绝对误差，检查逆矩阵一致性。  
5. `determinant_relative_error` 把数值 `det(g)` 与解析 `-Sigma^2 sin^2(theta)` 对比。  
6. `static_limit_radius_numeric` 把 `g_tt(r,theta)=0` 作为标量方程，用 `brentq` 在括号区间内求根。  
7. `static_limit_radius_analytic` 给出静止极限解析半径，并与数值根做逐项误差比对。  
8. `isco_radius` 用 Bardeen-Press-Teukolsky 公式计算顺/逆行 ISCO，体现旋转导致的不对称。  
9. `validate` 汇总全部阈值（视界顺序、ISCO 顺序、逆矩阵误差、行列式误差、静止极限误差、Schwarzschild 退化）并给出 PASS/FAIL。

第三方库说明：`numpy` 负责数组与线性代数，`pandas` 负责表格输出，`scipy.optimize.brentq` 仅用于一维根求解；核心 Kerr 公式与验证逻辑都在源码中逐步展开。

# 克尔-纽曼度规 (Kerr-Newman Metric)

- UID: `PHYS-0376`
- 学科: `物理`
- 分类: `广义相对论`
- 源序号: `395`
- 目标目录: `Algorithms/物理-广义相对论-0395-克尔-纽曼度规_(Kerr-Newman_Metric)`

## R01

问题定义：实现一个可复现、可验证的 Kerr-Newman 度规最小 MVP。

本条目聚焦三件事：
- 在 Boyer-Lindquist 坐标中计算带电旋转黑洞的 `g_{mu nu}` 与 `g^{mu nu}`；
- 计算视界半径、静止极限面（ergosphere 边界）和电磁四势；
- 用机器可检查阈值验证几何恒等式与极限退化（Kerr / Reissner-Nordstrom / Schwarzschild）。

## R02

Kerr-Newman 解是爱因斯坦-麦克斯韦方程在真空外部区域的轴对称定常精确解，描述“带电 + 旋转”黑洞。

它统一包含以下特例：
- `q=0` 时退化为 Kerr（旋转无电荷）；
- `a=0` 时退化为 Reissner-Nordstrom（带电无旋转）；
- `a=0,q=0` 时退化为 Schwarzschild（无电荷无旋转）。

本实现使用几何单位 `G=c=M=1`，并额外输出 `GM/c^2` 的 SI 尺度用于解释物理长度。

## R03

本实现的计算任务：
1. 对给定 `(a, q, r, theta)` 计算 Kerr-Newman 协变度规 `g_{mu nu}`；
2. 显式写出逆度规 `g^{mu nu}` 并检验 `g @ g_inv ≈ I`；
3. 验证 `det(g) = -Sigma^2 sin^2(theta)`；
4. 解析计算 `r_+`、`r_-` 与静止极限 `r_static(theta)`；
5. 用 `brentq` 数值求解 `g_tt=0`，并和解析静止极限半径比对；
6. 计算电磁四势 `A_mu`（固定规范）；
7. 验证三个极限：Kerr / RN / Schwarzschild。

## R04

建模假设（MVP 级别）：
- 时空为标准 Kerr-Newman（Boyer-Lindquist 坐标）；
- 参数满足 `a^2 + q^2 <= 1`（存在事件视界，避免裸奇点）；
- 采样点仅取外视界之外（`r > r_+`）；
- 不积分测地线，不做辐射转移、吸积盘微观模型或观测反演。

这些假设适用于“度规实现正确性验证”，不等价于完整天体物理拟合流程。

## R05

核心公式（`demo.py` 中全部显式实现）：

定义：
- `Sigma = r^2 + a^2 cos^2(theta)`
- `Delta = r^2 - 2r + a^2 + q^2`
- `sin2 = sin^2(theta)`
- `T = 2r - q^2`

协变度规关键分量：
- `g_tt = -(1 - T/Sigma)`
- `g_rr = Sigma/Delta`
- `g_thetatheta = Sigma`
- `g_tphi = -a T sin2 / Sigma`
- `g_phiphi = (r^2 + a^2 + a^2 T sin2 / Sigma) sin2`

逆度规关键分量：
- `A = (r^2 + a^2)^2 - a^2 Delta sin2`
- `g^tt = -A/(Sigma*Delta)`
- `g^tphi = -a T/(Sigma*Delta)`
- `g^phiphi = (Delta - a^2 sin2)/(Sigma*Delta*sin2)`
- `g^rr = Delta/Sigma`，`g^thetatheta = 1/Sigma`

边界与电磁势：
- 视界：`r_± = 1 ± sqrt(1 - a^2 - q^2)`
- 静止极限：`r_static(theta) = 1 + sqrt(1 - q^2 - a^2 cos^2(theta))`
- 电磁四势（本实现规范）：`A_mu dx^mu = -(q r/Sigma)(dt - a sin2 dphi)`

## R06

数值算法流程：
1. 读取配置（质量、参数组 `(a,q)`、网格与容差）；
2. 计算质量标尺 `GM/c^2`（米）；
3. 对每组 `(a,q)` 计算视界、赤道静止极限、拖曳角速度与电磁势；
4. 在采样网格计算 `g_{mu nu}` 与 `g^{mu nu}`；
5. 统计 `max|g g^{-1} - I|`；
6. 统计行列式解析恒等式相对误差；
7. 用 `brentq` 解 `g_tt(r,theta)=0` 并与解析静止极限比较；
8. 计算 Kerr/RN/Schwarzschild 三种极限误差；
9. 汇总阈值并输出 PASS/FAIL。

## R07

复杂度分析：
- 设参数组个数 `N_p`，半径采样数 `N_r`，角度采样数 `N_t`。
- 度规一致性检查复杂度约 `O(N_p * N_r * N_t)`；
- 静止极限求根复杂度约 `O(N_p * N_theta * N_iter)`，`N_iter` 为 Brent 迭代步数；
- 空间复杂度约 `O(N_p * N_r * N_t)`（结果表存储）。

默认规模很小，通常亚秒级运行完成。

## R08

数值稳定性策略：
- 参数先验检查 `a^2+q^2<=1`，避免无视界参数导致公式失效；
- 几何检查点要求 `r > r_+`，避免 `Delta<=0`；
- 避开 `theta=0,pi` 极点，防止显式 `g^{phiphi}` 分母退化；
- `g_tt=0` 用带括号区间的一维稳健根求解（`brentq`）；
- 同时使用三类诊断：逆矩阵误差、行列式误差、静止极限解析/数值误差。

## R09

适用场景：
- 广义相对论课程中“带电旋转黑洞”结构演示；
- 自研张量代码的单元测试/回归测试基线；
- 后续测地线积分或 ray tracing 的基础模块。

不适用场景：
- 含等离子体、辐射输运、MHD 的高保真吸积盘建模；
- 超极端参数（`a^2+q^2>1`）的裸奇点研究；
- 直接面向观测数据的端到端统计反演。

## R10

正确性检查点（脚本已实现阈值）：
1. `r_plus >= r_minus >= 0`；
2. `a^2 + q^2 <= 1`；
3. `r_static_eq >= r_plus`；
4. `max|g g^{-1} - I| < 1e-10`；
5. `max det(g) 相对误差 < 1e-10`；
6. `max |r_static_numeric - r_static_analytic| < 1e-11`；
7. `q=0` 的 Kerr 极限误差 `< 1e-12`；
8. `a=0` 的 RN 极限误差 `< 1e-12`；
9. `a=q=0` 的 Schwarzschild 极限误差 `< 1e-12`。

全部通过才输出 `Validation: PASS`。

## R11

默认参数（`KerrNewmanConfig`）：
- `mass_solar = 10.0`
- 参数组：`(a,q) = (0.0,0.0), (0.6,0.2), (0.8,0.3), (0.3,0.7)`
- `radii_for_checks = (3.0, 6.0, 10.0)`
- `theta_for_checks = (pi/6, pi/2)`
- `static_limit_thetas = (pi/4, pi/2)`
- 容差：
  - `tolerance_identity = 1e-10`
  - `tolerance_det_rel = 1e-10`
  - `tolerance_static_root = 1e-11`
  - `tolerance_kerr_limit = 1e-12`
  - `tolerance_rn_limit = 1e-12`
  - `tolerance_schwarzschild = 1e-12`

参数设计意图：覆盖无旋无电荷、较强旋转、较强电荷以及混合参数的代表点。

## R12

一次实测输出（`uv run python demo.py`，2026-04-07）：
- 质量标尺：`GM/c^2 = 14766.696910 m`（`10 M_sun`）
- 参数摘要示例：
  - `(a,q)=(0.8,0.3)`: `r_plus=1.519615`, `r_static_eq=1.953939`, `ergosphere_width_eq=0.434324`
  - `(a,q)=(0.3,0.7)`: `r_plus=1.648074`, `r_static_eq=1.714143`
- 误差量级：
  - 最大 `identity_max_abs_error = 4.441e-16`
  - 最大 `det_relative_error = 1.228e-15`
  - 最大静止极限求根误差 `5.009e-13`
- 极限检查：
  - `max |g_KN(q=0)-g_Kerr| = 0.0`
  - `max |g_KN(a=0)-g_RN| = 0.0`
  - `max |g_KN(a=q=0)-g_Schw| = 0.0`
- 最终结果：`Validation: PASS`

## R13

理论与验证边界：
- 本条目验证的是“度规表达式与几何恒等式的一致性”，不是完整黑洞动力学求解；
- 电磁势只给出一个常用规范，不涉及规范变换下的全部比较；
- 若进入观测层，需要加入发射模型、传播模型与噪声统计模型。

## R14

常见失败模式与修复：
- 失败：`Kerr-Newman parameters must satisfy a^2 + q^2 <= 1`
  - 修复：调整 `(a,q)` 到物理有视界区域。
- 失败：`Point must satisfy r > r_+`
  - 修复：把采样半径提高到外视界之外。
- 失败：`theta must avoid poles`
  - 修复：避免在 `theta=0` 或 `pi` 上直接检查显式 `g^{phiphi}`。
- 失败：`Failed to bracket static-limit root`
  - 修复：扩大求根上界，或先检查参数和角度是否导致无实根。

## R15

工程实践建议：
- 坐标顺序 `(t,r,theta,phi)` 固化到代码与注释，避免索引错位；
- 先做代数恒等式验证（逆矩阵/行列式），再做高层物理量验证；
- 保留解析公式与数值求根双路径，便于定位误差来源；
- 极限退化测试（Kerr/RN/Schwarzschild）应作为长期回归测试用例。

## R16

扩展方向：
1. 在该度规模块上增加 timelike/null 测地线积分；
2. 引入 Carter 常数相关轨道分类与频率分析；
3. 接入光线追踪，展示带电旋转背景下阴影/像差变化；
4. 加入更细参数扫描与误差热力图输出。

## R17

本目录交付内容：
- `demo.py`：`numpy + scipy + pandas` 的可运行 Kerr-Newman MVP；
- `README.md`：R01-R18 完整说明、公式、阈值与实测数据；
- `meta.json`：与任务元数据一致。

运行方式：

```bash
cd Algorithms/物理-广义相对论-0395-克尔-纽曼度规_(Kerr-Newman_Metric)
uv run python demo.py
```

脚本无交互输入，直接打印参数摘要、误差表与验证结果。

## R18

`demo.py` 源码级算法流拆解（9 步，非黑盒）：
1. `horizon_radii`：先检查 `a^2+q^2<=1`，再计算 `r_+`/`r_-`。  
2. `kerr_newman_metric_covariant`：逐项构造 `g_tt,g_rr,g_thetatheta,g_tphi,g_phiphi`。  
3. `kerr_newman_metric_contravariant`：用解析逆度规公式逐项实现 `g^{mu nu}`。  
4. `metric_identity_error`：计算 `g @ g_inv - I` 的最大绝对误差。  
5. `determinant_relative_error`：把数值 `det(g)` 与解析 `-Sigma^2 sin^2(theta)` 对比。  
6. `static_limit_radius_numeric`：把 `g_tt(r,theta)=0` 转成标量方程，使用 `brentq` 求根。  
7. `build_parameter_summary`：按参数组汇总视界、静止极限、拖曳角速度和电磁势。  
8. `kerr_limit_error`/`rn_limit_error`/`schwarzschild_limit_error`：执行三种解析退化一致性检查。  
9. `validate`：统一执行阈值判定并在 `main` 输出 `Validation: PASS/FAIL`。

第三方库说明：
- `numpy`：数组运算和线性代数；
- `pandas`：结构化结果表输出；
- `scipy.optimize.brentq`：一维有括号根求解；
- 核心 Kerr-Newman 公式与验证逻辑均在源码中显式展开，没有调用 GR 黑盒求解器。

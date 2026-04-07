# 爱因斯坦场方程 (Einstein Field Equations)

- UID: `PHYS-0045`
- 学科: `物理`
- 分类: `广义相对论`
- 源序号: `45`
- 目标目录: `Algorithms/物理-广义相对论-0045-爱因斯坦场方程_(Einstein_Field_Equations)`

## R01

本条目实现一个“可运行且可验证”的爱因斯坦场方程最小 MVP。  
核心思路：不直接求一般 4 维时空的度规，而是在空间平直（`k=0`）的 FRW 宇宙学背景下，验证由 EFE 推出的三条关键动力学方程是否数值一致。

## R02

采用的物理方程（含宇宙学常数）为：

- 爱因斯坦场方程：`G_{mu nu} + Lambda g_{mu nu} = (8*pi*G/c^4) T_{mu nu}`
- 第一条 Friedmann 方程：`H^2 = (8*pi*G/3) rho + (Lambda*c^2/3)`
- `dot(H)` 方程：`dot(H) = -4*pi*G * (rho + p/c^2)`
- 加速度方程：`ddot(a)/a = -(4*pi*G/3)*(rho + 3p/c^2) + (Lambda*c^2/3)`

这里 `rho, p` 仅由物质与辐射贡献，暗能量通过 `Lambda` 放在几何侧。

## R03

MVP 建模选择：

- 时空：空间平直 FRW（`Omega_m + Omega_r + Omega_lambda ~= 1`）；
- 成分：
  - 物质（`w=0`）：`rho_m(a) ~ a^-3`
  - 辐射（`w=1/3`）：`rho_r(a) ~ a^-4`
- 宇宙学常数：`Lambda = 3*Omega_lambda*H0^2/c^2`；
- 单位：全 SI 制（`G, c, rho, H, Lambda` 都是 SI 量）。

## R04

`demo.py` 做的事不是“黑箱调用 GR 库”，而是显式地：

1. 从 `H0, Omega_*` 构造 `rho_crit0` 和 `Lambda`；
2. 在尺度因子网格 `a in [a_min, a_max]` 上计算 `rho_m(a), rho_r(a), p(a)`；
3. 用 EFE 形式计算 `H(a)`；
4. 用标准 `E(a)^2` 参数化再算一遍 `H(a)` 交叉校验；
5. 用链式法则 `dot(H)=aH dH/da` 与理论 `dot(H)` 比较；
6. 用 `dot(H)+H^2` 与加速度方程右侧比较；
7. 输出残差与抽样表格，并通过断言判定 PASS/FAIL。

## R05

复杂度（设网格点数为 `N`）：

- 密度、压强、哈勃率、残差计算均为向量化 `O(N)`；
- 表格构建 `O(N)`；
- 总时间复杂度 `O(N)`，空间复杂度 `O(N)`。

本 MVP 默认 `N=320`，几乎瞬时运行。

## R06

输入与输出：

- 输入：脚本内置参数（无交互）
  - `h0_km_s_mpc`
  - `omega_m, omega_r, omega_lambda`
  - `a_min, a_max, n_points`
- 输出：
  - 三类相对残差（Eq1 / dotH / acceleration）；
  - `H` 两种计算路径的一致性指标；
  - 尺度因子抽样表（`rho_m, rho_r, H, dotH, q` 与残差列）。

## R07

正确性设计：

- 不是只验证一条式子，而是同时验证三条互相关联方程；
- 使用两条独立 `H(a)` 计算路径（EFE 形式 vs `Omega` 形式）；
- 所有验证均做成程序断言，确保 CI/批处理环境下自动判定。

## R08

数值策略与稳定性：

- 全部使用 `float64`；
- 尺度因子使用 `geomspace`，兼顾早期宇宙与晚期宇宙数量级；
- 用 `EPS` 防止极端情况下分母为零；
- 对平直性、参数非负、网格规模做显式输入检查。

## R09

适用范围：

- 广义相对论与宇宙学教学中的“EFE -> Friedmann 方程”数值核验；
- 作为更复杂宇宙学代码的基线测试。

不适用范围：

- 非平直或各向异性宇宙（Bianchi 等）；
- 非 FRW 的局域强场时空（黑洞并合、数值相对论全求解）；
- 含复杂暗能量模型 `w(a)` 或耦合暗部门模型。

## R10

技术栈：

- Python 3
- `numpy`：向量化数值计算
- `pandas`：结果表格化输出
- `scipy.constants`：`G, c, parsec` 物理常数

重点：核心物理流程由源码显式实现，不依赖“广义相对论黑箱求解器”。

## R11

运行方式：

```bash
cd Algorithms/物理-广义相对论-0045-爱因斯坦场方程_(Einstein_Field_Equations)
uv run python demo.py
```

脚本不需要命令行参数，不需要交互输入。

## R12

默认参数含义：

- `H0 = 67.66 km/s/Mpc`
- `Omega_m = 0.3111`
- `Omega_r = 9.2e-5`
- `Omega_lambda = 1 - Omega_m - Omega_r`
- `a` 取值范围：`[0.02, 1.5]`

这组参数用于展示：早期（小 `a`）辐射/物质主导与后期（大 `a`）Lambda 主导下，方程仍数值一致。

## R13

理论保证类型：

- 近似比保证：N/A（非优化问题）；
- 随机成功率：N/A（确定性计算）。

可验证保证：

- 第一条 Friedmann 方程残差足够小；
- `dot(H)` 关系残差足够小；
- 加速度方程残差足够小；
- 两种 `H(a)` 计算路径一致。

## R14

常见失败模式：

1. `Omega_m + Omega_r + Omega_lambda` 明显不等于 1（但仍强行当平直宇宙）；
2. 单位错配（`H0` 单位转换错误会导致全部量纲崩坏）；
3. 给出负密度参数，导致非物理结果；
4. 把暗能量既放进 `rho` 又放进 `Lambda`（双重计入）。

## R15

可扩展方向：

- 扩展到 `Omega_k != 0` 的非平直 FRW；
- 引入暗能量状态方程 `w(a)` 与数值积分求 `H(a)`；
- 增加宇宙年龄积分、共动距离积分等可观测量模块；
- 接入不确定度传播，研究参数误差对残差与可观测量的影响。

## R16

相关主题：

- Robertson-Walker 度规
- 宇宙连续性方程 `dot(rho)+3H(rho+p/c^2)=0`
- Lambda-CDM 参数化
- 弱场极限下 EFE 与牛顿引力的对应关系

## R17

交付内容与完成状态：

- `README.md`：R01-R18 已完整填写；
- `demo.py`：已实现可运行 MVP，无占位符；
- `meta.json`：保持 `PHYS-0045 / 物理 / 广义相对论 / source 45` 元数据一致。

脚本可在非交互模式直接运行并打印诊断。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 创建 `FRWConfig`，调用 `run_einstein_friedmann_mvp`。  
2. `validate_config` 检查 `H0`、`a` 范围、网格大小和“平直性约束”。  
3. `build_background` 先把 `H0` 转成 SI，再计算 `rho_crit0` 与 `Lambda`。  
4. `matter_radiation_fields` 在 `a` 网格上计算 `rho_m(a), rho_r(a), rho_total(a), p(a)`。  
5. `hubble_from_einstein` 按第一条 Friedmann 方程算 `H(a)`；`hubble_from_omegas` 用 `E(a)^2` 再算一遍 `H(a)` 交叉验证。  
6. `dot_h_from_a_derivative` 用 `dot(H)=aH dH/da` 得到左侧，并与理论右侧 `-4*pi*G*(rho+p/c^2)` 比较。  
7. 计算 `accel_lhs = dot(H)+H^2` 与加速度方程右侧，形成三组残差（Eq1/dotH/accel）和 `q` 参数表。  
8. `main` 对残差执行断言阈值检查并输出抽样 `pandas` 表格，形成“可运行 + 可验证”的最小闭环。  

第三方库角色说明：`numpy/pandas/scipy.constants` 仅用于数值与展示；爱因斯坦方程到 Friedmann 关系的算法链路全部在源码中逐步展开。

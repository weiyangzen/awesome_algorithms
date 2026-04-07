# 福克-普朗克方程 (Fokker-Planck Equation)

- UID: `PHYS-0307`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `310`
- 目标目录: `Algorithms/物理-统计力学-0310-福克-普朗克方程_(Fokker-Planck_Equation)`

## R01

福克-普朗克方程描述**随机过程的概率密度如何随时间演化**。对一维过程，若漂移项为 `A(x)`、扩散系数为 `D`，常见形式是：

`∂_t p(x,t) = -∂_x(A(x)p(x,t)) + D ∂_{xx}p(x,t)`。

它把“单粒子随机轨道的统计行为”提升为“群体概率密度场”的演化方程，是统计力学、随机过程、非平衡体系中的核心工具。

## R02

本条目采用最经典可验证场景：Ornstein-Uhlenbeck (OU) 过程。

- SDE 形式：`dX_t = -k X_t dt + sqrt(2D) dW_t`
- 对应 Fokker-Planck：`∂_t p = -∂_x((-k x)p) + D∂_{xx}p = ∂_x(kxp) + D∂_{xx}p`

该模型的解析解已知（高斯保持性），因此可以做严格数值校验：
- 均值：`m(t)=m0 exp(-kt)`
- 方差：`v(t)=D/k + (v0-D/k)exp(-2kt)`

## R03

MVP 目标不是做通用 PDE 框架，而是打通最小闭环：

1. 在有限区间上离散求解 1D Fokker-Planck；
2. 保证质量守恒（`∫p dx ≈ 1`）；
3. 在多个快照时刻和解析 OU 解比较（密度误差 + 矩误差）；
4. 用断言给出自动通过/失败结果，便于批量验证。

## R04

离散方法采用**保守型有限体积显式格式**：

- 写成守恒律：`∂_t p = -∂_x F`，`F = A p - D ∂_x p`；
- 单元更新：`p_i^{n+1} = p_i^n - (dt/dx)(F_{i+1/2}-F_{i-1/2})`；
- 对流通量 `A p` 用迎风（upwind）稳定化；
- 扩散项用中心差分梯度；
- 两端采用无通量边界：`F(x_min)=F(x_max)=0`。

这个离散选择使“守恒结构”在代码中可见，而不是黑盒 PDE 求解器。

## R05

`demo.py` 默认参数（`FPConfig`）：

- 空间区间：`[-6, 6]`
- 网格点：`nx=501`
- 漂移参数：`k=1.0`
- 扩散系数：`D=0.55`
- 初始高斯：`mean0=1.5, var0=0.18`
- 终止时间：`t_final=1.2`
- 时间步：`dt=2e-4`
- 快照时刻：`(0.0, 0.3, 0.7, 1.2)`

并内置容差用于自动校验：质量误差、均值误差、方差误差、最终 `L1` 密度误差、最小密度下界。

## R06

代码结构（`demo.py`）：

- `FPConfig`：集中管理物理参数、网格、容差；
- `analytic_ou_moments` / `analytic_ou_pdf`：给出 OU 解析参考；
- `compute_flux`：构造离散通量（迎风对流 + 中心扩散）；
- `step_fokker_planck`：做一次显式推进；
- `solve_fokker_planck`：主循环，抓取快照并汇总诊断；
- `evaluate_snapshot`：计算质量、矩误差、`L1/L2/L∞` 密度误差；
- `run_consistency_checks`：自动断言；
- `main`：打印稳定性信息和结果表。

## R07

伪代码：

```text
x, dx <- uniform grid
p <- normalized Gaussian(mean0, var0)

for n in 1..N:
  F <- A(x)*p - D*grad(p)
  impose no-flux boundaries
  p <- p - dt * div(F)
  if crossed snapshot time:
    store p(x,t)

for each snapshot t:
  p_ref <- analytical OU Gaussian at t
  compute mass, mean, variance, L1/L2/Linf errors

assert conservation + moment errors + final L1 within tolerances
print report table
```

## R08

复杂度（`nx` 网格点，`nt` 时间步，`ns` 快照数）：

- 单步更新：`O(nx)`
- 全时域推进：`O(nx * nt)`
- 快照评估：`O(ns * nx)`
- 总体：`O(nx*nt + ns*nx)`，主导项通常是 `O(nx*nt)`
- 空间复杂度：`O(nx + ns*nx)`（当前实现保存快照密度）

在默认配置下规模较小，CPU 几百毫秒可完成。

## R09

数值稳定与可靠性策略：

- `explicit_stability_bounds` 先计算保守 CFL 约束：
  - 对流近似约束：`dt <= dx/max|A|`
  - 扩散约束：`dt <= dx^2/(2D)`
- 若配置 `dt` 接近上限会直接报错，避免隐蔽发散；
- 无通量边界 + 守恒通量差分保证质量漂移可控；
- 通过 `min_density` 监测是否出现明显负密度。

## R10

为何不用“更高级库一把梭”：

- `scipy.integrate.solve_ivp` 可把半离散系统当 ODE 解，但会隐藏通量结构；
- 专用 PDE 包可更快搭建，但不利于展示 Fokker-Planck 的离散物理意义；
- 当前实现优先“可读、可核查、可验证”，更适合作为算法条目的最小原型。

## R11

调参建议：

- 想提高精度：减小 `dt`、增大 `nx`；
- 想更快运行：适度减小 `nx` 或缩短 `t_final`；
- 若断言失败：先检查 `dt` 是否过大，再扩大空间区间以降低边界截断影响；
- 若高斯尾部被边界截断：增大 `|x_min|, |x_max|`。

## R12

关键实现细节：

- `compute_flux` 中对流通量使用 `np.where(a_face>=0, p_left, p_right)`，显式体现迎风方向；
- 扩散通量 `-D*(p_right-p_left)/dx` 与对流通量统一到面通量框架；
- `step_fokker_planck` 只做“通量散度更新”，逻辑单一、便于审计；
- 误差评估不用黑盒统计函数，直接积分得到质量、均值、方差与范数误差。

## R13

运行方式：

```bash
cd "Algorithms/物理-统计力学-0310-福克-普朗克方程_(Fokker-Planck_Equation)"
uv run python demo.py
```

或在仓库根目录：

```bash
uv run python "Algorithms/物理-统计力学-0310-福克-普朗克方程_(Fokker-Planck_Equation)/demo.py"
```

脚本无交互输入。

## R14

输出表字段含义：

- `mass`：`∫p dx`，应接近 1；
- `mean_num / var_num`：数值均值与方差；
- `mean_ref / var_ref`：OU 解析均值与方差；
- `mean_abs_err / var_abs_err`：矩误差；
- `l1_err / l2_err / linf_err`：密度误差指标；
- `min_density`：最小密度，用于检测非物理负值。

典型趋势：随着时间推进，均值向 0 回缩，方差向稳态值 `D/k` 收敛。

## R15

常见问题排查：

- 报 `dt is too large`：减小 `dt` 或减少 `D` / 增大 `dx`；
- `Mass drift too large`：检查边界区间是否过窄，导致概率流被截断；
- `Final L1 density error too large`：通常需要更细网格或更小时间步；
- `Density became too negative`：显式格式过激进，优先减小 `dt`。

## R16

可扩展方向：

- 漂移改为非线性势场 `A(x)=-U'(x)`（双稳态势等）；
- 扩展到位置依赖扩散 `D(x)`；
- 与 Euler-Maruyama 粒子模拟做分布级交叉验证；
- 加入自适应时间步或半隐式扩散处理提高稳定性；
- 输出 CSV/图像用于更系统的误差收敛实验。

## R17

边界与限制：

- 当前是 1D、常系数扩散、线性漂移的 OU 特例；
- 空间上用有限区间近似无限域，边界过近会引入截断误差；
- 显式格式稳定性受限，长时或高分辨率场景成本会上升；
- 此 MVP 面向教学与算法验证，不是生产级 PDE 求解器。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main()` 创建 `FPConfig`，调用 `solve_fokker_planck()` 进入求解流程。  
2. `solve_fokker_planck()` 先 `build_grid()` 建立均匀网格，并通过 `explicit_stability_bounds()` 计算 `dt_adv` 与 `dt_diff`，在源码层显式执行稳定性门槛检查。  
3. 用 `gaussian_pdf()` 构造初始密度，再由 `normalize_density()` 通过 `scipy.integrate.trapezoid` 归一化质量。  
4. 每个时间步调用 `step_fokker_planck()`，其中 `compute_flux()` 先算面速度 `a_face`，再做迎风对流通量和中心扩散通量，最后拼接无通量边界。  
5. `step_fokker_planck()` 用通量散度 `(F_{i+1/2}-F_{i-1/2})/dx` 更新 `p^{n+1}`，保持守恒型结构。  
6. 在快照时刻，`evaluate_snapshot()` 计算数值质量/矩，并调用 `analytic_ou_pdf()`、`analytic_ou_moments()` 生成解析参考，得到 `L1/L2/L∞` 与矩误差。  
7. 全部快照汇总为 `pandas.DataFrame`；`run_consistency_checks()` 对质量漂移、矩误差、最终 `L1` 和最小密度执行断言。  
8. `main()` 打印网格、稳定性上界与诊断表，并给出最终快照摘要；断言通过则输出 `All checks passed.`。  

第三方库并未被当作黑盒：`numpy` 仅承担数组运算，`scipy.integrate.trapezoid` 只做显式积分，离散通量、更新公式、误差指标和校验逻辑均在源码中手写展开。

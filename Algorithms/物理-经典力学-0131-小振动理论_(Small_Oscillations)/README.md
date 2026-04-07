# 小振动理论 (Small Oscillations)

- UID: `PHYS-0131`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `131`
- 目标目录: `Algorithms/物理-经典力学-0131-小振动理论_(Small_Oscillations)`

## R01

小振动理论处理的是“系统在稳定平衡点附近的小位移运动”。在这一近似下，非线性系统可线性化为二阶常系数系统：

`M q_ddot + K q = 0`

其中 `M` 是质量矩阵，`K` 是势能 Hessian（刚度矩阵）。

本条目目标：
- 构建一个可运行的 2 自由度小振动 MVP；
- 用正规模（normal modes）分解得到本征频率和振型；
- 用全 ODE 数值积分做交叉验证；
- 输出正交性、能量守恒、重构误差诊断。

## R02

问题定义（本目录实现）：
- 输入：
  - 两质量参数 `m1, m2`；
  - 三刚度参数 `k1, k2, kc`（中间耦合弹簧 `kc`）；
  - 初始位移 `q0` 和初始速度 `v0`；
  - 时间区间和积分容差。
- 输出：
  - 两个本征角频率 `omega1, omega2`；
  - 振型矩阵 `Phi`；
  - 模态解析重构轨迹与 ODE 轨迹；
  - `M` 正交性误差、`K` 对角化误差、能量漂移、轨迹误差。

## R03

该模型来自平衡点附近势能二次展开：

`V(q) ~= V(0) + 1/2 * q^T K q`

动能为：

`T = 1/2 * q_dot^T M q_dot`

代入拉格朗日方程得到线性系统：

`M q_ddot + K q = 0`

对 2 自由度耦合弹簧系统，矩阵写成：

`M = [[m1, 0], [0, m2]]`

`K = [[k1+kc, -kc], [-kc, k2+kc]]`

## R04

小振动核心是广义特征值问题：

`K phi_i = omega_i^2 M phi_i`

性质：
- `M` 正定且 `K` 对称正定时，`omega_i^2 > 0`；
- 模态向量可选成 `M`-正交归一：`Phi^T M Phi = I`；
- 且有 `Phi^T K Phi = diag(omega^2)`，系统在模态坐标下解耦。

## R05

模态坐标定义：

`q = Phi eta`

代入后变成两条独立方程：

`eta_i_ddot + omega_i^2 eta_i = 0`

解析解：

`eta_i(t) = eta_i(0) cos(omega_i t) + eta_i_dot(0)/omega_i * sin(omega_i t)`

初值投影由 `M` 内积给出：

`eta(0) = Phi^T M q0`

`eta_dot(0) = Phi^T M v0`

## R06

本条目算法由两条路径并行组成：
1. 模态解析路径：
   - 解 `eigh(K, M)` 得到 `omega, Phi`；
   - 投影初值到模态坐标；
   - 用闭式解重构 `q_modal(t), v_modal(t)`。
2. 全系统数值路径：
   - 写成一阶状态 `y=[q1,q2,v1,v2]`；
   - 用 `solve_ivp(DOP853)` 积分 `q_ode(t), v_ode(t)`。

最后比较两条路径，验证小振动理论与实现的一致性。

## R07

伪代码：

```text
input: M, K, q0, v0, t_grid

[lambda, Phi] = eigh(K, M)
omega = sqrt(lambda)

eta0 = Phi^T M q0
eta_dot0 = Phi^T M v0
for each t:
  eta(t) = eta0*cos(omega*t) + (eta_dot0/omega)*sin(omega*t)
  eta_dot(t) = -eta0*omega*sin(omega*t) + eta_dot0*cos(omega*t)
  q_modal(t) = Phi * eta(t)
  v_modal(t) = Phi * eta_dot(t)

solve_ivp on y=[q,v] with q_ddot = -M^{-1}Kq -> q_ode(t), v_ode(t)

compute:
- orthogonality: Phi^T M Phi - I
- diagonalization: Phi^T K Phi - diag(omega^2)
- energy drift
- max ||q_modal-q_ode|| and ||v_modal-v_ode||
```

## R08

复杂度分析（自由度 `n`，采样点 `N`）：
- 特征分解：`O(n^3)`；
- 模态重构：`O(N n^2)`；
- ODE 积分：约 `O(N n^2)`（每步矩阵-向量乘）。

本 MVP 中 `n=2`，主开销几乎全部来自时间采样输出，整体非常轻量。

## R09

数值稳定性策略：
- 使用 `solve_ivp(method="DOP853")` + 严格容差 `rtol=1e-9, atol=1e-11`；
- 通过 `max_rel_energy_drift` 监控长期积分漂移；
- 检查 `Phi^T M Phi` 与 `Phi^T K Phi` 的最大绝对误差，确认模态分解质量；
- 在同一时间网格上直接对比模态解与 ODE 解，避免插值引入额外误差。

## R10

与相关方法对比：
- 小振动（本条目）：
  - 优点：可解析分解，物理解释清晰，计算便宜；
  - 局限：只在平衡点附近、小位移条件下成立。
- 完整非线性多体仿真：
  - 优点：适用范围更广；
  - 缺点：参数和实现复杂度高、解释性弱。

本 MVP 选择“小而诚实”的线性模型，并用数值积分做闭环验证。

## R11

默认参数（`demo.py`）：
- 质量：`m1=1.0`, `m2=1.4`
- 刚度：`k1=42.0`, `k2=55.0`, `kc=18.0`
- 初值：`q0=(0.08, -0.03)`, `v0=(0.00, 0.05)`
- 仿真：`t in [0,20]`, `num_points=2400`

调参建议：
- 保持小振动条件：`|q|` 建议不超过约 `0.1~0.2`（无量纲小位移尺度）；
- 若要看拍频效应，可让初值同时激发两个模态；
- 若要更严格校验，增大 `num_points` 并收紧容差。

## R12

实现细节说明：
- `build_mass_matrix / build_stiffness_matrix`：构建 `M, K`；
- `modal_decomposition`：用 `scipy.linalg.eigh(K, M)` 解广义特征值；
- `modal_time_solution`：按公式解析求 `eta, eta_dot` 并重构 `q, v`；
- `integrate_full_ode`：把二阶系统转一阶后调用 `solve_ivp`；
- `total_energy`：按 `0.5*v^T M v + 0.5*q^T K q` 向量化计算；
- `simulate`：组织全流程并输出 `pandas.DataFrame + summary`。

## R13

运行方式：

```bash
cd "Algorithms/物理-经典力学-0131-小振动理论_(Small_Oscillations)"
uv run python demo.py
```

或在仓库根目录：

```bash
uv run python Algorithms/物理-经典力学-0131-小振动理论_(Small_Oscillations)/demo.py
```

脚本无交互输入，直接打印 summary 与轨迹头尾。

## R14

输出字段解读：
- `omega1_rad_s`, `omega2_rad_s`：两模态角频率；
- `freq1_hz`, `freq2_hz`：对应线频率；
- `m_orthogonality_max_abs_err`：`Phi^T M Phi` 偏离单位阵的误差；
- `k_diagonalization_max_abs_err`：`Phi^T K Phi` 偏离对角阵误差；
- `max_q_modal_vs_ode_l2_err`、`max_v_modal_vs_ode_l2_err`：模态解析与 ODE 的最大轨迹差；
- `max_rel_energy_drift`：能量相对漂移（越小越好）。

理想情况下，误差量级应接近积分容差。

## R15

常见问题排查：
- `Expected strictly positive eigenvalues`：
  - 说明参数导致 `K` 非正定（非稳定平衡），检查 `k1,k2,kc`。
- 模态与 ODE 偏差较大：
  - 检查是否误改了 RHS 符号；
  - 收紧 `rtol/atol`。
- 能量漂移较大：
  - 增加采样精度或缩短积分区间；
  - 检查初值是否过大，导致线性小振动前提不再可信。

## R16

可扩展方向：
- 增加阻尼矩阵 `C`，研究阻尼模态；
- 将 2 自由度推广到 `n` 自由度链式系统；
- 由实验数据反推 `M/K`（参数辨识）；
- 加入轻度非线性项并与线性模型比较误差边界；
- 输出频谱（FFT）验证主频与本征频率一致。

## R17

适用边界与限制：
- 本模型只覆盖“平衡点附近小位移”线性区；
- 对大振幅、强非线性、碰撞、间隙等现象不适用；
- 默认无阻尼、无外激励，主要用于理论验证和教学演示；
- 结果是低维简化模型，不替代工程级高保真有限元/多体仿真。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `main()` 创建 `SmallOscillationConfig`，固定质量、刚度、初值、时间网格和积分容差。  
2. `simulate(cfg)` 调用 `build_mass_matrix` 与 `build_stiffness_matrix` 构造 `M, K`，并在 `validate_inputs` 做正定/维度/有限性检查。  
3. `modal_decomposition` 调用 `scipy.linalg.eigh(K, M)` 得到本征值 `lambda` 与振型 `Phi`，计算 `omega=sqrt(lambda)`。  
4. `modal_time_solution` 用 `eta0=Phi^T M q0`、`eta_dot0=Phi^T M v0` 将初值投影到模态空间。  
5. 同函数按解析式逐时刻计算 `eta(t), eta_dot(t)`，再由 `q=Phi eta`、`v=Phi eta_dot` 重构得到 `q_modal, v_modal`。  
6. `integrate_full_ode` 把系统改写为 `y=[q,v]` 的一阶方程，调用 `solve_ivp(DOP853)` 计算 `q_ode, v_ode`。  
7. `total_energy` 基于 `0.5*v^T M v + 0.5*q^T K q` 向量化计算全程能量，得到 `max_rel_energy_drift`。  
8. `simulate` 进一步计算 `Phi^T M Phi`、`Phi^T K Phi` 误差，以及 `max ||q_modal-q_ode||`、`max ||v_modal-v_ode||`。  
9. `main` 打印 summary 与轨迹样本，并通过断言阈值完成“模态分解-解析解-数值解-诊断”闭环验证。  

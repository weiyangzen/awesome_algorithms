# 开尔文-亥姆霍兹不稳定性 (Kelvin-Helmholtz Instability)

- UID: `PHYS-0090`
- 学科: `物理`
- 分类: `流体力学`
- 源序号: `90`
- 目标目录: `Algorithms/物理-流体力学-0090-开尔文-亥姆霍兹不稳定性_(Kelvin-Helmholtz_Instability)`

## R01

开尔文-亥姆霍兹不稳定性（KHI）描述的是：当两层流体在界面上存在速度剪切时，界面微小扰动可能被放大并发展成波纹、卷涡（billow）结构。

本条目聚焦二维双层不可压、无粘模型的线性阶段，目标是把“是否不稳定、增长率多大、哪些波数会增长”做成可计算、可断言、可复现实验。

## R02

MVP 目标是完成一个最小但闭环的验证流程，而不是完整 CFD 求解器：

- 从双层流体色散关系计算每个离散模态的增长率；
- 自动识别稳定模态与不稳定模态；
- 用多模态叠加构造 `eta(x,t)`（界面位移）时间演化；
- 用回归拟合测得增长率并和理论值对照；
- 输出表格和断言，保证脚本一键运行即可判定通过/失败。

## R03

建模假设：

- 两层流体均为不可压、无粘、无限延拓的经典界面模型；
- 扰动处于线性小振幅阶段（忽略非线性卷吸与破碎）；
- 允许重力和表面张力共同参与稳定/不稳定竞争；
- 背景速度为分层常值 `u_top`、`u_bottom`；
- 空间采用周期区间 `[0, L)`，用离散波数 `k_m = 2*pi*m/L` 采样。

这套设定适合做 KHI 的“第一性验证”，复杂物理（粘性、可压缩、磁场、三维涡动力学）不在此 MVP 范围。

## R04

线性色散关系采用两层界面经典形式：

`omega = k*U_bar +/- sqrt(D(k))`

其中：

- `U_bar = (rho_top*u_top + rho_bottom*u_bottom) / (rho_top + rho_bottom)`
- `D(k) = g*k*(rho_bottom-rho_top)/(rho_top+rho_bottom)`
  `       + sigma*k^3/(rho_top+rho_bottom)`
  `       - (rho_top*rho_bottom/(rho_top+rho_bottom)^2) * (DeltaU^2) * k^2`
- `DeltaU = u_top - u_bottom`

判据：

- 若 `D(k) >= 0`，模态稳定（仅振荡/平移）；
- 若 `D(k) < 0`，模态不稳定，增长率 `gamma(k) = sqrt(-D(k))`。

## R05

算法主流程（高层）如下：

1. 构造离散模态 `m=1..M` 与波数 `k_m`；
2. 对每个模态计算 `D(k_m)`、`gamma(k_m)`、`omega_real(k_m)`；
3. 用 `brentq` 在连续 `k` 区间定位 `D(k)=0` 根并给出不稳定窗口；
4. 生成确定性随机相位初始模态系数 `c_m(0)`；
5. 依据线性解 `c_m(t)=c_m(0)*exp((gamma_m+i*omega_m)t)` 叠加得到 `eta(x,t)`；
6. 对 `eta(x,t)` 做时序 FFT，提取主导模态振幅轨迹；
7. 用线性回归拟合 `log|A_m(t)|` 估计实验增长率；
8. 用 ODE 数值积分对照解析指数增长并执行阈值断言。

## R06

设空间点数 `nx`、时间步数 `nt`、模态数 `M`。

- 色散关系计算：`O(M)`；
- 连续波数根搜索：`O(G)`（`G` 为扫网格规模，默认常数量级）；
- 界面重建（模态叠加）：`O(nt * M * nx)`；
- FFT 与回归：`O(nt * nx log nx)` + `O(nt)`；
- 空间复杂度：`O(nt * nx + M)`。

在默认参数下（`M=12, nx=512, nt=260`）计算代价很低，适合 CI/批量校验。

## R07

正确性直觉：

- KHI 在线性阶段本质是模态增长问题，每个 `k` 模态独立演化；
- 色散关系给出“增长还是衰减/稳定”的解析判据 `D(k)`；
- 若代码重建的 `eta(x,t)` 来自同一组 `gamma(k)`，则主导模态的 `log|A|` 应近似线性；
- 回归斜率与理论 `gamma` 对齐，说明“判据 -> 演化 -> 观测”链路一致；
- 额外用 `solve_ivp` 对 `dA/dt=gamma*A` 做数值-解析对照，检查数值流程无偏差。

## R08

数值稳定策略：

- 使用小初值振幅（`eta0_scale=1e-5`）保持在线性区；
- 对 `gamma` 采用 `sqrt(max(0, -D))`，避免负值浮点抖动导致 NaN；
- 回归仅使用后段时间窗，减少早期多模态混合误差；
- ODE 校验用高阶 `DOP853` 且严格容差（`rtol=1e-11, atol=1e-13`）；
- 所有关键指标加断言，异常即失败，防止“看起来运行了但结果失真”。

## R09

`demo.py` 的最小工具栈：

- `numpy`：波数、复指数模态叠加、FFT；
- `scipy`：`brentq` 求临界波数，`solve_ivp` 做增长 ODE 对照；
- `pandas`：模态表和摘要指标表；
- `scikit-learn`：`LinearRegression` 拟合 `log|A|` 增长率，`r2_score/MAE` 评估拟合质量；
- `torch`：独立估计界面能量增长率（交叉诊断）。

## R10

演示覆盖三件事：

- 演示 A（谱判据）：打印每个离散模态的 `D(k)` 与 `gamma(k)`，区分稳定/不稳定；
- 演示 B（连续窗口）：用根搜索输出连续波数不稳定区间 `(k_left, k_right)`；
- 演示 C（时域验证）：从 `eta(x,t)` 的 FFT 反推主导模态增长率并与理论值对比。

脚本无交互输入，适合直接集成到验证流水线。

## R11

关键输出指标：

- `unstable_mode_count` / `stable_mode_count`：稳定性分区完整性；
- `dominant_mode`, `dominant_gamma`：理论主导增长模态；
- `fft_fitted_gamma`：从时域数据反演得到的增长率；
- `fft_fit_r2`, `fft_log_mae`：指数增长拟合质量；
- `ode_rel_error`：`solve_ivp` 与解析指数解误差；
- `dominant_mode_end_rel_amp_error`：主导模态末端振幅误差；
- `torch_energy_gamma`：基于能量轨迹的独立增长率估计。

## R12

默认参数（`KHIConfig`）：

- 密度：`rho_top=1.0, rho_bottom=2.0`；
- 背景速度：`u_top=1.0, u_bottom=-1.0`；
- 物理项：`gravity=9.81, surface_tension=0.02`；
- 离散规模：`max_mode=12, nx=512, nt=260, time_end=0.6`；
- 初始扰动：`eta0_scale=1e-5`，固定随机种子 `7`。

该参数组特意构造为“低模态稳定、高模态不稳定”的混合场景，便于校验判据正确性。

## R13

理论保证说明：

- 近似比保证：N/A（不是组合优化问题）；
- 概率成功率保证：N/A（流程确定性，固定随机种子后可复现）。

本条目可验证保证：

- 主导模态增长回归需满足 `R^2 > 0.995`；
- 回归增长率与理论增长率相对偏差小于 6%；
- ODE 数值解与解析解相对误差需低于 `5e-7`；
- 末端振幅相对误差需低于 8%。

## R14

常见失效模式：

1. 参数导致全部稳定或全部不稳定，无法做混合场景自检；
2. 初始振幅过大，线性模型失效却仍按指数拟合；
3. 时间窗过短，主导模态尚未从混合态中分离；
4. `nx` 太低导致频谱泄漏明显，FFT 读数偏差增大；
5. 把线性模型结果误解为非线性卷涡后期行为。

## R15

工程扩展方向：

- 加入粘性项，扩展到黏性 KHI 色散关系；
- 引入可压缩效应，比较低马赫与高马赫差异；
- 扩展到 MHD-KHI（磁张力稳定机制）；
- 用伪谱法或有限体积法推进二维涡量方程，进入非线性卷涡阶段；
- 输出 CSV/Parquet 供后续可视化与参数扫描。

## R16

相关主题：

- Rayleigh-Taylor 不稳定性（重力反转驱动）；
- Richtmyer-Meshkov 不稳定性（冲击波触发）；
- Orr-Sommerfeld 稳定性分析；
- 剪切层谱方法与模态分解；
- 天体喷流、海气界面与等离子体中的剪切不稳定过程。

## R17

MVP 功能清单：

- [x] 线性色散关系计算与稳定性判据；
- [x] 连续波数临界点求解与不稳定区间识别；
- [x] 多模态界面时域重建 `eta(x,t)`；
- [x] 基于 FFT + 线性回归的增长率反演；
- [x] `solve_ivp` 与解析指数增长对照；
- [x] `torch` 能量增长率交叉诊断；
- [x] 表格输出 + 自动断言（无交互运行）。

运行方式：

```bash
cd Algorithms/物理-流体力学-0090-开尔文-亥姆霍兹不稳定性_(Kelvin-Helmholtz_Instability)
uv run python demo.py
```

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `KHIConfig` 定义双层流体参数、离散规模和初值尺度，并提供 `rho_sum`、`delta_u` 等派生量。  
2. `dispersion_discriminant` 显式计算 `D(k)` 的浮力项、表面张力项和剪切项，给出每个波数的稳定性判据。  
3. `mode_table` 把离散模态 `m=1..M` 映射到 `k/gamma/omega`，形成可读的结构化表格。  
4. `find_unstable_intervals` 在连续波数网格上检测符号变化，并用 `scipy.optimize.brentq` 求 `D(k)=0` 根，再拼出不稳定区间。  
5. `simulate_interface` 构造初始复模态系数 `c_m(0)`，按 `exp((gamma+i*omega)t)` 时间推进并与 `exp(i*kx)` 空间基叠加得到 `eta(x,t)`。  
6. `estimate_growth_from_fft` 对 `eta(x,t)` 做时序 FFT，提取主导模态振幅，使用 `LinearRegression` 拟合 `log|A(t)|` 反演增长率。  
7. `integrate_scalar_growth_ode` 对 `dA/dt=gamma*A` 调用 `solve_ivp(DOP853)`，并和解析解比较相对误差，验证数值流程一致性。  
8. `main` 汇总 `pandas` 表格指标，结合 `torch` 能量增长率估计做交叉诊断，最终通过断言给出可自动化判定结果。  

第三方库没有被当作黑箱算法：核心 KHI 判据、模态增长建模、频谱反演和验收阈值都在源码里逐步展开；`scipy/sklearn/torch` 仅承担通用数值工具角色。

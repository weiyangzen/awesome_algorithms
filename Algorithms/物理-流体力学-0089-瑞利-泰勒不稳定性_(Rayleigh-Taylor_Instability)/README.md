# 瑞利-泰勒不稳定性 (Rayleigh-Taylor Instability)

- UID: `PHYS-0089`
- 学科: `物理`
- 分类: `流体力学`
- 源序号: `89`
- 目标目录: `Algorithms/物理-流体力学-0089-瑞利-泰勒不稳定性_(Rayleigh-Taylor_Instability)`

## R01

瑞利-泰勒不稳定性（RTI）描述的是：在加速度（典型是重力）方向上，当“重流体位于轻流体之上”时，界面微扰会被浮力驱动放大，最终形成尖刺（spike）与气泡（bubble）结构。

本条目聚焦二维周期界面、不可压无粘、线性小扰动阶段。目标是把 RTI 的核心问题做成可计算闭环：哪些波数不稳定、增长率是多少、时域重建能否反演回理论增长率。

## R02

MVP 关注“最小但可验证”的链路，不追求全量 CFD：

- 用线性色散关系计算离散模态增长率；
- 自动区分稳定/不稳定模态；
- 重建界面位移 `eta(x,t)` 的线性演化；
- 用 FFT + 回归从时域数据反推主导增长率；
- 用 ODE 数值积分与解析指数解做一致性校验。

## R03

建模假设如下：

- 上下两层流体不可压、无粘、无限深近似；
- 扰动振幅足够小，停留在线性阶段；
- 仅考虑重力与表面张力的竞争（无剪切背景流）；
- 周期区间 `[0, L)` 离散为模态 `k_m = 2*pi*m/L`；
- 采用固定参数的确定性随机相位初值，保证可复现。

这保证模型简洁透明，适合作为 RTI 入门级算法验证器。

## R04

线性两层界面 RTI 判据采用：

`D(k) = A*g*k - sigma*k^3/(rho_top + rho_bottom)`

其中 Atwood 数 `A = (rho_top-rho_bottom)/(rho_top+rho_bottom)`。

- 若 `D(k) > 0`，模态不稳定，增长率 `gamma(k)=sqrt(D(k))`；
- 若 `D(k) <= 0`，模态稳定（以振荡为主），频率 `omega(k)=sqrt(-D(k))`。

临界波数（`D(k)=0` 的正根）为：

`k_c = sqrt(A*g*(rho_top + rho_bottom)/sigma)`

## R05

算法主流程（高层）如下：

1. 生成离散模态 `m=1..M` 及波数 `k_m`；
2. 计算每个模态的 `D(k_m)`、`gamma(k_m)`、`omega(k_m)`；
3. 通过 `brentq` 在连续 `k` 轴定位临界点并识别不稳定区间；
4. 构造小振幅复系数 `c_m(0)`，按 `exp((gamma_m+i*omega_m)t)` 推进；
5. 叠加得到 `eta(x,t)`；
6. 对 `eta(x,t)` 做 FFT，提取主导模态振幅轨迹；
7. 线性回归 `log|A(t)|` 得实验增长率并与理论对比；
8. 用 `solve_ivp` 对单模 ODE 增长做数值-解析一致性校验。

## R06

记模态数为 `M`，空间点数 `nx`，时间步数 `nt`：

- 模态判据计算：`O(M)`；
- 临界波数搜索：`O(G)`（`G` 为一维扫描网格规模）；
- 界面重建：`O(nt * M * nx)`；
- FFT + 回归：`O(nt * nx log nx) + O(nt)`；
- 空间复杂度：`O(nt * nx + M)`。

默认参数规模很小，适合自动化验证与参数扫频。

## R07

正确性直觉：

- 在线性阶段，每个傅里叶模态独立演化；
- 色散关系直接给出该模态“增长还是稳定”；
- 若生成 `eta(x,t)` 的推进因子与判据一致，则同一模态的 `log|A|` 必须近似线性；
- 回归斜率贴近理论 `gamma`，说明“判据 -> 演化 -> 观测”闭环正确；
- ODE 数值积分与解析指数解一致，说明实现流程没有隐藏数值偏差。

## R08

数值稳健性策略：

- 初始振幅取 `eta0_scale=2e-7`，确保线性近似成立；
- `sqrt` 前使用 `clip` 防止浮点噪声导致负值开方；
- 回归使用后段时间窗，减少早期多模态混合影响；
- `solve_ivp` 使用 `DOP853` 且严格容差；
- 关键指标均设置断言，保证脚本可自动判定通过/失败。

## R09

`demo.py` 使用最小工具栈：

- `numpy`：波数计算、模态叠加、FFT；
- `scipy`：`brentq` 求临界波数、`solve_ivp` 校验 ODE；
- `pandas`：模式表与摘要指标表；
- `scikit-learn`：线性回归和拟合质量评估；
- `torch`：界面能量轨迹增长率的独立估计。

## R10

演示内容包含三段：

- 演示 A（离散判据）：输出每个模态的 `D(k)`、`gamma`、`omega`；
- 演示 B（连续区间）：输出连续波数轴上的不稳定窗口 `(k_left, k_right)`；
- 演示 C（时域反演）：从 `eta(x,t)` 反推主导模态增长率并与理论值比较。

全流程无交互输入，适合直接纳入流水线。

## R11

关键输出指标：

- `atwood_number`：不稳定驱动强度；
- `unstable_mode_count` / `stable_mode_count`：模态分区完整性；
- `dominant_mode`, `dominant_gamma`：主导不稳定模态及理论增长率；
- `fft_fitted_gamma`, `fft_fit_r2`, `fft_log_mae`：时域反演质量；
- `ode_rel_error`：ODE 数值解与解析解偏差；
- `critical_k_formula`, `critical_k_numeric`：临界波数公式值与数值值；
- `torch_energy_gamma`：基于能量轨迹的独立增长率估计。

## R12

默认参数（`RTIConfig`）设置为混合稳定场景：

- 密度：`rho_top=2.0, rho_bottom=1.0`（重上轻下）；
- 物理项：`gravity=9.81, surface_tension=0.06`；
- 离散规模：`max_mode=16, nx=512, nt=260, time_end=0.35`；
- 初值：`eta0_scale=2e-7, random_seed=19`。

该组参数会产生“低波数不稳定，高波数稳定”的典型 RTI 线性谱。

## R13

理论保证类型说明：

- 近似比保证：N/A（非组合优化问题）；
- 概率成功率保证：N/A（固定参数、固定种子、确定性流程）。

本实现的可验证验收条件：

- 主导模态回归 `R^2 > 0.995`；
- 回归增长率与理论值相对误差 `< 6%`；
- ODE 数值-解析相对误差 `< 1e-6`；
- 主导模态末端振幅相对误差 `< 8%`；
- 临界波数公式/数值相对误差 `< 5e-3`。

## R14

常见失效模式：

1. `A <= 0`（轻上重下）导致不存在 RTI 增长；
2. 表面张力过小导致离散模态几乎全不稳定，失去混合场景自检意义；
3. 时间窗过短，增长斜率拟合不稳；
4. `nx` 太低导致频谱读数偏差；
5. 把线性模型误用于非线性翻卷后期并过度解释结果。

## R15

工程扩展方向：

- 纳入黏性与扩散，研究黏性 RTI 增长抑制；
- 扩展到可压缩 RTI 或 MHD-RTI；
- 用伪谱/有限体积法推进二维 vorticity/level-set 场；
- 做参数扫描（Atwood、sigma、g）并导出 CSV/Parquet；
- 引入多层流体界面或时间变加速度场。

## R16

相关主题：

- Kelvin-Helmholtz 不稳定性（剪切驱动）；
- Richtmyer-Meshkov 不稳定性（冲击波触发）；
- 表面张力主导的毛细-重力波；
- 线性稳定性分析与谱方法；
- 惯性约束聚变、超新星混合、地球物理分层流。

## R17

MVP 功能清单：

- [x] RTI 线性色散判据与模态分类；
- [x] 连续波数临界点与不稳定区间定位；
- [x] 多模态 `eta(x,t)` 时域重建；
- [x] FFT + 回归反演增长率；
- [x] `solve_ivp` 与解析增长的一致性校验；
- [x] `torch` 能量增长率交叉诊断；
- [x] 自动断言与终端摘要输出。

运行方式：

```bash
cd Algorithms/物理-流体力学-0089-瑞利-泰勒不稳定性_(Rayleigh-Taylor_Instability)
uv run python demo.py
```

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `RTIConfig` 定义密度、重力、表面张力、离散规模和初值尺度，并派生 `rho_sum` 与 `atwood`。  
2. `dispersion_discriminant` 逐波数计算 `D(k)=A*g*k-sigma*k^3/rho_sum`，显式给出不稳定/稳定判据。  
3. `mode_table` 生成 `mode/k/discriminant/growth_rate/omega_stable` 结构化数据并标记 `is_unstable`。  
4. `find_unstable_intervals` 在连续 `k` 网格上检测符号翻转，再用 `scipy.optimize.brentq` 精确求根得到不稳定区间。  
5. `simulate_interface` 生成确定性随机相位初值 `c_m(0)`，按 `exp((gamma+i*omega)t)` 推进并叠加空间基 `exp(i*k*x)` 构造 `eta(x,t)`。  
6. `estimate_growth_from_fft` 对 `eta(x,t)` 做时序 FFT，抽取主导模态振幅并用 `LinearRegression` 拟合 `log|A(t)|` 得增长率。  
7. `integrate_scalar_growth_ode` 以 `solve_ivp(DOP853)` 求解 `dA/dt=gamma*A`，与解析指数解比较得到相对误差。  
8. `main` 汇总 `pandas` 指标表，加入 `torch` 能量增长率交叉检查，并通过断言给出自动化 PASS/FAIL。  

第三方库没有被当作黑箱：RTI 判据、模态推进、频谱反演和验收阈值都在源码中逐步展开；`scipy/sklearn/torch` 只承担通用数值工具角色。

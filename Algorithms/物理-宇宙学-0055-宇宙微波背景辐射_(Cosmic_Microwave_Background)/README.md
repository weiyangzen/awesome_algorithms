# 宇宙微波背景辐射 (Cosmic Microwave Background)

- UID: `PHYS-0055`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `55`
- 目标目录: `Algorithms/物理-宇宙学-0055-宇宙微波背景辐射_(Cosmic_Microwave_Background)`

## R01

宇宙微波背景辐射（CMB）是大爆炸后遗留下来的近乎完美黑体辐射，今天观测到的平均温度约 `2.725 K`。它包含两层关键信息：

- 频谱层：平均辐射频谱近似 Planck 黑体分布；
- 各向异性层：温度涨落可用角功率谱 `C_ell`（或 `D_ell = ell(ell+1)C_ell/(2pi)`）描述。

本条目把 CMB 的这两层信息做成一个可运行、可审计的最小算法流程。

## R02

本 MVP 目标是“用尽量少的依赖把 CMB 分析链跑通”，包括：

1. 合成 FIRAS 风格的黑体频谱数据；
2. 通过加权非线性最小二乘反演 `T_CMB` 与标定增益；
3. 构造一个透明的 toy `D_ell` 声学峰模型并采样噪声观测；
4. 检测第一声学峰位置；
5. 由 `C_ell` 重建角相关函数 `C(theta)` 的离散快照。

不依赖 CAMB/CLASS 等高层黑箱求解器。

## R03

核心方程：

1. Planck 黑体谱（按频率）：
`B_nu(T) = (2 h nu^3 / c^2) / (exp(h nu/(kT)) - 1)`

2. 角功率谱常用变换：
`D_ell = ell(ell+1)C_ell/(2pi)`，
`C_ell = 2pi D_ell/[ell(ell+1)]`

3. Cosmic variance（近似）：
`sigma(C_ell) ~= sqrt(2/(2ell+1)) * C_ell`

4. 两点角相关函数：
`C(theta) = Sum_{ell} ((2ell+1)/(4pi)) C_ell P_ell(cos theta)`

其中 `P_ell` 是 Legendre 多项式。

## R04

`demo.py` 的算法主流程：

1. 生成 `30-900 GHz` 频段的理想黑体谱，并加入相对高斯噪声；
2. 用 `scipy.optimize.least_squares` 对 `(T, gain)` 做加权拟合；
3. 构造 toy `D_ell`：低 `ell` 基线 + 三个高斯声学峰 + 高 `ell` 阻尼尾；
4. 转成 `C_ell` 后按 cosmic variance 采样噪声观测；
5. 用 Savitzky-Golay 平滑和 `find_peaks` 检测峰位置；
6. 在 `theta={0,30,...,180}` 度上计算 `C(theta)`；
7. 输出结果表并执行自动断言。

## R05

设频谱采样点数为 `N_nu`，多极矩上限为 `L`：

- 频谱生成与拟合残差评估：`O(N_nu)` 每次函数评估；
- `D_ell/C_ell` 构造与噪声采样：`O(L)`；
- 峰检测（含平滑）：`O(L)`；
- `C(theta)` 在 `N_theta` 个角度上计算：`O(L * N_theta)`。

本实现 `N_theta` 很小（7 个角度），总体复杂度近似 `O(N_nu + L)`。

## R06

默认参数下，脚本会输出三类结果：

- 黑体拟合：`T_hat` 接近 `2.7255 K`，增益 `gain_hat` 接近 `1.0`；
- 功率谱峰：第一峰应在 `ell≈220` 附近；
- 角相关：`C(0 deg)` 通常显著大于大角度相关值。

并给出 `Checks` 列表，全部通过则 `Run completed successfully.`。

## R07

优点：

- 流程透明：所有关键方程在源码中显式实现；
- 工程简洁：仅用 `numpy + scipy + pandas`；
- 可复现：固定随机种子，结果稳定。

局限：

- toy `D_ell` 不是精密玻尔兹曼求解结果；
- 未建模前景污染、束函数、天空掩膜、仪器系统误差；
- 只做温度各向异性，不含偏振谱（`EE/TE/BB`）。

## R08

前置知识：

- 黑体辐射与 Planck 常数、Boltzmann 常数；
- CMB 角功率谱 `C_ell` 与 `D_ell` 的关系；
- cosmic variance 的统计含义；
- Legendre 多项式展开。

环境要求：

- Python `>=3.10`
- `numpy`, `scipy`, `pandas`

## R09

适用场景：

- 教学中讲解 CMB 分析链路（频谱 -> 温度 -> 声学峰 -> 角相关）；
- 在接入高精度宇宙学库前做快速 sanity check；
- 需要可读、可审计的 baseline 脚本。

不适用场景：

- 精密宇宙学参数估计（如拟合 `Omega_b h^2`、`n_s`、`tau`）；
- 实际实验数据管线（需 beam/window/noise covariance/full-sky likelihood）；
- 偏振与 lensing 重建等高阶分析。

## R10

正确性检查点（实现层）：

1. `planck_radiance_hz` 在正温度下必须返回正值；
2. 拟合后 `T_hat` 应接近真值，且残差 RMS 不应异常增大；
3. `D_ell <-> C_ell` 变换需严格互逆（同一 `ell` 网格上）；
4. 第一声学峰应落在合理范围（本实现断言 `[190,250]`）；
5. `C(0)` 通常大于中等角度相关（本实现检查 `C(0)>C(60)`）。

## R11

数值稳定性策略：

- 在 `exp` 参数上做截断（`x <= 700`），避免溢出；
- 使用 `expm1(x)` 计算 `exp(x)-1`，提升小 `x` 区间精度；
- 噪声标准差和分母统一使用 `clip(..., 1e-30, None)` 防止除零；
- 对采样后的 `C_ell` 做下界截断保持正值；
- 峰检测前做平滑，降低随机起伏对峰位的误判。

## R12

关键参数：

- `true_temperature_k`：真值温度；
- `relative_noise_std`：频谱观测相对噪声；
- `ell_max`：分析到的最高多极矩；
- `cosmic_variance_scale`：控制 `C_ell` 随机扰动强度；
- `seed_spectrum` / `seed_cl`：复现性开关。

调参建议：

- 想更稳定峰位可适当减小 `cosmic_variance_scale`；
- 想看更明显阻尼尾可提高 `ell_max`；
- 想测试拟合鲁棒性可增加 `relative_noise_std`。

## R13

本条目不是近似优化问题，因此“近似比”不适用。可提供的保证是：

- 固定配置与随机种子时输出可复现；
- 脚本含内置阈值断言，保证得到物理上合理的温度与第一峰位置；
- 全流程失败会抛出异常并返回非零退出状态，不会静默失败。

## R14

常见失效模式：

1. 频谱噪声过大导致 `(T, gain)` 拟合退化；
2. 声学峰显著性不足，`find_peaks` 可能检不出第一峰；
3. 过高噪声使 `C(theta)` 顺序关系失真；
4. 若误改 `D_ell <-> C_ell` 换算公式，会导致峰幅和相关函数量级错误。

防护手段：保留断言、固定 seed、逐段输出中间量。

## R15

工程实践建议：

- 把 `CMBConfig` 固化到实验记录，保证结果可追踪；
- 在接入真实数据前先用合成数据做回归测试；
- 若要扩展到真实实验，优先补充 beam/window 和噪声协方差；
- 若要用于教学，建议把 `D_ell` 曲线和 `C(theta)` 结果可视化。

## R16

相关主题与后续扩展：

- 宇宙学参数估计与 MCMC；
- CAMB/CLASS 玻尔兹曼方程求解；
- CMB 偏振（`EE/TE/BB`）与再电离特征；
- lensing 重建与二次估计器；
- 与 BAO/SN/弱透镜做联合约束。

## R17

`demo.py` 能力摘要：

- 黑体谱合成：`make_synthetic_spectrum`；
- 参数拟合：`fit_temperature_and_gain`；
- toy 声学谱：`toy_acoustic_d_ell`；
- `C_ell` 噪声采样：`sample_observed_c_ell`；
- 峰检测：`detect_acoustic_peaks`；
- 角相关重建：`angular_correlation`。

运行方式：

```bash
cd Algorithms/物理-宇宙学-0055-宇宙微波背景辐射_(Cosmic_Microwave_Background)
uv run python demo.py
```

脚本不需要任何交互输入。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `main` 初始化 `CMBConfig`，给出温度真值、噪声等级、`ell` 范围和随机种子。  
2. `make_synthetic_spectrum` 在 `30-900 GHz` 上计算 `B_nu(T)`，生成带噪声的观测频谱。  
3. `fit_temperature_and_gain` 通过 `least_squares` 最小化加权残差，回收 `T_hat` 与 `gain_hat`。  
4. `toy_acoustic_d_ell` 构造由基线、三组高斯峰、阻尼尾组成的解析 `D_ell`。  
5. `d_ell_to_c_ell` 把理论 `D_ell` 转成 `C_ell`，`sample_observed_c_ell` 按 cosmic variance 加噪得到观测 `C_ell`。  
6. `c_ell_to_d_ell` 回到观测 `D_ell`，`detect_acoustic_peaks` 先平滑再峰检，输出峰位置和显著性。  
7. `angular_correlation` 用 `sum_l ((2l+1)/(4pi)) C_l P_l(cos theta)` 在多个角度重建 `C(theta)`。  
8. `main` 打印结果并执行 4 条断言（温度、增益、第一峰位置、相关函数关系）；全部通过即完成验收。  

说明：虽然使用了 `scipy` 的优化和信号工具，但核心 CMB 物理链路（黑体谱、`D_ell/C_ell` 变换、相关函数求和）均在源码中逐项展开，可直接追踪。

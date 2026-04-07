# 康普顿散射 (Compton Scattering)

- UID: `PHYS-0252`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `255`
- 目标目录: `Algorithms/物理-量子力学-0255-康普顿散射_(Compton_Scattering)`

## R01

康普顿散射描述高能光子与近似静止自由电子的非弹性散射过程。与经典弹性散射不同，散射后光子能量降低、波长增加，且变化量只与散射角有关。

该条目 MVP 的目标是把这一过程落成可执行计算：给定入射能量和角度，计算散射光子能量、波长位移与角分布微分截面，并通过内置物理一致性检查给出 `PASS/FAIL`。

## R02

MVP 输入输出范围：
- 输入（脚本内置参数）：入射光子能量 `E0`、角度网格 `theta in [0, pi]`。
- 输出：
1. `E'(theta)`：散射光子能量；
2. `Delta lambda(theta)`：波长位移；
3. `dσ/dΩ(theta)`：Klein-Nishina 微分截面；
4. 与 Thomson 低能极限的偏差指标。

脚本不依赖外部文件，不需要交互输入，直接运行即可复现实验级核心关系。

## R03

核心运动学公式：

`Delta lambda = lambda' - lambda = lambda_C (1 - cos(theta))`

`lambda_C = h / (m_e c)` 为电子康普顿波长。

等价能量形式：

`E' = E0 / (1 + (E0 / (m_e c^2)) (1 - cos(theta)))`

该式体现了散射角越大，光子反冲损失越明显。

## R04

本实现采用双通道自洽校验：
- 通道 A：先由 `E'(theta)` 计算 `lambda'(theta)=hc/E'`，得到数值位移 `Delta lambda_num`。
- 通道 B：直接用康普顿位移公式计算 `Delta lambda_pred`。

若两者在整个角度网格上误差接近机器精度，说明单位换算、常数和能量-波长映射链路一致。

## R05

角分布采用 Klein-Nishina 公式：

`dσ/dΩ = (r_e^2 / 2) (E'/E0)^2 (E'/E0 + E0/E' - sin^2(theta))`

其中 `r_e` 为经典电子半径。

低能极限 `E0 << m_e c^2` 时，Klein-Nishina 应退化到 Thomson 结果：

`dσ/dΩ = (r_e^2 / 2) (1 + cos^2(theta))`

MVP 使用该极限做正确性对照，而不把公式当黑箱直接输出。

## R06

`demo.py` 的输入输出约定：
- 默认主案例：`E0 = 661.7 keV`（常见伽马能量量级）。
- 低能校验案例：`E0 = 1.0 keV`，用于检验 Thomson 极限。
- 输出内容：
1. 常量与参数回显；
2. 角度抽样表（能量、波长位移、截面与比值）；
3. 验证指标与 `Validation: PASS/FAIL`。

## R07

高层算法流程：
1. 生成 `theta` 离散网格。
2. 计算电子静能 `m_e c^2` 与 `lambda_C`。
3. 由康普顿能量公式得到 `E'(theta)`。
4. 将能量映射为波长，得到 `Delta lambda_num`。
5. 独立计算 `Delta lambda_pred = lambda_C(1-cos theta)`。
6. 计算 Klein-Nishina 与 Thomson 微分截面。
7. 计算总截面和误差指标，输出验证结论。

## R08

设角度采样数为 `N`：
- 能量、波长、截面均为逐点向量化计算，时间复杂度 `O(N)`。
- 总截面积分为一次数值积分，时间复杂度 `O(N)`。
- 空间复杂度主要由若干长度 `N` 数组构成，为 `O(N)`。

## R09

数值稳定性处理：
- 相对误差分母使用 `max(|ref|, eps)`，避免 `theta≈0` 时除零放大。
- 对 `Delta lambda` 的相对误差只在非零位移区间统计，避免前向散射点的人为奇异。
- 校验 `finite` 与 `non_negative`，过滤 `nan/inf` 或负截面异常。

## R10

MVP 技术栈：
- Python 3
- `numpy`：向量化公式计算与数值积分
- `pandas`：结构化结果表输出
- `dataclasses`：参数配置封装

未调用专用散射黑箱库；所有物理公式与验证逻辑在源码中显式实现。

## R11

运行方式：

```bash
cd Algorithms/物理-量子力学-0255-康普顿散射_(Compton_Scattering)
uv run python demo.py
```

运行后应看到角度样表、误差指标与最终 `Validation: PASS`。

## R12

输出字段说明（样表）：
- `theta_deg`: 散射角（度）
- `E_prime_kev`: 散射光子能量
- `lambda_prime_pm`: 散射光子波长（pm）
- `delta_lambda_pm`: 由能量反算得到的波长位移（pm）
- `delta_lambda_pred_pm`: 由康普顿公式得到的理论位移（pm）
- `dsigma_kn_barn_sr`: Klein-Nishina 微分截面（barn/sr）
- `dsigma_th_barn_sr`: Thomson 微分截面（barn/sr）
- `kn_over_th`: 两者比值

## R13

内置验证规则：
1. `max_abs_shift_err_m` 小于绝对阈值；
2. `max_rel_shift_err` 小于相对阈值；
3. 低能点态误差 `max_low_energy_point_rel` 小于阈值；
4. 低能总截面误差 `low_energy_total_rel` 小于阈值；
5. 所有结果有限且 `dσ/dΩ >= 0`。

全部满足则输出 `Validation: PASS`，否则进程返回非零退出码。

## R14

当前实现边界：
- 假设电子初始静止且自由，未包含束缚态修正。
- 未包含极化依赖与探测器响应函数。
- 只做单次散射，不含多重散射传输。
- 角分布采用解析公式，未做蒙特卡洛几何追踪。

## R15

可扩展方向：
- 加入电子初始动量分布（多普勒展宽）以贴近材料谱线。
- 引入偏振态，扩展到偏振依赖 Klein-Nishina。
- 增加蒙特卡洛采样，模拟能谱与角分布直方图。
- 对接实验数据做参数拟合与不确定度评估。

## R16

应用场景：
- 核医学与辐射探测中的伽马能谱分析。
- 天体高能辐射传输建模。
- 粒子探测器课程中的角分布与能量损失教学。
- 作为更复杂 Geant4/MCNP 模拟前的快速 sanity check。

## R17

方法对比：
- 相比仅展示公式：本条目提供可运行的数值流程和自动校验。
- 相比完全蒙特卡洛：本 MVP 更轻量、可解释，适合快速验证。
- 相比黑箱库调用：能量映射、位移关系、截面积分和低能极限检查均在源码可审计展开。

## R18

`demo.py` 源码级算法拆解（8 步）：
1. `ComptonConfig` 定义主案例能量、低能校验能量、角度采样与误差阈值。
2. `electron_rest_energy_kev` 和 `compton_wavelength_m` 计算 `m_ec^2` 与 `lambda_C` 两个关键常量。
3. `scattered_photon_energy_kev` 按康普顿能量公式向量化生成 `E'(theta)`。
4. `wavelength_from_energy_m` 把 `E0` 与 `E'(theta)` 转成波长，构造 `Delta lambda_num`。
5. `compton_shift_predicted_m` 独立生成 `Delta lambda_pred`，与第 4 步做一致性比较。
6. `klein_nishina_differential_cross_section` 与 `thomson_differential_cross_section` 计算角分布，并在 `total_cross_section` 中积分成总截面。
7. `validate` 汇总 5 类检查（位移绝对/相对误差、低能点态/总截面误差、数值有限与非负）。
8. `main` 组装 `pandas` 表格、打印抽样结果和指标，并据 `PASS/FAIL` 返回进程状态码。

第三方库仅承担数组和表格基础能力，物理算法链条全部在本地源码显式实现。

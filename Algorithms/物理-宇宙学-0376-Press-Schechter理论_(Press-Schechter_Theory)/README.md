# Press-Schechter理论 (Press-Schechter Theory)

- UID: `PHYS-0358`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `376`
- 目标目录: `Algorithms/物理-宇宙学-0376-Press-Schechter理论_(Press-Schechter_Theory)`

## R01

Press-Schechter 理论用于从线性高斯密度涨落出发，估计不同质量暗晕的数密度分布。核心思想是：把线性场在质量尺度 `M` 上平滑后，若线性过密度 `δ` 超过球对称坍缩阈值 `δ_c`，则认为该尺度会形成坍缩天体。MVP 目标是数值计算 `sigma(M,z)` 与 `dn/dM`，并输出可检查的质量函数表。

## R02

本实现采用最小可运行的宇宙学闭环：

1. 平直 `LCDM` 背景：`Omega_m=0.315, Omega_Lambda=0.685, h=0.674`
2. 线性谱形状：`P(k) = A * k^{n_s} * T(k)^2`，`T(k)` 取 BBKS 近似
3. 归一化：通过 `sigma8=0.811` 反解振幅 `A`
4. 演化：`sigma(M,z)=D(z)*sigma(M,0)`，`D(z)` 用 Carroll-Press-Turner 近似

## R03

`demo.py` 的输入/输出约定：

- 输入（脚本内固定，无交互）：
1. `k` 网格：`1e-4` 到 `1e2` (`1/Mpc`)
2. 质量网格：`1e10` 到 `1e16` (`Msun`)
3. 红移：`z=0` 与 `z=1`

- 输出：
1. 归一化前后 `sigma8`（用于自检）
2. 采样质量点上的 `sigma`, `nu`, `dn/dM`, `F(>M)`
3. 有限质量区间上的质量分数积分检查

## R04

MVP 使用的关键公式：

1. 质量-半径映射（顶帽窗）：
`M = (4/3) * pi * rho_m0 * R^3`

2. 方差：
`sigma^2(R) = ∫ [k^2 P(k) W^2(kR)] / (2 pi^2) dk`

3. 峰高参数：
`nu = delta_c / sigma(M,z)`，其中 `delta_c = 1.686`

4. Press-Schechter 微分质量函数：
`dn/dM = sqrt(2/pi) * (rho_m0/M^2) * nu * exp(-nu^2/2) * |d ln sigma / d ln M|`

5. 累积坍缩分数：
`F(>M) = erfc(nu / sqrt(2))`

## R05

`demo.py` 的主流程：

1. 构建 `Cosmology` 常量与 `k`、`M` 网格
2. 计算 BBKS 传输函数并得到单位振幅功率谱
3. 用 `sigma(8/h Mpc)` 反解谱振幅 `A`
4. 对每个质量通过顶帽窗积分得到 `sigma(M,0)`
5. 用增长因子得到 `sigma(M,1)`
6. 用数值梯度 `dlnsigma/dlnM` 计算 `dn/dM`
7. 计算 `F(>M)` 并打印采样表
8. 输出有限区间质量守恒积分作为 sanity check

## R06

正确性依据：

1. `sigma(M)` 定义直接来自线性理论与平滑窗函数，公式标准且可复现
2. 振幅由 `sigma8` 归一化，避免随意尺度因子导致不可比较结果
3. `dn/dM` 由 Press-Schechter 闭式表达式直接实现，关键斜率项用 `np.gradient` 在对数网格上估计
4. 同时输出 `z=0` 与 `z=1`，应满足高红移下大质量端丰度更低的物理趋势

## R07

复杂度分析（设质量点数 `N_M`，波数点数 `N_k`）：

- 时间复杂度：`O(N_M * N_k)`（主耗时在 `sigma(R)` 的二维积分被积函数构建）
- 空间复杂度：`O(N_M * N_k)`（`outer(R, k)` 与窗函数矩阵）

当前参数 `N_M=120, N_k=4096` 在普通 CPU 上可快速运行。

## R08

数值稳定性处理：

1. 顶帽窗 `W(x)` 在 `x->0` 用泰勒展开，避免 `0/0` 取消误差
2. `sigma` 与 `nu` 计算前做 `clip`，避免 `log(0)` 与除零
3. `q` 变量设下界 `1e-12`，避免 BBKS 传输函数出现奇异
4. 积分使用 `scipy.integrate.simpson`，比粗糙矩形法更稳定

## R09

参数与单位约定：

1. 质量单位：`Msun`
2. 长度单位：`Mpc`
3. 波数单位：`1/Mpc`
4. 密度单位：`Msun/Mpc^3`

该单位体系保证 `M-R` 映射、`P(k)` 积分和 `dn/dM` 维度一致，可直接做数量级检查。

## R10

运行方式：

```bash
uv run python Algorithms/物理-宇宙学-0376-Press-Schechter理论_(Press-Schechter_Theory)/demo.py
```

或在目录内运行：

```bash
uv run python demo.py
```

## R11

输出字段说明：

1. `sigma(z=0), sigma(z=1)`：质量尺度 RMS 扰动
2. `nu(z=0), nu(z=1)`：坍缩阈值相对涨落幅度
3. `dn/dM`：单位体积单位质量下的暗晕丰度
4. `F(>M,z=0)`：高于该质量阈值的累计坍缩质量分数

## R12

可预期结果特征：

1. `sigma` 随 `M` 增大而下降
2. `nu` 随 `M` 增大而上升
3. `dn/dM` 在低质量端较大，高质量端指数压低
4. 同质量下 `z=1` 的 `dn/dM` 小于 `z=0`（尤其大质量端更明显）

## R13

模型边界与简化：

1. 该 MVP 是原始 Press-Schechter，不含 Sheth-Tormen 椭球修正
2. 传输函数使用 BBKS 近似，未显式加入重子声学振荡细节
3. 增长因子使用拟合式，不是数值解完整线性增长微分方程
4. 结果适合教学与流程验证，不替代高精度 Boltzmann 求解器

## R14

可能失败模式：

1. 若 `k` 积分区间过窄，`sigma(M)` 会被低估或高估
2. 若质量网格过稀，`dlnsigma/dlnM` 噪声增大，`dn/dM` 抖动
3. 若把 `M` 扩到极端范围，有限积分与近似传输函数误差会放大
4. 若参数设成非平直宇宙，当前增长因子近似需替换

## R15

最小测试建议：

1. 归一化测试：`sigma8(after normalization)` 应接近输入 `0.811`
2. 单调测试：`sigma(M)` 应基本单调下降
3. 正值测试：`dn/dM` 应非负
4. 红移对比：大质量端应满足 `dn/dM(z=1) < dn/dM(z=0)`

## R16

可扩展方向：

1. 将 multiplicity function 从 PS 替换为 Sheth-Tormen 并比较丰度差异
2. 接入 CAMB/CLASS 生成高精度线性谱替代 BBKS
3. 输出 `dn/dlnM`、累计丰度 `n(>M)` 等更常用观测量
4. 引入 `M200c/Mvir` 等质量定义转换，提高与模拟目录兼容性

## R17

应用场景：

1. 半解析星系形成模型中的暗晕先验丰度
2. 大尺度结构课题中的质量函数数量级估计
3. 课堂演示“线性涨落 -> 非线性坍缩统计”的桥接流程
4. 作为更复杂宇宙学生成管线的单元测试基线

## R18

`demo.py` 的源码级算法流（展开第三方库调用，不把其视作黑箱）：

1. 先在 `power_spectrum_unnormalized` 中显式构造 `k^{n_s}` 与 BBKS `T(k)`，得到单位振幅谱形。
2. 在 `sigma_r` 中构造 `x = outer(R, k)`，逐点计算顶帽窗 `W(x)`，形成二维积分被积函数矩阵。
3. `scipy.integrate.simpson` 对每个 `R` 方向沿 `k` 轴执行分段抛物线求积，得到 `sigma^2(R)`；再开方得到 `sigma(R)`。
4. `normalize_power_to_sigma8` 用 `R=8/h` 处的 `sigma8_unit` 反解振幅 `A=(sigma8_target/sigma8_unit)^2`，并线性缩放整条 `P(k)`。
5. `growth_factor_lcdm` 通过 `E(z)`、`Omega_m(z)`、`Omega_L(z)` 计算 `g(z)`，再归一化得到 `D(z)`，把 `sigma(M,0)` 映射到 `sigma(M,z)`。
6. `np.gradient(log(sigma), log(M))` 计算 `d ln sigma / d ln M`，这是 PS 质量函数把尺度概率映射到质量空间的关键雅可比项。
7. `press_schechter_dndm` 组合 `nu`、指数截断项和斜率项，得到 `dn/dM`；`scipy.special.erfc` 同时给出 `F(>M)`。
8. `pandas.DataFrame` 汇总关键质量点结果，保证输出结构稳定、可读、可被后续验证脚本直接解析。

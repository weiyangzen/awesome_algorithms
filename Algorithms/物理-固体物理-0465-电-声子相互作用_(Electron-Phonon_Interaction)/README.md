# 电-声子相互作用 (Electron-Phonon Interaction)

- UID: `PHYS-0442`
- 学科: `物理`
- 分类: `固体物理`
- 源序号: `465`
- 目标目录: `Algorithms/物理-固体物理-0465-电-声子相互作用_(Electron-Phonon_Interaction)`

## R01

问题定义：把“电-声子相互作用”落成可运行的数值 MVP，而不是只停留在教材公式。

本条目选择正常态（normal state）下的最小闭环：
- 构造并归一化 Eliashberg 谱函数 `alpha^2F(omega)`；
- 计算电子自能 `ImSigma(omega, T)` 与 `ReSigma(omega, T)`；
- 用 `ReSigma` 在费米面附近的斜率估计有效耦合常数 `lambda`；
- 计算 `Gamma(E_F, T) = -2 ImSigma(0, T)` 的温度增长行为。

## R02

物理背景（MVP 视角）：
- 晶格振动量子化后得到声子，电子与声子耦合后会导致有效质量增强、有限寿命（散射率）和谱函数重整化。
- 在 Migdal-Eliashberg 常用表述里，核心输入是 `alpha^2F(omega)`。
- 一个常用耦合指标是：
  `lambda = 2 * integral_0^inf [alpha^2F(omega) / omega] d omega`。

本实现不做超导能隙求解，只做正常态自能与耦合强度诊断，保证最小且诚实。

## R03

计算任务拆解：
1. 设定目标耦合常数 `lambda_target` 与 Einstein-like 声子峰参数 `(omega0, sigma)`。
2. 在声子频率网格上构造高斯形 `alpha^2F(omega)`，并按积分定义归一化。
3. 在电子频率网格上计算 `ImSigma(omega, T)`（含 Bose/Fermi 占据数核）。
4. 对 `ImSigma` 做离散 Kramers-Kronig 主值变换，得到 `ReSigma`。
5. 在 `omega≈0` 小窗口线性拟合 `ReSigma` 斜率，估计 `lambda_from_slope`。
6. 输出 summary + 温度扫描表，并执行质量门槛断言。

## R04

建模假设（有意简化）：
- 正常态单带近费米面近似；
- 使用经验化 `alpha^2F`（高斯峰），不从第一性原理声子色散直接构造；
- 自能核采用常见 normal-state 形式；
- Kramers-Kronig 在有限频窗离散近似，边界截断误差不可避免；
- `lambda` 的斜率提取在低温更可靠，高温仅作趋势参考。

## R05

核心公式（`demo.py` 对应实现）：

1. 耦合归一化：
`lambda = 2 * integral_0^inf [alpha^2F(Omega)/Omega] dOmega`

2. 正常态自能虚部（本实现核函数）：
`ImSigma(omega,T) = -pi * integral_0^inf alpha^2F(Omega) * [2 n_B(Omega,T) + f(Omega-omega,T) + f(Omega+omega,T)] dOmega`

3. Kramers-Kronig（主值积分）：
`ReSigma(omega) = (1/pi) P integral ImSigma(omega') / (omega' - omega) d omega'`

4. 有效耦合斜率估计：
`lambda_eff ~= - dReSigma/domega |_{omega->0}`

5. 费米面散射率：
`Gamma(E_F,T) = -2 ImSigma(0,T)`

## R06

数值算法流程：
1. 参数合法性检查（正值、网格奇偶性、温度正值）。
2. 构造 `omega_el`（对称含 0）和 `omega_ph`（正频率）网格。
3. 用 Simpson 积分确定 `alpha^2F` 振幅，使积分 `lambda` 命中目标值。
4. 对每个温度 `T`：
   - 广播计算 `ImSigma(omega,T)`；
   - 用离散主值核矩阵做 Kramers-Kronig 得 `ReSigma`；
   - 用 `scikit-learn` 线性回归拟合低能斜率并估计 `lambda`。
5. 用 PyTorch 对 `Gamma(E_F,T)` 做独立积分交叉校验。
6. 聚合 `pandas` 报表并执行断言门槛。

## R07

复杂度（记电子网格 `Nw`、声子网格 `Nph`、温度点数 `Nt`）：
- `ImSigma` 计算：`O(Nt * Nw * Nph)`；
- Kramers-Kronig 离散主值：`O(Nt * Nw^2)`；
- 空间复杂度主导项：`O(Nw^2)`（主值核矩阵）。

本参数下 (`Nw=801`, `Nph=1201`, `Nt=3`) 属于可直接运行的桌面级计算。

## R08

数值稳定性处理：
- Fermi/Bose 分布的指数输入做 `clip`，避免溢出；
- 声子网格从 `1e-5 eV` 起步，避免 `1/omega` 奇点；
- 使用奇数 `Nw` 保证 `omega=0` 恰在网格上；
- Kramers-Kronig 对角项显式置零实现主值处理；
- 质量门槛里只要求“低温斜率接近目标耦合”，避免对高温过度解释。

## R09

适用场景：
- 固体物理教学中演示 `alpha^2F -> self-energy -> lambda/Gamma` 的完整链路；
- 做快速参数敏感性扫描（改 `omega0/sigma/lambda_target` 看趋势）；
- 作为更复杂电子结构代码的原型验证前端。

不适用：
- 材料级精确预测（缺少真实能带、声子谱、矩阵元与自洽闭环）；
- 超导 `Tc` 的高精度计算（未解全 Eliashberg 方程）；
- 强关联系统或多带强各向异性体系。

## R10

脚本中的正确性门槛：
1. `abs(lambda_from_alpha2F - lambda_target) <= 5e-3`。
2. 低温斜率估计 `lambda_from_slope_lowT` 与目标偏差不超过 `0.20`。
3. `Gamma(E_F,T)` 随温度严格递增（该模型参数下的预期行为）。
4. `torch` 与 `numpy/scipy` 在 `Gamma(E_F,T)` 上的相对差不超过 `2e-2`。

这些门槛面向“可运行 MVP 的一致性验证”，不是材料数据库级精度标准。

## R11

默认参数（`ElectronPhononParams`）：
- `lambda_target = 0.85`
- `omega0_eV = 0.020`
- `sigma_eV = 0.004`
- `omega_ph_max_eV = 0.100`
- `n_omega_ph = 1201`
- `omega_el_max_eV = 0.250`
- `n_omega_el = 801`
- `fit_window_eV = 0.008`
- `temperatures_K = (20.0, 100.0, 300.0)`

单位约定：能量统一 `eV`（输出部分散射率/自能换算为 `meV`）。

## R12

本地实测（命令：`uv run python demo.py`）：

Summary:
- `lambda_target = 0.850000`
- `lambda_from_alpha2F = 0.850000`
- `alpha2F_amplitude = 8.102645e-01`
- `lambda_from_slope_lowT = 0.863328`
- `lambda_slope_rel_error = 1.568e-02`
- `mstar_over_m_lowT = 1.863328`
- `gamma_ratio_highT_over_lowT = 4510.794072`
- `max_torch_gamma_rel_error = 2.781e-03`

Temperature scan:
- `T=20 K`: `lambda_from_slope=0.863328`, `gamma_EF=0.027901 meV`
- `T=100 K`: `lambda_from_slope=0.698327`, `gamma_EF=22.871591 meV`
- `T=300 K`: `lambda_from_slope=-0.003697`, `gamma_EF=125.857879 meV`

说明：低温斜率提取与目标 `lambda` 一致性较好；高温下斜率解释性下降，但散射率温升趋势显著。

## R13

理论与结果一致性解释：
- 归一化后 `lambda_from_alpha2F` 精确命中目标，说明谱函数构造和积分实现正确；
- `ReSigma` 在低能区给出正质量增强（`m*/m > 1`），符合电-声子重整化直觉；
- `Gamma(E_F,T)` 随温度增强，符合热激发声子增加散射通道的机制；
- 高温斜率偏离目标属于“低能线性近似失效 + 有限窗口数值效应”，已在模型假设中声明。

## R14

常见失败模式与修复建议：
- 失败：`n_omega_el` 偶数导致 `omega=0` 不在网格。
  - 修复：使用奇数网格点（脚本已强校验）。
- 失败：`omega_ph` 从 0 起导致 `alpha^2F/omega` 发散。
  - 修复：频率下限设小正数（如 `1e-5 eV`）。
- 失败：指数溢出出现 `inf/nan`。
  - 修复：对无量纲指数输入做 `clip`。
- 失败：拟合窗口过宽，高温下 `lambda` 估计失真。
  - 修复：缩小 `fit_window_eV`，并以低温结果为主。

## R15

工程实践建议：
- 报告里分开给出“积分定义的 `lambda`”与“斜率估计的 `lambda`”，避免混淆；
- 任何 `ReSigma` 解释都应同时提供拟合 `R^2`；
- 若要提高 Kramers-Kronig 精度，可扩展频窗并增加尾部外推；
- 把 `Gamma(E_F,T)` 的单调性当作快速回归指标，适合批量自动化检查。

## R16

可扩展方向：
- 从单峰高斯 `alpha^2F` 扩展到多峰或第一性原理输入谱；
- 引入动量分辨 `Sigma(k,omega)` 与各向异性耦合；
- 增加 Bloch-Gruneisen 电阻率模块，与 `Gamma(E_F,T)` 联合诊断输运；
- 进一步接入 Eliashberg 方程求解 `Tc` 与能隙温标。

## R17

本目录交付内容：
- `demo.py`：可直接运行的最小实现（`numpy + scipy + pandas + scikit-learn + torch`）；
- `README.md`：R01-R18 完整说明、公式、参数、实测与限制；
- `meta.json`：任务元数据与目录信息。

运行方式：

```bash
cd Algorithms/物理-固体物理-0465-电-声子相互作用_(Electron-Phonon_Interaction)
uv run python demo.py
```

无需交互输入，程序直接打印 summary 与温度扫描表。

## R18

`demo.py` 源码级算法流拆解（8 步，非黑盒）：
1. `ElectronPhononParams` 定义目标耦合、谱峰位置/宽度、频率网格和温度列表。  
2. `check_params` 强制参数合法（正值、网格密度、奇数电子网格等）。  
3. `build_frequency_grids` 创建电子与声子离散频率网格。  
4. `normalized_alpha2f` 先生成高斯原型谱，再用 `lambda` 积分公式反算振幅并归一化。  
5. `imag_self_energy` 显式构造 `2n_B + f(Omega-omega) + f(Omega+omega)` 核并积分得到 `ImSigma`。  
6. `real_self_energy_kramers_kronig` 用离散主值核矩阵实现 `ImSigma -> ReSigma` 的 Kramers-Kronig 变换。  
7. `estimate_lambda_from_slope` 在低能窗口线性拟合 `ReSigma`，提取 `lambda=-dReSigma/domega|0`；`gamma_at_fermi_torch` 提供独立积分交叉校验。  
8. `main` 汇总 `summary/temperature_scan` 表，并执行 4 条质量门槛断言。  

第三方库仅承担数值积分、线性拟合和张量计算工具角色；核心物理流程（谱函数归一化、自能核、主值变换、耦合提取）均在源码中显式展开。

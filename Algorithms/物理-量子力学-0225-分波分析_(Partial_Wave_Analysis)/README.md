# 分波分析 (Partial Wave Analysis)

- UID: `PHYS-0224`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `225`
- 目标目录: `Algorithms/物理-量子力学-0225-分波分析_(Partial_Wave_Analysis)`

## R01

分波分析（Partial Wave Analysis）是中心势散射问题的标准方法。其核心是把散射波按角动量本征态 `l=0,1,2,...` 分解，每个分波由一个相移 `delta_l` 描述势场对该角动量通道的影响。总散射振幅与散射截面由所有分波叠加得到。

## R02

典型应用场景：
- 量子力学中的中心势弹性散射（如核散射、原子散射简化模型）。
- 从实验角分布 `dσ/dΩ` 反推出低阶相移（`s,p,d` 波主导区间）。
- 研究共振与阈值行为（某些 `l` 通道相移快速穿越）。

## R03

核心公式（弹性、中心势）：
- 散射振幅：
  `f(theta) = (1/k) * sum_{l=0..∞} (2l+1) exp(i*delta_l) sin(delta_l) P_l(cos(theta))`
- 微分截面：`dσ/dΩ = |f(theta)|^2`
- 总截面：
  `σ_tot = (4π/k^2) * sum_{l=0..∞} (2l+1) sin^2(delta_l)`

本 MVP 使用球形方势阱 `V(r)=-V0 (r<a), 0 (r>=a)`，在 `r=a` 处做径向波函数匹配，解析得到每个 `delta_l`。

## R04

本目录 `demo.py` 输入输出约定：
- 输入：无交互输入。脚本内部固定参数 `E=5, V0=12, a=1, l_max=8, m=1, hbar=1`。
- 输出：
  - 每个角动量通道 `l` 的相移 `delta_l`（弧度与角度）与分波截面 `sigma_l`。
  - 截断分波和下的总截面 `sigma_tot`。
  - 若干角度（0° 到 180°）的 `dσ/dΩ`。
  - 光学定理残差 `|Im f(0) - kσ_tot/(4π)|` 作为一致性检查。

## R05

伪代码（本实现）：

```text
给定 E, V0, a, l_max:
  计算外区与内区波数 k, q

for l in 0..l_max:
  计算 ka=k*a, qa=q*a
  计算球贝塞尔函数 j_l, y_l 及导数
  beta = q * j'_l(qa) / j_l(qa)
  tan(delta_l) = [k*j'_l(ka)-beta*j_l(ka)] / [k*y'_l(ka)-beta*y_l(ka)]
  delta_l = atan2(分子, 分母)

由 delta_l 计算:
  sigma_l = (4*pi/k^2) * (2l+1) * sin^2(delta_l)
  sigma_tot = sum_l sigma_l

for 每个散射角 theta:
  f(theta) = (1/k) * sum_l (2l+1)e^{i delta_l}sin(delta_l)P_l(cos(theta))
  dσ/dΩ = |f(theta)|^2

输出表格与光学定理残差
```

## R06

正确性要点：
- 对球形方势阱，径向方程在内外区分别由球贝塞尔/诺伊曼函数表示，匹配 `u_l` 与 `u_l'` 可唯一确定 `delta_l`。
- 使用 `atan2(num, den)` 比直接 `arctan(num/den)` 更稳健，避免象限错误。
- `σ_tot` 的分波求和与前向振幅虚部满足光学定理；本 MVP 将其作为数值一致性自检。

## R07

复杂度（`L=l_max+1`, 角度采样数 `N_theta`）：
- 相移计算：每个 `l` 仅常数次特殊函数评估，约 `O(L)`。
- 角分布计算：每个角度累加 `L` 个分波，约 `O(L * N_theta)`。
- 空间复杂度：`O(L + N_theta)`。

## R08

与其他散射求解方式对比：
- 相比直接数值积分径向方程，分波法对中心势有更清晰物理解释（“哪个角动量通道在贡献”）。
- 相比 Born 近似，分波法在中等耦合强度下通常更稳健，并显式满足幺正结构（通过相移表达）。
- 代价是需要截断 `l_max`；高能情况下需要更多分波以收敛。

## R09

数据结构与函数划分：
- `wave_numbers`：由能量与势阱参数计算 `k, q`。
- `phase_shift_square_well`：单个 `l` 的匹配公式求 `delta_l`。
- `compute_phase_shifts`：批量生成 `l=0..l_max` 的相移数组。
- `scattering_amplitude`：按勒让德多项式叠加得到 `f(theta)`。
- `partial_cross_sections` / `total_cross_section`：计算 `sigma_l` 与 `sigma_tot`。
- `optical_theorem_residual`：评估光学定理残差。
- `main`：固定参数、调用计算、打印结果。

## R10

边界与异常处理：
- `E<=0` 或 `E+V0<=0` 抛 `ValueError`（避免非散射态或虚波数）。
- `l<0`、`a<=0`、`l_max<0` 抛 `ValueError`。
- 当 `j_l(q a)` 过小（接近节点）时，匹配中的对数导数不稳定，抛 `RuntimeError` 防止静默错误。
- 输出前检查 `sigma_l>=0`，保证基本物理性。

## R11

MVP 选型说明：
- `numpy`：向量化数组与数值计算。
- `scipy.special`：`spherical_jn/spherical_yn/eval_legendre`，用于分波解析表达。

该组合足够小，且直接覆盖分波法的数学核心，不依赖重型框架。

## R12

运行方式（在本目录下）：

```bash
uv run python demo.py
```

脚本无任何交互输入。

## R13

预期输出特征：
- 打印 `l=0..8` 的相移与分波截面，低阶分波通常贡献更显著。
- 打印 `sigma_tot` 与一个很小的光学定理残差（截断误差与浮点误差导致非零）。
- 打印多个角度的 `dσ/dΩ`，通常呈现明显角向各向异性。

## R14

常见实现错误：
- 把外区解误写成仅 `j_l`，忘记 `y_l` 线性组合，导致无法编码相移。
- 相移公式使用 `arctan(num/den)` 丢失象限信息。
- 漏掉 `1/k` 前因子，导致 `f(theta)` 和截面量纲错误。
- `l_max` 截断过小却直接解释精确角分布，忽略截断误差。

## R15

最小测试清单：
- 运行性：`uv run python demo.py` 无报错结束。
- 数值性：所有 `sigma_l` 非负，`sigma_tot` 正值。
- 一致性：光学定理残差应明显小于 `sigma_tot` 的典型尺度。
- 可解释性：输出中至少前几阶分波有可见贡献，非全零相移。

## R16

可扩展方向：
- 将方势阱替换为一般短程势，通过径向方程数值积分提取 `delta_l`。
- 在多能量点扫描 `delta_l(E)`，寻找共振行为（相移穿越特定角度）。
- 引入自旋与耦合通道，扩展到更真实的散射模型。
- 增加角分布可视化与对 `l_max` 收敛性分析。

## R17

局限与取舍：
- 本 MVP 只做弹性散射、单通道、球对称势，不覆盖吸收或非中心势。
- 使用方势阱是教学化简化，真实体系可能需要更精细势模型。
- `l_max` 有限截断会带来高角动量尾项误差；高能时需更高截断。

## R18

源码级算法流程（本实现，非黑箱分解）：
1. `wave_numbers` 根据 `E, V0` 计算外区波数 `k` 与内区波数 `q`，并先做物理可行性检查。  
2. `compute_phase_shifts` 遍历 `l=0..l_max`，逐通道调用 `phase_shift_square_well`。  
3. 在 `phase_shift_square_well` 中，用 `scipy.special.spherical_jn/spherical_yn` 计算 `j_l(ka), y_l(ka)` 及导数，以及 `j_l(qa)` 与导数。  
4. 由 `beta = q*j'_l(qa)/j_l(qa)` 构建内区对数导数，再代入匹配公式得到 `num/den`，用 `atan2(num, den)` 得到 `delta_l`。  
5. `partial_cross_sections` 按 `sigma_l=(4π/k^2)(2l+1)sin^2(delta_l)` 计算每个分波贡献，并在 `total_cross_section` 汇总成 `sigma_tot`。  
6. `scattering_amplitude` 对每个角度计算 `f(theta)`：累加 `(2l+1)e^{i delta_l}sin(delta_l)P_l(cos theta)` 后除以 `k`。  
7. `differential_cross_section` 将每个角度的振幅取模平方，输出 `dσ/dΩ`。  
8. `optical_theorem_residual` 用前向振幅虚部与 `kσ_tot/(4π)` 比较，给出一致性残差并在 `main` 打印完整结果。  

# 渡越辐射 (Transition Radiation)

- UID: `PHYS-0187`
- 学科: `物理`
- 分类: `电动力学`
- 源序号: `188`
- 目标目录: `Algorithms/物理-电动力学-0188-渡越辐射_(Transition_Radiation)`

## R01

本条目实现一个最小可运行的渡越辐射（Transition Radiation, TR）算法 MVP：
- 场景：带电粒子以法向穿越单一平面界面（真空 -> 理想导体极限）。
- 目标：数值计算频谱-角分布 `d^2W/(dω dΩ)`、频带总辐射能量 `W_band`、频带光子数 `N_band`。
- 校验：利用公式可分离结构，验证 `W_band ~ Δω`、`N_band ~ ln(ω_max/ω_min)` 这两条关系。

## R02

渡越辐射的物理本质是：当带电粒子跨越介质边界时，其电磁自场需要在边界条件下重构，重构过程会向外辐射电磁波。它不是由“加速度弯轨”触发，而是由“介质边界不连续”触发，因此在高能粒子探测、TRD（Transition Radiation Detector）中很常见。

## R03

MVP 采用的经典 Ginzburg-Frank 极限模型（法向入射、真空到理想导体、后向半球）：

\[
\frac{d^2W}{d\omega d\Omega}=
\frac{z^2\alpha\hbar}{\pi^2}
\frac{\beta^2\sin^2\theta}{\left(1-\beta^2\cos^2\theta\right)^2}
\]

以及光子数谱角分布：

\[
\frac{d^2N}{d\omega d\Omega}=
\frac{1}{\hbar\omega}\frac{d^2W}{d\omega d\Omega}
=
\frac{z^2\alpha}{\pi^2\omega}
\frac{\beta^2\sin^2\theta}{\left(1-\beta^2\cos^2\theta\right)^2}
\]

其中 `z` 是电荷数、`β=v/c`、`γ=(1-β^2)^(-1/2)`。

## R04

输入/输出定义（由 `demo.py` 内部固定参数给出，无交互）：

输入：
- `gamma`：粒子洛伦兹因子。
- `charge_state_z`：电荷态（单位电荷倍数）。
- `omega_min_rad_s`, `omega_max_rad_s`：积分频带。
- `n_theta`, `n_omega`：离散积分网格精度。

输出：
- 角积分后谱密度 `dW/domega`。
- 总能量 `W_band_2d` 与分离形式 `W_band_sep`。
- 总光子数 `N_band_2d` 与分离形式 `N_band_sep`。
- 角分布峰值角 `theta_peak` 与特征角 `1/gamma` 的对比。

## R05

核心建模假设：
1. 单界面、法向穿越，不考虑多层膜干涉。
2. 理想导体近似，不显式引入频率色散的复介电常数。
3. 只做经典电动力学级别计算，不含量子修正与辐射反作用。
4. 仅计算后向半球积分（`theta in [0, pi/2]`）。
5. 使用数值积分验证，不调用外部黑盒辐射求解器。

## R06

离散化关系：

\[
W_{band} = \int_{\omega_{min}}^{\omega_{max}}\int_{\Omega}
\frac{d^2W}{d\omega d\Omega}\,d\Omega\,d\omega
\]

\[
N_{band} = \int_{\omega_{min}}^{\omega_{max}}\int_{\Omega}
\frac{d^2N}{d\omega d\Omega}\,d\Omega\,d\omega
\]

轴对称下 `dΩ = 2π sin(theta) dtheta`，因此角积分可写为 1D 数值积分。

## R07

算法主线：
1. 从 `gamma` 得到 `beta`。
2. 构造 `theta` 网格与几何频率网格 `omega`（`geomspace`）。
3. 计算角核函数 `K(theta)=β^2 sin^2(theta)/(1-β^2 cos^2(theta))^2`。
4. 得到 `d^2W/(dω dΩ)` 并在角度上积分成 `dW/domega`。
5. 在频率上积分获得 `W_band_2d`。
6. 独立用 `W_band_sep = (omega_max-omega_min) * dW/domega` 做交叉校验。
7. 对光子数执行同样流程并用 `ln` 结构校验。
8. 检查峰值角 `theta_peak ~ 1/gamma`。

## R08

伪代码：

```text
beta <- sqrt(1 - 1/gamma^2)
theta_grid <- linspace(0, pi/2, n_theta)
omega_grid <- geomspace(omega_min, omega_max, n_omega)

K(theta) <- beta^2 sin^2(theta) / (1 - beta^2 cos^2(theta))^2

d2W(theta) <- z^2 alpha hbar / pi^2 * K(theta)
dW_domega <- integrate_theta(d2W(theta) * 2pi sin(theta))
W_band_2d <- integrate_omega(dW_domega)
W_band_sep <- (omega_max - omega_min) * dW_domega

d2N(omega, theta) <- z^2 alpha/(pi^2*omega) * K(theta)
dN_domega(omega) <- integrate_theta(d2N * 2pi sin(theta))
N_band_2d <- integrate_omega(dN_domega)
N_band_sep <- angle_prefactor * ln(omega_max/omega_min)

assert W_band_2d ~= W_band_sep
assert N_band_2d ~= N_band_sep
assert theta_peak ~= 1/gamma
print diagnostics
```

## R09

`demo.py` 对应实现模块：
- `TRConfig`：参数容器。
- `beta_from_gamma`：从 `gamma` 得到 `beta`。
- `angular_kernel`：实现 `K(theta)`。
- `d2w_domega_domega`：能量谱角分布。
- `d2n_domega_domega`：光子数谱角分布。
- `integrate_over_theta`：执行 `2π sin(theta)` 权重角积分。
- `run_mvp`：组织全部数值计算与断言。
- `main`：固定参数运行并打印摘要。

## R10

运行方式：

```bash
cd Algorithms/物理-电动力学-0188-渡越辐射_(Transition_Radiation)
uv run python demo.py
```

脚本无需任何输入，直接输出能量/光子数积分结果、相对误差和角度采样值。

## R11

复杂度分析：
- 角积分成本：`O(n_theta)`。
- 2D 光子数积分成本：`O(n_theta * n_omega)`。
- 空间复杂度：`O(n_theta * n_omega)`（由 `d2n_grid` 主导）。

在默认参数下（`4000 x 2000`）可在普通 CPU 上快速完成。

## R12

数值稳定性与单位：
1. SI 单位统一：`W` 用焦耳，`ω` 用 `rad/s`，`d^2W/(dω dΩ)` 用 `J*s/sr`。
2. `theta=0` 处核函数极限为 0，数值上无发散。
3. 对 `N_band` 使用几何 `omega` 网格，匹配 `1/omega` 结构并降低离散误差。
4. 使用双精度 `float64` 和 `np.trapezoid` 保持可重复性。

## R13

正确性验证策略：
1. `W` 双路径一致性：`W_band_2d` vs `W_band_sep`。
2. `N` 双路径一致性：`N_band_2d` vs `N_band_sep`。
3. 角特征一致性：`theta_peak` 与 `1/gamma` 的相对误差应小。
4. 输出多个角度样本，人工检查分布先增后减并在小角附近达峰。

## R14

局限与边界：
1. 真实材料中介电常数 `epsilon(omega)` 频散会改变谱形，本 MVP 未覆盖。
2. 仅单界面，不含薄膜/多层结构中的干涉增强与抑制。
3. 不包含粒子束发散、入射角分布、有限束斑效应。
4. 对很高 `gamma` 的精细峰结构，需要自适应角网格进一步提升精度。

## R15

可扩展方向：
1. 将理想导体扩展到一般介质（复介电常数）并区分前向/后向辐射。
2. 增加薄膜双界面模型，研究干涉条纹与频带优化。
3. 对 `gamma` 扫描生成 `theta_peak` 与辐射产额缩放律。
4. 用 `pandas` 导出参数扫描表，为后续 TRD 设计提供数据基线。

## R16

工程检查清单：
1. `README.md` 保持 `R01-R18` 全部填写。
2. `demo.py` 无占位符残留，可直接运行。
3. 只使用最小工具栈（NumPy）并显式实现积分流程。
4. 提供断言防止静默错误。
5. 所有改动限制在任务专属目录内。

## R17

参考资料：
1. V. L. Ginzburg and I. M. Frank, *Radiation of a uniformly moving electron due to its transition from one medium into another*.
2. J. D. Jackson, *Classical Electrodynamics* (Transition Radiation 相关章节).
3. M. L. Ter-Mikaelian, *High-Energy Electromagnetic Processes in Condensed Media*.

## R18

源码级算法流程拆解（9 步）：
1. `main()` 构建 `TRConfig`，调用 `run_mvp()`。
2. `run_mvp()` 先做参数合法性检查（`gamma>1`, `omega_min<omega_max` 等）。
3. 调用 `beta_from_gamma()` 计算 `beta`，并生成 `theta/omega` 离散网格。
4. 调用 `angular_kernel()` 计算 `K(theta)`。
5. 调用 `d2w_domega_domega()` 生成 `d^2W/(dω dΩ)`，再用 `integrate_over_theta()` 得到 `dW/domega`，最后在 `omega` 维积分得到 `W_band_2d`。
6. 使用分离表达 `W_band_sep = (omega_max-omega_min) * dW/domega`，执行第一组一致性断言。
7. 调用 `d2n_domega_domega()` 构造 2D 光子数密度，先角积分再频率积分得到 `N_band_2d`。
8. 用 `N_band_sep = angle_prefactor * ln(omega_max/omega_min)` 执行第二组一致性断言，并检查 `theta_peak ~ 1/gamma`。
9. 返回结果字典并在 `main()` 中打印可审计数值摘要和角度采样点。

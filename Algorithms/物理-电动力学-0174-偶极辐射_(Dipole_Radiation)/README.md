# 偶极辐射 (Dipole Radiation)

- UID: `PHYS-0173`
- 学科: `物理`
- 分类: `电动力学`
- 源序号: `174`
- 目标目录: `Algorithms/物理-电动力学-0174-偶极辐射_(Dipole_Radiation)`

## R01

**问题定义**  
构建一个最小可运行算法，计算谐振电偶极子（Hertzian dipole）的远场辐射角分布 \(dP/d\Omega\) 与总辐射功率 \(P\)，并用数值积分验证经典解析公式：
\[
\frac{dP}{d\Omega}\propto \sin^2\theta,\qquad
P=\int \frac{dP}{d\Omega}\,d\Omega.
\]
脚本需非交互运行并输出可审计数值结果。

## R02

**物理背景**  
当电偶极矩随时间振荡 \(p(t)=p_0\cos(\omega t)\) 时，会向外辐射电磁波。  
在远场区（\(r\gg \lambda\)）辐射呈甜甜圈形分布：沿偶极轴（\(\theta=0,\pi\)）最弱，垂直偶极轴（\(\theta=\pi/2\)）最强。  
该问题是电动力学中最基础的辐射模型之一，可作为更复杂天线/散射模型的基线验证。

## R03

**数学模型**  
设偶极矩沿 \(z\) 轴振荡：
\[
\mathbf{p}(t)=p_0\cos(\omega t)\,\hat{\mathbf{z}}.
\]
远场时间平均角功率密度：
\[
\left\langle\frac{dP}{d\Omega}\right\rangle
=\frac{\omega^4 p_0^2}{32\pi^2\epsilon_0 c^3}\sin^2\theta.
\]
总功率解析解：
\[
\langle P\rangle
=\frac{\omega^4 p_0^2}{12\pi\epsilon_0 c^3}.
\]
远场电场分量（用于二次校验）：
\[
E_\theta(r,\theta,t)=
\frac{\mu_0 p_0\omega^2}{4\pi r}\sin\theta\cos\!\left[\omega\!\left(t-\frac{r}{c}\right)\right].
\]

## R04

**MVP 输入/输出定义**  
输入：`demo.py` 内部固定参数（非交互）。  
1. `p0_coulomb_meter`：偶极矩振幅 \(p_0\)（C·m）。  
2. `frequency_hz`：频率 \(f\)（Hz）。  
3. `observation_radius_m`：远场观测半径 \(r\)（m）。  
4. `n_theta, n_phi, n_time`：角度与时间离散分辨率。  

输出：  
1. \(P_{\text{theory}}\) 与两条数值路径计算的 \(P_{\text{numeric}}\)。  
2. 相对误差（解析角分布积分路径、场量平均路径）。  
3. 若断言失败则直接报错；若通过则打印多个角度样本 \(dP/d\Omega\)。

## R05

**建模假设**  
1. 真空环境（\(\epsilon_0,\mu_0,c\) 常数）。  
2. 远场近似成立（忽略近场 \(1/r^2,1/r^3\) 项）。  
3. 偶极子尺寸远小于波长（理想赫兹偶极子）。  
4. 仅考虑单频简谐振荡与时间平均辐射功率。  
5. 采用双精度浮点离散积分，不做符号积分。

## R06

**关键公式与约束**  
1. 角分布公式：
\[
\frac{dP}{d\Omega}=A\sin^2\theta,\quad
A=\frac{\omega^4 p_0^2}{32\pi^2\epsilon_0 c^3}.
\]
2. 球面积分：
\[
P=\int_0^{2\pi}\int_0^\pi
\frac{dP}{d\Omega}\sin\theta\,d\theta\,d\phi.
\]
3. 远场 Poynting 关系（瞬时）：
\[
S_r=\frac{E_\theta^2}{\mu_0 c},\qquad
\frac{dP}{d\Omega}=r^2\langle S_r\rangle_t.
\]
4. 物理约束：\(dP/d\Omega\ge 0\)，且轴向极小、赤道极大。

## R07

**算法思路**  
1. 在 \(\theta,\phi\) 上建立中点离散网格，构造 \(d\Omega=\sin\theta\,d\theta\,d\phi\)。  
2. 按解析式计算每个 \(\theta\) 的 \(dP/d\Omega\)。  
3. 在球面上离散积分得到总功率 \(P_{\text{numeric,analytic}}\)。  
4. 独立地由 \(E_\theta(r,\theta,t)\) 构造瞬时 \(S_r(t)\)，时间平均得到 \(dP/d\Omega\)。  
5. 再次积分得到 \(P_{\text{numeric,field}}\)。  
6. 将两条数值路径分别对比理论总功率并做断言。

## R08

**伪代码**

```text
set constants eps0, mu0, c0
set dipole parameters p0, f, r
build theta mid-point grid and phi grid

omega = 2*pi*f
A = omega^4 * p0^2 / (32*pi^2*eps0*c0^3)
dpdomega(theta) = A * sin(theta)^2

P_num_analytic = sum_over_sphere(dpdomega * dOmega)
P_theory = omega^4 * p0^2 / (12*pi*eps0*c0^3)

build one-period time grid t
E_theta(theta,t) = (mu0*p0*omega^2/(4*pi*r))*sin(theta)*cos(omega*(t-r/c0))
S_r(theta,t) = E_theta^2 / (mu0*c0)
dpdomega_from_fields(theta) = r^2 * mean_t(S_r)
P_num_fields = sum_over_sphere(dpdomega_from_fields * dOmega)

assert P_num_analytic ~= P_theory
assert P_num_fields ~= P_theory
assert dpdomega_from_fields ~= dpdomega
print diagnostics
```

## R09

**实现说明（对应 demo.py）**  
1. `DipoleConfig`：集中管理偶极参数与离散分辨率。  
2. `angular_power_density`：实现解析角功率密度 \(dP/d\Omega\)。  
3. `total_power_theory`：实现闭式总功率公式。  
4. `far_field_e_theta`：显式计算远场 \(E_\theta(\theta,t)\)。  
5. `integrate_total_power_from_domega`：在 \((\theta,\phi)\) 网格上做球面积分。  
6. `main`：组织两条计算路径、断言一致性并打印结果。

## R10

**运行方式**

```bash
cd Algorithms/物理-电动力学-0174-偶极辐射_(Dipole_Radiation)
uv run python demo.py
```

预期：程序无需输入，直接输出总功率对比与角度样本；若出现偏差超过阈值将触发断言。

## R11

**复杂度分析**  
设 \(\theta\) 网格点数为 \(N_\theta\)、\(\phi\) 网格点数为 \(N_\phi\)、时间采样点数为 \(N_t\)。  
1. 解析路径积分复杂度：\(O(N_\theta N_\phi)\)。  
2. 场量路径（含时间平均）复杂度：\(O(N_\theta N_t + N_\theta N_\phi)\)。  
3. 空间复杂度：\(O(N_\theta N_t + N_\theta N_\phi)\)（由中间数组主导）。

## R12

**数值稳定性与单位检查**  
1. 全部采用 SI 制：\(p_0\)(C·m), \(f\)(Hz), \(P\)(W), \(dP/d\Omega\)(W/sr)。  
2. 使用中点网格避开 \(\theta=0,\pi\) 端点处离散误差放大。  
3. 时间平均使用完整一个周期离散采样，抑制相位偏置。  
4. 两条独立路径（解析角分布与场量平均）互验，降低单公式误用风险。

## R13

**验证策略**  
1. **总功率验证 A**：\(\int(dP/d\Omega)\,d\Omega\) 数值积分对比解析 \(P\)。  
2. **总功率验证 B**：由 \(E_\theta\to S_r\to dP/d\Omega\) 的独立路径再积分对比 \(P\)。  
3. **逐角度一致性**：`dpdomega_from_fields` 与 `dpdomega_analytic` 点对点比对。  
4. **形状验证**：归一化分布应与 \(\sin^2\theta\) 形状一致。

## R14

**边界与局限**  
1. 仅适用远场区，不含近场反应功与储能项。  
2. 未考虑介质损耗、色散、边界反射与多极矩耦合。  
3. 单偶极单频模型，不覆盖宽带激励与瞬态脉冲。  
4. 未实现全波数值求解器（如 FDTD/FEM），仅做解析公式的 MVP 验证。

## R15

**可扩展方向**  
1. 引入有损介质参数，比较真空与介质中辐射效率变化。  
2. 扩展到短偶极天线输入阻抗与辐射电阻估计。  
3. 支持多偶极阵列，研究方向图叠加与波束赋形。  
4. 用 `pandas` 导出角分布表格，配合后续可视化或回归测试。

## R16

**工程化检查清单**  
1. `README.md` 的 `R01-R18` 均已填写。  
2. `demo.py` 不含占位符，且可 `uv run python demo.py` 非交互运行。  
3. 核心公式在代码中有一一对应函数。  
4. 断言覆盖总功率与角分布一致性。  
5. 所有改动仅位于本任务专属目录。

## R17

**参考资料**  
1. D. J. Griffiths, *Introduction to Electrodynamics*, 4th ed., dipole radiation chapter.  
2. J. D. Jackson, *Classical Electrodynamics*, 3rd ed., radiation from localized sources.  
3. Balanis, *Antenna Theory*, short dipole radiation pattern (工程视角补充)。

## R18

**源码级算法流程拆解（3-10步）**  
1. `main()` 初始化 `DipoleConfig` 并创建 \(\theta\) 中点网格。  
2. 调用 `angular_power_density` 计算解析 \(dP/d\Omega(\theta)\)。  
3. 调用 `integrate_total_power_from_domega` 在球面离散积分得到 `p_numeric_analytic`。  
4. 调用 `total_power_theory` 计算闭式总功率 `p_theory`。  
5. 构造一个周期时间网格，调用 `far_field_e_theta` 生成 \(E_\theta(\theta,t)\)。  
6. 由 `S_r=E_\theta^2/(\mu_0 c)` 求时间平均，再乘 \(r^2\) 得 `dpdomega_from_fields`。  
7. 对 `dpdomega_from_fields` 再做球面积分得到 `p_numeric_fields`。  
8. 使用 `np.testing.assert_allclose` 对三类一致性执行阈值校验。  
9. 打印理论值、数值值、相对误差和多个角度样本，形成可审计输出。

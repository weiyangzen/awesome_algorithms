# 电四极辐射 (Electric Quadrupole Radiation)

- UID: `PHYS-0175`
- 学科: `物理`
- 分类: `电动力学`
- 源序号: `176`
- 目标目录: `Algorithms/物理-电动力学-0176-电四极辐射_(Electric_Quadrupole_Radiation)`

## R01

电四极辐射是多极展开中高于电偶极、磁偶极的一阶辐射机制。该条目的目标是实现一个最小可运行 MVP：对给定谐振电四极矩张量模式，计算时间平均角功率密度 `dP/dΩ` 和总辐射功率 `P`，并通过数值积分验证与解析公式一致。

## R02

问题定义：
- 输入（demo 内固定参数）：四极矩振幅 `Q0`（C·m²）、频率 `f`（Hz）、角网格规模 `n_theta/n_phi`、时间采样 `n_time`。
- 输出：
  - 解析总功率 `P_theory`
  - 由角分布数值积分得到的 `P_numeric`
  - 两条角分布构造路径之间的一致性误差
  - 若误差超阈值则断言失败。

## R03

本 MVP 采用轴对称迹零（traceless）模式：

`Q(t) = Q0 cos(ωt) * diag(-1/2, -1/2, 1)`

其中 `ω = 2πf`。该模式满足 `Tr(Q)=0`，是电四极辐射常用基准模式，角分布关于方位角 `φ` 对称。

## R04

核心总功率公式（SI 制，时间平均）采用电四极矩不变量表达：

`<P> = (1 / (40π ε0 c^5)) * Σ_ij <(Q'''_ij)^2>`

对谐振 `Q_ij(t)=A_ij cos(ωt)`，有

`<(Q'''_ij)^2> = (ω^6 A_ij^2)/2`，

因此

`<P> = (ω^6 / (80π ε0 c^5)) * Σ_ij A_ij^2`。

## R05

对本条目模式 `A = Q0 * diag(-1/2,-1/2,1)`：

`Σ_ij A_ij^2 = (3/2) Q0^2`

故解析总功率为：

`<P> = (3 ω^6 Q0^2) / (80π ε0 c^5)`。

这是 demo 中用于最终校验的闭式基准值。

## R06

角分布采用两种等价构造：

1. 轴对称解析角公式：
`<dP/dΩ> = (9 ω^6 Q0^2 / (128 π^2 ε0 c^5)) * sin^2θ * cos^2θ`

2. 张量核公式（更通用）：
`<dP/dΩ> = (1 / (16 π^2 ε0 c^5)) * <| n × (n × (Q''' · n)) |^2>`

其中 `n` 是观测方向单位向量。

## R07

算法思路：
1. 建立 `θ-φ` 中点网格并构造球面元 `dΩ = sinθ dθ dφ`。
2. 按解析角公式得到 `dP/dΩ(θ)`。
3. 按张量核公式得到 `dP/dΩ(θ,φ)`，其中时间平均因子由 `sin^2(ωt)` 离散平均给出。
4. 对两条角分布分别球面积分得到总功率。
5. 与解析总功率 `P_theory` 做一致性断言。

## R08

伪代码：

```text
set constants eps0, mu0, c
set config (Q0, f, n_theta, n_phi, n_time)
omega = 2*pi*f

build theta-phi midpoint grid
build dOmega weights

dp_analytic(theta) = C1 * sin^2(theta)*cos^2(theta)
P_num_analytic = integrate(dp_analytic over sphere)

shape = diag(-1/2, -1/2, 1)
kernel(theta,phi) = |n x (n x (shape*n))|^2
avg_phase2 = mean_t[sin^2(omega*t)]
dp_tensor = C2 * (omega^6*Q0^2) * avg_phase2 * kernel
P_num_tensor = integrate(dp_tensor over sphere)

P_theory = 3*omega^6*Q0^2/(80*pi*eps0*c^5)
assert P_num_analytic ~= P_theory
assert P_num_tensor ~= P_theory
assert dp_analytic(theta,phi) ~= dp_tensor(theta,phi)
print diagnostics
```

## R09

`demo.py` 实现要点：
- `QuadrupoleConfig`：集中定义参数。
- `quadrupole_shape_tensor()`：返回迹零模式张量。
- `angular_power_density_axisymmetric()`：实现解析角分布。
- `angular_power_density_from_tensor_kernel()`：实现方向向量-张量核构造。
- `integrate_over_sphere()`：对 `dP/dΩ` 做离散球面积分。
- `main()`：执行两条路径、断言、输出。

## R10

运行方式：

```bash
cd Algorithms/物理-电动力学-0176-电四极辐射_(Electric_Quadrupole_Radiation)
uv run python demo.py
```

程序无交互输入，直接输出理论值、数值值和相对误差。

## R11

复杂度分析：
- 设角网格为 `Nθ, Nφ`，时间采样为 `Nt`。
- 角分布构造复杂度：`O(Nθ Nφ)`。
- 时间平均仅计算 `sin^2` 均值：`O(Nt)`。
- 总复杂度：`O(Nθ Nφ + Nt)`。
- 空间复杂度：`O(Nθ Nφ)`。

## R12

数值与单位检查：
- 单位：`Q0` (C·m²), `ω` (s⁻¹), `dP/dΩ` (W/sr), `P` (W)。
- 中点网格避免在 `θ=0,π` 端点采样导致离散误差放大。
- 使用双精度浮点与 `numpy.testing.assert_allclose` 做阈值校验。

## R13

验证策略：
1. `P_num_analytic` vs `P_theory`。
2. `P_num_tensor` vs `P_theory`。
3. `dP/dΩ` 的两条路径点对点比对。
4. 额外检查方位角不变性（轴对称模式下 `φ` 方向应近似常值）。

## R14

边界与局限：
- 仅覆盖经典、远场、非介质吸收场景。
- 仅演示单频谐振四极矩，不含瞬态脉冲。
- 未引入更高阶多极项（八极等）和介质边界条件。

## R15

可扩展方向：
- 支持任意对称迹零 `Q_ij` 模式与旋转坐标系。
- 引入 `pandas` 导出 `θ,φ,dP/dΩ` 表格用于可视化。
- 对接数值电磁解算器（FDTD/FEM）做交叉验证。

## R16

工程化检查清单：
- `README.md` 保留并填充 `R01-R18`。
- `demo.py` 无占位符，且可直接运行。
- 输出包含可审计指标和误差。
- 改动仅限本任务专属目录。

## R17

参考资料：
1. J. D. Jackson, *Classical Electrodynamics*（多极辐射章节）。
2. D. J. Griffiths, *Introduction to Electrodynamics*（辐射与多极展开）。
3. L. D. Landau & E. M. Lifshitz, *The Classical Theory of Fields*（辐射多极矩表述）。

## R18

`demo.py` 源码级流程拆解（9 步）：
1. 在 `main()` 初始化 `QuadrupoleConfig` 与常数 `ε0, μ0, c`。
2. 用 `build_angular_grid()` 生成 `θ/φ` 中点网格和积分权重 `dΩ`。
3. 调用 `angular_power_density_axisymmetric()` 计算解析角分布。
4. 调用 `integrate_over_sphere()` 得到 `P_num_analytic`。
5. 调用 `angular_power_density_from_tensor_kernel()`：先构造方向向量 `n`，再计算 `n × (n × (Q'''·n))` 的平方模并做时间平均。
6. 再次调用 `integrate_over_sphere()` 得到 `P_num_tensor`。
7. 调用 `total_power_theory_axisymmetric()` 计算闭式 `P_theory`。
8. 使用 `np.testing.assert_allclose` 做三类校验：两条总功率与理论值一致、两条角分布逐点一致。
9. 打印配置、三种功率结果、相对误差与多个角度样本，形成可复现实验输出。

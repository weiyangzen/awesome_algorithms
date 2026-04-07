# 费米-狄拉克统计 (Fermi-Dirac Statistics)

- UID: `PHYS-0034`
- 学科: `物理`
- 分类: `量子统计`
- 源序号: `34`
- 目标目录: `Algorithms/物理-量子统计-0034-费米-狄拉克统计_(Fermi-Dirac_Statistics)`

## R01

费米-狄拉克统计用于描述不可分辨费米子（如电子）在热平衡下的平均占据数：

`f(E) = 1 / (exp((E-μ)/(k_B T)) + 1)`。

其核心物理特征是泡利不相容原理导致的占据上限：每个单粒子态平均占据数都满足 `0 <= f(E) <= 1`。在低温下，分布在费米能附近呈现陡峭“台阶”；温度升高后台阶被热展宽。

本条目实现一个最小可运行数值实验：在固定总粒子数约束下，求理想三维费米气体的 `μ(T)`，并输出占据数与热力学量。

## R02

采用无量纲单位 `k_B=1`，设三维自由粒子态密度近似为：

`g(E) = A * sqrt(E)`，`E >= 0`。

固定粒子数条件：

`n = ∫_0^∞ g(E) f(E; μ, T) dE`。

能量密度：

`u = ∫_0^∞ E g(E) f(E; μ, T) dE`。

当 `T=0` 时，`f(E)` 退化为阶跃函数，满足

`n = A * ∫_0^{E_F} sqrt(E) dE = A*(2/3)*E_F^(3/2)`，

据此可由给定 `E_F` 与 `n` 反推系数 `A`。

## R03

MVP 目标：
- 固定 `E_F=1`、`n=1`，先由 `T=0` 归一化得到 `A`；
- 对温度序列 `T=[0.05, 0.10, 0.20, 0.40, 0.80, 1.20]`，数值求解 `μ(T)`；
- 输出两张表：
  - 热力学汇总：`T, μ, μ_sommerfeld, density, energy_density`；
  - 代表性能级占据数：`f(E=...)`。

这样既覆盖费米-狄拉克分布的核心公式，也展示“粒子数守恒 -> 化学势反求”的常见数值流程。

## R04

数值方法：
1. 对积分做变量替换 `E=t^2`，将半无限区间积分改写为 `t in [0, +∞)` 的平滑形式；
2. 用 `scipy.integrate.quad` 执行自适应求积；
3. 通过 `scipy.optimize.brentq` 解方程 `n(T,μ)-n_target=0`；
4. 用 `scipy.special.expit` 计算 `1/(1+exp(x))`，避免指数溢出；
5. 追加一致性检查（占据数边界、`μ(T)` 单调性、高能 MB 极限）。

## R05

`demo.py` 默认参数（`FDConfig`）：
- `fermi_energy = 1.0`
- `total_density = 1.0`
- `temperatures = (0.05, 0.10, 0.20, 0.40, 0.80, 1.20)`
- `energy_levels = (0.10, 0.30, 0.80, 1.00, 1.40, 2.20)`
- 积分容差：`quad_epsabs = 1e-9`, `quad_epsrel = 1e-8`
- 夹根扩展上限：`max_bracket_expand = 80`

脚本无需命令行参数和交互输入，可直接运行。

## R06

代码结构：
- `FDConfig`：集中管理模型参数；
- `dos_prefactor_from_fermi_energy`：按 `T=0` 关系归一化 `A`；
- `fermi_dirac_occupation`：计算占据数；
- `number_density` / `energy_density`：数值积分计算 `n` 与 `u`；
- `solve_mu_for_fixed_density`：在固定粒子数下反解 `μ(T)`；
- `sommerfeld_mu_approx`：给出低温近似 `μ` 供对照；
- `analyze`：批量生成结果表；
- `run_consistency_checks`：执行物理一致性断言；
- `main`：汇总打印。

## R07

伪代码：

```text
读取配置 cfg
A <- 由 E_F 与 n 计算态密度系数

for each T in cfg.temperatures:
  解方程 n(T, mu) = n_target 得到 mu
  计算 density = n(T, mu)
  计算 energy_density = u(T, mu)
  计算 Sommerfeld 近似 mu_sommerfeld
  对给定能级计算 f(E)
  写入汇总表与占据数表

执行一致性检查
打印两张表
```

## R08

复杂度（`M` 个温度点，`Q` 次根迭代，每次积分代价 `K_quad`）：
- 单次积分近似 `O(K_quad)`；
- 单温度求 `μ` 约 `O(Q * K_quad)`；
- 总时间复杂度约 `O(M * Q * K_quad)`；
- 空间复杂度 `O(M)`（主要是结果表）。

默认参数规模很小，CPU 下通常秒级完成。

## R09

数值稳定性处理：
- 使用 `expit` 代替直接 `exp` 公式，降低溢出风险；
- 上限 `t_upper` 随 `T`、`μ` 自适应放大，确保高能尾部覆盖；
- 夹根时动态扩展上下界，避免根不在初始区间；
- 全流程无随机数，结果可复现。

## R10

与替代路径对比：
- 纯解析：可借助费米积分与多重对数函数，但对“如何数值反求 `μ`”展示不足；
- 全微观仿真：如分子动力学/蒙特卡洛，成本高且超出最小示例需求；
- 当前实现：保留关键物理关系并将算法流程控制在可读、可运行的 MVP 范围内。

## R11

调参建议：
- 想更细看低温行为：在 `temperatures` 增加 `0.02, 0.03` 等点；
- 想提高积分精度：减小 `quad_epsabs/quad_epsrel`；
- 想更快运行：减少温度点或放宽容差；
- 若出现夹根失败：提高 `max_bracket_expand`，或增大积分上限估计策略。

## R12

实现要点：
- `A` 的归一化来自 `T=0` 封闭解，避免人为常数误差；
- `solve_mu_for_fixed_density` 利用 `n(T,μ)` 对 `μ` 的单调性，用 `brentq` 稳定求根；
- `sommerfeld_mu_approx` 仅用于低温近似对照，不参与主求解；
- 一致性检查包括：
  - `density` 接近目标 `n=1`；
  - 占据数严格位于 `[0,1]`；
  - 高温高能区接近 Maxwell-Boltzmann 尾部。

## R13

运行方式（在仓库根目录）：

```bash
uv run python "Algorithms/物理-量子统计-0034-费米-狄拉克统计_(Fermi-Dirac_Statistics)/demo.py"
```

或先进入目录再运行：

```bash
cd "Algorithms/物理-量子统计-0034-费米-狄拉克统计_(Fermi-Dirac_Statistics)"
uv run python demo.py
```

## R14

输出解释：
- `mu`：数值反解得到的化学势；
- `mu_sommerfeld`：低温近似值；
- `delta_mu`：数值解与近似解差值；
- `density`：回代得到的粒子数密度（应接近 1）；
- `energy_density`：能量密度；
- `f(E=...)`：指定能量点的平均占据数。

典型趋势：
- `T` 升高时 `μ` 下降；
- 低能级占据数较高、高能级占据数较低；
- 在高能尾部，FD 分布接近 MB 指数衰减。

## R15

常见问题排查：
- `Failed to bracket chemical potential root`：
  - 增大 `max_bracket_expand`；
  - 检查是否误把 `temperature` 设为 0 或负数。
- 密度断言失败：
  - 收紧积分容差；
  - 增大积分上限估计中的系数。
- 出现极端慢速：
  - 减少温度点数；
  - 放宽容差用于快速验证，再做精细计算。

## R16

可扩展方向：
- 使用 `scipy.special` 的费米积分函数（若可用）做交叉验证；
- 从能量密度继续推导比热并绘制 `C_V(T)`；
- 引入外场或有限体积离散能级，研究偏离理想模型的效应；
- 扩展到二维/一维态密度，比较维度依赖行为。

## R17

边界与限制：
- 当前模型是理想费米气体，不含相互作用；
- 态密度采用连续近似 `A*sqrt(E)`，忽略材料能带细节；
- 低温极限下仍是有限温近似，不等价于严格 `T=0`；
- 该 MVP 适用于算法与教学演示，不用于高精度实验拟合。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main()` 创建 `FDConfig`，固定 `E_F`、`n`、温度和采样能级。  
2. `analyze()` 调 `dos_prefactor_from_fermi_energy()`，由 `n=A*(2/3)E_F^(3/2)` 算出 `A`。  
3. 对每个温度 `T`，调用 `solve_mu_for_fixed_density()` 构造残差 `number_density(mu)-n_target`。  
4. `solve_mu_for_fixed_density()` 先动态扩展 `mu_low/mu_high` 直到夹住根，再用 `scipy.optimize.brentq` 求 `μ(T)`。  
5. `number_density()` 与 `energy_density()` 通过 `E=t^2` 变量替换，把积分交给 `scipy.integrate.quad` 自适应求积。  
6. 在积分核中，`fermi_dirac_occupation()` 使用 `scipy.special.expit` 稳定计算 `f(E)`，避免直接指数溢出。  
7. 每个温度点生成两类输出：汇总量（`μ`, `density`, `energy_density`）和离散能级占据数 `f(E_i)`。  
8. `run_consistency_checks()` 做边界/极限断言（`0<=f<=1`、密度守恒、`μ(T)` 下降、FD 高能尾部逼近 MB），通过后统一打印结果。  

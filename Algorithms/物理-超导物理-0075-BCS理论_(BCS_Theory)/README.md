# BCS理论 (BCS Theory)

- UID: `PHYS-0075`
- 学科: `物理`
- 分类: `超导物理`
- 源序号: `75`
- 目标目录: `Algorithms/物理-超导物理-0075-BCS理论_(BCS_Theory)`

## R01

BCS 理论（Bardeen-Cooper-Schrieffer）给出常规超导的微观机制：
- 费米面附近电子通过有效吸引相互作用形成 Cooper 对；
- 基态出现能隙 `Delta(T)`，导致单粒子激发谱 `E = sqrt(xi^2 + Delta^2)`；
- 在临界温度 `Tc` 处能隙闭合，系统从超导态转入正常态。

本目录实现一个“可运行、可追踪”的最小数值版 BCS 能隙方程求解器。

## R02

MVP 目标：
1. 数值求解线性化 BCS 方程，得到临界温度 `Tc`；
2. 在多个温度点求解有限温度能隙 `Delta(T)`；
3. 验证弱耦合极限中的经典关系：`2Delta(0)/Tc ~= 3.53`；
4. 输出结构化表格，展示温度-能隙-残差趋势。

实现强调“从源代码看清算法流”，不把核心方程交给黑盒求解器。

## R03

本实现使用各向同性 s-wave、弱耦合、常密度状态近似：

- 维度无关化参数：`lambda = V*N(0)`；
- Debye 截断：`omega_D`；
- 能隙方程：

`1/lambda = int_0^{omega_D} [tanh(E/(2T))/E] dxi`, `E=sqrt(xi^2 + Delta^2)`。

线性化极限（`Delta -> 0`）给出 `Tc` 方程：

`1/lambda = int_0^{omega_D} [tanh(xi/(2T))/xi] dxi`。

## R04

理论对照（弱耦合近似）：

- `Tc ~= 1.13 * omega_D * exp(-1/lambda)`
- `Delta(0) ~= 2 * omega_D * exp(-1/lambda)`
- `2Delta(0)/Tc ~= 3.53`

`demo.py` 同时输出“数值解”和“弱耦合近似”用于交叉检查。

## R05

数值策略（非黑盒）：

1. 在 `xi in [0, omega_D]` 上构造“近 0 更密”的几何网格；
2. 用 `numpy.trapezoid` 显式做能量积分；
3. 手写二分法 `bisect_root` 求解 `Tc` 与 `Delta(T)` 的根；
4. 对每个温度点计算残差 `f(Delta,T)=1/lambda-I(Delta,T)`，保证求解可验证。

核心求根与积分过程都在源码中展开，可逐行追踪。

## R06

`demo.py` 主要输出字段：

- `temperature`：温度 `T`
- `reduced_T`：归一化温度 `T/Tc`
- `gap_delta`：能隙 `Delta(T)`
- `gap_over_cutoff`：`Delta/omega_D`
- `quasiparticle_min_energy`：准粒子最小激发能（等于 `Delta`）
- `condensation_energy_density`：近似凝聚能密度 `-0.5*Delta^2`
- `residual_at_solution`：方程残差，检查数值收敛

## R07

算法优点：

- 保留 BCS 方程的核心物理结构；
- 依赖少（`numpy + pandas`），可快速复现；
- 每一步计算都可解释，适合教学与验证。

局限：

- 未包含 Eliashberg 频率依赖、强耦合修正；
- 未包含非常规配对（d-wave/p-wave）与各向异性费米面；
- 仅做静态平衡，不处理时变超导动力学。

## R08

前置知识：

- 费米统计与 Cooper 配对概念；
- 有限温度下 `tanh(E/2T)` 的含义；
- 一维数值积分和二分法根求解。

运行环境：

- Python `>=3.10`
- `numpy`
- `pandas`

## R09

适用场景：

- BCS 理论入门教学；
- 快速做 `Tc` 与 `Delta(T)` 的数量级估算；
- 作为更复杂超导模型（多带、强耦合）前的基线实现。

不适用场景：

- 需要实验级精度拟合材料参数；
- 需要包含频率依赖自能、各向异性或杂质散射效应；
- 需要第一性原理电子结构输入的高保真模拟。

## R10

正确性直觉：

1. BCS 自洽方程根对应允许的平衡能隙；
2. `T` 升高会削弱 `tanh(E/2T)` 有效配对核，从而压低 `Delta`；
3. 到 `Tc` 时非零根消失，仅剩 `Delta=0` 正常态；
4. 若低温 `Delta`、`Tc` 关系与弱耦合比例一致，说明模型与实现链路基本正确。

## R11

数值稳定性设计：

- `xi=0` 处在线性化积分使用极限值 `1/(2T)`，避免 `0/0`；
- 能量网格采用几何分布，专门提升小 `xi` 区域分辨率；
- 二分法强制夹根，避免牛顿法在临界附近发散；
- 保留 `residual_at_solution` 直接检查收敛质量。

## R12

关键参数（`BCSConfig`）：

- `coupling_lambda`：耦合强度（默认 `0.30`）
- `debye_cutoff`：Debye 截断（默认 `1.0`）
- `n_energy_grid`：能量网格点数（默认 `6000`）
- `n_temperatures`：温度采样点数（默认 `32`）
- `temperature_min_factor` / `temperature_max_factor`：扫描范围相对 `Tc` 的比例
- `root_tol` / `max_bisect_iter`：二分精度控制

调参建议：先固定 `lambda`，再提高 `n_energy_grid` 检查收敛。

## R13

保证说明：

- 近似比保证：N/A（物理方程求解问题，不是近似优化问题）。
- 数值求解保证：
  - 在已夹根区间内，二分法必然收敛；
  - 残差列可验证根是否达到设定容差。

工程上，脚本无交互输入且支持直接自动化运行。

## R14

常见失败模式：

1. 网格太粗，导致 `Tc` 偏差大；
2. 临界点附近使用不稳的单点迭代，根搜索震荡；
3. 线性化积分未正确处理 `xi->0` 极限，出现 `nan`；
4. 求根区间未夹根就直接迭代，导致错误解。

本实现通过“近零加密网格 + 极限值处理 + 夹根二分法”规避这些问题。

## R15

可扩展方向：

1. 引入能量依赖耦合 `V(xi,xi')` 与离散化积分方程；
2. 扩展到多带超导（多 `Delta_i` 联立自洽）；
3. 加入磁场下 Zeeman/轨道破缺对 `Tc` 的影响；
4. 与实验隧穿谱数据做反演拟合；
5. 引入 Eliashberg 方程比较强耦合修正。

## R16

相关主题：

- Ginzburg-Landau 理论（近 `Tc` 的序参量展开）
- Bogoliubov 变换与准粒子谱
- Anderson 定理与杂质效应
- Eliashberg 强耦合超导理论
- BEC-BCS 跨越问题

## R17

运行方式（无交互）：

```bash
cd "Algorithms/物理-超导物理-0075-BCS理论_(BCS_Theory)"
uv run python demo.py
```

交付核对：

- `README.md` 的 `R01-R18` 已全部填写；
- `demo.py` 可直接运行并打印结果；
- `meta.json` 与任务元数据一致；
- 目录内容可独立验证。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main()` 创建 `BCSConfig`，并调用 `build_energy_grid()` 构造近零加密能量网格。  
2. `solve_critical_temperature()` 构造线性化残差 `f(T)=1/lambda-I(Delta=0,T)`。  
3. 在 `T` 轴上先做夹根，再用手写 `bisect_root()` 求 `Tc`。  
4. `run_temperature_scan()` 生成 `T` 网格，对每个温度调用 `solve_gap_at_temperature()`。  
5. `solve_gap_at_temperature()` 先检查 `f(0,T)`：若非负则返回 `Delta=0`（正常态）；若负则继续找非零根。  
6. 对非零根分支，代码自适应扩大 `Delta` 上界直到夹住符号变化，再用 `bisect_root()` 求 `Delta(T)`。  
7. `pairing_integral()` 在每次残差评估中显式计算 `int tanh(E/2T)/E dxi`，并对 `xi=0` 采用解析极限防止数值奇异。  
8. `run_consistency_checks()` 验证单调性与比例关系（含 `2Delta(0)/Tc`），`main()` 最后打印完整数据表和校验指标。  

第三方库边界：`numpy` 仅用于数组和积分求和，`pandas` 仅用于结果表格；核心物理方程、夹根策略和二分求解全部在源码中实现。

# 恒压系综 (NPT Ensemble)

- UID: `PHYS-0331`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `338`
- 目标目录: `Algorithms/物理-统计力学-0338-恒压系综_(NPT_Ensemble)`

## R01

恒压系综（NPT, isothermal-isobaric ensemble）描述的是粒子数 `N`、外压 `P`、温度 `T` 固定，而体积 `V` 可以涨落的平衡体系。  
在经典统计力学中，构型概率权重可写为：

`pi(r^N, V) ∝ exp[-beta * (U(r^N; V) + P V)]`

其中 `beta = 1/(k_B T)`，`U` 为势能。由于体积变化会带来坐标积分测度变化，实际 Metropolis 接受率中会出现 `N ln(V'/V)` 项。

## R02

NPT 系综是从理论到工程都非常常用的工作系综：

- 理论上，它对应热力学势 `G = H - TS`（吉布斯自由能）框架；
- 分子模拟中，它能直接预测平衡密度、体积涨落、等温压缩性；
- 对比 NVT，NPT 更适合“给定温压、求结构/密度”的实验条件。

本目录 MVP 的重点是把 NPT 的核心接受率写清楚并跑通，而不是构建大规模高性能分子模拟平台。

## R03

本实现采用最小但诚实的 `Metropolis Monte Carlo`：

- 体系：二维 Lennard-Jones 粒子流体（周期边界）；
- move 1：粒子位移提案（固定体积）；
- move 2：各向同性体积提案（坐标按比例缩放）；
- 输出：体积、密度、每粒子势能、每粒子焓、接受率；
- 验证：在同温下比较两组压力，检查高压是否导致更小平均体积。

## R04

关键公式（采用约化单位 `k_B = 1`）：

1. 粒子位移接受率（`V` 不变）：
   - `A_disp = min(1, exp[-beta * DeltaU])`
2. 体积变换 `V -> V'`、`r_i -> s r_i`，`s = (V'/V)^(1/d)`：
   - `A_vol = min(1, exp[-beta*(DeltaU + P*DeltaV) + N*ln(V'/V)])`
3. Lennard-Jones 势（截断并平移）：
   - `u(r)=4*epsilon*((sigma/r)^12-(sigma/r)^6)-u(rc), r<rc`
   - `u(r)=0, r>=rc`

其中 `N ln(V'/V)` 来自坐标缩放后的雅可比因子，是 NPT 算法和 NVT 算法的关键差异之一。

## R05

复杂度分析（单次 sweep）：

- `N` 次粒子位移尝试，每次局部能量差为 `O(N)`，合计 `O(N^2)`；
- 1 次体积尝试，需重算总能量，复杂度 `O(N^2)`；
- 因此单 sweep 总复杂度 `O(N^2)`。

总复杂度约为 `O(S * N^2)`（`S` 为总 sweep 数），空间复杂度为 `O(N)`（粒子坐标与采样记录）。

## R06

`demo.py` 的主要输出包括：

- `mean_volume`：平均体积；
- `mean_density`：平均密度 `N/<V>`；
- `mean_energy_per_particle`：平均每粒子势能；
- `mean_enthalpy_per_particle`：平均每粒子焓 `(U+PV)/N`；
- `particle_acceptance`、`volume_acceptance`：两类 move 的接受率。

脚本默认跑两组压力（`0.5` 与 `1.4`），并通过断言检查：高压组平均体积更小、平均密度更高。

## R07

优点：

- 代码完整覆盖 NPT-MC 核心流程（位移 + 体积 move）；
- 接受率公式显式实现，可追踪每个物理项来源；
- 运行时间短，便于教学和快速验证。

局限：

- 只做二维、各向同性盒子涨落；
- 未引入长程修正、邻居表、并行优化；
- 结果是教学级/验证级，不是发表级精度。

## R08

前置知识：

- 系综理论（NVT/NPT 区别）；
- Metropolis-Hastings 接受-拒绝机制；
- Lennard-Jones 势与周期边界条件（minimum image）。

运行环境：

- Python `>=3.10`
- `numpy`
- `pandas`

## R09

适用场景：

- 学习 NPT 系综如何在代码层落地；
- 对比不同压力下的平衡体积与密度；
- 在更大分子模拟框架前做方法学原型验证。

不适用场景：

- 大规模高精度材料模拟；
- 需要严格误差条估计与有限尺寸外推；
- 需要复杂势函数（电荷、键角、多体势）与高性能实现。

## R10

正确性直觉：

1. 位移 move 保持体积不变，遵循玻尔兹曼因子 `exp(-beta*DeltaU)`；
2. 体积 move 需要同时平衡 `DeltaU`、`P*DeltaV` 与测度项 `N ln(V'/V)`；
3. 若 NPT 接受率实现正确，则提高外压会抑制大体积态权重；
4. 因而在同温下应观察到 `<V>` 随 `P` 增大而降低。

`demo.py` 的断言正是对第 4 点进行最小可执行检验。

## R11

数值稳定性与工程细节：

- 初态使用规则晶格，降低粒子重叠造成的极端能量；
- Lennard-Jones 使用截断平移势，避免势能在截断点不连续；
- `volume_step` 设为中等幅度，平衡探索能力与接受率；
- 使用固定随机种子，保证输出可复现；
- 采用对数接受准则 `log(u) < min(0, logA)`，减少下溢风险。

## R12

关键参数（`NPTConfig`）说明：

- `n_particles`：粒子数，影响统计波动和计算量；
- `temperature`：温度，控制热涨落强度；
- `pressure`：外压，主导平衡体积水平；
- `move_step`：粒子位移步长，过小混合慢、过大接受率低；
- `volume_step`：体积提案幅度，影响体积空间采样效率；
- `burn_in_sweeps` / `sample_sweeps` / `sample_interval`：决定热化长度与采样质量。

调参经验：若接受率长期低于 `~0.1`，先减小 `move_step` 或 `volume_step`。

## R13

理论保证类型：

- 近似比保证：N/A（非优化问题）。
- 随机成功率保证：无封闭形式下界（MCMC 混合速度依赖模型与参数）。

本 MVP 的可执行保证：

- 不需要交互输入即可运行；
- 输出包含关键统计量与接受率；
- 内置断言验证了一个核心物理趋势（高压 -> 小体积）。

## R14

常见失效模式：

1. 忘记 `N ln(V'/V)` 项，导致采样分布偏差；
2. 周期边界最小镜像实现错误，造成能量异常；
3. 步长过大，接受率接近 0，链几乎不动；
4. 热化不足就采样，统计量偏向初始态；
5. 粒子重叠导致 `r -> 0`，势能发散。

本实现通过晶格初态、截断势和平衡步数来降低上述风险。

## R15

可扩展方向：

1. 引入邻居表/Verlet list，把位移 move 加速到近似 `O(N)`；
2. 扩展到三维 `d=3`，并比较文献中的 LJ 状态方程；
3. 增加 block averaging / autocorrelation 分析输出误差条；
4. 增加可视化（体积时间序列、密度直方图）；
5. 结合 NVT/NPT 对照实验展示压强耦合效应。

## R16

相关主题：

- NVT 与 NPT 系综变换；
- Gibbs 集综与相平衡采样；
- 分子动力学中的 barostat（Berendsen、Parrinello-Rahman 等）；
- 体积涨落与等温压缩性 `kappa_T` 的估计。

## R17

运行方式（无交互）：

```bash
cd Algorithms/物理-统计力学-0338-恒压系综_(NPT_Ensemble)
uv run python demo.py
```

交付核对：

- `README.md` 的 `R01-R18` 均已填写；
- `demo.py` 为可运行 MVP；
- `meta.json` 与任务元数据保持一致；
- 目录可独立验证。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 定义两组压力（低压 `0.5`、高压 `1.4`），分别构造 `NPTConfig` 并调用 `run_npt_simulation`。  
2. `run_npt_simulation` 依据初始密度生成盒长，使用 `init_lattice_positions` 初始化粒子，并用 `total_energy` 计算初始势能。  
3. 在每个 sweep 中，执行 `N` 次 `attempt_particle_move`：对单粒子做随机位移，调用 `local_energy_of_particle` 计算 `DeltaU`，按 `exp(-beta*DeltaU)` 决定接受。  
4. 每个 sweep 再执行 1 次 `attempt_volume_move`：提议 `V'`，按比例缩放坐标和盒长，重算 `trial_energy`。  
5. `attempt_volume_move` 用 NPT 对数接受率 `-beta*(DeltaU + P*DeltaV) + N*ln(V'/V)` 进行接受/拒绝，并在接受时同步更新坐标、盒长、总能量。  
6. 热化后按 `sample_interval` 记录 `volume`、`density`、`energy_per_particle`、`enthalpy_per_particle`，最终整理为 `pandas.DataFrame`。  
7. 由采样数据计算 `RunSummary`（均值统计 + 两类接受率），`print_summary_table` 以表格形式输出。  
8. `main` 做自校验断言：样本数充足、接受率在合理区间，且高压组 `mean_volume` 小于低压组，验证 NPT 的核心物理趋势。  

第三方库说明：`numpy` 仅用于数值数组与随机数，`pandas` 仅用于结果汇总展示；NPT 采样流程（提案、能量差、接受率、统计）均在源码中逐步实现，无黑盒求解器。

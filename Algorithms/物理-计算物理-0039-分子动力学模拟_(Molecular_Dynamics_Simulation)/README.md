# 分子动力学模拟 (Molecular Dynamics Simulation)

- UID: `PHYS-0039`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `39`
- 目标目录: `Algorithms/物理-计算物理-0039-分子动力学模拟_(Molecular_Dynamics_Simulation)`

## R01

分子动力学模拟（Molecular Dynamics, MD）通过对牛顿方程进行离散时间积分，追踪粒子体系在相空间中的演化轨迹。  
本条目给出一个最小可运行 MVP，聚焦经典 `Lennard-Jones` 流体在二维周期盒中的演化，目标是把“力的计算 + 时间推进 + 观测统计”完整串起来。

实现要点：
- 势函数：截断平移的 Lennard-Jones 势；
- 积分器：`Velocity-Verlet`；
- 边界：周期边界 + 最小镜像；
- 输出：温度、总能量均值/方差、相对能量漂移。

## R02

本目录实现的问题定义（约化单位）：

- 输入（由 `MDConfig` 固定给出）：
  - 粒子数 `N`、密度 `rho`、目标温度 `T`；
  - 势参数 `epsilon`、`sigma`、截断半径 `rc`；
  - 质量 `m`、时间步长 `dt`、总步数 `n_steps`；
  - 热化步数与采样间隔等控制参数。
- 动力学模型：
  - `m * d^2 r_i / dt^2 = F_i`；
  - `F_i = -grad_i U(r_1, ..., r_N)`；
  - `U` 使用两体 Lennard-Jones 势累加。
- 输出：
  - 采样表（step、势能、动能、总能、瞬时温度）；
  - 汇总统计（均值、标准差、相对能量漂移）。

## R03

核心数学关系：

1. Lennard-Jones 两体势（截断平移）  
   `u(r) = 4*epsilon*((sigma/r)^12 - (sigma/r)^6) - u(rc), r < rc`，  
   `u(r) = 0, r >= rc`。
2. 两体作用力大小  
   `F(r) = 24*epsilon*(2*(sigma/r)^12 - (sigma/r)^6)/r`。  
   在代码中使用向量形式并写成 `1/r^2` 因子以减少重复开方。
3. 周期边界最小镜像  
   `delta = delta - L*round(delta/L)`，保证粒子对采用最近镜像距离。
4. Velocity-Verlet  
   - `r(t+dt) = r(t) + v(t)dt + 0.5*a(t)dt^2`  
   - `v(t+dt) = v(t) + 0.5*(a(t)+a(t+dt))dt`
5. 瞬时温度（二维，去除质心平动自由度）  
   `T_inst = 2K / dof`，其中 `K = (m/2) * sum_i |v_i|^2`。

## R04

算法高层流程：
1. 根据 `N` 与 `rho` 计算盒长 `L = sqrt(N/rho)`。  
2. 在规则晶格上初始化粒子位置并加小扰动。  
3. 由高斯分布初始化速度，去掉质心漂移并重标定到目标温度。  
4. 计算初始力与势能（两两粒子求和，含最小镜像与截断）。  
5. 循环执行 Velocity-Verlet 时间推进。  
6. 热化阶段周期性做速度重标定（简单 thermostat）。  
7. 在生产阶段按固定间隔采样能量与温度。  
8. 汇总样本统计并输出自检结论。

## R05

核心数据结构：

- `MDConfig`（`dataclass`）：
  - 集中管理物理参数与数值参数，保证可复现配置。
- `positions: np.ndarray`，形状 `(N, 2)`：
  - 粒子空间坐标。
- `velocities: np.ndarray`，形状 `(N, 2)`：
  - 粒子速度。
- `forces: np.ndarray`，形状 `(N, 2)`：
  - 当前步每个粒子的合力。
- `records: list[dict[str, float]]`：
  - 采样序列（step、能量、温度）。
- `samples: pandas.DataFrame`：
  - 统一展示与统计采样结果。

## R06

正确性与物理一致性检查点：

- 力和势来自同一势函数表达式，避免“势-力不一致”；
- 粒子对 `(i, j)` 的力按牛顿第三定律成对累加：`F_ij = -F_ji`；
- 使用最小镜像保证周期边界距离计算正确；
- Velocity-Verlet 为时间可逆/辛结构友好的标准选择，适合 NVE 段能量监控；
- 脚本输出 `relative_energy_drift` 作为数值稳定性指标；
- `main` 中内置断言检查样本数、温度范围、能量漂移上界。

## R07

复杂度分析（`N` 粒子，采样前后总步数 `T`）：

- 两体相互作用遍历所有粒子对：`O(N^2)`；
- 单步积分主要成本是一次力重算：`O(N^2)`；
- 总时间复杂度：`O(T * N^2)`；
- 空间复杂度：
  - 状态数组（位置/速度/力）：`O(N)`；
  - 采样记录：`O(S)`（`S` 为采样点数）。

该 MVP 未使用邻居表，优先保证源码透明度而非极限性能。

## R08

边界与异常处理：

- 配置非法（如 `density <= 0`、`dt <= 0`、`n_steps <= thermalization_steps`）时抛 `ValueError`；
- 仅支持二维实现，若 `dimension != 2` 抛 `ValueError`；
- 若出现粒子极端重叠（`r^2 < 1e-12`）抛 `RuntimeError`；
- 若采样为空（参数导致无生产期记录）抛 `RuntimeError`；
- 速度初始化后若温度异常接近 0，抛 `RuntimeError`。

## R09

MVP 取舍说明：

- 选择最基础的 LJ 两体势，不引入键角、电荷、多体势；
- 只做二维体系，减少几何与可视化复杂度；
- 热浴采用简化速度重标定，而非 Nose-Hoover/Langevin；
- 不依赖专用 MD 框架（如 OpenMM/LAMMPS），核心流程全部源码显式实现；
- 使用 `numpy + pandas` 作为最小通用工具栈。

## R10

`demo.py` 主要函数职责：

- `_validate_config`：参数合法性校验；
- `_build_lattice_positions`：构建初始晶格位置；
- `_initialize_velocities`：高斯抽样速度并校正温度；
- `_forces_and_potential`：计算合力与总势能（核心物理计算）；
- `_velocity_verlet_step`：单步时间推进；
- `_rescale_to_temperature`：热化阶段速度重标定；
- `run_md_simulation`：完整仿真循环 + 采样 + 汇总；
- `main`：打印样本预览、统计指标并执行自检断言。

## R11

运行方式（无交互）：

```bash
cd Algorithms/物理-计算物理-0039-分子动力学模拟_(Molecular_Dynamics_Simulation)
uv run python demo.py
```

脚本使用固定随机种子，默认输出可复现。

## R12

输出字段说明：

- `step`：采样时刻对应的积分步编号；
- `potential_energy`：体系总势能；
- `kinetic_energy`：体系总动能；
- `total_energy`：总能量 `E = U + K`；
- `temperature`：瞬时温度估计；
- `mean_temperature / std_temperature`：温度均值与波动；
- `mean_total_energy / std_total_energy`：总能均值与波动；
- `relative_energy_drift`：生产期首尾总能相对漂移。

## R13

最小测试建议（当前脚本已覆盖核心检查）：

1. 正常运行测试：`uv run python demo.py`，应看到样本表与 `Self-check passed.`。  
2. 参数异常测试：
   - 把 `density` 改成负值，应触发 `ValueError`；
   - 把 `n_steps` 调小到不大于 `thermalization_steps`，应触发 `ValueError`。
3. 稳定性测试：
   - 将 `dt` 逐步调大，观察 `relative_energy_drift` 增大并可能触发断言失败。

## R14

关键参数及调参方向：

- `dt`：步长；越大越快但数值误差和能量漂移会加重；
- `cutoff`：截断半径；越大越接近完整势，但计算更慢；
- `density`：影响平衡结构与碰撞频率；
- `temperature`：控制速度尺度与热涨落幅度；
- `thermalization_steps`：热化长度，太短会导致统计受初态污染；
- `sample_interval`：采样间隔，过小会增加自相关。

经验上，先调 `dt` 保证稳定，再调热化和采样参数。

## R15

与相关方法对比：

- 对比 Monte Carlo：
  - MD 给出真实时间演化轨迹；
  - MC 更偏向平衡采样，不直接提供动力学时间信息。
- 对比 Euler 显式积分：
  - Velocity-Verlet 对哈密顿系统长期能量行为更稳定；
  - Euler 实现更简单但误差累积明显。
- 对比专业 MD 引擎：
  - 本实现可读性高，便于教学/审计；
  - 专业引擎在邻居表、并行、长程相互作用上性能更强。

## R16

典型应用场景：

- 计算物理课程中讲解“势函数 -> 力 -> 积分器 -> 观测量”完整链路；
- 作为更复杂模拟器开发前的基线原型；
- 用于快速验证新势函数或新积分策略的最小实验台；
- 作为后续扩展到 NVT/NPT、扩散系数计算、径向分布函数的起点。

## R17

可扩展方向：

1. 增加 Verlet neighbor list / cell list，将步进复杂度降到近线性平均表现；  
2. 扩展到三维并加入压力估计（virial pressure）；  
3. 增加更规范 thermostat（Langevin 或 Nose-Hoover）；  
4. 输出轨迹文件并计算 `g(r)`、均方位移 `MSD`；  
5. 将双循环力计算替换为 `numba`/`PyTorch` 向量化加速版本。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 构造默认 `MDConfig` 并调用 `run_md_simulation`，确保整条链路无交互运行。  
2. `run_md_simulation` 先执行 `_validate_config`，然后由 `N` 与 `rho` 计算盒长，生成初始位置和速度。  
3. `_initialize_velocities` 会去掉质心漂移并按目标温度重标定，得到可控初始态。  
4. `_forces_and_potential` 对所有粒子对做最小镜像距离计算，在截断范围内累加 LJ 势和作用力。  
5. 每个时间步执行 `_velocity_verlet_step`：先更新位置，再用新位置重算力，最后更新速度。  
6. 在热化阶段按 `thermostat_interval` 调用 `_rescale_to_temperature`，将系统温度拉回目标值附近。  
7. 进入生产阶段后按 `sample_interval` 记录 `potential/kinetic/total/temperature` 到 `records`，随后转成 `DataFrame`。  
8. 计算首尾总能相对漂移与均值统计，`main` 打印结果并用断言进行最小可执行自检。

第三方库说明：`numpy` 用于数值数组与随机采样，`pandas` 仅用于采样结果汇总；核心 MD 算法（力计算、积分、温控、采样）均在源码中显式展开，无黑盒求解器。

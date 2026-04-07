# 分子动力学积分器 (MD Integrators)

- UID: `PHYS-0323`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `327`
- 目标目录: `Algorithms/物理-计算物理-0327-分子动力学积分器_(MD_Integrators)`

## R01

分子动力学积分器的核心任务是：在给定粒子间作用力后，将牛顿方程

`m * d2r_i/dt2 = F_i(r_1, ..., r_N)`

离散化为稳定、可计算的时间推进公式。  
在 MD 中，积分器不仅影响单步精度，还直接决定长期守恒性质（尤其是总能量漂移和相空间结构保持）。

## R02

本条目实现一个最小但诚实的 NVE（恒粒子数/体积/能量）MVP：

- 体系：二维 Lennard-Jones 粒子，周期边界条件；
- 力场：截断 Lennard-Jones 势（含势能平移）；
- 对比积分器：
  - 显式 Euler；
  - Symplectic Euler（kick-drift）；
  - Velocity Verlet（kick-drift-kick）。

目标不是做完整 MD 引擎，而是可运行地比较积分器在同一物理系统上的数值行为。

## R03

模型方程与势函数：

1. 粒子动力学：`dr_i/dt = v_i`, `dv_i/dt = F_i/m`。
2. Lennard-Jones 势（截断半径 `r_c` 内）：
   - `u(r)=4*epsilon*((sigma/r)^12-(sigma/r)^6)-u(r_c)`。
3. 力：
   - `F_ij = 24*epsilon*(2*(sigma/r)^12-(sigma/r)^6) * (r_ij / r^2)`。
4. 总能量：
   - `E = K + U = 0.5*m*sum_i |v_i|^2 + sum_{i<j} u(r_ij)`。

`demo.py` 显式实现以上公式，不调用黑盒动力学库。

## R04

三种积分器更新公式如下（`a_n = F(r_n)/m`）：

1. 显式 Euler：
   - `r_{n+1} = r_n + dt * v_n`
   - `v_{n+1} = v_n + dt * a_n`
2. Symplectic Euler（kick-drift）：
   - `v_{n+1} = v_n + dt * a_n`
   - `r_{n+1} = r_n + dt * v_{n+1}`
3. Velocity Verlet：
   - `v_{n+1/2} = v_n + 0.5*dt*a_n`
   - `r_{n+1} = r_n + dt*v_{n+1/2}`
   - `a_{n+1} = F(r_{n+1})/m`
   - `v_{n+1} = v_{n+1/2} + 0.5*dt*a_{n+1}`

Velocity Verlet 是经典 MD 的主流基线，因为它二阶、辛、且时间可逆。

## R05

本 MVP 的前提与边界：

- 单组分粒子，质量统一为 `m=1`；
- 只考虑二维、周期边界；
- 不含温控器/压控器（纯 NVE）；
- 势能只做截断平移，不做尾修正或长程电荷；
- 粒子数较小（`N=16`），重点在方法对比而非规模性能。

这些限制使代码短小透明，便于核对积分器本身。

## R06

`demo.py` 输入输出约定：

- 输入：脚本内置 `MDConfig` 参数（粒子数、密度、温度、步长、步数、随机种子等）。
- 输出：一张对比表，字段包括：
  - `max_rel_energy_drift`：最大相对能量漂移；
  - `final_rms_pos_error`：相对高分辨率参考解的末态位置 RMS 误差；
  - `final_rms_vel_error`：末态速度 RMS 误差；
  - 以及积分器阶数/是否辛/是否时间可逆标签。
- 额外输出：阈值检查与 `Validation: PASS/FAIL`。

脚本无交互输入，可直接执行。

## R07

高层流程：

1. 初始化粒子位置（晶格 + 微扰）与速度（去质心速度并按目标温度缩放）；
2. 构造统一初态，分别运行三种积分器；
3. 用更小步长的 Velocity Verlet 生成内部参考轨迹；
4. 计算每种积分器的能量漂移与末态误差；
5. 汇总为 `pandas` 表格并打印；
6. 执行自检断言（Verlet 应优于 Euler）并输出 PASS/FAIL。

## R08

复杂度分析（`N` 粒子，`S` 时间步）：

- 每次力计算是双循环：`O(N^2)`；
- 每个时间步至少一次力评估，故单积分器约 `O(S*N^2)`；
- 本脚本同时跑 3 个被测积分器 + 1 个参考积分器，常数更大但数量级不变；
- 空间复杂度 `O(N)`（粒子状态 + 能量时间序列 `O(S)`）。

在 `N=16` 的教学规模下可在秒级完成。

## R09

数值稳定性与误差控制策略：

- 使用周期边界的 minimum-image 处理，避免坐标无限增长；
- 对极小粒子间距设下限保护（避免 `1/r^12` 奇异爆炸）；
- 采用固定随机种子保证可复现；
- 用高分辨率 Verlet 作为内部参考，减少“只看能量漂移”的片面性；
- 用相对能量漂移 `|E-E0|/|E0|` 归一化不同初态下量级差异。

## R10

MVP 技术栈：

- Python 3
- `numpy`：向量与数值计算
- `pandas`：结果汇总表格展示
- 标准库：`dataclasses`、`math`

未使用现成黑盒 MD 包（如 OpenMM、LAMMPS Python 封装）；积分流程全部在源码中展开。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0327-分子动力学积分器_(MD_Integrators)
uv run python demo.py
```

脚本不接收命令行参数，也不需要交互输入。

## R12

输出字段说明：

- `integrator`：积分器名称；
- `order`：时间离散阶数（1 或 2）；
- `symplectic`：是否保持辛结构；
- `time_reversible`：是否时间可逆；
- `dt`, `steps`：离散步长与步数；
- `max_rel_energy_drift`：全过程最大相对能量漂移；
- `final_rms_pos_error`：末态位置 RMS 误差（对 PBC 做 minimum-image）；
- `final_rms_vel_error`：末态速度 RMS 误差。

## R13

正确性验证（脚本内置）：

1. `VelocityVerlet` 的能量漂移应小于 `Euler`；
2. `SymplecticEuler` 的能量漂移应小于 `Euler`；
3. `VelocityVerlet` 的末态位置误差应小于 `Euler`；
4. `VelocityVerlet` 的能量漂移需低于基础阈值（`< 0.20`）。

若全部满足，脚本输出 `Validation: PASS`；否则退出非零状态。

## R14

当前实现局限：

- 仅二维且粒子数小，不代表大规模生产模拟性能；
- 没有邻居表/Verlet list，力计算仍为 `O(N^2)`；
- 未实现温度控制（Langevin/Nosé-Hoover）与压力控制（barostat）；
- 势函数仅 Lennard-Jones，未覆盖多体势或带电体系。

## R15

可扩展方向：

1. 加入邻居表和 cell list，将大体系复杂度降到近似 `O(N)`；
2. 增加 Leapfrog、RK4、BAOAB(Langevin) 等积分器对比；
3. 引入 NVT/NPT 控制器，比较不同系综下统计量；
4. 输出 RDF、MSD、VACF 等动力学观测量；
5. 扩展到三维多组分体系与并行化实现。

## R16

典型应用：

- 流体与软物质的微观动力学教学实验；
- 新积分器原型验证（先在小体系比较守恒与稳定性）；
- 分子模拟课程中演示“辛 vs 非辛”长期行为差异；
- 为后续高性能 MD 代码提供最小可信参考实现。

## R17

本方案与常见替代方案对比：

- 只用单一积分器：实现更短，但无法看到方法差异；
- 直接依赖外部 MD 包：工程效率高，但算法细节不透明；
- 本方案：同一物理模型下并列实现三种积分器，并加入参考解与断言验证，兼顾可解释性与可执行性。

因此该 MVP 适合“方法学习 + 小规模验证”的场景。

## R18

`demo.py` 源码级流程拆解（9 步）：

1. `initialize_state` 根据密度得到盒长，生成晶格初始坐标并加小扰动；再生成速度、去除质心漂移并按目标温度重标定。  
2. `compute_forces_and_potential` 对每一对粒子应用 minimum-image，显式计算 LJ 力和截断平移势能，得到总力与总势能。  
3. `integrate_euler` 用 `r_{n+1}=r_n+dt*v_n`、`v_{n+1}=v_n+dt*a_n` 推进，并记录每步总能量。  
4. `integrate_symplectic_euler` 用 kick-drift 顺序（先速度后位置）推进，同样记录能量序列。  
5. `integrate_velocity_verlet` 用半步速度 + 整步位置 + 新加速度收尾，完成二阶时间可逆推进。  
6. `main` 先运行三种被测积分器，再用更小步长的 Velocity Verlet 运行参考轨迹（相同总物理时间）。  
7. `final_state_rms_errors` 计算各积分器末态相对参考解的位置/速度 RMS 误差；位置误差通过 minimum-image 处理周期边界。  
8. `summarize_results` 组装 `pandas.DataFrame`，输出阶数、辛性、时间可逆性、能量漂移和末态误差。  
9. `main` 执行阈值检查并输出 `Validation: PASS/FAIL`，形成可自动验证的最小闭环。

第三方库使用边界：`numpy`/`pandas` 仅承担数值运算与表格打印；积分器更新、力学模型、误差指标与验证逻辑均为源码手工实现，没有黑盒求解器。

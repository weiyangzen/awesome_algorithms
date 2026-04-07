# 蛙跳算法 (Leapfrog Algorithm)

- UID: `PHYS-0324`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `329`
- 目标目录: `Algorithms/物理-计算物理-0329-蛙跳算法_(Leapfrog_Algorithm)`

## R01

蛙跳算法（Leapfrog）是计算物理中最常见的辛积分器之一，典型用于哈密顿系统时间推进。它的关键价值不是“局部误差最低”，而是“长期结构保持好”：

- 时间可逆；
- 近似保持辛结构；
- 在长时间积分时能量误差通常呈有界振荡，而非单向漂移。

本条目实现一个最小可运行 MVP，使用 1D 谐振子作为可解析基准，展示 Leapfrog 在精度、守恒性质和可逆性上的核心行为。

## R02

问题定义（固定步长、单自由度）：

- 动力学方程：`dq/dt = p/m`，`dp/dt = -k q`
- 初值：`q(0)=q0`，`p(0)=p0`
- 输入：`q0, p0, dt, n_steps, mass, spring_k`
- 输出：离散轨迹 `t_n, q_n, p_n` 与能量 `H_n = p_n^2/(2m) + k q_n^2/2`

目标是在不依赖 ODE 黑盒求解器的前提下，给出可审计的 Leapfrog 实现，并用可量化指标验证算法行为。

## R03

物理与数值模型：

- 哈密顿量：`H(q,p) = T(p) + V(q) = p^2/(2m) + k q^2/2`
- 力：`F(q) = -dV/dq = -kq`
- 角频率：`omega = sqrt(k/m)`

解析解（用于误差验证）：

- `q(t) = q0 cos(omega t) + (p0/(m omega)) sin(omega t)`
- `p(t) = p0 cos(omega t) - m omega q0 sin(omega t)`

## R04

Leapfrog（kick-drift-kick）单步更新：

1. `p_{n+1/2} = p_n + (dt/2) * F(q_n)`
2. `q_{n+1} = q_n + dt * p_{n+1/2}/m`
3. `p_{n+1} = p_{n+1/2} + (dt/2) * F(q_{n+1})`

该格式与 velocity-Verlet 等价，是分子动力学和哈密顿采样中的标准推进器。

## R05

`demo.py` 的 MVP 设计包含三组实验：

1. 收敛实验：步长逐次减半，观察终点相空间误差比例，验证二阶精度。
2. 长时能量实验：与显式 Euler 对照，比较 `max |H-H0|`。
3. 可逆性实验：先正向积分再反向积分，检查往返误差。

这三组实验分别覆盖“精度、稳定性、几何结构”三个维度。

## R06

正确性依据：

- 代码严格按 Leapfrog 半步动量-整步位置-半步动量顺序实现；
- 使用谐振子解析解直接计算终点相空间误差；
- 步长减半后误差比应接近 `4`（二阶方法特征）；
- 长时积分中，Leapfrog 能量误差应明显小于显式 Euler；
- 时间反演检查应返回接近初值状态。

## R07

复杂度分析（`N = n_steps`）：

- 时间复杂度：`O(N)`（每步常数次数值运算，Leapfrog 每步 2 次力评估）
- 空间复杂度：`O(N)`（MVP 保存完整轨迹用于诊断）
- 若仅保留当前状态，可降为 `O(1)` 存储

## R08

异常与边界处理：

- `n_steps <= 0`：抛 `ValueError`
- `dt == 0` 或非有限：抛 `ValueError`
- `mass <= 0` 或 `spring_k <= 0`：抛 `ValueError`
- `q0/p0` 非有限：抛 `ValueError`
- 积分过程中出现 `nan/inf`：抛 `RuntimeError`

该策略保证失败显式可见，避免静默污染后续结果。

## R09

MVP 边界（有意不做）：

- 仅实现 1D 线性谐振子，不扩展多体耦合系统；
- 仅固定步长，不做自适应步长控制；
- 不引入高阶组合辛积分（如 Yoshida）；
- 不接入现成 ODE 黑盒。

优先保证核心算法链条清晰、可复现、可验证。

## R10

技术栈：

- Python 3
- `numpy`：数组计算与向量化
- `pandas`：实验结果表格化展示

实现中未调用 `scipy.integrate.solve_ivp` 等黑盒求解器，Leapfrog 更新在源码中完全展开。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0329-蛙跳算法_(Leapfrog_Algorithm)
uv run python demo.py
```

脚本无交互输入，执行后自动打印三组实验结果，并在检查通过后输出 `All checks passed.`。

## R12

主要输出字段说明：

- 收敛表：
  - `h`：步长
  - `steps`：积分步数
  - `phase_error_T`：终点相空间误差
  - `prev_over_cur`：相邻误差比（理想接近 4）
- 能量表：
  - `method`：积分器名称
  - `max_abs_energy_drift`：全程最大能量漂移
  - `nfev`：力评估次数
- 额外诊断：`drift_ratio (Euler / Leapfrog)`、`round_trip_phase_error`

## R13

内置验证阈值：

1. 收敛比检查：最细两档步长误差比 `> 3.5`。
2. Leapfrog 长时能量漂移：`max |H-H0| < 5e-3`。
3. Euler 漂移应显著更大：`drift_ratio > 1000`。
4. 往返可逆误差：`round_trip_phase_error < 1e-10`。

任一条件不满足会触发 `assert`，使失败可直接暴露。

## R14

关键可调参数：

- `dt`：步长，影响截断误差与稳定性；
- `n_steps`：积分长度，决定是否进入长期行为区间；
- `mass` 与 `spring_k`：决定系统频率 `omega = sqrt(k/m)`；
- 初值 `q0, p0`：决定轨道振幅与相位。

建议先在 `mass=1, spring_k=1` 下调试，再扩展到其他尺度。

## R15

同类方法对比：

- 显式 Euler：一阶，简单但长期能量漂移严重；
- RK4：短时精度高，但非辛结构，超长时守恒量可能慢漂；
- Leapfrog：二阶但结构保持更好，常用于长期哈密顿积分。

因此在“长期物理可信性”优先时，Leapfrog 常优于同复杂度的非辛方法。

## R16

典型应用场景：

- 分子动力学中的 Verlet / velocity-Verlet 推进；
- 天体轨道积分与长期稳定性分析；
- 哈密顿蒙特卡洛（HMC）中的提议轨迹积分；
- 计算物理课程中辛积分器教学与基准测试。

## R17

可扩展方向：

- 从 1D 扩展到 `q, p in R^d` 的多自由度系统；
- 支持一般势能 `V(q)`，将力计算改为 `F(q) = -∇V(q)`；
- 增加多体相互作用、邻域列表、边界条件；
- 加入 RK4 / implicit 方法基准，形成误差-代价-守恒三维对比；
- 在保持算法透明前提下引入 `numba`/`torch` 做性能优化。

## R18

`demo.py` 源码级算法流程拆解（9 步）：

1. `OscillatorParams` 与 `validate_inputs` 定义物理参数与输入约束（步长、步数、参数正定性）。
2. `HarmonicForce` 提供 `F(q)=-kq` 并累计 `nfev`，让计算开销可观测。
3. `leapfrog_solve` 初始化 `t/q/p` 数组与初始状态 `(q0,p0)`。
4. 每个时间步先执行第一半步动量更新：`p_half = p_n + 0.5*dt*F(q_n)`。
5. 用半步动量做整步位置更新：`q_{n+1} = q_n + dt*(p_half/m)`。
6. 基于新位置计算 `F(q_{n+1})`，完成第二半步动量更新得到 `p_{n+1}`。
7. 步进过程中进行非有限值检查；循环结束后计算离散能量序列并返回 `Trajectory`。
8. `run_convergence_demo`、`run_long_time_energy_demo`、`run_reversibility_demo` 分别验证二阶精度、长期能量行为和时间可逆性。
9. `main` 串行执行三组实验并用断言门限给出通过/失败结论。

实现没有把 Leapfrog 委托给第三方求解器；算法更新路径在源码中可逐行追踪。

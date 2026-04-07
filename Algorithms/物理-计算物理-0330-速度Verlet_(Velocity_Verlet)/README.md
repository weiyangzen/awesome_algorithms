# 速度Verlet (Velocity Verlet)

- UID: `PHYS-0325`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `330`
- 目标目录: `Algorithms/物理-计算物理-0330-速度Verlet_(Velocity_Verlet)`

## R01

速度Verlet（Velocity Verlet）是分子动力学与哈密顿系统数值积分中的经典方法。它在工程上常被选为默认积分器，核心原因是：

1. 二阶精度（全局误差 `O(dt^2)`）；
2. 时间可逆；
3. 辛结构友好，长期能量误差通常有界振荡而非单向漂移。

本条目实现一个最小可运行 MVP：在 1D 谐振子上显式实现速度Verlet，并用可量化实验验证其关键性质。

## R02

问题定义：求解单自由度系统

- 动力学方程：`m * d2q/dt2 = -k q`
- 初值：`q(0)=q0`，`v(0)=v0`
- 输入：`q0, v0, dt, n_steps, mass, spring_k`
- 输出：离散轨迹 `t_n, q_n, v_n` 与能量
  `E_n = 0.5 * m * v_n^2 + 0.5 * k * q_n^2`

该系统有解析解，便于直接验证数值误差和阶数行为。

## R03

谐振子解析解（`omega = sqrt(k/m)`）：

- `q(t) = q0 cos(omega t) + (v0/omega) sin(omega t)`
- `v(t) = -q0 omega sin(omega t) + v0 cos(omega t)`

这些表达式在 `demo.py` 中通过 `exact_state()` 用作真值基线，用于收敛性和误差评估。

## R04

速度Verlet单步更新公式：

1. `a_n = a(q_n)`
2. `q_{n+1} = q_n + v_n*dt + 0.5*a_n*dt^2`
3. `a_{n+1} = a(q_{n+1})`
4. `v_{n+1} = v_n + 0.5*(a_n + a_{n+1})*dt`

本例中 `a(q) = -(k/m) q`。与 leapfrog 的 kick-drift-kick 形式等价，但以 `(q_n, v_n)` 同步状态输出更直观。

## R05

复杂度分析（`N = n_steps`）：

- 时间复杂度：`O(N)`（每步常数次算术和 1 次新位置加速度评估）；
- 空间复杂度：`O(N)`（MVP 保留完整轨迹用于诊断）；
- 若只保留当前状态，空间可降至 `O(1)`。

## R06

`demo.py` 模块职责：

1. `OscillatorParams`：封装 `mass` 与 `spring_k`；
2. `HarmonicAcceleration`：实现 `a(q)`，并统计力评估次数；
3. `velocity_verlet_integrate`：核心速度Verlet推进；
4. `explicit_euler_integrate`：一阶基线方法；
5. `run_convergence_study`：验证二阶收敛；
6. `run_energy_benchmark`：长时间能量漂移对比；
7. `run_reversibility_check`：前向+反向往返误差；
8. `main`：统一执行并断言。

## R07

正确性闭环设计：

1. 与解析解比较终点相空间误差 `phase_error`；
2. 步长减半时误差比应接近 `4`（二阶特征）；
3. 长时间积分中，速度Verlet的能量漂移应显著小于显式Euler；
4. 反向积分后应接近初始状态（时间可逆验证）。

脚本内置断言，不满足即抛错，避免“看起来跑完了但结果不可信”。

## R08

可复现与数值稳定策略：

- 全流程无随机项，结果由参数唯一决定；
- 参数入口统一校验（有限值、正质量、正刚度、非零步长、正步数）；
- `steps_for_duration` 强制 `t_end` 为 `|dt|` 的整数倍，避免隐含对齐误差；
- 每步检查 `q/v` 是否有限值，捕获数值爆炸。

## R09

默认实验参数（已在 `main` 固定）：

- 收敛实验：`t_end=2.0`, `h=[0.2, 0.1, 0.05, 0.025]`；
- 能量实验：`dt=0.1`, `t_end=200.0`；
- 可逆性实验：`q0=0.7`, `v0=-0.2`, `dt=0.05`, `n_steps=800`；
- 物理参数：`mass=1.0`, `spring_k=1.0`。

## R10

运行方式（无交互）：

```bash
cd Algorithms/物理-计算物理-0330-速度Verlet_(Velocity_Verlet)
uv run python demo.py
```

或在仓库根目录执行：

```bash
uv run python Algorithms/物理-计算物理-0330-速度Verlet_(Velocity_Verlet)/demo.py
```

## R11

脚本输出三段结果：

1. `Convergence study`：
   - `h`, `steps`, `phase_error`, `prev_over_cur`, `n_force_evals`
2. `Long-time energy benchmark`：
   - 每个方法的 `max_abs_energy_drift` 和力评估次数
3. `Time reversibility check`：
   - `round_trip_phase_error`

最后输出 `All checks passed.`。

## R12

核心指标定义：

- `phase_error = sqrt((q_num-q_ref)^2 + (v_num-v_ref)^2)`；
- `prev_over_cur = error(h) / error(h/2)`，二阶方法应接近 `4`；
- `max_abs_energy_drift = max_t |E(t)-E(0)|`；
- `drift_ratio = drift(Euler) / drift(VelocityVerlet)`；
- `round_trip_phase_error`：正向后反向积分返回初态的误差。

## R13

适用场景：

- 哈密顿系统长期积分（分子动力学、轨道问题）中的基线推进器；
- 教学场景下展示“二阶 + 可逆 + 近辛结构”的最小样例；
- 作为更复杂系统（多粒子、一般势能）的可审计起点。

不适用场景：

- 强刚性系统、需要自适应步长误差控制的 ODE；
- 需要高阶单步精度而非长期结构保持的短时问题。

## R14

常见失效模式与排查：

1. 步长过大导致状态发散或能量漂移异常；
2. `t_end` 与 `dt` 不整除造成末时刻偏差混入误差评估；
3. 势能与加速度公式不一致导致守恒量异常；
4. 把速度Verlet误写成“先全步速度再更新位置”的非等价流程。

## R15

可扩展方向：

1. 扩展到 `q, v in R^d` 多自由度系统；
2. 将 `a(q)` 替换为一般势能梯度 `-∇V(q)/m`；
3. 引入 N 体势并接入周期边界；
4. 增加 RK4、Symplectic Euler、Leapfrog 的统一对照基准；
5. 在保持算法透明前提下使用 `numba` 或 `torch` 做性能优化。

## R16

与相关算法关系：

- 与显式Euler相比：速度Verlet同阶成本下长期守恒表现显著更好；
- 与蛙跳（Leapfrog）相比：本质等价，仅状态组织方式不同；
- 与高阶非辛方法相比：短时局部精度不一定占优，但长期物理可信性通常更稳定。

## R17

最小可交付能力清单（本条目已覆盖）：

1. 显式写出速度Verlet更新公式并代码实现；
2. 提供解析基准用于误差定量验证；
3. 给出至少一个非辛基线（Euler）作对比；
4. 输出结构化表格结果（`pandas`）；
5. 提供自动断言，保证可批处理验证。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main()` 固定物理参数并依次调用收敛、能量、可逆性三组实验。  
2. `run_convergence_study()` 遍历步长网格，每次调用 `velocity_verlet_integrate()` 得到终点状态。  
3. `exact_state()` 在同一终止时刻计算解析 `q,v`，并由 `phase_error()` 形成误差；通过相邻误差比验证二阶收敛。  
4. `velocity_verlet_integrate()` 按 `q_{n+1}` 与 `v_{n+1}` 的半步加速度平均公式推进，并同步记录离散能量。  
5. `run_energy_benchmark()` 在长时间区间分别运行速度Verlet和 `explicit_euler_integrate()`，统计 `max_abs_energy_drift`。  
6. 脚本打印 `drift_ratio (Euler / VelocityVerlet)`，并用断言要求速度Verlet显著优于Euler。  
7. `run_reversibility_check()` 先用正步长积分，再以负步长从末态反向积分，计算往返 `round_trip_phase_error`。  
8. 三组断言全部通过后输出 `All checks passed.`；任一失败即抛异常并退出非零状态。

第三方库边界说明：`numpy` 仅用于数组与数学运算，`pandas` 仅用于结果表格展示；速度Verlet的更新、误差评估与验证流程全部在源码中显式实现，不依赖黑盒 ODE 求解器。

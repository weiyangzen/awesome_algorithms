# 三体问题 (Three-Body Problem)

- UID: `PHYS-0123`
- 学科: `物理`
- 分类: `天体力学`
- 源序号: `123`
- 目标目录: `Algorithms/物理-天体力学-0123-三体问题_(Three-Body_Problem)`

## R01

三体问题研究三个质量点在牛顿引力作用下的联立运动。与两体问题不同，三体问题一般不存在全局闭式解，系统对初值高度敏感，常呈现混沌行为。

本目录 MVP 聚焦一个标准可复现实例：
- 二维平面、三质量相等；
- 使用经典 figure-eight 初值；
- 通过数值积分得到轨迹；
- 用能量、总动量、质心漂移和周期闭合误差验证结果质量。

## R02

MVP 的计算任务定义：

输入：
- 初始状态 `state0 = [r1,r2,r3,v1,v2,v3]`，其中每个 `r_i,v_i` 都是二维向量；
- 质量向量 `m = [m1,m2,m3]`；
- 数值参数：`G`、软化参数 `epsilon`、时间区间 `[t0,t1]`、积分容差 `rtol/atol`、采样点数。

输出：
- 每个时刻三体的位置与速度；
- 全局诊断表：总能量、相对能量漂移、总动量范数、质心范数、最小两体距离；
- 验收指标：`max_abs_rel_energy_drift`、`max_momentum_norm`、`max_com_norm`、`period_closure_error`。

## R03

物理与建模假设：

- 采用经典牛顿引力，不含相对论修正；
- 三个天体视作点质量；
- 运动限制在二维平面；
- 单位制做无量纲化处理，取 `G=1`；
- 为避免极近距离导致数值发散，在势能与加速度中加入小软化项 `epsilon`。

这些假设使 MVP 保持简洁，同时保留三体问题核心动力学特征。

## R04

核心方程（`i=1,2,3`）：

1. 运动学方程：

`d r_i / dt = v_i`

2. 动力学方程：

`d v_i / dt = G * sum_{j!=i} m_j * (r_j-r_i) / (||r_j-r_i||^2 + epsilon^2)^(3/2)`

3. 总能量：

`E = sum_i 0.5*m_i*||v_i||^2 - sum_{i<j} G*m_i*m_j/sqrt(||r_i-r_j||^2+epsilon^2)`

4. 总动量与质心：

`P = sum_i m_i*v_i`

`R_cm = (sum_i m_i*r_i) / (sum_i m_i)`

理想连续系统中，`E`、`P` 与 `R_cm` 常量；数值积分中可作为误差监控量。

## R05

算法流程（高层）：

1. 构造 figure-eight 初始状态（等质量三体）。
2. 将状态打包为一维向量以适配 ODE 求解器。
3. 在 RHS 中显式计算 3x2 加速度矩阵。
4. 调用 `scipy.integrate.solve_ivp(method="DOP853")` 做高精度积分。
5. 将每个时刻状态解包为位置与速度。
6. 计算全局守恒量与两体距离诊断。
7. 构建轨迹表与全局指标表。
8. 按阈值做自动验收，失败即抛异常。

## R06

复杂度分析（设采样点数 `T`，天体数 `N=3`）：

- 单次 RHS 计算包含 `N*(N-1)` 对相互作用，复杂度 `O(N^2)`；
- 在 `N=3` 固定情况下为常数级，但一般 `N` 体扩展下是二次复杂度；
- 总体计算开销约为 `O(T * N^2)`；
- 存储所有时刻状态为 `O(T * N * d)`，本例 `d=2`。

## R07

正确性检查思路：

- 方程一致性：RHS 直接对应牛顿引力公式，无黑盒力模型；
- 物理一致性：监控 `E/P/R_cm` 是否保持近似守恒；
- 几何合理性：监控最小两体距离，避免穿越型数值伪解；
- 轨道一致性：figure-eight 近周期，检查 `state(t1)` 与 `state(t0)` 的闭合误差。

脚本中所有检查都以阈值断言落地，便于自动化验证。

## R08

数值稳定性与误差控制：

- 采用高阶显式方法 `DOP853`，兼顾精度与效率；
- 设置严格容差 `rtol=1e-10`、`atol=1e-12`；
- 引力软化 `epsilon=1e-6` 缓和近碰时的刚性；
- 通过能量相对漂移而非绝对值评估误差，提升尺度鲁棒性。

注意：三体系统本身对初值敏感，长时间积分会放大微小误差，MVP 只在有限时间窗内进行可信演示。

## R09

适用范围与局限：

- 适用：教学演示、动力学数值实验、守恒量监控示例；
- 局限：
  - 不包含碰撞处理与合并模型；
  - 不做辛积分（长时能量守恒更优的专用方法）；
  - 不含外力、非点质量、相对论修正；
  - 不用于高精度星历生产。

## R10

`demo.py` 函数分工：

- `ThreeBodyConfig`：统一管理积分与模型超参数；
- `build_figure_eight_initial_state`：提供可复现实验初值；
- `pack_state / unpack_state`：状态向量与结构化张量互转；
- `compute_accelerations`：显式两两引力加速度计算；
- `three_body_rhs`：拼装 ODE 的右端函数；
- `total_energy / total_momentum / center_of_mass`：守恒量诊断；
- `run_simulation`：调用 `solve_ivp` 执行积分；
- `build_diagnostics`：构建表格与汇总指标；
- `main`：串联流程、打印结果、执行阈值断言。

## R11

运行方式：

```bash
cd Algorithms/物理-天体力学-0123-三体问题_(Three-Body_Problem)
uv run python demo.py
```

脚本无交互输入，运行后会输出：
- 全局诊断表头部；
- 轨迹采样表头部；
- 汇总检查指标与最终通过/失败状态。

## R12

输出字段说明：

全局诊断表 `global_df`：
- `t`：时间；
- `energy`：总能量；
- `rel_energy_drift`：相对初始能量漂移；
- `momentum_norm`：总动量范数；
- `com_norm`：质心位置范数；
- `d01/d02/d12`：三组两体距离；
- `min_pair_dist`：当前时刻最小两体距离。

轨迹表 `body_df`：
- `t, body, x, y, vx, vy`：单体状态；
- `radius`：到原点距离；
- `speed`：速度模长。

## R13

脚本内置验收阈值：

- `max_abs_rel_energy_drift <= 5e-7`；
- `max_momentum_norm <= 5e-9`；
- `max_com_norm <= 5e-9`；
- `min_pair_dist >= 0.1`（避免过近导致不稳定）；
- `period_closure_error <= 5e-3`。

任何一条失败都会抛 `RuntimeError`，便于在流水线中直接判定失败。

## R14

关键参数与调参建议：

- `t1`：积分总时长；越长越能观察混沌，但累计误差也更大；
- `num_steps`：输出采样密度；影响可视化分辨率；
- `rtol/atol`：精度与性能权衡；
- `softening`：太小可能导致近碰刚性，太大则改变真实动力学；
- 初值：更换初值会显著改变轨迹形态，这是三体问题本性而非代码异常。

## R15

与其他数值方案对比：

- `solve_ivp(DOP853)`（本实现）：实现简洁、精度高、工程上易用；
- RK4 固定步长：更可控但需手动调步长，误差控制弱于自适应方法；
- 辛积分（如 leapfrog / Yoshida）：长时哈密顿守恒更好，但实现与调参更复杂；
- Barnes-Hut/FMM：适合大规模 N 体，不适合本条目最小三体演示。

## R16

典型应用场景：

- 天体力学课程中的“不可积系统”演示；
- 初值敏感性与混沌行为教学；
- 多体积分器验收时的基准案例；
- 后续扩展到 `N>3` 的算法原型验证。

## R17

可扩展方向：

- 换成辛积分器，评估长时间能量漂移优势；
- 加入三维坐标与任意质量分布；
- 加入碰撞检测与事件终止；
- 接入可视化（轨迹动画、庞加莱截面）；
- 扩展到 `N` 体并引入近似加速（Barnes-Hut）。

## R18

`demo.py` 源码级算法流程拆解（8 步）：

1. `main` 创建 `ThreeBodyConfig`，调用 `build_figure_eight_initial_state` 生成 `masses` 与 `state0`。  
2. `run_simulation` 构造 `t_eval` 并调用 `solve_ivp`；求解器每一步都回调 `three_body_rhs`。  
3. `three_body_rhs` 先 `unpack_state` 得到 `(positions, velocities)`，再调用 `compute_accelerations`。  
4. `compute_accelerations` 对三体做双重循环，按 `G*m_j*(r_j-r_i)/(||r_j-r_i||^2+epsilon^2)^(3/2)` 累加每个体的加速度。  
5. `three_body_rhs` 将 `(dr/dt, dv/dt)` 用 `pack_state` 拼回一维导数向量返回给积分器。  
6. 积分结束后 `build_diagnostics` 逐时刻计算 `energy`、`momentum_norm`、`com_norm` 与最小两体距离，并构建 `global_df/body_df`。  
7. `build_diagnostics` 汇总 `max_abs_rel_energy_drift`、`min_pair_dist`、`period_closure_error` 等指标作为 `summary`。  
8. `main` 打印表格和指标，执行阈值断言；全部通过则输出 `All checks passed.`。

说明：虽然 ODE 时间推进由 `solve_ivp` 完成，但力学核心并非黑箱；引力计算、状态更新方程与守恒量诊断均在源码中显式可追踪。

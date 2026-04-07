# 维里定理 (Virial Theorem)

- UID: `PHYS-0130`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `130`
- 目标目录: `Algorithms/物理-经典力学-0130-维里定理_(Virial_Theorem)`

## R01

维里定理把“轨道动力学”与“平均能量关系”连接起来。对有界运动，定义标量维里量 `G = r·p`，有：

`dG/dt = 2T - r·∇V`

对足够长时间平均，若 `⟨dG/dt⟩=0`，则：

`2⟨T⟩ = ⟨r·∇V⟩`

若势能是齐次势 `V(r)=alpha*r^n`，则 `r·∇V = nV`，进一步得到经典形式：

`2⟨T⟩ = n⟨V⟩`

本条目通过最小可运行数值实验验证该关系。

## R02

本目录 MVP 的问题定义：

- 输入：
  - 中心势参数 `alpha, n`（`V=alpha*r^n`）；
  - 初始条件 `(x0,y0,vx0,vy0)`；
  - 仿真时间 `t_end`、采样点 `num_points`、积分容差 `rtol/atol`。
- 输出：
  - 轨迹时间序列 `x,y,vx,vy,r,T,V,E,G`；
  - 维里定理平均关系诊断：`2⟨T⟩`、`n⟨V⟩`、相对误差；
  - 辅助诊断：`⟨2T-nV⟩`、`⟨dG/dt⟩` 估计、能量漂移。

## R03

模型采用二维单位质量粒子在中心势中的运动：

`V(r) = alpha * r^n`

`F = -∇V = -alpha*n*r^(n-2) * r_vec`

对应状态变量 `y=[x,y,vx,vy]` 的常微分方程：

`x_dot = vx, y_dot = vy`

`vx_dot = ax, vy_dot = ay`

其中 `a = F/m`，本实现取 `m=1`。

## R04

维里定理在本模型中的两层表达：

1. 瞬时表达：
   - `dG/dt = 2T - r·∇V`
2. 齐次势平均表达：
   - `r·∇V = nV`
   - `2⟨T⟩ = n⟨V⟩`

因此只要轨道有界且积分足够长，`2⟨T⟩` 与 `n⟨V⟩` 应收敛一致。

## R05

本 MVP 预置两个场景，避免只验证单一特例：

- `harmonic_oscillator_n2`：
  - `n=2`, `alpha=0.5`，即 `V=0.5*r^2`。
  - 理论期望：`2⟨T⟩ = 2⟨V⟩`。
- `kepler_like_n_minus_1`：
  - `n=-1`, `alpha=-1`，即 `V=-1/r`。
  - 理论期望：`2⟨T⟩ = -⟨V⟩`。

这两个场景分别对应“束缚谐振子”和“开普勒型引力势”。

## R06

算法流程分为两段：

1. 动力学积分：
   - 用 `solve_ivp(DOP853)` 积分状态方程，得到全时域轨迹。
2. 维里统计：
   - 计算 `T,V,E,G` 序列；
   - 计算 `2⟨T⟩` 与 `n⟨V⟩` 的差异；
   - 用 `G` 首尾差估计 `⟨dG/dt⟩`；
   - 检查能量漂移，确保数值误差可控。

## R07

伪代码：

```text
for scenario in scenarios:
  integrate y=[x,y,vx,vy] over t in [0, t_end]
  r = sqrt(x^2 + y^2)
  T = 0.5*(vx^2 + vy^2)
  V = alpha * r^n
  E = T + V
  G = x*vx + y*vy

  lhs = 2*mean(T)
  rhs = n*mean(V)
  virial_rel_error = |lhs-rhs| / max(|lhs|,|rhs|,eps)
  dg_dt_time_avg = (G_end - G_start)/t_end
  energy_drift = max(|(E-E0)/E0|)

  report summary + trajectory samples
```

## R08

复杂度分析（每个场景）：

- ODE 积分主开销近似 `O(N)` 到 `O(c*N)`（`N=num_points`，`c` 与自适应步长相关）；
- 后处理（`T,V,E,G` 及统计量）为 `O(N)`；
- 空间复杂度 `O(N)`。

该条目仅二维状态、两场景，运行成本非常低。

## R09

数值稳定性策略：

- 积分器选 `DOP853`，并设置严格容差 `rtol=1e-9, atol=1e-11`；
- 对 `r` 做 `1e-10` 安全下界，避免 `n<0` 时奇点除零；
- 用 `max_rel_energy_drift` 监控长期积分误差；
- 用 `virial_rel_error` 与 `⟨dG/dt⟩` 双指标判断维里收敛质量。

## R10

与“直接代数套公式”相比，本实现优势在于：

- 不是把 `2⟨T⟩=n⟨V⟩` 当黑箱，而是从轨道积分原始数据得到统计量；
- 同时给出能量守恒与 `G` 漂移诊断，可区分“理论失效”与“数值失效”；
- 可替换参数做实验，观察不同初值/积分窗口对收敛的影响。

## R11

默认参数（见 `demo.py`）：

- 谐振子：
  - `n=2`, `alpha=0.5`, `r0=(1.2,0.0)`, `v0=(0.0,0.7)`, `t_end=125.663706(=40π)`, `num_points=10000`
- 开普勒型：
  - `n=-1`, `alpha=-1`, `r0=(1.0,0.0)`, `v0=(0.0,0.85)`, `t_end=120`, `num_points=12000`

调参建议：

- 若要更小统计误差，优先增大 `t_end` 与 `num_points`；
- 对 `n<0` 场景，避免让初始半径过小或速度指向中心导致近奇点；
- 若能量漂移升高，先收紧积分容差。

## R12

代码结构对应关系：

- `Scenario`：单场景参数定义；
- `validate_scenario`：输入合法性和奇点风险检查；
- `acceleration_xy`：由势能参数显式计算力与加速度；
- `rhs`：一阶 ODE 右端；
- `run_scenario`：积分+统计的主流程；
- `format_summary_table`：整理输出指标；
- `main`：执行两个场景并做阈值断言。

## R13

运行方式：

```bash
cd "Algorithms/物理-经典力学-0130-维里定理_(Virial_Theorem)"
uv run python demo.py
```

或在仓库根目录：

```bash
uv run python Algorithms/物理-经典力学-0130-维里定理_(Virial_Theorem)/demo.py
```

脚本无交互输入，直接打印 summary 与轨迹头尾样本。

## R14

输出字段解释：

- `2<T>` 与 `n<V>`：维里定理两侧时间平均值；
- `rel_err`：两侧相对差异，越小越好；
- `mean(2T-nV)`：瞬时平衡量的时间平均，理想应接近 0；
- `<dG/dt>`：由 `G` 首尾差估计的平均导数，理想接近 0；
- `max_energy_drift`：最大相对能量漂移，用于评估数值可靠性；
- `r_range`：半径范围，辅助判断轨道是否有界。

## R15

常见问题排查：

- `Virial average mismatch too large`：
  - 积分窗口不够长（增大 `t_end`）；
  - 采样点不足（增大 `num_points`）；
  - 初值导致轨道不够有界或偏近奇点。
- `Energy drift too large`：
  - 收紧 `rtol/atol`；
  - 检查 `n<0` 场景是否出现极近径掠过。
- `Initial radius too close to zero`：
  - 修改 `r0`，避免奇点附近动力学。

## R16

可扩展方向：

- 增加非齐次势（如 `V = a r^2 + b/r`），比较一般式 `2⟨T⟩=⟨r·∇V⟩`；
- 引入阻尼或外驱动，研究非保守系统下维里关系的修正；
- 扩展到多体系统并统计群体平均维里量；
- 加入相空间图、频谱等可视化辅助诊断。

## R17

适用边界与限制：

- 当前实现是“单位质量 + 中心齐次势 + 二维”教学型 MVP；
- 维里等式依赖“有界运动 + 足够长时间平均”，短窗口会有偏差；
- 奇点势（`n<0`）附近对积分精度敏感；
- 该实现重在验证机制，不替代高保真天体力学/分子动力学引擎。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `main()` 调用 `default_scenarios()` 构造两组参数（`n=2` 与 `n=-1`）。
2. 对每个场景调用 `run_scenario(s)`，先经 `validate_scenario` 检查维度、时间、采样点和奇点风险。
3. 在 `run_scenario` 内构造时间网格 `t` 与初始状态 `y0=[x0,y0,vx0,vy0]`。
4. `solve_ivp` 调用 `rhs` 积分轨道；`rhs` 再调用 `acceleration_xy`，按 `F=-alpha*n*r^(n-2)*r_vec` 计算加速度。
5. 积分完成后从 `sol.y` 取出 `x,y,vx,vy`，向量化计算 `r`、动能 `T=0.5*v^2`、势能 `V=alpha*r^n`、总能 `E=T+V`。
6. 同步计算维里相关量：`G=x*vx+y*vy`、瞬时平衡 `2T-nV`，并构造时间平均 `2⟨T⟩`、`n⟨V⟩`。
7. 计算误差指标：`virial_rel_error`、`mean(2T-nV)`、`(G_end-G_start)/t_end`、`max_rel_energy_drift`。
8. 把全时域序列写入 `pandas.DataFrame`（`trajectory`），把统计量写入 `summary`，组成 `SimulationResult`。
9. `main` 汇总打印表格与轨迹样本，并用阈值断言（维里误差、能量漂移）完成自动正确性检查。

# 测地线计算

- UID: `MATH-0255`
- 学科: `数学`
- 分类: `微分几何`
- 源序号: `255`
- 目标目录: `Algorithms/数学-微分几何-0255-测地线计算`

## R01

本条目实现“测地线计算”的最小可运行版本（MVP）：  
在二维流形坐标系中，根据度量张量 `g_ij(q)` 自动构造 Christoffel 符号，并数值积分测地线方程。

实现目标：
- 不依赖专用几何黑盒库，直接按定义实现测地线 ODE；
- 给出可复现实验（单位球面上的两条测地线）；
- 输出可验证指标（能量守恒误差、是否落在大圆平面）。

## R02

问题定义（本目录实现范围）：
- 流形：单位球面 `S^2`，采用局部坐标 `q=(theta, phi)`。
- 度量：
  - `g(theta,phi) = [[1, 0], [0, sin^2(theta)]]`。
- 测地线方程（二阶）：
  - `d^2 q^k / dt^2 + Gamma^k_{ij}(q) * dq^i/dt * dq^j/dt = 0`。
- 输入：
  - 初值 `q0`（坐标），`v0`（初速度）；
  - 积分终止时间 `t_end`、离散点数 `num_points`。
- 输出：
  - 轨迹 `q(t), v(t)`；
  - 动能序列 `E(t)=0.5*v^T g v`；
  - 大圆平面残差与能量漂移统计。

## R03

核心数学关系：

1. Christoffel 符号定义：
   `Gamma^k_{ij} = 0.5 * g^{k l} (∂_i g_{j l} + ∂_j g_{i l} - ∂_l g_{i j})`。
2. 本实现对 `∂_a g_{ij}` 使用中心差分近似：
   `∂_a g_{ij}(q) ≈ [g_{ij}(q+h e_a)-g_{ij}(q-h e_a)] / (2h)`。
3. 将二阶系统改写为一阶系统：
   - `q' = v`
   - `v'^k = -Gamma^k_{ij}(q) v^i v^j`
4. 单位球面测地线应对应三维空间中过原点平面与球面的交线（大圆），可用于几何一致性验证。

## R04

算法流程（高层）：
1. 读入 `q0, v0` 并做形状/有限值检查。  
2. 在每个状态点计算度量 `g` 与度量偏导 `∂g`。  
3. 由 `g^{-1}` 与 `∂g` 计算 `Gamma^k_{ij}`。  
4. 组装一阶 RHS：`[q', v'] = [v, -Gamma(v,v)]`。  
5. 使用固定步长 RK4 在 `t_eval` 网格上积分。  
6. 从积分结果中拆出 `q_path` 与 `v_path`。  
7. 计算能量序列及相对漂移。  
8. 将轨迹映射到三维，计算与初始法向确定平面的残差并输出结果。

## R05

核心数据结构：
- `GeodesicResult`（`dataclass`）：
  - `name`：案例名；
  - `t`：时间网格；
  - `q`：`(N,2)` 坐标轨迹；
  - `v`：`(N,2)` 速度轨迹；
  - `energy`：每个时刻动能；
  - `energy_rel_drift`：能量相对漂移；
  - `plane_residual_max/mean`：大圆平面残差；
  - `extra_metric`：额外指标（赤道案例的 `theta` 偏移）。
- 中间张量：
  - `dg[a,i,j]`：度量偏导；
  - `gamma[k,i,j]`：Christoffel 符号。

## R06

正确性要点：
- 几何方程正确：RHS 严格对应测地线方程的一阶化形式。  
- Christoffel 来源可审计：由度量定义直接计算，不依赖黑盒 API。  
- 数值积分可复核：RK4 步进公式完全显式实现。  
- 物理一致性检验：
  - 对无外力测地线，动能应近似守恒；
  - 在单位球面上，测地线应近似落在某个过原点平面（大圆）。

## R07

复杂度分析（设维度 `n`，时间点数 `N`）：
- 单次 RHS 计算：
  - 度量偏导（中心差分）约 `O(n^3)`；
  - Christoffel 组装约 `O(n^4)`（四重循环）。
- RK4 每步调用 RHS 4 次，共 `N-1` 步。  
- 总时间复杂度（常数项略）：
  - `O(N * n^4)`；本题 `n=2`，实际非常轻量。
- 空间复杂度：
  - 轨迹存储 `O(N * n)`；
  - 张量临时量 `O(n^3)`。

## R08

边界与异常处理：
- `q0/v0` 维度不是 `(2,)` 或含 `nan/inf`：`ValueError`。  
- `t_end <= 0`、`num_points < 3`：`ValueError`。  
- `theta` 接近极点（`sin(theta)` 过小）触发坐标奇异：`ValueError`。  
- RK4 积分中若状态出现非有限值：`RuntimeError`。  
- 平面法向退化或轨迹点非有限：`RuntimeError`。

## R09

MVP 取舍：
- 选择单位球面作为演示流形，便于几何直觉和可验证性。  
- 使用 `numpy` 手写 RK4，避免环境缺失 `scipy` 时不可运行。  
- 使用数值偏导计算 Christoffel，保留“通用度量接口”而非手写球面特例公式。  
- 不实现自适应步长、事件检测、多图册切换，保持最小但诚实的可运行实现。

## R10

`demo.py` 主要函数职责：
- `sphere_metric`：定义球面度量。  
- `metric_partials`：中心差分计算度量偏导。  
- `christoffel_symbols`：由度量与偏导计算 `Gamma^k_{ij}`。  
- `geodesic_rhs`：构建测地线一阶系统 RHS。  
- `rk4_integrate`：固定步长 RK4 积分器。  
- `integrate_geodesic`：封装积分入口并返回 `t/q/v`。  
- `kinetic_energy`：计算 `0.5*v^T g v`。  
- `spherical_to_cartesian`、`spherical_velocity_to_cartesian`：球坐标映射。  
- `great_circle_plane_residual`：计算轨迹对大圆平面的偏差。  
- `run_case` / `print_case` / `main`：组织案例、打印结果与执行断言。

## R11

运行方式：

```bash
cd Algorithms/数学-微分几何-0255-测地线计算
python3 demo.py
```

脚本无交互输入，直接打印两组案例结果与最终校验状态。

## R12

输出字段解读：
- `energy drift`：`max_t |E(t)-E(0)| / max(1, |E(0)|)`。  
- `plane residual (max/mean)`：三维轨迹点到“由初始位置与初始速度确定平面”的偏差。  
- `equator theta deviation max`：赤道案例中 `theta` 偏离 `pi/2` 的最大值。  
- `samples` 行：
  - `t`：时间；
  - `theta, phi`：坐标；
  - `theta_dot, phi_dot`：速度；
  - `energy`：动能。  
- `All checks passed.`：表示内置数值一致性断言全部通过。

## R13

建议最小测试集（已内置）：
- `equator geodesic`：
  - 初值满足赤道大圆条件，理论上 `theta` 恒为 `pi/2`。
- `oblique great-circle geodesic`：
  - 一般初值，检验 Christoffel + ODE 是否产生稳定大圆轨迹。

建议补充异常测试：
- `theta` 贴近 `0` 或 `pi`（应触发坐标奇异报错）；  
- `num_points=2`（应报参数错误）；  
- `q0` 含 `nan`（应报有限值错误）。

## R14

可调参数：
- `integrate_geodesic`：
  - `num_points`（默认 `1600`）：积分网格密度；
  - `diff_step`（默认 `1e-6`）：中心差分步长；
  - `t_end`：积分总时长。
- `main` 中每个案例的 `q0, v0, t_end`。
- 末尾断言阈值：
  - `energy drift`、`plane residual`、`equator deviation`。

调参建议：
- 需要更高精度时可提高 `num_points`；
- 数值噪声偏大时可微调 `diff_step`（过大截断误差、过小舍入误差）。

## R15

方法对比：
- 对比“仅手写球面特例方程”：
  - 特例更快，但泛化差；
  - 本实现通过 `g -> Gamma -> ODE` 更通用，可迁移到其它流形。  
- 对比黑盒几何库的一行求解：
  - 黑盒更省代码；
  - 本实现可清晰追踪每个公式环节，便于教学与审计。  
- 对比自适应 ODE 积分器：
  - 自适应更高效稳健；
  - 本实现固定步长更简单、可控、无额外依赖。

## R16

典型应用场景：
- 微分几何教学：从度量到测地线方程的完整计算链。  
- 机器人与控制中的流形路径局部建模（概念验证）。  
- 计算机图形中的曲面最短路径近似（基础模块）。  
- 几何优化算法中的 baseline 轨迹积分器。

## R17

可扩展方向：
- 接入自适应步长 RK45/DOP853 提升效率与鲁棒性。  
- 增加更多流形（双曲面、旋转曲面、自定义参数曲面）及度量工厂。  
- 引入边值型测地线（给定起终点）求解，如 shooting / multiple shooting。  
- 对接可视化（matplotlib 3D）展示轨迹与法平面。  
- 将 Christoffel 计算改为符号/自动微分以降低差分误差。

## R18

`demo.py` 源码级算法流程拆解（8 步）：
1. `main` 定义两组固定初值（赤道与斜向）并逐个调用 `run_case`。  
2. `run_case` 调用 `integrate_geodesic`，将初值拼成状态 `y=[q,v]` 并构造 `t_eval`。  
3. `integrate_geodesic` 调用 `rk4_integrate`；每个步长内 RK4 会评估 4 次 `geodesic_rhs`。  
4. 在 `geodesic_rhs` 中，先用 `metric_partials` 对 `g_ij` 做中心差分，得到 `dg[a,i,j]`。  
5. `christoffel_symbols` 按 `Gamma^k_{ij}=0.5 g^{kl}(∂_i g_{jl}+∂_j g_{il}-∂_l g_{ij})` 组装联络系数。  
6. `geodesic_rhs` 用 `v'^k = -Gamma^k_{ij} v^i v^j` 计算加速度项，并返回 `[q', v']=[v, v']`。  
7. 积分完成后，`run_case` 计算动能序列与相对漂移，再把轨迹映射到三维并计算大圆平面残差。  
8. `main` 汇总最大误差指标并执行阈值断言；若全部通过则打印 `All checks passed.`。  

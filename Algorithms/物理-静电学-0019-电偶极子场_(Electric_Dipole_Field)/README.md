# 电偶极子场 (Electric Dipole Field)

- UID: `PHYS-0019`
- 学科: `物理`
- 分类: `静电学`
- 源序号: `19`
- 目标目录: `Algorithms/物理-静电学-0019-电偶极子场_(Electric_Dipole_Field)`

## R01

本条目实现“电偶极子场”的最小可运行 MVP，核心目标是把两种常用模型放在同一计算框架下比较：
- 精确模型：由 `+q` 与 `-q` 两点电荷直接叠加得到电场；
- 近似模型：使用偶极远场公式 `E_far`；
- 通过数值实验验证：当 `r/d` 增大时（观察点距离远大于电荷间距），近似误差按幂律下降。

## R02

问题定义（MVP 范围）：
- 输入：
  - 电荷量 `q > 0`；
  - 电荷间距 `d > 0`（沿 `x` 轴放置 `+q` 与 `-q`）；
  - 一组球壳半径 `r_i`（严格递增）；
  - 每个球壳上的采样方向数 `n_dirs`。
- 输出：
  - 每个 `r_i` 下远场近似的平均相对误差、`95%` 分位误差、最大误差；
  - `log(mean_error)` 对 `log(r/d)` 的线性拟合斜率。
- 约束：
  - 仅处理静态真空、点电荷模型；
  - 不处理介质边界、时变电磁场、辐射项。

## R03

数学模型（SI 单位）：
1. 库仑常数：`k = 1 / (4π ε0)`。  
2. 点电荷场：对电荷 `q_j`（位置 `r_j`），`E_j(r) = k q_j (r-r_j)/|r-r_j|^3`。  
3. 偶极子的精确场（两电荷叠加）：
`E_exact(r) = E_+(r) + E_-(r)`，其中 `q_+ = +q`，`q_- = -q`。  
4. 偶极矩：`p = q * d_vector`，本实现中 `d_vector = (d, 0, 0)`，故 `p = (q d, 0, 0)`。  
5. 远场近似（`r >> d`）：
`E_far(r) = k * (3 (p·r_hat) r_hat - p) / r^3`。  
6. 误差定义：
`rel_err = |E_far - E_exact| / |E_exact|`。

## R04

算法流程（MVP）：
1. 生成球面近均匀方向集（Fibonacci sphere）。
2. 对每个半径 `r_i` 构造采样点 `points = r_i * directions`。
3. 计算 `E_exact(points)`。
4. 计算 `E_far(points)`。
5. 逐点计算相对误差 `rel_err`。
6. 汇总每个壳层的 `mean / p95 / max` 误差。
7. 在 `log-log` 坐标下拟合误差衰减斜率。
8. 做断言检查并输出表格报告。

## R05

核心数据结构：
- `numpy.ndarray`：
  - 三维向量数组：`points`、`E_exact`、`E_far`；
  - 标量序列：`radii_m`、`r_over_d`、`mean_relative_error` 等。
- `DipoleFieldResult`（`dataclass`）：
  - 统一封装参数、误差统计和拟合斜率。
- `pandas.DataFrame`：
  - 用于终端打印壳层误差诊断表。

## R06

正确性要点：
- 精确解部分直接按库仑定律向量叠加，不依赖黑盒求解器；
- 近似解部分使用标准偶极远场封闭公式；
- 当 `r/d` 增大时，高阶多极项贡献减弱，`E_far` 应更接近 `E_exact`；
- `run_checks` 要求误差随 `r/d` 单调下降，且远端误差足够小，并检查 `log-log` 斜率在合理区间（接近 `-2`）。

## R07

复杂度分析：
- 设壳层数为 `S`，每层方向采样数为 `N`。
- 时间复杂度：`O(SN)`，每个采样点做常数次向量运算。
- 空间复杂度：`O(N)`（逐壳层计算），外加 `O(S)` 统计量存储。
- 本实现完全向量化，避免 Python 层双重循环开销。

## R08

边界与异常处理：
- `q <= 0` 或 `d <= 0`：抛出 `ValueError`；
- `radii_m` 非一维、非递增、包含非正值或非有限值：抛出 `ValueError`；
- 采样点落在电荷奇点附近（距离过小）：抛出 `ValueError`；
- 方向数过小（`n_points < 8`）导致球面覆盖过差：抛出 `ValueError`。

## R09

MVP 取舍说明：
- 选择“精确解 vs 远场近似”这条最短闭环，突出偶极子场的核心物理含义；
- 不引入网格 PDE（泊松方程）求解，以免偏离本条目重点；
- 不依赖 `scipy` 或 `sklearn` 黑盒函数，所有关键公式均在源码中可追踪；
- 保留可重复、可扩展结构，便于后续增加势函数或更高阶多极展开。

## R10

`demo.py` 模块职责：
- `fibonacci_sphere`：生成球面准均匀单位方向；
- `exact_dipole_field`：实现双点电荷精确场；
- `dipole_far_field`：实现偶极远场近似；
- `evaluate_shell_errors`：统计壳层误差并拟合斜率；
- `run_dipole_field_mvp`：组织参数与主流程；
- `run_checks`：执行自动质量门槛；
- `preview_table` 与 `main`：打印无交互报告。

## R11

运行方式：

```bash
cd Algorithms/物理-静电学-0019-电偶极子场_(Electric_Dipole_Field)
uv run python demo.py
```

脚本不需要任何交互输入，运行后会输出误差统计和壳层表格。

## R12

输出字段解读：
- `mean rel error @ min r/d`：最近壳层的平均相对误差；
- `mean rel error @ max r/d`：最远壳层的平均相对误差；
- `error reduction ratio`：误差衰减倍数（首层/末层）；
- `log-log slope`：`mean_error ~ (r/d)^slope` 的拟合指数；
- 壳层表列：
  - `radius_m`：壳层半径；
  - `r_over_d`：无量纲距离比例；
  - `mean_rel_error`、`p95_rel_error`、`max_rel_error`：误差统计。

## R13

建议最小测试集：
- 基线场景：默认参数（`q=2e-9 C, d=0.04 m`）应通过全部断言；
- 增密采样：提高 `directions_per_shell`（如 `3000`）验证统计稳定性；
- 更远场：把最大半径增大，观察末端误差继续下降；
- 异常输入：
  - `q<=0`；
  - 非递增半径；
  - 壳层半径接近 `d/2` 导致奇点风险。

## R14

关键可调参数：
- `charge_c`：电荷量，影响场强量级；
- `separation_m`：电荷间距，决定偶极矩 `|p|=q d`；
- `radii_m`：评估半径序列，决定 `r/d` 覆盖范围；
- `directions_per_shell`：角向采样密度，影响统计噪声与运行时间。

经验：若要更平滑的误差曲线，优先增大 `directions_per_shell`。

## R15

方法对比：
- 对比“只用远场公式”：
  - 本实现同时计算精确解，能量化近似误差；
  - 只用远场无法知道在近场是否失真。
- 对比“纯数值网格法（PDE）”：
  - 网格法更通用，可处理复杂边界；
  - 本问题有解析表达式，直接向量公式更轻量、更透明。

## R16

应用场景：
- 电磁学教学：直观看到偶极近似的适用区间；
- 传感器与分子建模中的远场估计前验证；
- 多极展开截断误差的定量参考；
- 仿真前参数预估与数量级检查。

## R17

可扩展方向：
- 增加电势 `V_exact` 与 `V_far = k (p·r)/r^3` 的误差对比；
- 从单偶极子扩展到偶极阵列；
- 加入介质（有效介电常数）或边界条件；
- 用球谐展开显式展示更高阶多极项；
- 输出 CSV 或图像用于自动报告。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 调用 `run_dipole_field_mvp`，设置 `q`、`d`、壳层半径与采样方向数。  
2. `run_dipole_field_mvp` 调用 `evaluate_shell_errors` 进入主计算。  
3. `evaluate_shell_errors` 先调用 `fibonacci_sphere` 生成单位方向 `directions`。  
4. 对每个半径 `r_i`，构造采样点 `points = r_i * directions`。  
5. `exact_dipole_field` 依据两点电荷库仑场叠加得到 `E_exact`。  
6. `dipole_far_field` 用 `E = k*(3(p·r_hat)r_hat - p)/r^3` 计算 `E_far`。  
7. 在每个壳层计算逐点相对误差并汇总 `mean/p95/max`，随后对 `log(mean_error)` 与 `log(r/d)` 做线性拟合得到衰减斜率。  
8. `run_checks` 校验误差单调下降、远端误差阈值与斜率范围，最后 `main` 打印诊断表并输出 `All checks passed.`。  

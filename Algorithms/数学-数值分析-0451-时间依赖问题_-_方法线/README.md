# 时间依赖问题 - 方法线

- UID: `MATH-0451`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `451`
- 目标目录: `Algorithms/数学-数值分析-0451-时间依赖问题_-_方法线`

## R01

方法线（Method of Lines, MOL）的核心思想是“先离散空间、后积分时间”：
- 把时间依赖 PDE 在空间上离散成一个 ODE 系统；
- 再用常微分方程时间积分器推进该系统。

本条目给出一个最小可运行、可验证的 MVP：一维热方程的 MOL 实现，并用解析解做误差对照。

## R02

问题定义（本实现）：
- PDE：`u_t = alpha * u_xx`，`x in (0,1), t>0`
- 边界：`u(0,t)=0, u(1,t)=0`（齐次 Dirichlet）
- 初值：`u(x,0)=sin(pi x)`

输入：
- 扩散系数 `alpha`
- 空间网格点数 `n_space`
- 终止时间 `t_end`
- 时间步长 `dt`

输出：
- 不同空间分辨率下的终态误差（`max_abs_error`, `l2_error`）
- 误差收敛阶估计。

## R03

方法线离散过程：
1. 空间离散：对内部节点使用二阶中心差分近似 `u_xx`。  
2. 得到半离散系统：`y'(t) = alpha * L * y(t)`，其中 `L` 是离散 Laplace 矩阵。  
3. 时间离散：对 ODE 系统应用经典四阶 Runge-Kutta（RK4）推进。

对当前基准问题，解析解为：
`u(x,t)=exp(-alpha*pi^2*t)*sin(pi*x)`，可直接用于精度验证。

## R04

算法主流程：
1. 构造均匀空间网格与内部节点离散 Laplace 矩阵。  
2. 由初值构造内部状态向量 `y0`。  
3. 循环执行 RK4 时间步，得到 `y_{n+1}`。  
4. 在终止时刻将内部状态恢复为全网格解（边界补零）。  
5. 与解析解比较，计算误差。  
6. 在多组 `n_space` 下重复并估计经验收敛阶。

## R05

核心数据结构（`demo.py`）：
- `MOLConfig`：单次仿真配置（`alpha/n_space/t_end/dt` 等）。
- `x: np.ndarray`：空间网格。
- `y: np.ndarray`：内部节点状态向量（不含两端边界）。
- `y_history: np.ndarray`：时间推进中的内部状态轨迹（`steps+1, n_interior`）。
- `rows: list[(dx, steps, max_err, l2_err)]`：收敛实验汇总表。

## R06

正确性依据（工程可检验）：
- 空间算子严格对应中心差分公式：`(u_{i-1}-2u_i+u_{i+1})/dx^2`；
- 时间推进严格对应 RK4 公式（`k1,k2,k3,k4` 组合）；
- 对有解析解的问题，若实现正确，网格加密后误差应下降；
- 本脚本要求误差随 `n_space` 提升严格下降，且经验阶不低于阈值（`1.8`）。

## R07

复杂度分析：
- 设内部自由度 `m = n_space - 2`，时间步数 `N_t`。
- 每个 RK4 子步通过三对角模板（邻接三点）计算离散 Laplace，单次 `O(m)`。
- 每步 4 个子步，时间复杂度仍是 `O(N_t * m)`。
- 空间复杂度：`O(N_t*m)`（轨迹存储；若只保留当前状态可降到 `O(m)`）。

## R08

边界与异常处理：
- `n_space < 3`、`alpha<=0`、`dt<=0`、`t_end<=0`：抛 `ValueError`；
- 区间端点非法（`x_right <= x_left`）抛 `ValueError`；
- `t_end` 不是 `dt` 的整步倍数（本 MVP 约束）抛 `ValueError`；
- RK4 中若出现 `nan/inf` 状态或 RHS，抛 `RuntimeError`；
- 收敛检查不通过时抛 `AssertionError`。

## R09

MVP 取舍说明：
- 只实现单方程线性扩散模型，目标是把 MOL 主线讲清楚；
- 使用 `numpy + 标准库`，不依赖黑箱 PDE/ODE 求解器；
- 使用 RK4 而非自适应步长，便于逐行审计公式；
- 选择带解析解的基准问题，确保结果可验证而非仅“跑通”。

## R10

`demo.py` 关键函数职责：
- `validate_config`：参数合法性校验。
- `make_grid`：生成空间网格与 `dx`。
- `apply_laplacian_dirichlet_zero`：按零边界三对角模板作用离散 Laplace。
- `semi_discrete_rhs`：计算 `y' = alpha * L * y`。
- `rk4_step`：执行一个 RK4 时间步。
- `integrate_mol`：完成全时域积分，返回轨迹。
- `run_single_case`：单分辨率误差评估。
- `estimate_orders`：计算经验收敛阶。
- `print_report` / `run_quality_checks`：输出与自动验收。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0451-时间依赖问题_-_方法线
python3 demo.py
```

脚本无交互输入，运行后自动打印收敛表和阶数，并进行内置校验。

## R12

输出字段解读：
- `n_space`：空间总网格点数（含两端边界）。
- `dx`：空间步长。
- `steps`：时间步数。
- `max_abs_error`：终止时刻网格上的最大绝对误差。
- `l2_error`：终止时刻网格误差的离散 `L2` 指标。
- `p`：由相邻网格误差比估计的经验收敛阶。

## R13

建议最小测试集：
- 正常场景：`n_space = 21, 41, 61`（脚本默认）。
- 参数异常：`dt<=0`、`alpha<=0`、`n_space<3`。
- 对齐异常：手动设 `dt` 使 `t_end/dt` 非整数。
- 数值异常：极端参数导致不稳定时，检查非有限值保护是否触发。

## R14

关键参数与调节建议：
- `alpha`：扩散强度；越大系统越“快”衰减。
- `n_space`：空间精度主控参数；增大可降低空间截断误差。
- `dt`：时间步长；本实现采用固定步，需与稳定性约束兼容。
- `safety`（`choose_stable_dt`）：控制 `dt ~ safety * dx^2 / alpha` 的保守度。

工程上通常先固定 `alpha,t_end`，再通过增大 `n_space` 观察误差阶。

## R15

与相关方法对比：
- 方法线（本实现）：结构清晰，便于复用成熟 ODE 积分思想。
- 全离散显式 Euler：实现更短，但时间精度更低、稳定域更小。
- Crank-Nicolson / 隐式法：稳定性更好，需线性求解器。
- 谱法 + MOL：高精度但实现门槛更高。

本条目优先“透明 + 可验证”，故选择中心差分 + RK4。

## R16

典型应用场景：
- 扩散/传热类瞬态问题原型；
- 反应扩散模型的教学与基线验证；
- 从 PDE 到 ODE 的算法教学（数值分析课程）；
- 更复杂多物理模型前的数值骨架搭建。

## R17

可扩展方向：
- 改为稀疏三对角实现，提高大规模性能；
- 加入非齐次边界或源项 `f(x,t)`；
- 时间积分替换为自适应 RK 或隐式 BDF；
- 扩展到二维/三维网格；
- 增加误差-耗时基准对比和可视化输出。

## R18

源码级算法流程（`demo.py`，8 步）：
1. `main` 设定 `alpha/t_end/n_space_levels`，启动多分辨率实验。  
2. `run_single_case` 按 `dx` 计算稳定时间步（`choose_stable_dt`），生成 `MOLConfig`。  
3. `integrate_mol` 校验配置并通过 `make_grid` 得到网格与 `dx`。  
4. 用初值 `sin(pi x)` 生成内部向量 `y0`，进入时间循环。  
5. 每个 RK4 子步通过 `apply_laplacian_dirichlet_zero` 计算离散 `u_xx`（三对角模板，不走黑盒）。  
6. `rk4_step` 组合 `k1/k2/k3/k4` 得到下一时刻内部状态 `y`。  
7. 终止后 `reconstruct_full_state` 补上边界，与 `exact_solution` 对比得到 `max/l2` 误差。  
8. `estimate_orders` 计算经验阶，`run_quality_checks` 验证误差递减与阶数阈值，最终打印结果。  

说明：本实现没有调用第三方“黑箱 ODE/PDE 求解器”；半离散方程构造与时间推进均在源码中显式展开。

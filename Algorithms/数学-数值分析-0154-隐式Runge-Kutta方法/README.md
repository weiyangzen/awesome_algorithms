# 隐式Runge-Kutta方法

- UID: `MATH-0154`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `154`
- 目标目录: `Algorithms/数学-数值分析-0154-隐式Runge-Kutta方法`

## R01

本条目实现隐式 Runge-Kutta（IRK）方法的最小可运行版本，采用二阶段 Gauss-Legendre 格式（四阶、A-稳定）求解常微分方程初值问题：

- `y' = f(t, y), y(t0) = y0`
- 固定步长积分
- 每一步通过牛顿迭代求解隐式 stage 方程

MVP 目标：
- 给出 IRK 从 Butcher 系数到代码实现的完整落地路径；
- 不依赖黑盒 ODE 求解器；
- 提供误差与收敛阶验证，并展示对刚性测试方程的稳定性特征。

## R02

问题定义（本 MVP，标量状态）：

- 输入：
  - 右端函数 `f(t, y)`；
  - 区间 `[t0, t_end]`（`t_end > t0`）；
  - 初值 `y0`；
  - 步数 `N >= 1`（步长 `h = (t_end - t0) / N`）；
  - 牛顿参数 `tol` 与 `max_iter`。
- 输出：
  - 网格点 `t_0...t_N`；
  - 数值解 `y_0...y_N`；
  - 每步隐式求解诊断（牛顿迭代次数、stage 残差）。

演示中包含两类实验：
- 非刚性方程：用于验证四阶收敛；
- 刚性线性方程：用于观察隐式方法在较大步长下的稳定行为。

## R03

数学基础（Gauss-Legendre 二阶段 IRK）：

1. 一般 IRK 形式（`s` 阶段）
- Stage 方程：
  - `K_i = f(t_n + c_i h, y_n + h * sum_j a_ij K_j), i=1..s`
- 更新：
  - `y_{n+1} = y_n + h * sum_i b_i K_i`

2. 本实现使用 `s=2` 的 Gauss-Legendre 系数（四阶）：
- `c1 = 1/2 - sqrt(3)/6`
- `c2 = 1/2 + sqrt(3)/6`
- `A = [[1/4, 1/4 - sqrt(3)/6], [1/4 + sqrt(3)/6, 1/4]]`
- `b = [1/2, 1/2]`

3. 非线性方程组写成残差形式
- 定义 `G_i(K) = K_i - f(t_n + c_i h, y_n + h * sum_j a_ij K_j)`
- 求解 `G(K)=0`

4. 牛顿迭代
- 线性化：`J(K^m) * delta = -G(K^m)`
- 更新：`K^{m+1} = K^m + delta`

局部误差 `O(h^5)`、全局误差 `O(h^4)`（在足够光滑条件下）。

## R04

算法流程（单步）：

1. 已知 `(t_n, y_n)` 与步长 `h`。
2. 构造 stage 初值 `K^(0)`（用当前斜率填充）。
3. 计算残差向量 `G(K)`。
4. 用有限差分构造雅可比近似 `J(K)`。
5. 解线性方程 `J delta = -G` 并更新 `K`。
6. 若 `||G||_inf` 或 `||delta||_inf` 小于阈值则收敛，否则继续迭代。
7. 收敛后用 `y_{n+1} = y_n + h * b^T K` 推进一步。
8. 记录该步迭代次数与残差，进入下一时间步。

## R05

核心数据结构：

- `IRKTableau(dataclass)`：封装 `A, b, c`；
- `StepDiagnostic(dataclass)`：记录每步 `step_index`, `newton_iters`, `residual_inf_norm`；
- `IRKResult(dataclass)`：聚合 `t_values`, `y_values`, `diagnostics`；
- `numpy.ndarray`：
  - `t_values` 长度 `N+1`；
  - `y_values` 长度 `N+1`；
  - stage 向量 `k` 长度 `s=2`。

## R06

正确性要点：

- 离散公式正确性：代码严格实现 Gauss-Legendre 二阶段 Butcher 系数；
- 隐式求解正确性：每步求解 `G(K)=0`，并输出残差确认收敛质量；
- 数值验证正确性：
  - 非刚性样例下观测误差随 `h` 减小呈四阶趋势；
  - 与显式 Euler 对照可看到刚性样例中的稳定性差异。

## R07

复杂度分析：

设总步数为 `N`，阶段数 `s=2`，每步牛顿迭代平均 `M` 次。

- 时间复杂度：
  - 每次牛顿需构造数值雅可比（`O(s^2)` 次残差调用）并解 `s x s` 线性系统（`O(s^3)`）；
  - 总体约 `O(N * M * (s^3 + s^2))`；
  - 在本实现 `s=2` 为小常数，整体近似 `O(N * M)`。
- 空间复杂度：
  - 轨线存储 `O(N)`；
  - 单步临时变量 `O(s^2)`。

## R08

边界与异常处理：

- `N < 1`、`t_end <= t0`、`tol <= 0`、`max_iter <= 0` 抛 `ValueError`；
- `t0/t_end/y0` 非有限数抛 `ValueError`；
- 残差、雅可比、Newton 增量出现 `nan/inf` 抛 `RuntimeError`；
- 线性方程组奇异（`np.linalg.LinAlgError`）转为 `RuntimeError`；
- 超过 `max_iter` 未收敛抛 `RuntimeError`。

## R09

MVP 取舍：

- 只做标量 ODE，降低实现复杂度并强化可读性；
- 选用 Gauss-Legendre 二阶段：兼顾“隐式特性、A-稳定、四阶精度”；
- 雅可比使用数值差分，不要求用户提供 `df/dy`；
- 不实现自适应步长和事件检测，聚焦核心 IRK 机制。

## R10

`demo.py` 函数职责：

- `gauss_legendre_2stage_tableau`：返回本算法 Butcher 系数；
- `check_inputs`：输入参数校验；
- `stage_residual`：构造 `G(K)`；
- `numerical_jacobian`：有限差分雅可比；
- `newton_solve_stage_equations`：每步 Newton 求解隐式 stage；
- `implicit_runge_kutta_solve`：整段区间积分；
- `explicit_euler_solve`：刚性对照基线；
- `estimate_order`：实验收敛阶估计；
- `run_convergence_demo` / `run_stiff_demo`：两类实验；
- `main`：统一组织输出。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0154-隐式Runge-Kutta方法
python3 demo.py
```

脚本无交互输入，直接打印误差表、阶数估计和稳定性对比。

## R12

输出解读：

- `N`：步数；
- `h`：步长；
- `IRK_max_err`：IRK 在全网格上的最大绝对误差；
- `mean_newton_iters`：每步 Newton 平均迭代轮数；
- `max_stage_residual`：步内 stage 方程残差的最大无穷范数；
- `Estimated order`：基于 `log(error)-log(h)` 的经验收敛阶；
- 刚性对比中的 `IRK(T)`、`Euler(T)`：终点值及其误差。

## R13

建议最小测试集：

1. 收敛测试：`y' = y - t^2 + 1`，`N in {10,20,40,80,160}`。
2. 刚性测试：`y' = -15y`，取较大步长（如 `h=0.2`）。
3. 异常输入：`N=0`、`t_end<=t0`、`tol<=0`。
4. 迭代压力测试：故意把 `max_iter` 设得很小（例如 `1`）应触发不收敛异常。

## R14

关键参数建议：

- `step_list`：控制误差实验精度；
- `tol`：Newton 终止阈值（默认 `1e-12`）；
- `max_iter`：Newton 最大迭代次数（默认 `20`）；
- `fd_eps`：差分雅可比扰动量（默认 `1e-8`）。

建议先使用默认参数验证流程，再调大 `N` 或调小 `tol` 观察误差与成本变化。

## R15

方法对比：

- 对比显式 RK4：
  - 二者同为四阶；
  - RK4 单步显式更便宜；
  - IRK 需解隐式方程，但稳定域更好（A-稳定）。
- 对比 Euler：
  - Euler 一阶且对刚性问题步长受限明显；
  - IRK 在刚性问题上可用更大步长保持稳定。
- 对比 BDF：
  - BDF 是隐式多步法；
  - IRK 是隐式单步法，更便于变步长/自起步设计。

## R16

应用场景：

- 刚性或近刚性 ODE 的高精度积分；
- 需要较强稳定性且保持高阶精度的仿真任务；
- 数值分析教学中“显式 RK 与隐式 RK”差异演示；
- 作为后续实现 Rosenbrock、Radau、BDF 的概念基础。

## R17

后续扩展方向：

- 扩展到向量状态 `y in R^d` 与高维系统；
- 支持用户提供解析 Jacobian 以减少 Newton 成本；
- 增加自适应步长（嵌入误差估计）；
- 引入稀疏线性代数以支持大规模问题；
- 增加单元测试（收敛阶断言、异常路径断言）。

## R18

源码级算法流（对应 `demo.py`，8 步）：

1. `main` 先构造 Gauss-Legendre 二阶段 tableau，并配置步长序列与 Newton 参数。  
2. `run_convergence_demo` 对每个 `N` 调用 `implicit_runge_kutta_solve`，该函数先做输入校验并初始化网格。  
3. 每个时间步中，`newton_solve_stage_equations` 以当前斜率组成 stage 初值 `k`，准备求解 `G(k)=0`。  
4. `stage_residual` 按 `k_i - f(t_n+c_i h, y_n + h * (A k)_i)` 计算残差向量，暴露隐式方程真实结构。  
5. `numerical_jacobian` 对残差做分量扰动，显式构造 `J = dG/dk`（非黑盒）；随后解 `J*delta=-G`。  
6. Newton 迭代更新 `k <- k + delta`，以残差范数和增量范数双准则判停，并记录迭代次数/残差。  
7. stage 收敛后按 `y_{n+1} = y_n + h * b^T k` 更新主变量，循环完成整段积分。  
8. 程序汇总误差与实验阶，并运行刚性样例对照显式 Euler，展示隐式 RK 的稳定性收益。  

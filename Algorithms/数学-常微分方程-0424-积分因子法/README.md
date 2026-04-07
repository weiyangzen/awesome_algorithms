# 积分因子法

- UID: `MATH-0424`
- 学科: `数学`
- 分类: `常微分方程`
- 源序号: `424`
- 目标目录: `Algorithms/数学-常微分方程-0424-积分因子法`

## R01

积分因子法（Integrating Factor Method）用于求解一阶线性常微分方程：

`y'(x) + p(x) y(x) = q(x)`

核心思想是构造一个函数 `mu(x)`，使左侧变为一个乘积的导数，从而把微分方程转成一次积分问题。

## R02

本 MVP 解决的问题：
- 给定连续函数 `p(x), q(x)`、区间 `[x0, x1]` 和初值 `y(x0)=y0`；
- 使用积分因子法计算网格 `x_grid` 上的近似解 `y_if`；
- 用 `scipy.solve_ivp` 作为数值参考解 `y_ref`，并与解析解（若已知）做误差对照。

演示方程选为：
`y' + 2xy = x,  y(0)=1`
其解析解是：`y(x)=0.5+0.5*exp(-x^2)`。

## R03

选择该问题作为演示的原因：
- 这是积分因子法最标准的适用类型（一阶线性 ODE）；
- `p(x)=2x`、`q(x)=x` 可得到闭式解，便于审计实现正确性；
- 既能展示公式推导，也能展示工程实现中的离散积分细节。

## R04

方法推导：

原方程 `y' + p y = q`，构造积分因子
`mu(x) = exp( integral p(s) ds )`。

则
`(mu y)' = mu y' + mu' y = mu(y' + p y) = mu q`。

两侧从 `x0` 积分到 `x`：
`mu(x) y(x) - mu(x0) y0 = integral_{x0}^{x} mu(t) q(t) dt`。

解得
`y(x) = ( mu(x0) y0 + integral_{x0}^{x} mu(t) q(t) dt ) / mu(x)`。

## R05

离散算法流程：
1. 在 `[x0, x1]` 上构造均匀网格 `x_grid`。
2. 计算 `p_vals=p(x_grid)`、`q_vals=q(x_grid)`。
3. 用累积梯形积分得到 `ip(x)=integral p`。
4. 得积分因子 `mu=exp(ip)`。
5. 构造被积函数 `mu*q`，再次累积梯形积分得 `iq(x)=integral(mu*q)`。
6. 常数项 `c = mu(x0)*y0`。
7. 输出 `y_if = (c + iq) / mu`。

## R06

正确性要点：
- 理论层面：若 `p, q` 连续，积分因子法推导严格成立；
- 数值层面：误差主要来自对两个积分的离散化（梯形公式）与浮点误差；
- 审计层面：脚本同时输出
  - 与解析解的误差 `|y_if - y_exact|`
  - 与 `solve_ivp` 的误差 `|y_if - y_ref|`
  - 方程残差 `y'_if + p y_if - q`
  作为实现正确性证据链。

## R07

复杂度分析（网格点数为 `n`）：
- 时间复杂度：`O(n)`
  - 两次累积积分各 `O(n)`，其余逐点运算也为 `O(n)`；
- 空间复杂度：`O(n)`
  - 存储 `x, p, q, mu, iq, y` 等向量。

## R08

边界与异常处理：
- `x1` 必须大于 `x0`；
- `num_points >= 2`；
- `p(x), q(x)` 的返回值必须与 `x` 同形状且全部有限；
- 若出现 `nan/inf` 或数值溢出，立即报错。

说明：当 `integral p` 很大时，`exp(integral p)` 可能溢出，这是积分因子法在数值实现上的常见风险。

## R09

MVP 取舍：
- 仅实现“一阶线性方程 + 均匀网格 + 梯形积分”的最小闭环；
- 不引入自适应网格、刚性方程专用技巧、多重精度等高级特性；
- 通过一个可验证案例保证“方法可运行且可审计”，优先透明性。

## R10

`demo.py` 函数职责：
- `validate_setup`：校验区间、网格和初值参数；
- `ensure_vectorized_output`：校验 `p/q` 的向量化输出合法性；
- `solve_with_integrating_factor`：积分因子法核心实现；
- `solve_with_scipy_reference`：调用 `solve_ivp` 给出参考数值解；
- `exact_solution`：案例解析解；
- `ode_residual`：计算离散残差；
- `run_demo`：组织整套实验与指标输出；
- `main`：固定参数执行，无交互输入。

## R11

运行方式：

```bash
cd Algorithms/数学-常微分方程-0424-积分因子法
uv run python demo.py
```

脚本会直接输出误差指标和若干采样点结果。

## R12

主要输出字段说明：
- `max_abs_err_vs_exact`：积分因子离散解与解析解最大绝对误差；
- `max_abs_err_vs_scipy`：积分因子离散解与 `solve_ivp` 参考解最大绝对误差；
- `mean_abs_residual`：方程残差均值；
- `max_abs_residual`：方程残差最大绝对值；
- 采样表中的 `x/y_if/y_exact/y_scipy`：用于人工 spot check。

## R13

最小测试覆盖（脚本内置）：
- 方程：`y' + 2xy = x, y(0)=1`；
- 区间：`[0, 2]`；
- 网格：`401` 点；
- 核查项：
  - 对解析解误差是否足够小；
  - 对 `solve_ivp` 参考解误差是否足够小；
  - 残差是否接近 0。

## R14

关键参数与调参建议：
- `num_points`：越大，梯形积分误差通常越小，但计算量线性增加；
- `rtol/atol`（`solve_ivp`）：只影响参考解精度，不影响积分因子主算法；
- 若问题区间较大且 `p(x)` 为正且增长快，应警惕 `mu=exp(∫p)` 溢出。

## R15

与其他 ODE 方法对比：
- 对比欧拉/Runge-Kutta：
  - 它们是通用时间推进法；
  - 积分因子法利用线性结构，能直接把问题化为积分，结构更“解析化”。
- 对比纯黑盒 `solve_ivp`：
  - 黑盒更通用；
  - 积分因子法在匹配问题类别时更可解释，便于教学与验证。

## R16

典型应用：
- 电路与控制中的一阶线性动态模型；
- 化学动力学中的一阶线性近似；
- 概率与统计里某些一阶线性期望方程；
- 作为更复杂 ODE/PDE 推导中的基础模块。

## R17

可扩展方向：
- 改为自适应网格或高阶求积（Simpson、Gauss）；
- 对长区间引入缩放/对数技巧缓解 `mu` 溢出；
- 扩展到参数扫描和批量方程求解；
- 与符号推导（如解析积分可得时）联动，提高基准测试覆盖。

## R18

`demo.py` 的源码级算法流（8 步）：

1. `main` 固定案例参数（区间、网格、初值）并调用 `run_demo`。
2. `run_demo` 先定义 `p(x)=2x`、`q(x)=x`，并构造解析解函数。
3. `solve_with_integrating_factor` 调用 `validate_setup` 后生成 `x_grid`。
4. 计算 `p_vals/q_vals` 并用 `cumulative_trapezoid` 得到 `ip=∫p`。
5. 计算积分因子 `mu=exp(ip)`，再对 `mu*q` 做累积梯形积分得到 `iq=∫(mu*q)`。
6. 根据 `y=(mu(x0)*y0 + iq)/mu` 得到离散解 `y_if`。
7. `solve_with_scipy_reference` 使用同一 `p/q` 通过 `solve_ivp` 在同一网格产生 `y_scipy`，仅作参考核验。
8. `run_demo` 计算 `y_if` 相对解析解和参考解的误差，以及 `ode_residual`，最后打印统计与采样点表。

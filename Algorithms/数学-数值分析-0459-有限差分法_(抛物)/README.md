# 有限差分法 (抛物)

- UID: `MATH-0459`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `459`
- 目标目录: `Algorithms/数学-数值分析-0459-有限差分法_(抛物)`

## R01

有限差分法（抛物型）用于求解扩散/热传导类偏微分方程。最典型模型是一维热方程：
\[
u_t = \alpha u_{xx},\quad \alpha>0
\]
它描述物理量在空间上由高到低逐渐“抹平”的过程。本文给出一个最小可运行 MVP：
- 方程：一维热方程
- 离散：显式 FTCS（Forward Time, Central Space）
- 验证：与已知解析解逐点对比误差

## R02

问题设定：
- 空间区间：\(x\in[0,1]\)
- 时间区间：\(t\in[0,T]\)
- 边界条件：\(u(0,t)=u(1,t)=0\)（齐次 Dirichlet）
- 初值：\(u(x,0)=\sin(\pi x)\)

该初边值问题有解析解：
\[
u(x,t)=e^{-\alpha\pi^2 t}\sin(\pi x)
\]
因此可直接用来检验数值方法的正确性与精度。

## R03

网格离散：
- 空间网格：\(x_j=j\Delta x,\ j=0,1,\dots,N\)，\(\Delta x=1/N\)
- 时间网格：\(t^n=n\Delta t,\ n=0,1,\dots,N_t\)
- 离散未知量：\(u_j^n\approx u(x_j,t^n)\)

空间二阶导数采用中心差分：
\[
u_{xx}(x_j,t^n)\approx \frac{u_{j+1}^n-2u_j^n+u_{j-1}^n}{\Delta x^2}
\]

## R04

FTCS 更新公式（内部点 \(j=1,\dots,N-1\)）：
\[
u_j^{n+1}=u_j^n+r\left(u_{j+1}^n-2u_j^n+u_{j-1}^n\right),\quad r=\frac{\alpha\Delta t}{\Delta x^2}
\]
边界点直接强制：
\[
u_0^{n}=u_N^{n}=0
\]
该格式实现非常直接，不依赖黑盒 PDE 求解器。

## R05

显式 FTCS 的稳定性条件：
\[
r\le \frac{1}{2}
\]
若 \(r>1/2\)，误差模态会被放大，数值解容易振荡或发散。`demo.py` 中会在开算前检查 `r_actual <= 0.5`，不满足即抛出异常，避免产生“看似运行成功但结果错误”的情况。

## R06

一致性与收敛阶：
- 时间离散：前向差分，一阶 \(O(\Delta t)\)
- 空间离散：中心差分，二阶 \(O(\Delta x^2)\)
- 组合后全局精度通常写作 \(O(\Delta t+\Delta x^2)\)

在稳定条件满足时，网格加密会让数值解逐步逼近解析解。

## R07

复杂度分析：
- 每个时间步更新一次长度 \(N+1\) 的向量，时间复杂度 \(O(N)\)
- 共 \(N_t\) 步，总时间复杂度 \(O(NN_t)\)
- 内存主要为当前解与下一步解，空间复杂度 \(O(N)\)

这使其非常适合作为教学与原型阶段的基线实现。

## R08

数据结构与输出指标：
- `x: np.ndarray`：空间网格
- `u_num: np.ndarray`：数值解
- `u_ref: np.ndarray`：解析解
- `errors: dict`：`L1/L2/Linf` 三种误差

脚本还输出中心点值与离散能量（\(\sum u_j^2\Delta x\)）用于额外 sanity check。

## R09

伪代码：

```text
input alpha, nx, t_end, target_r
x <- uniform grid on [0,1]
dx <- 1/nx
dt_guess <- target_r * dx^2 / alpha
n_steps <- ceil(t_end / dt_guess)
dt <- t_end / n_steps
r <- alpha * dt / dx^2
if r > 0.5: raise error
u <- sin(pi*x), and enforce boundary u[0]=u[-1]=0
repeat n_steps:
    u_next[1:-1] <- u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
    u_next[0] <- 0, u_next[-1] <- 0
    u <- u_next
u_ref <- exp(-alpha*pi^2*t_end) * sin(pi*x)
compute L1, L2, Linf errors
```

## R10

MVP 默认参数（`demo.py`）：
- `alpha = 1.0`
- `nx = 80`
- `t_end = 0.05`
- `target_r = 0.45`

这些参数满足稳定性，同时步长足够细，能展示较小误差而不过度增加运行时间。

## R11

预期输出内容：
- 网格信息：`dx`, `dt`, `n_steps`
- 稳定性关键值：`r_actual`
- 误差范数：`L1`, `L2`, `Linf`
- 校验量：`u_num(0.5)`, `u_ref(0.5)`, `energy_num`, `energy_ref`

一次运行即可得到完整结果，不需要交互输入。

## R12

`demo.py` 函数职责：
- `initial_condition`：构造初值
- `exact_solution`：构造解析解
- `solve_heat_ftcs`：执行 FTCS 时间推进
- `compute_errors`：计算误差范数
- `main`：组织参数、求解、打印摘要

函数划分保持最小但清晰，便于后续扩展成隐式法或 Crank-Nicolson。

## R13

运行方式：

```bash
python3 demo.py
```

脚本默认在当前目录运行即可，无需额外参数。

## R14

常见错误与规避：
- 把 `r = alpha * dt / dx^2` 写错成 `alpha * dt / dx`。
- 忘记 `j=1..N-1` 的内部点范围，导致越界或错误覆盖边界。
- 直接采用 `dt_guess` 而不按 `t_end` 修正步数，最终时刻偏离目标。
- 仅看“程序不报错”而不做解析解对比，难以及时发现离散实现错误。

当前实现对这些点均有明确处理。

## R15

最小验证策略：
- 固定 `alpha,t_end,target_r`，逐步增大 `nx`（如 40, 80, 160）。
- 观察 `L2` 误差是否整体下降。
- 验证 `r_actual <= 0.5` 始终成立。
- 检查中心点 `u_num(0.5)` 与解析值 `u_ref(0.5)` 的差值是否减小。

## R16

适用场景与局限：
- 适用：抛物型 PDE 的入门教学、算法验证、原型基线。
- 局限：显式格式受稳定性限制，`dt` 必须与 `dx^2` 同阶，细网格时步数很多。
- 对大规模问题通常会转向隐式方法（如后向欧拉、Crank-Nicolson）以放宽稳定性约束。

## R17

可扩展方向：
- 改为后向欧拉（需解线性方程组，通常无条件稳定）。
- 改为 Crank-Nicolson（时间二阶精度）。
- 增加非齐次源项 \(f(x,t)\) 与一般边界条件。
- 扩展到二维热方程，并使用稀疏矩阵提升效率。

## R18

`demo.py` 源码级算法流程（8 步）：
1. 在 `main` 中设定 `alpha, nx, t_end, target_r`，调用 `solve_heat_ftcs`。
2. `solve_heat_ftcs` 先构造均匀网格 `x`，并根据 `target_r` 计算 `dt_guess`。
3. 用 `ceil(t_end / dt_guess)` 得到整数步数 `n_steps`，再回写 `dt=t_end/n_steps`，保证终止时刻严格等于 `t_end`。
4. 计算实际稳定参数 `r_actual = alpha*dt/dx^2`，若超过 `0.5` 直接报错退出。
5. 调用 `initial_condition(x)` 得到 `u^0`，并强制两端边界为 0。
6. 对每个时间步，用向量切片实现内部点更新：`u_next[1:-1] = u[1:-1] + r*(u[2:] - 2u[1:-1] + u[:-2])`，再覆盖 `u = u_next`。
7. 时间循环结束后，调用 `exact_solution(x, t_end, alpha)` 构造解析参考解 `u_ref`。
8. `compute_errors` 逐点计算误差向量并汇总 `L1/L2/Linf`，`main` 打印误差与能量等校验指标。

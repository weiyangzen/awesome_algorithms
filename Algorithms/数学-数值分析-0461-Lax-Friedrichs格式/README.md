# Lax-Friedrichs格式

- UID: `MATH-0461`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `461`
- 目标目录: `Algorithms/数学-数值分析-0461-Lax-Friedrichs格式`

## R01

Lax-Friedrichs 格式是求解双曲守恒律的经典显式差分/有限体积方法，核心特征是“中心通量 + 人工黏性”。

本条目给出一个可运行最小 MVP：
- 目标方程选用一维线性平流 `u_t + a u_x = 0`；
- 用守恒形式实现 Lax-Friedrichs 数值通量；
- 提供周期边界下的精确解对比、误差范数与质量守恒检查；
- 给出多分辨率误差收敛阶估计。

## R02

数学问题：
\[
\partial_t u + \partial_x f(u) = 0,
\]
在本 MVP 中取 `f(u)=a u`（`a` 为常数），即
\[
u_t + a u_x = 0,\quad x\in[0,1),\ t\in[0,T].
\]

周期边界条件：`u(0,t)=u(1,t)`。

若初值为 `u_0(x)`，精确解为平移：
\[
u(x,t)=u_0((x-a t)\bmod 1).
\]

## R03

网格离散：
- 空间：`x_j = jΔx`, `j=0,...,N-1`, `Δx=1/N`；
- 时间：`t^n = nΔt`；
- 近似：`u_j^n ≈ u(x_j,t^n)`。

CFL 数：
\[
\text{CFL}=\frac{a\Delta t}{\Delta x}.
\]

对线性平流的 Lax-Friedrichs 显式格式，稳定性要求近似满足 `|CFL|<=1`。

## R04

经典 Lax-Friedrichs 形式：
\[
u_j^{n+1}=\frac{1}{2}(u_{j+1}^n+u_{j-1}^n)-\frac{\Delta t}{2\Delta x}\left[f(u_{j+1}^n)-f(u_{j-1}^n)\right].
\]

等价的守恒通量写法（本实现采用）：
\[
F_{j+1/2}=\frac{f(u_j)+f(u_{j+1})}{2}-\frac{\alpha}{2}(u_{j+1}-u_j),\quad \alpha=\frac{\Delta x}{\Delta t},
\]
\[
u_j^{n+1}=u_j^n-\frac{\Delta t}{\Delta x}\left(F_{j+1/2}-F_{j-1/2}\right).
\]

这种写法直接体现保守性，便于扩展到一般守恒律。

## R05

稳定性与数值耗散：
- Lax-Friedrichs 通过 `-(alpha/2)*(u_{j+1}-u_j)` 引入黏性项，提高鲁棒性；
- 黏性越大（或网格越粗）越稳定，但间断/陡梯度被抹平更明显；
- 在本问题中 `|CFL|<=1` 是关键检查条件。

## R06

精度特性：
- 时间一阶、空间一阶；
- 对光滑解通常观察到近似一阶收敛；
- 对间断解虽稳定，但会出现明显数值扩散（边沿钝化）。

因此它适合作为“稳健基线”，不适合高保真间断捕捉。

## R07

边界与初值设计：
- 周期边界用 `np.roll` 实现邻点访问，逻辑简洁且不需额外幽灵单元；
- `demo.py` 提供两类初值：
  - 光滑初值（高斯 + 正弦）：用于误差与收敛阶评估；
  - 方波初值：用于观察总变差下降和数值扩散。

## R08

复杂度：
- 单步更新为向量常数次运算，`O(N)`；
- 总时间复杂度 `O(N * N_t)`；
- 空间复杂度 `O(N)`（仅保留当前解）。

## R09

算法伪代码：

```text
输入: nx, a, t_end, cfl_target, u0(x)
构建网格 x, 初始化 u=u0(x)
根据 cfl_target 计算 dt0, 取 n_steps=ceil(t_end/dt0)
回算 dt=t_end/n_steps, 校验 |a*dt/dx|<=1
alpha = dx/dt
循环 n_steps 次:
    uR = roll(u,-1)
    F  = 0.5*(f(u)+f(uR)) - 0.5*alpha*(uR-u)
    u  = u - (dt/dx)*(F - roll(F,1))
构造精确解 u_exact(x,t_end)=u0((x-a*t_end) mod 1)
输出误差范数、质量误差、收敛阶估计
```

## R10

`demo.py` 的函数分工：
- `initial_condition_smooth`：光滑初值；
- `initial_condition_square`：方波初值；
- `linear_flux`：线性通量 `f(u)=a*u`；
- `lax_friedrichs_step`：单步更新（守恒通量形式）；
- `solve_lax_friedrichs`：完整时间推进；
- `exact_periodic_solution`：周期精确解；
- `error_norms`：`L1/L2/Linf`；
- `run_resolution_case`：单分辨率实验；
- `main`：多分辨率与方波实验汇总。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0461-Lax-Friedrichs格式
python3 demo.py
```

脚本无交互输入，执行后直接打印结果。

## R12

输出指标说明：
- `nx`：空间网格数；
- `n_steps`：时间步数；
- `actual_cfl`：回算后的 CFL；
- `L1/L2/Linf`：终止时刻数值解与精确解误差；
- `mass_error`：离散质量初末差；
- `p(100->200), p(200->400)`：经验收敛阶；
- `TV0/TVT`（方波）：总变差初值/末值，用于观察扩散。

## R13

最小验证集：
1. 光滑初值 + `nx=100,200,400`：检验误差随网格细化下降；
2. 计算经验收敛阶，期望接近一阶；
3. 方波初值：验证方法稳定且总变差不增（通常下降）；
4. 质量守恒检查：`mass_error` 接近机器精度。

## R14

参数建议：
- `cfl_target` 建议 `0.7~0.95`；
- `t_end` 建议 `0.3~1.0`；
- 若追求更清晰收敛趋势，优先提高 `nx`；
- 若出现振荡/发散，先检查 `|actual_cfl|` 是否超过 1。

## R15

常见实现错误：
- 把 `alpha` 写成 `|a|` 却仍称“经典 Lax-Friedrichs”；
- 忘记周期回卷导致边界错位；
- 用 `round` 取步数导致 `dt` 偏大，实际 CFL 可能超阈值；
- 忽略非有限数检查，数值爆炸后不易定位问题。

本实现用 `ceil` 控制步长，并做有限性与 CFL 显式校验。

## R16

与相关格式对比：
- 对比迎风格式：两者都一阶，Lax-Friedrichs 通常更“黏”；
- 对比 Lax-Wendroff：后者二阶更锐利，但在间断附近更易振荡；
- 对比 Godunov/Rusanov：后者可视为更物理的数值通量构造，激波问题表现更好。

## R17

可扩展方向：
- 将 `linear_flux` 替换为 Burgers 通量 `f(u)=u^2/2`；
- 把常数 `alpha=dx/dt` 替换为局部波速，得到 Rusanov(LF local)；
- 引入 TVD 限制器构造二阶格式；
- 扩展到二维守恒律与 Strang 分裂。

## R18

`demo.py` 源码级流程（8 步）：
1. `main` 设定 `a, t_end, cfl_target`，并准备多个 `nx` 做网格收敛实验。  
2. `run_resolution_case` 调用 `solve_lax_friedrichs`，构建网格和初值 `u0`。  
3. `solve_lax_friedrichs` 先由 `dt0=cfl_target*dx/|a|` 计算初始步长，再用 `n_steps=ceil(t_end/dt0)` 回算 `dt`，保证达到终止时刻且不放大 CFL。  
4. 每一步调用 `lax_friedrichs_step`：先计算右邻状态 `uR=roll(u,-1)`，再算通量 `f(u), f(uR)`。  
5. 用 `F_{j+1/2}=0.5*(f(u_j)+f(u_{j+1}))-0.5*alpha*(u_{j+1}-u_j)` 组装数值界面通量。  
6. 通过离散守恒更新 `u^{n+1}=u^n-(dt/dx)*(F_{j+1/2}-F_{j-1/2})`，并检查结果全为有限数。  
7. 时间推进结束后，`exact_periodic_solution` 用回溯坐标 `(x-a*t)%1` 构造精确解，`error_norms` 计算 `L1/L2/Linf`。  
8. `main` 汇总不同 `nx` 的误差并计算经验阶，再运行方波案例输出 `TV0/TVT` 与 `mass_error`，验证耗散与守恒行为。  

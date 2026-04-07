# MUSCL格式

- UID: `MATH-0466`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `466`
- 目标目录: `Algorithms/数学-数值分析-0466-MUSCL格式`

## R01

MUSCL（Monotonic Upstream-centered Schemes for Conservation Laws）格式是有限体积法中的高分辨率重构框架，核心思想是：
- 在每个网格单元内用分片线性函数重构（而不是分片常数）；
- 用限制器（limiter）约束斜率，避免 Gibbs 振荡；
- 结合守恒通量与显式时间推进，在“高精度”和“无振荡”之间折中。

本条目提供一个最小可运行 MVP：
- 方程：一维线性平流 `u_t + a u_x = 0`（周期边界）；
- 空间离散：有限体积 + MUSCL 线性重构 + `minmod` 限制器；
- 数值通量：Rusanov（对线性方程等价于稳健上风型通量）；
- 时间离散：SSP-RK2（TVD Runge-Kutta 二阶）；
- 输出：误差范数、经验收敛阶、质量守恒误差、总变差变化。

## R02

目标守恒律：
\[
\partial_t u + \partial_x f(u)=0,\quad f(u)=a u,\quad x\in[0,1),\ t\in[0,T].
\]

周期边界：`u(0,t)=u(1,t)`。  
给定初值 `u_0(x)` 后，精确解为平移：
\[
u(x,t)=u_0((x-a t)\bmod 1).
\]

这使我们可以直接计算终止时刻误差并评估收敛阶。

## R03

有限体积半离散形式：
\[
\frac{d\bar u_j}{dt}=-\frac{1}{\Delta x}\left(F_{j+1/2}-F_{j-1/2}\right),
\]
其中 `\bar u_j` 是单元平均值。

MUSCL 的关键在界面状态重构。对单元 `j`：
- 左差分：`Δ^-_j = u_j-u_{j-1}`；
- 右差分：`Δ^+_j = u_{j+1}-u_j`；
- 限幅斜率：`s_j = minmod(Δ^-_j, Δ^+_j)`；
- 得到 `j+1/2` 处左右状态：
\[
u_{j+1/2}^{L}=u_j+\frac{1}{2}s_j,\qquad
u_{j+1/2}^{R}=u_{j+1}-\frac{1}{2}s_{j+1}.
\]

## R04

`minmod` 限制器定义：
\[
minmod(a,b)=
\begin{cases}
\mathrm{sign}(a)\min(|a|,|b|), & ab>0,\\
0, & ab\le 0.
\end{cases}
\]

它的性质：
- 若局部单调（同号），保留较小斜率，抑制过冲；
- 若出现拐点或极值邻域（异号），直接将斜率降为 0；
- 通常可显著改善间断附近振荡问题，但会增加耗散、在极值处降阶。

## R05

界面通量使用 Rusanov（局部 Lax-Friedrichs）：
\[
F(u_L,u_R)=\frac{f(u_L)+f(u_R)}{2}-\frac{\alpha}{2}(u_R-u_L),
\]
对线性通量 `f(u)=a u` 取 `\alpha=|a|`。

该通量具有：
- 保守性（通过界面通量差更新）；
- 鲁棒性（内置耗散项）；
- 对标量双曲方程实现简单，适合最小 MVP 演示。

## R06

时间推进采用 SSP-RK2：
1. `u^(1) = u^n + dt * L(u^n)`  
2. `u^(n+1) = 0.5*u^n + 0.5*(u^(1) + dt*L(u^(1)))`

其中 `L(u)` 是空间离散算子。  
该组合（TVD 空间离散 + SSP-RK）在合适 CFL 下可兼顾稳定性与二阶时间精度。

## R07

网格与步长设置：
- `x_j = jΔx`, `Δx=1/nx`，周期索引用 `np.roll`；
- `dt0 = cfl_target * dx / |a|`；
- `n_steps = ceil(t_end / dt0)`，再回算 `dt = t_end / n_steps`；
- 实际 `CFL = |a| dt / dx`，代码中显式校验 `CFL<=1`。

初值：
- 光滑测试：高斯包 + 正弦，用于收敛阶；
- 间断测试：方波，用于总变差/无振荡诊断。

## R08

复杂度：
- 单步主要是若干 `np.roll`、逐点运算，时间 `O(N)`；
- 总复杂度 `O(N * N_t)`；
- 额外内存 `O(N)`。

这是典型“高分辨率但轻量”的一维教学实现。

## R09

算法伪代码：

```text
输入: nx, a, t_end, cfl_target, u0(x)
生成网格 x，u <- u0(x)
计算 dx, dt0, n_steps, dt，校验 CFL
重复 n_steps 次:
    # 空间算子 L(u)
    计算 u_left=roll(u,1), u_right=roll(u,-1)
    slope = minmod(u-u_left, u_right-u)
    uL = u + 0.5*slope
    uR = roll(u,-1) - 0.5*roll(slope,-1)
    F  = Rusanov(uL, uR)
    L(u) = -(F - roll(F,1))/dx

    # SSP-RK2
    u1 = u + dt*L(u)
    u  = 0.5*u + 0.5*(u1 + dt*L(u1))

输出: 数值解、误差范数、收敛阶、质量误差、TV 变化
```

## R10

`demo.py` 结构：
- `initial_condition_smooth / initial_condition_square`：两类初值；
- `minmod`：限制器；
- `muscl_reconstruct`：构造界面左右状态；
- `rusanov_flux_linear`：数值通量；
- `muscl_spatial_operator`：半离散算子 `L(u)`；
- `ssp_rk2_step`：单步时间推进；
- `solve_muscl`：总求解器；
- `exact_periodic_solution`：解析参考解；
- `error_norms / total_variation`：误差与 TV 指标；
- `main`：多网格实验与打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0466-MUSCL格式
python3 demo.py
```

脚本无交互输入，执行后直接输出实验表格与校验结论。

## R12

输出字段解释：
- `nx`：空间网格数；
- `n_steps`：实际时间步数；
- `actual_cfl`：回算后的 CFL；
- `L1/L2/Linf`：与精确解的误差范数；
- `mass_error`：离散质量初末差；
- `p(80->160), p(160->320)`：经验收敛阶；
- `TV0/TVT`：间断测试的总变差初值与末值。

正常现象：
- 光滑解误差随网格加密下降；
- 经验阶通常大于一阶（受限制器影响，不必苛求理想 2）；
- `mass_error` 接近机器精度；
- `TVT` 通常不大于 `TV0`。

## R13

最小验证清单：
1. 光滑初值 `nx=80,160,320`，检查误差单调下降；
2. 计算两段经验阶，确认不是退化到明显一阶以下；
3. 间断初值下检查 `TVT <= TV0`（允许微小数值噪声）；
4. 检查 `mass_error` 量级（通常约 `1e-15 ~ 1e-13`）。

`demo.py` 对经验阶与 TV 增长设置了断言，便于自动化验收。

## R14

参数建议：
- `cfl_target`：建议 `0.6 ~ 0.9`；
- `t_end`：建议 `0.2 ~ 1.0`；
- 若要更清晰收敛趋势，优先增加 `nx`；
- 若关注间断形状，可缩短 `t_end` 观察早期演化。

若出现发散或异常振荡，首先检查：
- `CFL` 是否超阈值；
- 限制器实现是否正确处理符号；
- 周期索引方向是否一致。

## R15

常见错误：
- 将限制器写成“无条件中心差分”，导致间断处振荡；
- 界面左右状态索引错位（`j+1/2` 处混淆 `j` 与 `j+1`）；
- 通量差分方向写反，导致“反扩散”发散；
- 用 `round` 估计步数造成 CFL 偶发超限；
- 忽略有限数检查，溢出后难定位。

本实现用 `ceil` 控步长、显式 CFL 校验和 `isfinite` 检查降低这些风险。

## R16

与相邻格式对比：
- 相比一阶上风/Lax-Friedrichs：MUSCL 在光滑区更精确、数值耗散更小；
- 相比 Lax-Wendroff（无限制）：MUSCL 在间断附近更稳健、振荡更少；
- 相比 WENO：MUSCL 实现更简单、开销更低，但在高阶精度与复杂波结构上不如 WENO。

定位上，MUSCL 常是“工程稳健 + 中等精度 + 低复杂度”的实用方案。

## R17

可扩展方向：
- 将线性平流替换为 Burgers 方程 `f(u)=u^2/2`；
- 将 `minmod` 改为 MC / van Leer / Superbee 等 limiter 对比；
- 在系统方程（Euler）上与 HLL/HLLC 黎曼求解器组合；
- 从一维扩展到二维（维度分裂或真正多维重构）；
- 增加误差-耗时基准，对比 MUSCL/WENO/DG。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 设定 `a, t_end, cfl_target` 和网格列表，启动收敛与间断两组实验。  
2. `run_resolution_case` 调用 `solve_muscl`，在 `[0,1)` 上生成网格并初始化 `u0`。  
3. `solve_muscl` 用 `dt0 = cfl_target*dx/|a|` 估计步长，`ceil` 得到 `n_steps` 后回算 `dt`，并校验实际 `CFL<=1`。  
4. 每个时间步执行 `ssp_rk2_step`：先在当前 `u` 上调用 `muscl_spatial_operator` 得到 `k1`。  
5. `muscl_spatial_operator` 内部先 `muscl_reconstruct`：通过 `minmod` 斜率生成每个界面的 `u_L, u_R`。  
6. 用 `rusanov_flux_linear(u_L,u_R,a)` 计算所有界面通量 `F_{j+1/2}`，再以 `-(F_{j+1/2}-F_{j-1/2})/dx` 形成半离散残差。  
7. SSP-RK2 第二阶段在预测解 `u1` 上再次计算空间残差 `k2`，再凸组合得到 `u^{n+1}`，并检查结果全为有限数。  
8. 仿真结束后构造精确平移解，计算 `L1/L2/Linf`、`mass_error`、`TV` 变化与经验阶，输出并用断言做最小正确性门禁。  

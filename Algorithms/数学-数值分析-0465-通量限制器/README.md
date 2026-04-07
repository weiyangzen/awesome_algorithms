# 通量限制器

- UID: `MATH-0465`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `465`
- 目标目录: `Algorithms/数学-数值分析-0465-通量限制器`

## R01

通量限制器（Flux Limiter）用于在双曲守恒律离散中折中“高阶精度”和“无振荡稳定性”：
- 在光滑区尽量接近二阶（减少数值扩散）；
- 在陡梯度/间断附近自动退化到一阶单调格式（抑制伪振荡）。

本条目实现一个最小可运行 MVP：
- 方程：一维线性平流 `u_t + a u_x = 0`；
- 对比格式：一阶迎风 vs TVD 通量限制器；
- 输出：误差范数、经验收敛阶、总变差、过冲量、质量误差。

## R02

目标问题：
\[
\partial_t u + \partial_x(a u)=0,\quad x\in[0,1),\ t\in[0,T],\ a>0.
\]

边界条件采用周期边界：`u(0,t)=u(1,t)`。

若初值为 `u_0(x)`，解析解为平移：
\[
u(x,t)=u_0((x-a t)\bmod 1).
\]

该模型既能做光滑收敛实验，也能用方波初值测试无振荡性质。

## R03

离散记号：
- 网格点：`x_j=jΔx`, `j=0,...,N-1`, `Δx=1/N`；
- 时间层：`t^n=nΔt`；
- 离散解：`u_j^n`。

CFL 数：
\[
\nu = a\Delta t/\Delta x.
\]

本 MVP 使用显式推进，并在代码中强制检查 `0 < ν <= 1`。

## R04

构造思想：把一阶迎风通量与二阶 Lax-Wendroff 修正做受限混合。

对 `a>0`：
- 低阶迎风通量：
\[
F^{\text{up}}_{j+1/2}=a u_j;
\]
- 二阶修正差：
\[
F^{\text{LW}}_{j+1/2}-F^{\text{up}}_{j+1/2}=\frac{a}{2}(1-\nu)(u_{j+1}-u_j).
\]

限制后通量：
\[
F_{j+1/2}=F^{\text{up}}_{j+1/2}+\phi(r_j)\big(F^{\text{LW}}_{j+1/2}-F^{\text{up}}_{j+1/2}\big),
\]
\[
r_j=\frac{u_j-u_{j-1}}{u_{j+1}-u_j}.
\]

守恒更新：
\[
u_j^{n+1}=u_j^n-\frac{\Delta t}{\Delta x}(F_{j+1/2}-F_{j-1/2}).
\]

## R05

`demo.py` 实现了 4 个常见限制器：
- `minmod`：最保守，扩散较大；
- `vanleer`：平滑，常作默认；
- `superbee`：更锐利，但更激进；
- `mc`（monotonized central）：折中方案。

默认使用 `vanleer`，兼顾光滑区精度和间断区稳定性。

## R06

数值性质（本 MVP 场景）：
- 一阶迎风：鲁棒、单调，但扩散明显；
- 通量限制器：在光滑区可接近二阶，在间断区自动降阶，保持 TVD 倾向；
- 代码通过方波案例检查总变差和过冲量，避免“看起来高阶但振荡失控”。

## R07

初值与边界设置：
- 光滑初值：`sin(2πx) + 0.2 cos(4πx)`，用于收敛阶测试；
- 方波初值：区间 `[0.2,0.5]` 为 1，其余为 0，用于间断诊断；
- 周期边界用 `np.roll` 实现，不额外引入幽灵单元。

## R08

复杂度：
- 单步更新均为向量化常数次运算，时间复杂度 `O(N)`；
- 总复杂度 `O(N * N_t)`；
- 空间复杂度 `O(N)`。

这是“教学/验证级”最小实现，便于进一步扩展到更复杂守恒律。

## R09

算法伪代码：

```text
输入: nx, a>0, t_end, cfl_target, u0(x), limiter
构建网格 x, 初始化 u=u0(x)
计算 dt0=cfl_target*dx/a
取 n_steps=ceil(t_end/dt0), 回算 dt=t_end/n_steps
计算 cfl=a*dt/dx, 校验 0<cfl<=1
循环 n_steps:
    if scheme==upwind:
        F = u
    else:
        du_minus = u - roll(u,1)
        du_plus  = roll(u,-1) - u
        r = safe_div(du_minus, du_plus)
        phi = limiter(r)
        F = u + 0.5*(1-cfl)*phi*du_plus
    u = u - cfl*(F - roll(F,1))
构造精确解 u_exact(x)=u0((x-a*t_end) mod 1)
输出误差范数、TV、过冲、质量误差
```

## R10

`demo.py` 模块对应关系：
- `limiter_phi`：限制器函数集合；
- `upwind_step`：一阶迎风单步；
- `flux_limiter_step`：通量限制器单步；
- `solve_advection`：完整时间推进和 CFL 管理；
- `exact_solution_periodic`：精确解；
- `error_norms` / `total_variation` / `overshoot_amount`：诊断指标；
- `run_case`：单实验包装；
- `main`：批量实验与断言校验。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0465-通量限制器
python3 demo.py
```

脚本无交互输入，执行后直接输出实验表格和检查结论。

## R12

输出指标说明：
- `L1/L2/Linf`：数值解与解析解误差；
- `mass_err`：离散质量初末差；
- `Empirical order`：多网格误差估计阶；
- `TV0/TVT/TV_drop`：初末总变差变化；
- `overshoot`：超出初值范围 `[min(u0), max(u0)]` 的总量。

这些指标一起判断：精度、守恒、无振荡是否同时达标。

## R13

最小验证集：
1. 光滑初值 + `nx=100,200,400`：比较迎风与限制器误差和收敛阶；
2. 方波初值 + `nx=400`：比较 TV 与过冲；
3. 每组都检查质量误差接近 0；
4. 通过断言保证回归时不悄悄退化。

## R14

参数建议：
- `cfl_target`：`0.7~0.9`；
- `t_end`：`0.2~0.8`（足够观察传播，又不至于累积过多误差）；
- 限制器优先顺序：`vanleer`（默认）→ `mc` → `minmod/superbee` 按需求试验。

如果发现异常振荡，先降低 `cfl_target` 并检查 `r` 的分母零除处理。

## R15

常见实现错误：
- 把限制修正项乘在错误差分方向（应使用 `u_{j+1}-u_j` 的高阶修正项）；
- `r` 比值直接相除导致除零警告或 NaN 传播；
- 只看 `L2` 误差，不检查间断处过冲与 TV 变化；
- 忘记按实际 `dt` 重算 CFL，导致表面参数合法但真实步长超阈值。

本实现使用 `np.divide(..., where=...)` 做安全比值，并内置轻量断言。

## R16

与相邻条目关系：
- 相比 `Lax-Friedrichs`：限制器显著降低数值扩散；
- 相比 `Lax-Wendroff`：限制器在间断附近更稳，减少 Gibbs 型振荡；
- 与一阶迎风关系：迎风是限制器退化状态（`phi=0`）的稳定基线。

因此通量限制器是从“一阶稳健”过渡到“高分辨率格式”的关键桥梁。

## R17

可扩展方向：
- 支持 `a<0` 与变系数 `a(x,t)` 的统一上风方向实现；
- 扩展到非线性守恒律（如 Burgers/Euler）并接 Riemann 通量；
- 引入 MUSCL-Hancock、SSP-RK 时间推进；
- 与 ENO/WENO 做精度-代价基准对比。

## R18

`demo.py` 源码级流程（8 步）：
1. `main` 设定 `a, t_end, cfl_target, limiter`，并准备光滑/方波两类实验。  
2. `run_case` 调用 `solve_advection`，在 `[0,1)` 上构建均匀网格并离散初值。  
3. `solve_advection` 用 `dt0=cfl_target*dx/a` 估计步长，再用 `ceil` 反推 `n_steps` 和实际 `dt`，计算真实 CFL。  
4. 在每个时间步，若是基线格式则走 `upwind_step`，即 `F=u` 的守恒更新。  
5. 若是限制器格式则进入 `flux_limiter_step`：先算 `du_minus`、`du_plus`，再用 `np.divide(..., where=...)` 安全构造 `r`。  
6. `limiter_phi` 根据 `r` 逐点给出 `phi(r)`，并用 `F = u + 0.5*(1-cfl)*phi*du_plus` 形成受限高阶通量。  
7. 用守恒差分 `u <- u - cfl*(F-roll(F,1))` 推进；推进结束后由 `exact_solution_periodic` 构造解析参考解。  
8. `error_norms/total_variation/overshoot_amount` 汇总精度与无振荡指标，`main` 打印表格并执行阈值断言。  

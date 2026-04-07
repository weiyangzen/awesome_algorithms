# MacCormack格式

- UID: `MATH-0463`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `463`
- 目标目录: `Algorithms/数学-数值分析-0463-MacCormack格式`

## R01

MacCormack 格式是求解双曲型守恒律/平流方程的经典显式二阶预测-校正（predictor-corrector）方法。它用一次单边差分做预测、再用反向单边差分做校正，在保持实现简单的同时获得二阶精度。

本条目给出一个可运行最小 MVP：
- 目标方程取一维线性平流 `u_t + a u_x = 0`；
- 周期边界下实现 MacCormack 时间推进；
- 输出与精确解对比的误差范数、质量误差、经验收敛阶；
- 附加间断初值诊断观察色散振荡特征。

## R02

数学模型：
\[
\partial_t u + a\,\partial_x u = 0,
\quad x\in[0,1),\ t\in[0,T],
\]
其中 `a` 为常数传播速度。

边界条件采用周期形式 `u(0,t)=u(1,t)`。若初值为 `u_0(x)`，精确解为平移：
\[
u(x,t)=u_0((x-a t)\bmod 1).
\]

## R03

离散定义：
- 空间网格：`x_j=jΔx`, `j=0,...,N-1`, `Δx=1/N`；
- 时间网格：`t^n=nΔt`；
- 网格函数：`u_j^n ≈ u(x_j,t^n)`。

CFL 数定义为：
\[
\nu = \frac{a\Delta t}{\Delta x}.
\]
对线性平流 MacCormack 显式格式，通常要求 `|ν|<=1` 保持稳定。

## R04

MacCormack 两步法（以 `a>0` 为例）：

预测步（前向差分）：
\[
\tilde u_j = u_j^n - \nu\,(u_{j+1}^n-u_j^n).
\]

校正步（后向差分，作用于预测场）：
\[
u_j^{n+1} = \frac12\left(u_j^n + \tilde u_j - \nu\,(\tilde u_j-\tilde u_{j-1})\right).
\]

当 `a<0` 时交换差分方向（预测用后向、校正用前向），避免迎风方向错误。

## R05

稳定性与误差行为：
- 线性常系数情形在 `|ν|<=1` 时可稳定计算；
- 对平滑解表现二阶收敛（空间/时间耦合）；
- 对间断数据会出现色散型过冲/欠冲（Gibbs 类现象）；
- 相比一阶迎风，耗散更小但振荡风险更高。

## R06

与相邻格式对比：
- 对比迎风格式：MacCormack 精度更高、边沿更锐利；
- 对比 Lax-Friedrichs：MacCormack 人工黏性更低，细节保持更好；
- 对比 Lax-Wendroff：在线性平流上两者可视为等价阶次方案，误差形态相近；
- 对比 TVD/WENO：MacCormack 更轻量，但间断抑振能力不足。

## R07

边界处理策略：
- 本实现采用周期边界；
- 通过 `np.roll(u, -1)` 与 `np.roll(u, 1)` 获取相邻点；
- 无需显式构造 ghost cells，索引逻辑清晰。

该处理方式适合教学与小规模基准验证。

## R08

复杂度：
- 单步计算由常数次向量运算组成，时间复杂度 `O(N)`；
- 总复杂度 `O(N * N_t)`；
- 额外内存约 `O(N)`（当前场 + 预测场）。

相比更复杂高分辨率格式，MVP 具有极低实现和维护成本。

## R09

算法伪代码：

```text
输入: nx, a, t_end, cfl_target, u0(x)
构建网格 x 并初始化 u=u0(x)
计算 dx=1/nx, dt0=cfl_target*dx/|a|
取 n_steps=ceil(t_end/dt0), 回算 dt=t_end/n_steps
计算 ν=a*dt/dx, 校验 |ν|<=1
循环 n_steps 次:
    若 ν>=0:
        u_pred = u - ν*(roll(u,-1)-u)
        u      = 0.5*(u + u_pred - ν*(u_pred-roll(u_pred,1)))
    否则:
        u_pred = u - ν*(u-roll(u,1))
        u      = 0.5*(u + u_pred - ν*(roll(u_pred,-1)-u_pred))
构造精确解 u_exact(x,t_end)=u0((x-a*t_end) mod 1)
输出误差范数、守恒误差与收敛阶
```

## R10

`demo.py` 结构：
- `initial_condition_smooth`：平滑初值（收敛实验）；
- `initial_condition_square`：方波初值（振荡诊断）；
- `maccormack_step`：单步预测-校正核心；
- `solve_maccormack`：完整时间推进与 CFL 检查；
- `exact_periodic_solution`：周期精确解；
- `error_norms`：`L1/L2/Linf` 误差；
- `run_resolution_case`：单分辨率测试；
- `main`：多网格汇总和断点评估。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0463-MacCormack格式
python3 demo.py
```

脚本无交互输入，运行后直接输出表格与诊断结果。

## R12

输出字段说明：
- `nx`：空间网格数；
- `n_steps`：时间步数；
- `actual_cfl`：回算后的实际 CFL；
- `L1/L2/Linf`：终时刻误差范数；
- `mass_error`：离散总量初末差；
- `p(100->200), p(200->400)`：经验收敛阶估计；
- `TV0/TVT, min/max`：间断案例的总变差和过冲范围。

## R13

最小验证集建议：
1. 平滑初值 + `nx=100/200/400`，检查误差随网格细化下降；
2. 计算经验阶并验证接近二阶；
3. 间断初值检查 `min(u), max(u)` 观察色散振荡；
4. 检查 `mass_error` 是否接近机器精度。

## R14

参数建议：
- `cfl_target`: `0.7~0.9`；
- `t_end`: `0.2~0.8`；
- 收敛实验优先增加 `nx`；
- 若出现发散，优先核对 `|actual_cfl|` 与方向差分实现。

## R15

常见实现错误：
- 忘记在 `a<0` 时反转预测/校正差分方向；
- 使用 `round` 取步数导致 `dt` 变大从而 CFL 失控；
- 周期索引写反，导致相位错误；
- 跳过有限值检查，数值炸解难以及时发现。

本实现用 `ceil` 回算 `dt` 并显式检查稳定性与有限性。

## R16

工程使用建议：
- 作为二阶显式基线用于新问题原型验证；
- 与 Lax-Wendroff、迎风、TVD/WENO 做统一误差-成本对照；
- 间断主导问题应配合限制器或改用 TVD/WENO；
- 在教学场景中适合演示“二阶精度与色散振荡的权衡”。

## R17

可扩展方向：
- 非线性通量（如 Burgers 方程）并增加局部波速控制；
- 叠加人工黏性或通量限制器构建 TVD-MacCormack；
- 扩展到二维平流（方向分裂或无分裂）；
- 增加误差随 CFL 变化曲线，研究稳定裕量与相位误差。

## R18

`demo.py` 的源码级执行流程（8 步）：
1. `main` 设定 `a, t_end, cfl_target` 与多组 `nx`，用于收敛测试。  
2. `run_resolution_case` 调用 `solve_maccormack`，创建网格 `x` 与离散初值 `u0`。  
3. `solve_maccormack` 先用 `dt0=cfl_target*dx/|a|` 估算步长，再通过 `n_steps=ceil(t_end/dt0)` 回算 `dt`，保证终止时刻精确覆盖且 CFL 不被放大。  
4. 计算真实 `ν=a*dt/dx` 并做 `|ν|<=1` 稳定性检查；若不满足直接报错。  
5. 每个时间步调用 `maccormack_step`：先生成预测场 `u_pred`，再根据速度符号选择反向差分进行校正。  
6. 校正后得到 `u_next`，并通过 `ensure_finite_array` 检查是否出现 `NaN/Inf`。  
7. 时间推进完成后，`exact_periodic_solution` 用 `(x-a*t)%1` 构造精确平移解，`error_norms` 计算 `L1/L2/Linf`。  
8. `main` 汇总多网格误差并估计收敛阶，再运行方波诊断输出 `TV0/TVT` 与 `min/max`，展示 MacCormack 在间断处的色散特征。  

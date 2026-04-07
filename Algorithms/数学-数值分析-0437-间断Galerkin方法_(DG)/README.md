# 间断Galerkin方法 (DG)

- UID: `MATH-0437`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `437`
- 目标目录: `Algorithms/数学-数值分析-0437-间断Galerkin方法_(DG)`

## R01

间断 Galerkin 方法（Discontinuous Galerkin, DG）是一类把“有限元局部多项式表示”与“有限体积界面通量思想”结合起来的高阶守恒律离散方法。
它允许单元间解不连续，通过数值通量在界面耦合各单元，因此同时具备：

- 局部高阶表示能力（每个单元内可用高阶多项式）
- 严格局部守恒结构
- 便于并行的单元级计算模式

本条目给出一个最小可运行 MVP：一维线性平流方程上的 `DG(P1) + 上风/LF 通量 + SSP-RK3`。

## R02

模型方程：
\[
u_t + a u_x = 0,\quad x\in[0,1),\ t\in[0,T],
\]
其中 `a` 为常数传播速度，边界取周期条件。

周期问题有解析解：
\[
u(x,t)=u_0((x-a t)\bmod 1),
\]
因此非常适合验证 DG 实现是否正确、是否达到理论收敛阶。

## R03

空间离散设置：

- 区间划分为 `N` 个单元，单元长度 `h=1/N`
- 每个单元上使用一次多项式（`P1`）局部近似
- 参考单元基函数选为模态基 \(\phi_0(\xi)=1,\ \phi_1(\xi)=\xi\), \(\xi\in[-1,1]\)

于是单元内近似写作：
\[
u_h|_{K_j}(\xi,t)=u_{j,0}(t)\phi_0(\xi)+u_{j,1}(t)\phi_1(\xi).
\]
代码里 `coeffs[:,0]` 和 `coeffs[:,1]` 分别对应 \(u_{j,0},u_{j,1}\)。

## R04

DG 弱形式（守恒型）对每个单元 \(K_j\) 与测试函数 \(v\)：
\[
\int_{K_j} u_t v\,dx
-\int_{K_j} a u v_x\,dx
+\hat f_{j+1/2}v^-_{j+1/2}
-\hat f_{j-1/2}v^+_{j-1/2}=0.
\]

这里 \(\hat f\) 是数值通量，负责处理单元间的不连续跳变。
与连续有限元不同，DG 不强制单元边界连续，而是通过通量实现信息交换和稳定性控制。

## R05

本实现采用局部 Lax-Friedrichs 通量：
\[
\hat f(u_L,u_R)=\frac12 a(u_L+u_R)-\frac12|a|(u_R-u_L).
\]

对于线性平流，这个通量等价于经典上风通量：

- `a>0` 时取左侧状态
- `a<0` 时取右侧状态

因此实现既简洁又具备物理传播方向的一致性。

## R06

在 \(\{1,\xi\}\) 模态基下，参考单元矩阵为：

- 质量矩阵 \(M=\mathrm{diag}(2,\ 2/3)\)
- 刚度矩阵 \(S=\begin{bmatrix}0&0\\2&0\end{bmatrix}\)

代入弱形式后，得到本代码使用的半离散右端（每个单元）：

- \(\dot u_{0}=(\hat f_R-\hat f_L)/h\)
- \(\dot u_{1}=(3/h)\,(-2au_0+\hat f_R+\hat f_L)\)

其中 \(\hat f_R,\hat f_L\) 分别是该单元右/左界面数值通量。
`demo.py` 的 `dg_rhs` 正是这个显式公式。

## R07

时间离散采用三阶 SSP-RK3（强稳定保持 Runge-Kutta）：

1. \(U^{(1)}=U^n+\Delta t\,L(U^n)\)
2. \(U^{(2)}=\frac34 U^n+\frac14(U^{(1)}+\Delta t\,L(U^{(1)}))\)
3. \(U^{n+1}=\frac13 U^n+\frac23(U^{(2)}+\Delta t\,L(U^{(2)}))\)

其中 \(L(\cdot)\) 是 DG 空间离散右端。
SSP-RK3 在双曲问题中是常用、稳定性表现较好的显式时间推进器。

## R08

对显式 DG，稳定步长通常满足
\[
\Delta t \propto \frac{h}{(2p+1)|a|}.
\]
本 MVP 固定 `p=1`，使用
\[
\Delta t_{\text{guess}}=\text{cfl\_target}\cdot \frac{h}{3|a|},
\]
然后按终止时刻反算整数步数得到实际 `dt`。

程序会输出 `cfl(actual)=|a|*dt/h*(2p+1)`，便于检查是否处于安全范围。

## R09

伪代码：

```text
input: n_cells, a, t_end, cfl_target
project u0(x) onto local P1 basis -> coeffs
h <- 1 / n_cells
dt_guess <- cfl_target * h / (|a|*(2p+1)), p=1
n_steps <- ceil(t_end / dt_guess), dt <- t_end / n_steps

repeat n_steps:
    reconstruct boundary traces in each cell:
        u^-_{j+1/2} = u_{j,0}+u_{j,1}
        u^+_{j-1/2} = u_{j,0}-u_{j,1}
    compute all interface fluxes by LF/upwind
    build rhs by modal semi-discrete formulas
    update coeffs with SSP-RK3

evaluate exact solution at quadrature points
report L1/L2/Linf and mass error
```

## R10

`demo.py` 默认参数：

- `a = 1.0`
- `t_end = 0.3`
- `cfl_target = 0.25`
- 网格数：`[40, 80, 160]`

初值为平滑周期函数：
\[
u_0(x)=\sin(2\pi x)+0.25\cos(4\pi x).
\]
这组设置可较稳定地观察到接近二阶的误差收敛趋势（`P1` 对平滑解通常是二阶到更高的 L2 表现，受时间离散与常数影响）。

## R11

程序输出字段：

- `cells`: 单元数
- `steps`: 时间步数
- `cfl(actual)`: 实际 DG CFL
- `L1/L2/Linf`: 与解析解的误差范数
- `mass_error`: 质量守恒误差

此外还会打印相邻网格间基于 `L2` 的实验收敛阶：
\[
p\approx \log_2(e_h/e_{h/2}).
\]
若实现正确，`L2` 阶通常应接近 2（在该 MVP 参数下）。

## R12

代码结构说明：

- `initial_condition`：定义周期光滑初值
- `project_to_p1_dg`：把初值做单元内 `L2` 投影到 P1 模态系数
- `numerical_flux`：界面 LF/上风通量
- `dg_rhs`：DG 半离散右端（核心公式）
- `ssp_rk3_step`：单步三阶 SSP-RK
- `solve_dg_advection`：完整时间推进
- `error_metrics`：用高阶求积计算误差
- `mass_from_coeffs`：计算守恒量
- `run_case/main`：批量实验和汇总打印

## R13

运行方式（仓库根目录）：

```bash
uv run python Algorithms/数学-数值分析-0437-间断Galerkin方法_(DG)/demo.py
```

或进入目录后：

```bash
uv run python demo.py
```

脚本无交互输入，执行后直接在终端打印误差与收敛阶结果。

## R14

DG 初学实现常见错误：

- 把左右迹值方向写反，导致通量取值错误
- 忘记在周期边界处做 `roll` 闭合
- 只写空间离散却用过大 `dt`，出现显式发散
- 误把单元系数当作节点值，导致后处理和误差评估错误
- 用单点误差代替积分误差，掩盖真实精度行为

本实现分别通过 `np.roll`、统一通量函数和求积误差规避这些问题。

## R15

复杂度分析（`N` 为单元数，`T` 为时间步数）：

- 单次 `dg_rhs` 复杂度：`O(N)`
- SSP-RK3 每步调用 `dg_rhs` 3 次，仍是 `O(N)`
- 总复杂度：`O(N*T)`
- 存储复杂度：`O(N)`（每单元仅 2 个模态系数）

向量化实现常数较小，作为教学级/原型级 DG 基线足够轻量。

## R16

建议的验证清单：

- 网格加密：`40 -> 80 -> 160 -> 320`，检查 `L2` 阶稳定性
- CFL 扫描：例如 `0.1/0.2/0.3`，观察稳定性与误差变化
- 速度符号测试：`a=1` 与 `a=-1` 应都能正确平移
- 守恒测试：`mass_error` 应保持在较小量级
- 初值替换测试：换为单一正弦波，便于与解析解对照

## R17

适用性边界：

- 适用：双曲守恒律、平流主导问题、需要局部高阶与局部守恒的场景
- 谨慎：强间断/激波下会出现振荡，通常需要限幅器或 WENO 重构
- 不适用：把 DG 当作“无条件稳定”的黑盒显式方法（其时间步受 CFL 强约束）

因此该 MVP 更适合作为 DG 核心机制演示与后续高分辨率扩展起点。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `project_to_p1_dg` 在每个单元上用 Gauss 求积做 `L2` 投影，得到初始模态系数 `(u0,u1)`。
2. `solve_dg_advection` 根据 `h`、`a`、`p=1` 和 `cfl_target` 计算 `dt_guess`，再反推整数步数与实际 `dt`。
3. 每个 RK 子步里，`dg_rhs` 先把模态系数重构为左右边界迹值 `u0±u1`。
4. 用相邻单元迹值组装所有界面左右态，调用 `numerical_flux` 计算 LF/上风通量。
5. 把右界面通量与左界面通量代入模态半离散公式，得到每个单元的 \(\dot u_0,\dot u_1\)。
6. `ssp_rk3_step` 按 3 个 stage 组合右端，完成一步三阶显式时间推进。
7. 时间循环结束后，`error_metrics` 在单元内高阶求积点重构数值解，并与周期解析解对比得到 `L1/L2/Linf`。
8. `main` 汇总多组网格结果，打印实际 CFL、守恒误差与 `L2` 实验收敛阶。

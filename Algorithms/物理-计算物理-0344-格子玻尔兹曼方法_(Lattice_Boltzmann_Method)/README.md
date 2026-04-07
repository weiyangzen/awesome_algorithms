# 格子玻尔兹曼方法 (Lattice Boltzmann Method)

- UID: `PHYS-0337`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `344`
- 目标目录: `Algorithms/物理-计算物理-0344-格子玻尔兹曼方法_(Lattice_Boltzmann_Method)`

## R01

格子玻尔兹曼方法（Lattice Boltzmann Method, LBM）是一种介于微观动力学与宏观流体方程之间的数值方法。  
它不直接离散 Navier-Stokes 方程，而是离散化速度空间中的粒子分布函数 `f_i`，再通过碰撞和迁移（streaming）演化。

核心更新形式（BGK 单松弛模型）：

`f_i(x + c_i*dt, t+dt) = f_i(x,t) - ω [f_i(x,t) - f_i^eq(x,t)] + S_i`

其中 `ω=1/τ`，`f_i^eq` 是局部平衡分布，`S_i` 是外力源项。

## R02

本条目 MVP 选择最经典、可验证的测试问题：二维通道 Poiseuille 流。

- 空间：`nx x ny` 规则格点；
- x 方向：周期边界；
- y 方向：上下固壁（无滑移，bounce-back）；
- 驱动：恒定体力 `g_x`；
- 目标：收敛后速度剖面应接近抛物线解析解。

这样可以在小规模代码里完整体现 LBM 的“碰撞-迁移-边界-宏观量恢复-解析对照”闭环。

## R03

离散速度模型采用 D2Q9：

- 9 个离散方向 `c_i=(c_{ix}, c_{iy})`；
- 权重 `w_i = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]`；
- 晶格声速平方 `c_s^2 = 1/3`。

平衡分布使用二阶展开：

`f_i^eq = w_i * rho * [1 + 3(c_i·u) + 4.5(c_i·u)^2 - 1.5|u|^2]`

这也是工程上最常见的弱可压 LBM 配置。

## R04

粘性参数与松弛时间关系：

`nu = c_s^2 * (tau - 0.5) = (tau - 0.5)/3`

因此 `tau > 0.5` 才有正粘性。  
本 MVP 默认 `tau=0.80`，对应 `nu=0.1`（晶格单位），在稳定性和收敛速度间较平衡。

## R05

外力处理采用 Guo forcing（源码显式实现，不依赖黑盒 CFD 库）：

`S_i = w_i*(1-1/(2*tau))*[((c_i-u)/c_s^2) + ((c_i·u)/c_s^4)c_i]·g`

并在宏观速度恢复中使用半步修正：

`u = (sum_i f_i*c_i + 0.5*g) / rho`

该处理比“直接改速度”更一致，尤其在低马赫稳态流中误差更小。

## R06

边界条件实现策略：

- x 向周期：通过 `np.roll` 做 streaming 时天然实现；
- y 向固壁：上下边界执行 on-node bounce-back：
  - 底壁：`f2<-f4, f5<-f7, f6<-f8`
  - 顶壁：`f4<-f2, f7<-f5, f8<-f6`

这是一种简洁可靠的无滑移近似，适合 MVP。

## R07

`demo.py` 的主流程：

1. 初始化 `rho=1, u=0, f=f_eq`。
2. 从 `f` 恢复 `rho, ux, uy`（含半力修正）。
3. 在壁面将 `u` 设为 0，构造 `f_eq`。
4. 计算 Guo 力项 `S_i`。
5. 进行 BGK 碰撞：`f_post = f - ω(f-feq) + S_i`。
6. 按离散速度做 streaming。
7. 应用上下壁 bounce-back。
8. 周期记录收敛历史（`u_max`、密度范围、剖面残差）。
9. 结束后与解析抛物线剖面对比并断言。

## R08

复杂度估计（设总格点 `N = nx * ny`，离散方向 `Q=9`，步数 `T`）：

- 每一步主要是 `Q` 个方向上的数组运算，时间复杂度 `O(T * Q * N)`；
- 存储 `f`、`feq` 及若干标量场，空间复杂度 `O(Q * N)`。

在本配置 `64 x 24 x 5000` 下，CPU 即可在短时间内完成。

## R09

数值稳定与工程约束：

- 低马赫约束：通过小体力保持 `Ma << 1`；
- `tau > 0.5`：保证正粘性；
- 使用密度漂移监控 `max|rho-1|`；
- 使用剖面增量残差监控稳态收敛；
- 解析剖面对照防止“仅看残差但解偏离物理真值”。

## R10

MVP 技术栈：

- Python 3
- `numpy`：LBM 核心数组计算（碰撞/迁移/宏观量恢复）
- `pandas`：收敛历史和汇总表输出

没有使用专用 CFD 包。关键算法步骤都在源码中逐行可追踪。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0344-格子玻尔兹曼方法_(Lattice_Boltzmann_Method)
uv run python demo.py
```

脚本无交互输入，会打印：

- 按 `report_every` 记录的历史表；
- 最终摘要（误差、马赫数、质量漂移、收敛残差）；
- `Validation: PASS` 或断言失败信息。

## R12

主要输出指标解释：

- `u_max`：通道内最大平均流速；
- `rho_min/rho_max`：密度上下界，反映弱可压偏差；
- `profile_residual`：相邻两次报告的速度剖面最大变化；
- `relative_L2_error_profile`：数值剖面与解析抛物线的相对 `L2` 误差；
- `max_mach = sqrt(3)*u_max`：低马赫性检查；
- `mass_drift_inf = max|rho-1|`：整体守恒漂移量。

## R13

内置验收阈值（`demo.py` 断言）：

1. `relative_L2_error_profile < 0.12`
2. `max_mach < 0.1`
3. `mass_drift_inf < 8e-3`
4. `final_profile_residual < 2e-4`

全部满足则输出 `Validation: PASS`。

## R14

当前 MVP 的局限：

- 仅 D2Q9 单相、弱可压、层流示例；
- 仅 BGK 单松弛，不含 MRT/Entropic 等更强稳定模型；
- 边界仅使用简单 bounce-back，未实现高阶曲壁/入口出口条件；
- 未并行化（单机 numpy）。

## R15

可扩展方向：

- 用 MRT-LBM 提升高雷诺数稳定性；
- 加入 Zou/He 压力或速度边界，支持入口/出口问题；
- 扩展到障碍绕流（圆柱流）并提取阻力系数、Strouhal 数；
- 引入热 LBM 或多相 LBM（Shan-Chen 等）处理更复杂物理；
- 使用 `numba` 或 GPU 框架加速大规模网格。

## R16

典型应用场景：

- 多孔介质渗流模拟；
- 微流控低速流动与混合；
- 复杂几何通道内流动（LBM 对网格和边界处理较友好）；
- 作为 CFD 教学与原型验证工具。

## R17

与传统 Navier-Stokes 有限体积/有限差分法相比：

- LBM 优势：局部更新结构规整，易并行，复杂边界处理直观；
- LBM 劣势：本质弱可压，参数区间和马赫数限制更严格；
- 本条目选择 LBM MVP：能在少量代码内展示完整算法链路，并可用解析解做强校验。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `d2q9_lattice` 定义 D2Q9 的 `cx/cy/w` 与对向索引，建立离散速度空间。  
2. `equilibrium` 根据 `rho,u` 计算 `f_eq`，实现二阶 Maxwell-Boltzmann 展开。  
3. `macroscopic` 从分布函数求 `rho,momentum`，并加入 `0.5*g` 半步力修正得到速度。  
4. `guo_forcing_term` 按 Guo 公式逐方向构造外力源项 `S_i`。  
5. `run_lbm` 每个时间步执行 BGK 碰撞：`f_post = f - omega*(f-feq) + S_i`。  
6. `stream` 对每个离散方向做 `np.roll` 迁移，自动形成 x 周期边界。  
7. `apply_bounce_back_walls` 在上下壁做反弹回跳，形成无滑移固壁条件。  
8. 迭代过程中记录 `u_max/rho_range/profile_residual`，用于监控收敛与守恒。  
9. `main` 调用 `analytic_poiseuille_profile` 生成解析抛物线，计算相对 `L2` 误差、马赫数、质量漂移并执行断言。  

说明：`numpy/pandas` 只承担通用数组和表格功能；LBM 的碰撞、迁移、力项和边界逻辑均在源码中显式实现，没有把算法交给第三方黑盒。

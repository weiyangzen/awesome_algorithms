# Crank-Nicolson方法

- UID: `MATH-0452`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `452`
- 目标目录: `Algorithms/数学-数值分析-0452-Crank-Nicolson方法`

## R01

本条目实现 Crank-Nicolson 方法（简称 CN）求解一维热方程的最小可运行版本，并给出可验证误差的数值实验。

MVP 目标：
- 展示从 PDE 到离散线性系统的完整链路；
- 不把核心线性求解当黑盒，手写三对角 Thomas 算法；
- 通过与解析解对比，输出误差与收敛阶。

## R02

问题定义（初边值问题）：
- 方程：`u_t = alpha * u_xx`，`x in [0, 1]`，`t > 0`；
- 边界：`u(0, t) = 0`，`u(1, t) = 0`（零 Dirichlet）；
- 初值：`u(x, 0) = sin(pi x)`；
- 参数：`alpha > 0`，终止时间 `t_final > 0`。

该设置下解析解已知：
`u(x, t) = exp(-alpha*pi^2*t) * sin(pi*x)`，
可直接用于误差验证。

## R03

离散化与 CN 公式：
- 空间等距网格：`x_i = i*dx`，`dx = 1/nx`，仅保留内部点 `i=1..nx-1`；
- 时间网格：`t^n = n*dt`，`dt = t_final/nt`；
- 定义 `r = alpha*dt/dx^2`。

记离散 Laplacian 模板 `L u_i = u_{i-1} - 2u_i + u_{i+1}`，
CN 在内部点上的矩阵形式为：
`(I - 0.5*r*L) u^{n+1} = (I + 0.5*r*L) u^n`。

对应三对角系数：
- 左矩阵：主对角 `1+r`，上下对角 `-r/2`；
- 右矩阵：主对角 `1-r`，上下对角 `+r/2`。

## R04

算法流程：
1. 生成网格与初值 `u(x,0)`（仅内部节点）。
2. 根据 `r` 组装 CN 左右三对角系数。
3. 每个时间步先构造右端向量 `rhs = B*u^n`。
4. 求解三对角线性系统 `A*u^{n+1}=rhs`。
5. 在选定步（起点/中点/终点）计算 `L2` 误差快照。
6. 终止后计算最终 `L2` 与 `L_inf` 误差。
7. 对多组网格输出误差表与观测收敛阶。

## R05

核心数据结构：
- `RunResult`（`dataclass`）：
  - `nx`, `nt`, `dx`, `dt`, `r`；
  - `l2_error`, `max_error`；
  - `snapshots: list[(step, time, l2_error)]`。
- 向量存储：
  - 内部解向量 `u` 长度 `nx-1`；
  - 三对角向量 `lower/diag/upper`。

## R06

正确性依据：
- CN 对扩散方程是时间二阶、空间二阶（在本问题光滑解下可观测）；
- 本实现每步严格解离散方程 `A*u^{n+1}=B*u^n`；
- 用解析解逐点对比计算误差，避免“自洽但不正确”的假收敛；
- `solve_tridiagonal` 显式实现 Thomas 前消元+回代，数值路径可追踪。

## R07

复杂度分析：
- 设内部点数 `m = nx-1`，时间步数 `nt`。
- 每步：
  - 构造 `rhs`：`O(m)`；
  - Thomas 求解：`O(m)`。
- 总时间复杂度：`O(nt * m)`；
- 空间复杂度：`O(m)`（不存全时序，仅存当前步与少量快照）。

## R08

边界与异常处理：
- `alpha <= 0`、`t_final <= 0`、`nx < 2`、`nt < 1` 会抛 `ValueError`；
- 零 Dirichlet 边界在代码中通过“不把边界纳入未知向量 + 回填 0”实现；
- `solve_tridiagonal` 对 `n=0/1` 做特判，避免越界。

## R09

MVP 取舍说明：
- 选择最经典热方程样例，优先保证可验证性与实现透明度；
- 不依赖 `scipy` 线性求解器，避免把关键步骤隐藏在库内部；
- 只实现零 Dirichlet 边界，不引入额外工程复杂度（足够说明 CN 核心机制）。

## R10

`demo.py` 函数职责：
- `initial_condition` / `exact_solution`：给出初值与解析解；
- `l2_error`：统一 `L2` 误差计算；
- `solve_tridiagonal`：Thomas 三对角解法；
- `crank_nicolson_heat_1d`：执行一次完整 CN 仿真并返回 `RunResult`；
- `print_convergence_table`：输出网格误差表与观测阶；
- `print_snapshots`：输出关键时间点误差；
- `main`：配置多组网格并串联整个实验流程。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0452-Crank-Nicolson方法
python3 demo.py
```

脚本无需交互输入，会自动运行 3 组网格：
`(nx, nt) = (40,400), (80,800), (160,1600)`。

## R12

输出解读：
- 表格列 `nx/nt/dx/dt/r`：网格与 CFL 类参数；
- `L2 error`：离散 `L2` 范数误差；
- `Linf error`：最大点误差；
- `Observed order`：相邻两组网格误差比得到的实验收敛阶；
- `snapshots`：`step=0`、中间步、终止步误差，观察随时间演化的误差水平。

## R13

最小测试覆盖建议：
- 正常案例：默认 3 组网格（用于收敛阶验证）；
- 参数边界：
  - `alpha <= 0`；
  - `t_final <= 0`；
  - `nx = 1`；
  - `nt = 0`；
- 稳定性观察：固定 `nx` 增大 `dt`（即增大 `r`）时，CN 仍可运行但精度会下降。

## R14

关键可调参数：
- `alpha`：扩散系数；
- `t_final`：模拟终止时间；
- `nx`：空间划分数；
- `nt`：时间步数。

调参经验：
- 若看重精度，通常同步加密 `dx` 与 `dt`（例如都减半）；
- 若看重速度，在可接受误差下适当减小 `nt`。

## R15

与常见方法对比：
- 显式 FTCS：
  - 优点：实现最简单；
  - 缺点：受稳定性约束（`r` 不能太大）。
- 后向欧拉：
  - 优点：隐式稳定；
  - 缺点：时间一阶精度。
- Crank-Nicolson（本条目）：
  - 隐式求解，时间二阶，通常在精度/稳定性之间更平衡。

## R16

应用场景：
- 热传导/扩散类 PDE 教学与快速原型；
- 金融工程中的抛物型方程（如 Black-Scholes 离散）；
- 作为更高维隐式方法或 ADI 方法的基础模块。

## R17

可扩展方向：
- 支持非零或时间相关 Dirichlet/Neumann/Robin 边界；
- 扩展到二维网格并结合 ADI 分裂；
- 加入自适应时间步控制；
- 以 `scipy.sparse` 重构为大规模稀疏矩阵版本；
- 与显式、后向欧拉做统一 benchmark（误差-耗时曲线）。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 设置 `alpha/t_final` 与三组 `(nx, nt)`，逐组调用 `crank_nicolson_heat_1d`。  
2. `crank_nicolson_heat_1d` 校验参数后创建网格，计算 `dx`、`dt`、`r`。  
3. 用 `initial_condition` 生成内部初值向量 `u`，并构造 CN 左右三对角系数。  
4. 在每个时间步先按右矩阵模板拼装 `rhs = B*u^n`（仅向量邻接运算）。  
5. 调用 `solve_tridiagonal` 对 `A*u^{n+1}=rhs` 做 Thomas 前消元与回代。  
6. 在起始/中间/结束步，把内部解回填到全网格并用 `exact_solution` 计算快照误差。  
7. 全部时间步完成后，计算最终 `L2` 与 `L_inf` 误差并封装成 `RunResult`。  
8. `print_convergence_table` 输出误差与观测阶，`print_snapshots` 输出关键步误差。  

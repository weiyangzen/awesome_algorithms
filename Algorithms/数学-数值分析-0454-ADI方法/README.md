# ADI方法

- UID: `MATH-0454`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `454`
- 目标目录: `Algorithms/数学-数值分析-0454-ADI方法`

## R01

本条目给出 ADI（Alternating Direction Implicit，交替方向隐式）方法的最小可运行实现，用于求解二维热方程：

`u_t = alpha * (u_xx + u_yy)`。

MVP 目标：
- 明确展示 ADI 的“两次一维隐式求解”机制；
- 给出可复现的误差指标（`L2` 与 `L∞`）；
- 保持代码短小、可读、可直接执行。

## R02

问题设置（demo.py 内固定）：
- 空间区域：`[0,1] x [0,1]`；
- 边界条件：齐次 Dirichlet（边界恒为 0）；
- 初值：`u(x,y,0)=sin(pi x) sin(pi y)`；
- 扩散系数：`alpha=0.1`；
- 网格：`41 x 41`（含边界）；
- 时间步：`dt=0.002`，步数 `50`（终止时刻 `t=0.1`）。

该初值有解析解，便于直接做数值误差验证。

## R03

方法原理（Peaceman-Rachford ADI 分裂）：

记 `r_x = alpha*dt/(2*hx^2)`，`r_y = alpha*dt/(2*hy^2)`，离散二阶差分算子分别为 `D_xx`、`D_yy`。

每个时间步 `n -> n+1` 分两半步：
1. `(I - r_x D_xx) U* = (I + r_y D_yy) U^n`
2. `(I - r_y D_yy) U^{n+1} = (I + r_x D_xx) U*`

核心收益：把二维隐式问题分解为两组一维三对角线系统，计算效率和实现复杂度都显著下降。

## R04

离散与未知量组织：
- 仅存储内部网格点（去掉边界后大小 `(N-2) x (N-2)`）；
- 边界值通过“外侧补零”自然进入差分算子；
- `laplacian_x/laplacian_y` 分别沿 x、y 方向施加二阶差分；
- 每个方向隐式矩阵都是同一个三对角矩阵，可重复使用。

## R05

代码中的关键数据结构：
- `StepRecord(step, time, l2_error, linf_error, interior_energy)`：记录每一步诊断信息；
- `abx/aby`：`scipy.linalg.solve_banded` 使用的三对角带状矩阵表示（3 x n）；
- `u`：内部网格解；
- `records: list[StepRecord]`：完整时间推进轨迹。

## R06

正确性与一致性要点：
- 空间离散使用标准二阶中心差分；
- ADI 两半步合并后与 Crank-Nicolson 思路一致，保留高稳定性；
- 通过解析解 `exp(-2*pi^2*alpha*t) sin(pi x) sin(pi y)` 验证；
- `demo.py` 最后包含误差阈值断言（`linf_error <= 1e-2`）。

## R07

复杂度分析（每个时间步）：
- 需要两轮三对角线求解：
  - x-隐式：对每一列求解，约 `O(nx*ny)`；
  - y-隐式：对每一行求解，约 `O(nx*ny)`；
- 总体每步时间复杂度 `O(nx*ny)`；
- 空间复杂度 `O(nx*ny)`。

与直接解二维稠密线性系统相比，ADI 的规模优势非常明显。

## R08

边界与异常处理：
- `n_points < 3` 会抛错；
- `alpha <= 0`、`dt <= 0`、`n_steps <= 0` 会抛错；
- 线性求解器优先使用 `scipy.solve_banded`，若不可用则自动回退到内置 Thomas 算法；
- 不依赖交互输入，保证批处理场景稳定执行。

## R09

MVP 取舍说明：
- 只实现最核心的二维热方程 ADI，不引入多物理场和复杂边界；
- 采用规则网格和固定参数，突出方法本身；
- 输出误差和能量衰减轨迹，不做可视化依赖；
- 保持单文件可读性，便于教学和复审。

## R10

`demo.py` 函数职责：
- `build_tridiagonal_banded`：构造 ADI 隐式步的三对角带状矩阵；
- `thomas_solve_banded`：无 SciPy 环境下的三对角批量求解回退实现；
- `solve_tridiagonal`：求解器分发；
- `laplacian_x/laplacian_y`：方向差分算子；
- `one_adi_step`：执行一次 ADI 全步；
- `run_adi_heat`：完整时间推进并记录误差；
- `print_trace`：格式化输出迭代轨迹；
- `main`：固定参数执行和断言。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0454-ADI方法
python3 demo.py
```

运行后会打印：
- `rx/ry` 参数；
- 最终 `L2`、`L∞` 误差；
- 中心点数值；
- 若干时间步的误差与能量轨迹。

## R12

输出字段说明：
- `step`：时间步编号；
- `time`：当前时刻；
- `l2_error`：与解析解的均方根误差；
- `linf_error`：最大绝对误差；
- `interior_energy`：内部点 `u^2` 平均值，用于观察扩散衰减趋势。

## R13

建议最小验证集合：
- 当前默认参数（`41x41`, `dt=0.002`, `t=0.1`）应稳定通过；
- 粗网格测试：`n_points=21`，验证仍可收敛；
- 精网格测试：`n_points=81`，观察误差下降；
- 时间步敏感性：固定网格下减半 `dt`，对比最终误差变化。

## R14

可调参数与建议：
- `n_points`：空间分辨率，增大可提升精度但增加计算量；
- `dt`：时间步长，过大虽可稳定但会影响精度；
- `n_steps`：模拟终止时刻由 `n_steps * dt` 决定；
- `alpha`：扩散系数，影响衰减速度。

实务建议：先锁定 `alpha` 和终止时刻，再做网格/步长收敛实验。

## R15

与其他方法的关系：
- 对比显式差分：显式法受 CFL 强约束，ADI 在扩散问题上更稳；
- 对比全隐式二维求解：ADI 把二维系统拆成一维系统，计算更经济；
- 对比纯 Crank-Nicolson 组装大矩阵：ADI 更容易工程化并利用三对角高效求解。

## R16

典型应用场景：
- 二维/三维抛物型 PDE（热传导、扩散方程）；
- 需要长时间积分且希望避免显式法严格稳定条件的场景；
- 教学中用于展示“算子分裂 + 方向隐式”的数值思想。

## R17

后续扩展方向：
- 增加非齐次源项 `f(x,y,t)`；
- 扩展到非均匀网格与各向异性扩散；
- 添加 Neumann/Robin 边界条件；
- 做参数扫描并输出 CSV 结果（便于后处理）；
- 增加单元测试覆盖异常路径与收敛性检查。

## R18

源码级算法流程（对应 `demo.py`，9 步）：
1. `main` 固定 `n_points/alpha/dt/n_steps`，调用 `run_adi_heat`，确保脚本无需交互输入。  
2. `run_adi_heat` 构建网格并计算 `rx=ry=alpha*dt/(2*h^2)`，只保留内部未知量 `u`。  
3. `build_tridiagonal_banded` 生成三对角带状矩阵 `abx/aby`，其对角为 `1+2r`，上下对角为 `-r`。  
4. 每个时间步先执行 `one_adi_step` 的半步：`rhs_half = u + ry*laplacian_y(u)`，即 y 方向显式。  
5. 对 `rhs_half` 的每一列解 `abx * u_half = rhs_half`（x 方向隐式）；求解器由 `solve_tridiagonal` 分发到 `scipy.solve_banded` 或 `thomas_solve_banded`。  
6. 再构造 `rhs_full = u_half + rx*laplacian_x(u_half)`，即 x 方向显式。  
7. 对 `rhs_full` 的每一行解 `aby * u_next = rhs_full`（代码中通过转置把“按行”转为“按列”批量求解），完成 y 方向隐式。  
8. 回到 `run_adi_heat`，重建含边界的全场解，与解析解比较并记录 `l2_error/linf_error/interior_energy` 到 `StepRecord`。  
9. `main` 打印轨迹并做最终误差阈值断言，若超阈值则抛异常，保证 MVP 可自动验收。  

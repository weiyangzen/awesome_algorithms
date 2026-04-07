# 线性规划 - 原始-对偶内点法

- UID: `MATH-0367`
- 学科: `数学`
- 分类: `优化`
- 源序号: `367`
- 目标目录: `Algorithms/数学-优化-0367-线性规划_-_原始-对偶内点法`

## R01

原始-对偶内点法（Primal-Dual Interior-Point Method, PD-IPM）是求解线性规划（LP）的主流数值算法之一。它同时维护原始变量 `x`、对偶变量 `y` 与对偶松弛变量 `s`，在每次迭代中沿着 KKT 方程的牛顿方向前进，并通过“中心路径”保持 `x>0, s>0`。

本目录给出一个可运行、可审计的 MVP：
- `demo.py` 用 `numpy` 手写 Mehrotra predictor-corrector 原始-对偶内点法。
- 算法不依赖黑盒 LP 求解器作为主流程。
- 运行后输出迭代日志、最终解、残差与互补间隙，并可选和 `scipy.optimize.linprog` 对照结果。

## R02

本实现求解的标准型为：

- 原始问题（P）：
  - `min c^T x`
  - `s.t. A x = b`
  - `x >= 0`
- 对偶问题（D）：
  - `max b^T y`
  - `s.t. A^T y + s = c`
  - `s >= 0`

其中：
- `A in R^{m x n}`，`b in R^m`，`c in R^n`。
- `x` 是原始变量，`y` 是等式约束乘子，`s` 是对偶松弛变量。

`demo.py` 内置一个 4 变量、2 等式约束的 LP 示例（含 2 个显式 slack 变量），无需任何输入即可运行。

## R03

KKT 条件写成残差形式：

1. 原始可行残差：`r_p = A x - b`
2. 对偶可行残差：`r_d = A^T y + s - c`
3. 互补残差：`r_c = X S e = x .* s`

内点法不直接令 `x_i s_i = 0`，而是走中心路径：
- `x_i s_i = sigma * mu`
- `mu = (x^T s) / n`

对应牛顿线性化系统：
- `A dx = -r_p`
- `A^T dy + ds = -r_d`
- `S dx + X ds = rhs3`

其中 `rhs3` 在 predictor / corrector 阶段有不同构造（见 R04）。

## R04

算法采用 Mehrotra predictor-corrector 框架：

1. 计算当前 `r_p, r_d, r_c, mu`。
2. **预测步（affine scaling）**：
   - 取 `rhs3_aff = -r_c`，求解一次牛顿方向 `(dx_aff, dy_aff, ds_aff)`。
   - 估计最大可行步长 `alpha_aff_pri, alpha_aff_dual`（不留安全边界，`tau=1`）。
   - 得到 `mu_aff` 并计算 `sigma = (mu_aff/mu)^3`。
3. **校正步（centering + second-order correction）**：
   - 使用 `rhs3_corr = -r_c - dx_aff .* ds_aff + sigma * mu * e`。
   - 再求解一次牛顿方向 `(dx, dy, ds)`。
4. 按 fraction-to-boundary 规则取步：
   - `alpha_pri = min(1, tau * min(-x_i/dx_i))`（仅对 `dx_i<0`）
   - `alpha_dual = min(1, tau * min(-s_i/ds_i))`（仅对 `ds_i<0`）
5. 更新 `x,y,s`，并保持数值正性（`x,s >= 1e-15`）。

## R05

`demo.py` 中核心数据结构：

- `IterationRecord`
  - `iter_id`：迭代编号
  - `primal_res`：归一化原始残差
  - `dual_res`：归一化对偶残差
  - `mu`：平均互补间隙 `(x^T s)/n`
  - `alpha_pri`、`alpha_dual`：本轮原始/对偶步长
  - `sigma`：中心参数
  - `objective`：当前目标值 `c^T x`
- `PDIPMResult`
  - `x, y, s`：最终变量
  - `converged`：是否达到阈值
  - `iterations`：迭代次数
  - `history`：完整迭代记录

## R06

示例问题（脚本内置）：

- 目标：`min -3x1 - x2`
- 约束：
  - `x1 + x2 + x3 = 4`
  - `2x1 + x2 + x4 = 5`
  - `x >= 0`

这里 `x3, x4` 为 slack 变量。该问题解析最优解可写为：
- `x* = [2.5, 0, 1.5, 0]`
- `c^T x* = -7.5`

`demo.py` 会输出算法估计值，并计算 `||x - x_ref||_inf` 与残差指标。

## R07

复杂度（稠密矩阵情形）：

- 每轮主要代价是求解法方程：
  - `M dy = rhs`，其中 `M = A diag(x/s) A^T`，维度 `m x m`
- 构建 `M` 约 `O(m n + m^2 n)`（广播实现下核心约 `A diag(d) A^T`）
- 稠密线性求解约 `O(m^3)`
- 因此单轮主成本可近似记为 `O(m^2 n + m^3)`
- 总体约 `O(T * (m^2 n + m^3))`，`T` 为迭代轮数

空间复杂度：
- 存储 `A`、`M`、向量状态，约 `O(mn + m^2 + n)`。

## R08

正确性与收敛检查（实现层面）：

- 维度与有限值检查：`A,b,c` 维度、有限性、非空性。
- 步长安全性：只在负方向分量上限制步长，保证更新后 `x,s` 保持正。
- 终止条件：
  - `primal_res <= tol`
  - `dual_res <= tol`
  - `gap <= tol`，其中 `gap = mu/(1+|c^T x|)`
- 输出后再验：`||Ax-b||_inf`、`||A^T y + s - c||_inf`、`mu`。

## R09

数值稳定性策略：

- 法方程加小正则：`M <- M + reg * I`（默认 `1e-10`），缓解近奇异情况。
- `solve` 失败时回退到 `lstsq`，避免直接中断。
- 每轮后执行 `x = max(x,1e-15)`、`s = max(s,1e-15)` 抑制浮点负零噪声。
- `sigma` 做截断到 `[0,1]`，避免极端数值导致过激中心化。

## R10

主要函数说明：

- `validate_lp_data`：检查 LP 输入合法性。
- `residual_metrics`：计算 `r_p,r_d,r_c` 与归一化残差/间隙。
- `step_to_boundary`：计算满足正性约束的最大步长。
- `solve_newton_direction`：解 KKT 消元后的法方程，返回 `(dx,dy,ds)`。
- `primal_dual_interior_point`：主循环（预测步 + 校正步 + 更新 + 收敛判定）。
- `print_history`：打印前若干行迭代日志。
- `run_demo_case`：构造固定 LP、运行求解、汇总指标并可选 SciPy 对照。
- `main`：汇总阈值校验，失败时抛异常。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0367-线性规划_-_原始-对偶内点法
python3 demo.py
```

脚本不需要任何交互输入。

## R12

输出字段解释：

- 迭代表：
  - `primal_res`：归一化原始残差
  - `dual_res`：归一化对偶残差
  - `mu`：平均互补间隙
  - `alpha_p/alpha_d`：原始/对偶步长
  - `sigma`：中心参数
  - `c^T x`：当前目标
- 终态：
  - `x*, y*, s*`
  - `objective` 与 `reference objective`
  - `primal residual inf-norm`
  - `dual residual inf-norm`
  - `average complementarity mu`
  - `||x* - x_ref||_inf`
  - 可选 `SciPy linprog objective`
- `Summary`：最终是否通过阈值断言。

## R13

最小测试建议：

1. 主案例（README R06）
- 检查是否收敛。
- 检查目标值接近 `-7.5`。
- 检查 `primal/dual residual` 与 `mu` 足够小。

2. 输入异常测试
- `A` 维度错误或含 NaN/Inf。
- `b,c` 维度不匹配。
- `tau` 不在 `(0,1)` 或 `max_iter<=0`。

3. 数值稳健性测试
- 轻度病态 `A`，验证正则与 `lstsq` 回退逻辑。

## R14

可调参数（`primal_dual_interior_point`）：

- `max_iter`：最大迭代数（默认 `80`）
- `tol`：收敛阈值（默认 `1e-8`）
- `tau`：fraction-to-boundary 系数（默认 `0.995`）
- `regularization`：法方程正则系数（默认 `1e-10`）

调参建议：
- 不收敛时先增大 `max_iter`。
- 残差抖动时增大 `regularization`（例如 `1e-9 ~ 1e-7`）。
- 若步长过于保守，可尝试 `tau=0.997`；若出现负值风险，减小 `tau`。

## R15

与其他 LP 方法对比：

- 与单纯形法：
  - 单纯形沿顶点移动，解释性强；
  - 内点法走可行域内部，通常在大规模稠密问题更稳定。

- 与纯障碍法（只看原始）：
  - 原始-对偶法同时控制原始与对偶残差，终止判据更全面。

- 与黑盒 `linprog`：
  - 黑盒在工程上更省心；
  - 本实现强调“可审计”与“可教学”，每个线性代数步骤均可追踪。

## R16

典型应用场景：

- 资源分配与生产计划（线性成本 + 线性约束）。
- 网络流与运输问题的线性松弛。
- 作为更复杂算法的子模块：
  - 二次规划/锥规划内点法的 LP 子问题
  - 分支定界中的 LP 松弛求解。

## R17

可扩展方向：

- 稀疏矩阵版本：使用稀疏线性代数以支持更大规模问题。
- 更完整预处理：自动缩放、行列重排、冗余约束检测。
- 更强步长/中心参数策略：自适应 `sigma` 与残差平衡。
- 支持不等式输入接口：自动引入 slack 并转换到标准型。
- 增加 benchmark 套件：随机 LP、Netlib 风格小样本回归测试。

## R18

`demo.py` 源码级流程拆解（8 步）：

1. `run_demo_case` 构造标准型 LP：给定 `A,b,c` 以及解析参考解 `x_ref`。
2. `primal_dual_interior_point` 用 `x=1,s=1,y=0` 初始化一个严格正但可不可行的起点。
3. 每轮调用 `residual_metrics` 计算 `r_p,r_d,r_c` 和归一化 `primal_res/dual_res/gap`，先判断是否收敛。
4. 预测步阶段令 `rhs3_aff=-r_c`，通过 `solve_newton_direction` 解法方程得到 `(dx_aff,dy_aff,ds_aff)`。
5. 由 `alpha_aff_pri/alpha_aff_dual` 估计 `mu_aff`，再按 `sigma=(mu_aff/mu)^3` 计算中心参数。
6. 校正步构造 `rhs3_corr=-r_c-dx_aff.*ds_aff+sigma*mu*e`，再次求解牛顿方向 `(dx,dy,ds)`。
7. 用 `step_to_boundary` 计算 `alpha_pri/alpha_dual`，更新 `x,y,s` 并做正性截断；同时记录 `IterationRecord`。
8. 迭代结束后输出 `x*,y*,s*`、残差、互补间隙和目标值，并可选调用 SciPy `linprog` 仅做结果对照。

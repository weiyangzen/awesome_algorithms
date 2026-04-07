# 高斯-赛德尔迭代

- UID: `CS-0323`
- 学科: `计算机`
- 分类: `数值计算`
- 源序号: `491`
- 目标目录: `Algorithms/计算机-数值计算-0491-高斯-赛德尔迭代`

## R01

高斯-赛德尔迭代（Gauss-Seidel Iteration）用于求解线性方程组 `Ax=b`，属于经典定常迭代法。  
与雅可比法不同，它在同一轮迭代中会立刻使用“刚更新出的新分量”，因此通常收敛更快。

设 `A = D + L + U`：
- `D`：对角矩阵；
- `L`：严格下三角部分；
- `U`：严格上三角部分。

迭代写成：
`(D+L)x^(k+1) = b - Ux^(k)`。

## R02

本目录 MVP 的问题定义：

- 输入：
  - 方阵 `A` 与向量 `b`；
  - 初始向量 `x0`（可选，默认零向量）；
  - 收敛阈值 `tol` 与最大迭代次数 `max_iter`。
- 输出：
  - 近似解 `x`；
  - 迭代次数 `iterations`；
  - 是否收敛 `converged`；
  - 最后一步步长误差 `||x^(k+1)-x^(k)||_inf`；
  - 残差 `||Ax-b||_2`；
  - 迭代矩阵 `T_GS = -(D+L)^{-1}U` 的谱半径估计 `rho(T_GS)`。

`demo.py` 使用固定随机种子构造样例，运行时无需交互输入。

## R03

关键数学关系：

1. 矩阵分裂：`A = D + L + U`。  
2. Gauss-Seidel 更新：`(D+L)x^(k+1) = b - Ux^(k)`。  
3. 固定点形式：`x^(k+1) = T_GS x^(k) + c`，其中  
   `T_GS = -(D+L)^{-1}U`，`c = (D+L)^{-1}b`。  
4. 误差递推：`e^(k+1) = T_GS e^(k)`。  
5. 收敛判据：`rho(T_GS) < 1` 时线性收敛。  
6. 常见充分条件：`A` 严格对角占优时，Gauss-Seidel 收敛。

## R04

算法高层流程：

1. 校验输入合法性（维度、有限值、非零对角、参数范围）。
2. 初始化 `x`（默认零向量或给定 `x0`）。
3. 保存旧向量 `x_old`，用于“上三角部分仍用旧值”。
4. 对每个分量 `i` 依次更新：
   - `sigma_lower = sum_{j<i} a_ij x_j`（新值）；
   - `sigma_upper = sum_{j>i} a_ij x_old_j`（旧值）；
   - `x_i = (b_i - sigma_lower - sigma_upper) / a_ii`。
5. 计算步长误差 `||x-x_old||_inf`，低于 `tol` 即停止。
6. 结束后计算残差 `||Ax-b||_2`。
7. 构造 `T_GS` 并计算谱半径作为收敛诊断。

## R05

核心数据结构：

- `GaussSeidelResult`（`dataclass`）
  - `x`：近似解；
  - `iterations`：实际迭代次数；
  - `converged`：是否在上限内收敛；
  - `step_error_inf`：末轮步长误差；
  - `residual_l2`：最终残差；
  - `spectral_radius_T`：`T_GS` 谱半径估计。
- `A, b, x_true`：由固定随机种子构造的可复现测试系统。

## R06

正确性要点（可在代码中验证）：

1. 真解 `x*` 满足固定点方程 `x* = T_GS x* + c`。  
2. 若 `rho(T_GS)<1`，则 `T_GS^k -> 0`，从而误差 `e^(k)->0`。  
3. `demo.py` 通过三类指标验证：
  - `converged` 必须为真；
  - 残差足够小；
  - 与 `np.linalg.solve(A,b)` 参考解误差足够小。

## R07

时间复杂度（稠密矩阵）：

- 单轮迭代需要对每行做内积，复杂度 `O(n^2)`；
- `k` 轮总复杂度 `O(k*n^2)`；
- 对稀疏矩阵可近似为 `O(k*nnz(A))`。

## R08

空间复杂度：

- 输入矩阵 `A`（稠密）占 `O(n^2)`；
- 迭代向量 `x` 与 `x_old` 占 `O(n)`；
- 构造诊断矩阵 `T_GS` 时需要额外 `O(n^2)`。

若只执行求解、不计算谱半径，算法附加空间主要为 `O(n)`。

## R09

边界与异常处理：

- `A` 非方阵：抛 `ValueError`。
- `b` 或 `x0` 维度不匹配：抛 `ValueError`。
- 输入存在 `nan/inf`：抛 `ValueError`。
- 对角线存在零或近零元素：抛 `ValueError`（更新公式不可定义）。
- `tol<=0` 或 `max_iter<=0`：抛 `ValueError`。
- 超过 `max_iter` 未收敛：返回 `converged=False`，并由检查函数触发失败。

## R10

MVP 设计取舍：

- 用 `numpy` 显式实现分量级更新，不调用黑盒迭代求解器。
- 样例使用严格对角占优系统，优先保证稳定、可复现收敛。
- 输出收敛标记、残差、谱半径与解误差，便于教学演示与回归检查。
- 暂不引入预条件、SOR 参数扫描、稀疏存储优化，保持最小可运行。

## R11

`demo.py` 的函数职责：

- `validate_inputs`：统一做输入与参数合法性检查。
- `gauss_seidel_solve`：执行 Gauss-Seidel 主迭代并输出诊断。
- `build_strictly_diagonally_dominant_system`：构造可复现实验系统。
- `run_checks`：执行收敛性与精度断言。
- `main`：串联实验、打印报告。

## R12

运行方式：

```bash
cd Algorithms/计算机-数值计算-0491-高斯-赛德尔迭代
uv run python demo.py
```

脚本无命令行参数、无交互输入。

## R13

输出字段说明：

- `matrix_size`：系统维度；
- `max_iter`：最大迭代轮数；
- `tol`：停止阈值；
- `iterations_used`：实际迭代次数；
- `converged`：是否收敛；
- `spectral_radius_T`：迭代矩阵谱半径估计；
- `step_error_inf`：末轮步长误差；
- `residual_l2`：最终残差；
- `solution_error_inf_vs_x_true`：相对构造真解误差；
- `solution_error_inf_vs_numpy`：相对参考直接解误差。

## R14

最小测试设计：

- 固定 `n=8`、随机种子 `491` 构造严格对角占优 `A`；
- 通过 `x_true` 生成 `b=A@x_true`；
- 使用 `np.linalg.solve` 作为独立参考；
- 断言项：
  - 必须收敛；
  - `rho(T_GS) < 1`；
  - `||Ax-b||_2 < 1e-10`；
  - 相对 `x_true` 与参考解的误差都 `< 1e-10`。

## R15

与相关算法对比：

- 对比 Jacobi：
  - Gauss-Seidel 一轮内复用新值，通常迭代次数更少；
  - Jacobi 全分量解耦，更易并行。
- 对比 SOR：
  - SOR 在合适松弛因子下可更快；
  - Gauss-Seidel 无需调参，作为基线更稳定。
- 对比直接法（LU）：
  - 直接法一次求解精度高；
  - 迭代法在大规模稀疏场景更有内存与可扩展性优势。

## R16

典型应用：

- 线性系统迭代求解的教学与工程基线；
- 偏微分方程离散后的线性子问题；
- 多重网格中的平滑步骤（smoother）；
- SOR、SSOR、预条件设计的基础构件。

## R17

可扩展方向：

- 支持 CSR/CSC 稀疏矩阵，降低大规模问题开销；
- 增加 SOR 参数 `omega`，形成加速版本；
- 记录完整迭代历史并绘制收敛曲线；
- 与 Jacobi/CG/SOR 做统一 benchmark；
- 针对非对角占优系统加入重排或预条件策略。

## R18

`demo.py` 源码级算法流程（9 步，非黑盒）：

1. `main` 设置 `n=8`、`tol`、`max_iter`，调用 `build_strictly_diagonally_dominant_system` 生成 `A,b,x_true`。  
2. `build_strictly_diagonally_dominant_system` 用固定随机种子生成随机矩阵，并把对角线替换为“行绝对值和 + 1”，保证严格对角占优。  
3. `main` 调用 `gauss_seidel_solve`；该函数先用 `validate_inputs` 检查方阵、维度、有限值、非零对角与参数合法性。  
4. 迭代开始时复制 `x_old`，随后按行 `i=0..n-1` 更新：低索引分量用本轮新值，高索引分量用上一轮旧值。  
5. 每个分量按公式 `x_i = (b_i - sigma_lower - sigma_upper)/a_ii` 更新，完整体现 Gauss-Seidel 的“即算即用”机制。  
6. 每轮结束计算 `step_error_inf = ||x - x_old||_inf`，若小于 `tol` 则标记 `converged=True` 并停止。  
7. 迭代后计算 `residual_l2 = ||Ax-b||_2`，再构造 `T_GS = -(D+L)^{-1}U` 并计算其谱半径 `spectral_radius_T`。  
8. `main` 计算与 `x_true`、`np.linalg.solve(A,b)` 的无穷范数误差；`run_checks` 对收敛、谱半径、残差与误差做硬断言。  
9. 全部断言通过后打印报告与近似解向量，输出 `All checks passed.`。  

说明：第三方库仅用于线性代数基础运算与参考解对比，Gauss-Seidel 迭代主逻辑在源码中逐行展开实现。

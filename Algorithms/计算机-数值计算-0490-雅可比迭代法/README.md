# 雅可比迭代法

- UID: `CS-0322`
- 学科: `计算机`
- 分类: `数值计算`
- 源序号: `490`
- 目标目录: `Algorithms/计算机-数值计算-0490-雅可比迭代法`

## R01

雅可比迭代法（Jacobi Iteration）用于求解线性方程组 `Ax=b`，属于经典定常迭代法（stationary iterative method）。

- 核心思想：把 `A` 分裂为 `A=D+(L+U)`，其中 `D` 是对角矩阵，`L` 是严格下三角部分，`U` 是严格上三角部分。
- 迭代公式：
  - `x^(k+1)=D^{-1}(b-(L+U)x^(k))`
  - 或写成固定点形式 `x^(k+1)=Bx^(k)+c`，其中 `B=-D^{-1}(L+U)`，`c=D^{-1}b`。
- 每个分量更新只依赖“上一轮”向量，因此天然适合并行计算。

## R02

本目录 MVP 解决的问题：

- 输入：
  - 方阵 `A` 与向量 `b`；
  - 初始向量 `x0`（可选，默认零向量）；
  - 停止阈值 `tol` 与最大迭代次数 `max_iter`。
- 输出：
  - 近似解 `x`；
  - 迭代次数、收敛标记 `converged`；
  - 步长误差 `||x^(k+1)-x^(k)||_inf`；
  - 残差 `||Ax-b||_2`；
  - 迭代矩阵 `B` 的谱半径估计 `rho(B)`。

`demo.py` 使用固定随机种子构造样例，运行时无需交互输入。

## R03

关键数学关系：

1. 矩阵分裂：`A=D+L+U`。
2. Jacobi 更新：`x^(k+1)=D^{-1}(b-(L+U)x^(k))`。
3. 固定点形式：`x^(k+1)=Bx^(k)+c`。
4. 误差递推：`e^(k+1)=Be^(k)`，`e^(k)=x^(k)-x*`。
5. 收敛判据：`rho(B)<1` 为线性固定点收敛的充要条件。
6. 常用充分条件：若 `A` 严格对角占优，则 Jacobi 收敛。

## R04

算法高层流程：

1. 检查输入合法性（维度、有限值、对角线非零、阈值合法）。
2. 提取 `diag(A)`，构造 `R=A-diag(diag(A))`。
3. 初始化 `x`（默认零向量或用户给定 `x0`）。
4. 迭代更新 `x_new=(b-R@x)/diag(A)`。
5. 计算步长误差 `||x_new-x||_inf`，判断是否达到 `tol`。
6. 达标则停止，否则进入下一轮。
7. 结束后计算残差 `||Ax-b||_2` 和谱半径估计 `rho(B)` 并输出诊断。

## R05

核心数据结构：

- `JacobiResult`（`dataclass`）：
  - `x`: 近似解向量；
  - `iterations`: 实际迭代次数；
  - `converged`: 是否在上限内收敛；
  - `step_error_inf`: 最后一轮步长误差；
  - `residual_l2`: 最终残差；
  - `spectral_radius_B`: 迭代矩阵谱半径估计。
- `A, b, x_true`：由固定随机种子构造的可复现实验数据。

## R06

正确性要点（实现可验证）：

1. `x*` 为真解时满足 `x*=Bx*+c`，即固定点方程成立。
2. 若 `rho(B)<1`，则 `B^k->0`，因此 `e^(k)->0`，迭代收敛到唯一解。
3. 在 `demo.py` 中，使用三类指标验证正确性：
  - 收敛标记必须为真；
  - `||Ax-b||_2` 必须足够小；
  - 与 `np.linalg.solve(A,b)` 的解差必须足够小。

## R07

时间复杂度（稠密矩阵）：

- 单次迭代主开销是矩阵向量乘法 `R@x`，复杂度 `O(n^2)`。
- 进行 `k` 轮迭代时，总复杂度 `O(k*n^2)`。
- 若使用稀疏矩阵存储，单轮可近似降为 `O(nnz(A))`，总复杂度 `O(k*nnz(A))`。

## R08

空间复杂度：

- 输入矩阵 `A`（稠密）占 `O(n^2)`；
- 迭代向量与中间向量（`x`、`x_new`、`diag`）占 `O(n)`；
- 谱半径估计中构造 `B`（稠密）额外 `O(n^2)`。

若仅看算法附加状态且不保留 `B`，可认为主要为 `O(n)`。

## R09

边界与异常处理：

- `A` 不是方阵：抛 `ValueError`。
- `b` 或 `x0` 维度与 `A` 不匹配：抛 `ValueError`。
- 输入出现 `nan/inf`：抛 `ValueError`。
- 对角线存在零或近零元素：抛 `ValueError`（Jacobi 不可定义）。
- `tol<=0` 或 `max_iter<=0`：抛 `ValueError`。
- 超过 `max_iter` 未达阈值：`converged=False`，由 `run_checks` 触发断言失败。

## R10

MVP 设计取舍：

- 使用 `numpy` 实现核心迭代，避免黑盒求解器替代算法主体。
- 样例选“严格对角占优”系统，优先保障可复现和稳定收敛。
- 保持代码小而完整：有输入校验、有诊断输出、有自动断言。
- 暂不引入稀疏格式、预条件或并行框架，避免超出 MVP 范围。

## R11

`demo.py` 的函数职责：

- `validate_inputs`：检查矩阵/向量形状、有限性、对角线合法性、参数范围。
- `jacobi_solve`：执行 Jacobi 主循环并返回诊断结构。
- `build_strictly_diagonally_dominant_system`：构造可复现、可收敛测试系统。
- `run_checks`：统一执行收敛与精度断言，失败即抛异常。
- `main`：串联实验流程，打印报告。

## R12

运行方式：

```bash
cd Algorithms/计算机-数值计算-0490-雅可比迭代法
uv run python demo.py
```

脚本无命令行参数、无交互输入，直接输出结果与校验结论。

## R13

输出字段说明：

- `matrix_size`：线性系统维度。
- `max_iter`：最大迭代轮数。
- `tol`：停止阈值。
- `iterations_used`：实际迭代轮数。
- `converged`：是否收敛。
- `spectral_radius_B`：迭代矩阵谱半径估计。
- `step_error_inf`：末轮步长误差。
- `residual_l2`：最终残差范数。
- `solution_error_inf_vs_x_true`：与构造真解的误差。
- `solution_error_inf_vs_numpy`：与直接法参考解的误差。

## R14

最小测试设计：

- `n=8` 的严格对角占优系统，随机种子固定为 `490`。
- 由 `x_true` 构造 `b=A@x_true`，可直接评估解误差。
- 同时对比 `np.linalg.solve` 作为独立参考。
- 四类硬性检查：
  - 必须收敛；
  - 谱半径 `<1`；
  - 残差 `<1e-8`；
  - 解误差（对真解与参考解）均 `<1e-8`。

## R15

与相关算法对比：

- 对比 Gauss-Seidel：
  - Gauss-Seidel 使用“新值立刻回代”，通常迭代数更少；
  - Jacobi 每轮完全解耦，更利于并行。
- 对比 SOR：
  - SOR 在合适松弛因子下可更快收敛；
  - Jacobi 不需额外松弛参数，调参负担更小。
- 对比直接法（LU）：
  - 直接法步数少但可能更占内存；
  - Jacobi 适合大规模、可迭代逼近场景。

## R16

典型应用场景：

- 大规模线性系统的基线求解器。
- 多重网格中的平滑器（smoother）。
- 预条件构造中的对角预条件（Jacobi preconditioner）思想来源。
- 并行计算教学与数值实验中的入门迭代方法。

## R17

可扩展方向：

- 引入稀疏矩阵格式（CSR/CSC）降低内存与算力开销。
- 增加阻尼因子形成加权 Jacobi。
- 输出完整迭代历史并绘制收敛曲线。
- 与 Gauss-Seidel、SOR、CG 做统一 benchmark。
- 加入自动参数扫描（`tol`、`max_iter`、系统规模）用于性能评估。

## R18

`demo.py` 源码级算法流程（9 步，非黑盒）：

1. `main` 设定 `n=8`、`tol`、`max_iter`，并调用 `build_strictly_diagonally_dominant_system` 生成 `A,b,x_true`。  
2. `build_strictly_diagonally_dominant_system` 用固定随机种子构造随机矩阵，并把对角线改为“行绝对值和 + 1”，确保严格对角占优。  
3. `main` 调用 `jacobi_solve`，该函数先通过 `validate_inputs` 检查方阵、维度、有限值、对角线非零和参数合法性。  
4. `jacobi_solve` 拆分出 `diag(A)` 与 `R=A-diag(diag(A))`，并初始化 `x`（默认零向量或传入 `x0`）。  
5. 每轮按 `x_new=(b-R@x)/diag(A)` 完成一次 Jacobi 更新；该式严格对应理论公式 `D^{-1}(b-(L+U)x)`。  
6. 计算 `step_error_inf=||x_new-x||_inf`，若小于 `tol` 则标记 `converged=True` 并停止。  
7. 迭代结束后计算 `residual_l2=||Ax-b||_2`，再构造迭代矩阵 `B=-(R/diag[:,None])` 并用 `eigvals` 得到 `spectral_radius_B`。  
8. `main` 使用 `np.linalg.solve(A,b)` 获取参考解，计算对 `x_true` 和参考解的无穷范数误差。  
9. `run_checks` 对收敛标记、谱半径、残差、误差四类指标做硬断言，全部通过后打印报告与 `All checks passed.`。  

说明：第三方库仅用于基础向量化运算和参考验算，Jacobi 主迭代流程由源码逐步显式实现。

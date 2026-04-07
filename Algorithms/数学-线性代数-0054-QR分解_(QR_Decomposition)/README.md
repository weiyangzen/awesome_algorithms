# QR分解 (QR Decomposition)

- UID: `MATH-0054`
- 学科: `数学`
- 分类: `线性代数`
- 源序号: `54`
- 目标目录: `Algorithms/数学-线性代数-0054-QR分解_(QR_Decomposition)`

## R01

QR 分解把矩阵写成“正交矩阵 × 上三角矩阵”的形式：
\[
A = QR
\]
其中 `Q` 的列向量两两正交且范数为 1，`R` 为上三角矩阵。  
它是数值线性代数中求解最小二乘、构造正交基、稳定替代正规方程的重要基础模块。

## R02

本条目 MVP 目标：
- 输入：实数矩阵 `A (m x n, m>=n)` 与向量 `b`（用于最小二乘演示）。
- 输出：薄 QR 因子 `Q (m x n)`、`R (n x n)`，满足 `A≈QR`。
- 进一步输出：通过 QR 求得的最小二乘解 `x`，并与 `numpy.linalg.lstsq` 对照。

`demo.py` 采用固定样例，无需交互输入。

## R03

关键数学关系：
- 线性最小二乘问题
\[
\min_x \|Ax-b\|_2
\]
在 `A=QR`（`Q^TQ=I`）下可改写为：
\[
\min_x \|QRx-b\|_2 = \min_x \|Rx-Q^Tb\|_2
\]
- 当 `A` 满列秩时，`R` 可逆，解由上三角回代得到：
\[
Rx = Q^Tb
\]
- 因为正交变换保持 2-范数，QR 比直接解正规方程 `A^TAx=A^Tb` 更稳定。

## R04

算法路线（本实现）：
1. 对输入矩阵做合法性检查（二维、有限值、`m>=n`）。
2. 使用 Householder 反射逐列消元，把 `A` 变换为上三角结构 `R`。
3. 同步累积正交变换，得到 `Q`。
4. 截取薄 QR：`Q_thin = Q[:, :n]`，`R_thin = R[:n, :]`。
5. 计算重构误差与正交误差。
6. 用 `R x = Q^T b` 回代求解最小二乘。
7. 与 `numpy.linalg.lstsq` 的结果做数值对照。

## R05

核心数据结构：
- `a: np.ndarray (m, n)`：输入矩阵。
- `q_full: np.ndarray (m, m)`：累积反射后的完整正交矩阵。
- `r_full: np.ndarray (m, n)`：经反射更新后的矩阵。
- `q: np.ndarray (m, n)`：薄 Q。
- `r: np.ndarray (n, n)`：薄 R。
- `b: np.ndarray (m,)`：右端向量。
- `x: np.ndarray (n,)`：最小二乘解。

## R06

正确性要点：
- 每一步 Householder 反射都可写为 `H = I - 2vv^T`，满足正交性 `H^TH=I`。
- 连乘后的 `Q = H1H2...Hk` 仍正交。
- 左乘反射会把当前列主元以下元素消成 0，最终形成上三角 `R`。
- 因而得到 `A = QR`，并可安全用于三角回代。

## R07

复杂度分析（`m>=n`）：
- Householder QR 时间复杂度：`O(m n^2)`。
- 回代求解 `Rx=y`：`O(n^2)`。
- 空间复杂度：`O(m^2 + mn)`（本 MVP 显式存储 `Q_full`，更易理解与审计）。

## R08

边界与异常处理：
- `A` 非二维或含 `nan/inf`：抛出 `ValueError`。
- `m < n`：本 MVP 不支持宽矩阵最小二乘路径，抛出 `ValueError`。
- `tol <= 0`：抛出 `ValueError`。
- 回代时若 `R` 对角元过小：判定为秩亏，抛出 `ValueError`。
- `b` 维度与 `A` 不一致：抛出 `ValueError`。

## R09

MVP 取舍：
- 主算法手写 Householder QR，避免把核心逻辑直接外包给 `numpy.linalg.qr`。
- 只依赖 `numpy`，保持运行环境轻量。
- `numpy.linalg.lstsq` 仅作为参考校验，不参与主路径求解。

## R10

`demo.py` 函数职责：
- `validate_matrix`：检查矩阵输入合法性。
- `householder_qr`：执行 QR 分解并返回指标。
- `backward_substitution`：解上三角系统。
- `solve_least_squares_qr`：通过 QR 解最小二乘。
- `build_demo_problem`：构造固定可复现样例。
- `main`：组织运行、打印指标、演示秩亏失败路径。

## R11

运行方式：

```bash
cd Algorithms/数学-线性代数-0054-QR分解_(QR_Decomposition)
uv run python demo.py
```

脚本无命令行参数，无交互输入。

## R12

输出字段说明：
- `||A - Q@R||_F`：分解重构误差。
- `||Q^TQ - I||_F`：正交性误差。
- `||tril(R,-1)||_F`：`R` 的下三角残量。
- `x (QR)`：手写 QR 得到的解。
- `x (NumPy lstsq)`：NumPy 参考解。
- `||x_qr - x_np||_2`：两者解差。
- `||A@x - b||_2`：最小二乘残差。
- `||x_qr - x_true||_2`：与构造真值的偏差。

## R13

最小测试建议：
- 满列秩高矩阵（默认样例），验证重构和解精度。
- 近奇异矩阵，观察 `tol` 对稳定性的影响。
- 明确秩亏矩阵，验证回代阶段是否正确报错。
- 随机多组 `A,b`，与 `numpy.linalg.lstsq` 对比误差。

## R14

可调参数：
- `tol`：零判定阈值，默认 `1e-12`。
- 样例规模 `m,n`：可放大测试性能。
- `np.set_printoptions`：调节输出精度与展示。

工程建议：当数据噪声较大时，可适度放宽 `tol`，并同时监控残差与秩判断的一致性。

## R15

与相关方法对比：
- 与正规方程：QR 通常更稳定，避免 `A^TA` 条件数平方放大。
- 与 LU：LU 主要用于方阵线性系统；QR 更适合最小二乘。
- 与 SVD：SVD 更稳健但代价更高；QR 是很多工程场景中的效率优先解法。

## R16

典型应用：
- 线性回归最小二乘求解。
- 正交基构造与子空间投影。
- 数值优化中的线性子问题。
- 作为迭代特征值算法与矩阵分解流程的基础算子。

## R17

可扩展方向：
- 改为原地存储 Householder 向量，降低 `Q_full` 的内存开销。
- 加入列主元 QR（CPQR）以处理秩揭示问题。
- 扩展到复数矩阵（共轭转置版本）。
- 结合 `scipy.linalg.qr` 做大规模性能对比基准。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 调用 `build_demo_problem` 生成固定的高矩阵 `A`、真值 `x_true` 与 `b=A@x_true`。  
2. `householder_qr` 先校验 `A`，初始化 `Q_full=I`、`R_full=A.copy()`。  
3. 对列 `k=0..n-1` 取子向量 `x=R_full[k:,k]`，构造 Householder 向量 `v`。  
4. 用反射 `H=I-2vv^T` 左乘当前尾块：`R_full[k:,k:] -= 2 v (v^T R_full[k:,k:])`，把主元下方元素消零。  
5. 同步更新 `Q_full[:,k:] -= 2 (Q_full[:,k:] v) v^T` 累积正交变换。  
6. 循环结束后截取薄因子 `Q=Q_full[:,:n]`、`R=R_full[:n,:]`，并把数值噪声级下三角清零。  
7. `solve_least_squares_qr` 计算 `y=Q^Tb`，再由 `backward_substitution` 解 `Rx=y` 得到 `x`。  
8. `main` 打印重构误差、正交误差、解差和残差，并用秩亏矩阵示例验证异常分支。  

说明：第三方库在本条目中只提供基础数组运算和参考验算；QR 分解主流程（反射构造、逐列消元、Q 累积、回代求解）均在源码中显式实现。

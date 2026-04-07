# Givens旋转与Householder变换

- UID: `MATH-0062`
- 学科: `数学`
- 分类: `线性代数`
- 源序号: `62`
- 目标目录: `Algorithms/数学-线性代数-0062-Givens旋转与Householder变换`

## R01

Givens 旋转与 Householder 变换都是“正交相似/正交投影”思想下的基础构件，核心用途是把矩阵逐步化为更易处理的结构（尤其是 QR 分解中的上三角 `R`）。

- Householder：一次反射可同时消去一整列的多个元素；
- Givens：一次旋转只作用两个分量，精细消去单个元素；
- 二者都保持 2-范数与条件稳定性（正交变换不放大向量范数）。

## R02

历史定位（简要）：

- Householder 反射由 Alston S. Householder 在 20 世纪中期系统化，用于高稳定矩阵分解；
- Givens 旋转由 Wallace Givens 推广到数值线性代数，尤其适合稀疏和逐元素消元；
- 现代 LAPACK/BLAS 生态中的 QR、最小二乘、特征值算法都以这两类变换为基础模块。

## R03

本条目 MVP 解决的问题：

- 输入：
  - 一个实数满列秩矩阵 `A in R^{m x n}`（`m >= n`）；
  - 观测向量 `b in R^m`；
- 输出：
  - 通过 Householder 构造的 `Q_h, R_h`；
  - 通过 Givens 构造的 `Q_g, R_g`；
  - 正交误差、重构误差、下三角残量；
  - 基于 QR 的最小二乘解与 `numpy.linalg.lstsq` 的对比误差。

## R04

时间复杂度（稠密矩阵，`m >= n`）：

- Householder QR：约 `O(2mn^2 - 2n^3/3)`；
- Givens QR：数量级同为 `O(mn^2)`，但常数通常更大（逐元素旋转次数更多）；
- 基于 QR 的回代求解最小二乘：`O(n^2)`。

在稀疏场景下，Givens 可按非零结构选择性应用，可能获得更好的结构保持。

## R05

空间复杂度：

- 若显式形成完整 `Q`：`O(m^2)`；
- 存储 `R`：`O(mn)`；
- 本 MVP 选择“可读性优先”，显式存储 `Q` 和 `R` 以便做数值验证。

## R06

直觉例子：令 `x = [a, b]^T`。

- Givens 选择 `c,s` 使
  `[[c, s], [-s, c]] [a, b]^T = [r, 0]^T`，从而精准消去 `b`；
- Householder 构造反射向量 `v`，令
  `H = I - 2vv^T`，把高维向量 `x` 一步反射到某个坐标轴方向（除首元外其余归零）。

这就是“Givens 局部旋转、Householder 整段反射”的核心区别。

## R07

算法意义：

- 提供稳定 QR 分解路径，是最小二乘与特征值算法的重要前置；
- 数值上优于直接法方程 `A^T A x = A^T b`（后者会平方条件数）；
- 在工程中可按数据形态选型：
  - 稠密批量：Householder 常更快；
  - 稀疏/在线更新：Givens 常更灵活。

## R08

核心公式：

1. Householder 反射：
   - `H = I - 2vv^T`（`||v||_2 = 1`）
   - 左乘更新：`R <- H R`
2. Givens 旋转：
   - `G = [[c, s], [-s, c]]`，其中 `c = a/r, s = b/r, r = hypot(a,b)`
   - 左乘更新两行，逐个消去列下方元素
3. QR 关系：
   - 经过一系列正交变换后 `A = Q R`
   - `Q^T Q = I`，`R` 为上三角（或上梯形）。

## R09

适用条件与局限：

适用：

- 需要稳定最小二乘解；
- 需要显式或隐式正交化；
- 需要构建上三角结构用于回代。

局限：

- 显式形成完整 `Q` 会消耗 `O(m^2)` 内存；
- Givens 在稠密问题上通常慢于 Householder；
- 若矩阵列不满秩，直接 `np.linalg.solve(R1, y)` 会失败，需要改用带主元策略或 SVD。

## R10

正确性要点（工程可验证版）：

1. 每步变换矩阵均正交，因此不改变欧氏范数。  
2. Householder 每轮把当前列主对角线下方元素同时压到 0。  
3. Givens 每次只旋转两行，精确把一个目标元素变为 0。  
4. 累积后得到上三角 `R`，并保有 `A = QR`。  
5. `demo.py` 用 `||Q^TQ-I||`、`||A-QR||`、`||tril(R,-1)||` 三项数值指标联合验算。

## R11

数值稳定性：

- Householder 和 Givens 都属于正交变换路径，稳定性通常优于法方程；
- Givens 系数使用 `hypot(a,b)` 计算，避免 `a^2+b^2` 的上溢/下溢风险；
- Householder 向量用“符号选择 + 归一化”策略，减轻消差误差；
- 本实现在双精度下通常可达到 `1e-12` 量级的结构误差。

## R12

性能视角：

- Householder：反射次数少，适合稠密块计算；
- Givens：局部操作细粒度强，适合结构化稀疏消元和增量更新；
- 本 MVP 不做并行/分块优化，目标是透明呈现算法本体。

## R13

本目录 MVP 的可验证保证：

- 无交互输入，固定随机种子可复现；
- 同时运行 Householder 与 Givens 两套 QR；
- 自动检查：
  - 正交误差
  - 重构误差
  - 上三角结构误差
  - 最小二乘解与 `lstsq` 的偏差
- 通过阈值后输出 `All checks passed.`。

## R14

常见失效模式：

- 输入含 `NaN/Inf`；
- `m < n`（本 MVP 面向 `m >= n`）；
- `A` 列不满秩导致回代矩阵奇异；
- 阈值设得过严导致在极端机器误差下误判失败。

## R15

实现设计（`demo.py`）：

- `QRResult`：封装分解结果与诊断指标；
- `validate_matrix`：检查维度、有限性与列秩；
- `householder_vector` / `householder_qr`：手写反射向量与分解主循环；
- `givens_coeffs` / `apply_givens_left` / `givens_qr`：手写旋转参数与两行更新；
- `least_squares_via_qr`：用 `Q,R` 解最小二乘；
- `run_checks`：统一门槛断言；
- `main`：构造样例、运行对比、打印报告。

## R16

相关算法链路：

- QR 特征值算法：先做 Hessenberg 化，再反复 QR 迭代；
- 最小二乘家族：线性回归、Kalman 子步骤等常依赖 QR 稳定求解；
- 稀疏数值线代：Givens 常配合稀疏结构更新；
- SVD 与正交化：Householder 同样是双对角化和正交基构造的基础工具。

## R17

运行方式：

```bash
cd Algorithms/数学-线性代数-0062-Givens旋转与Householder变换
python3 demo.py
```

依赖：

- `numpy`
- Python 标准库：`dataclasses`

脚本不需要交互输入，会直接打印两种方法的误差指标与最小二乘对比结果。

## R18

`demo.py` 的源码级算法流程（9 步，非黑盒）如下：

1. `main` 固定随机种子，构造满列秩矩阵 `A` 与向量 `b`，并做输入合法性检查。  
2. `householder_qr` 从第 0 列到第 `n-1` 列循环，取当前子向量 `x = R[k:,k]`。  
3. `householder_vector` 构造单位向量 `v`，形成反射 `H = I - 2vv^T`。  
4. 用 `R[k:,k:] -= 2 v (v^T R[k:,k:])` 左乘更新，批量清零当前列对角线下方元素；同时右乘更新 `Q`。  
5. `givens_qr` 对每列从下往上处理，每次取 `(a,b)=(R[i-1,j],R[i,j])`。  
6. `givens_coeffs` 计算 `c,s`，`apply_givens_left` 只更新两行，把 `R[i,j]` 旋到 0，并同步累积 `Qt`。  
7. 结束后设 `Q = Qt^T`，分别得到 Householder 与 Givens 的 `Q,R`。  
8. `least_squares_via_qr` 取经济型块 `Q1,R1`，先算 `y=Q1^T b`，再解上三角方程 `R1 x = y`。  
9. `run_checks` 检查 `||Q^TQ-I||`、`||A-QR||`、`||tril(R,-1)||` 及与 `numpy.linalg.lstsq` 的解差，全部通过后输出成功信息。

说明：`numpy` 仅用于基础数组运算与参考解校验，Householder/Givens 主流程在源码中逐步展开实现。

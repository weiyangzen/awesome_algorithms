# Arnoldi迭代

- UID: `MATH-0059`
- 学科: `数学`
- 分类: `线性代数`
- 源序号: `59`
- 目标目录: `Algorithms/数学-线性代数-0059-Arnoldi迭代`

## R01

Arnoldi 迭代用于在 Krylov 子空间中构造一组正交基，并把一般方阵 `A` 投影成小规模上 Hessenberg 矩阵，从而近似其特征信息。

给定 `A in R^{n x n}` 与初始向量 `b != 0`，Arnoldi 生成：

- 正交基 `Q_m = [q1, q2, ..., qm]`
- 上 Hessenberg 矩阵 `H_m`
- 满足关系 `A Q_m = Q_{m+1} Hbar_m`

其中 `Hbar_m` 为 `(m+1) x m` 的 Hessenberg 形式。`H_m` 的特征值称为 Ritz 值，是 `A` 特征值的近似。

## R02

历史定位（简要）：

- Arnoldi 方法由 W. E. Arnoldi 在 1951 年提出；
- 它把“幂法的一维方向迭代”推广为“子空间正交化迭代”；
- 对一般非对称矩阵，它是 GMRES、IRAM（ARPACK/eigs 家族）等方法的核心基础。

## R03

本条目 MVP 解决的问题：

- 输入：
  - 实方阵 `A`；
  - 初始向量 `b`；
  - Arnoldi 步数 `m`；
  - breakdown 阈值 `tol`；
- 输出：
  - Arnoldi 基 `Q_m`；
  - Hessenberg 投影 `H_m` 与 `Hbar_m`；
  - 正交性误差、Arnoldi 关系残差；
  - Ritz 值与参考特征值之间的匹配误差。

## R04

时间复杂度（稠密情形）：

- 第 `k` 步一次 matvec：`A @ q_k`，代价 `O(n^2)`；
- 第 `k` 步 Gram-Schmidt 正交化约 `O(k n)`；
- 总计 `k=1..m`，得到 `O(m n^2 + m^2 n)`；
- 当 `m << n` 时，通常以 `O(m n^2)` 为主导。

若 `A` 稀疏，matvec 部分可降为 `O(nnz(A))`，总复杂度近似 `O(m * nnz(A) + m^2 n)`。

## R05

空间复杂度：

- 存储 `Q` 约 `O(n m)`；
- 存储 `Hbar` 约 `O(m^2)`；
- 输入矩阵稠密存储 `O(n^2)`。

若只看 Arnoldi 迭代附加开销（不含输入 `A`），为 `O(nm + m^2)`。

## R06

微型直觉示例（`A` 为循环移位矩阵）：

- `q1 = e1`
- `A q1 = e2`，与 `q1` 正交，归一化得 `q2 = e2`
- `A q2 = e3`，继续得到 `q3 = e3`

因此 Arnoldi 会沿 Krylov 序列 `span{b, Ab, A^2b, ...}` 逐步构造正交基。达到不变子空间后会出现 breakdown（`h_{k+1,k}≈0`）。

## R07

算法意义：

- 让“大矩阵特征问题”转化为“小 Hessenberg 特征问题”；
- 对非对称矩阵同样适用（相较于 Lanczos 对称约束更宽）；
- 为 GMRES 残差最小化、隐式重启特征求解提供统一子空间框架。

## R08

核心公式（MGS 版本）：

1. `q1 = b / ||b||`
2. 对 `k = 1..m`：
   - `v = A qk`
   - 对 `j = 1..k`：
     - `h_{j,k} = qj^T v`
     - `v = v - h_{j,k} qj`
   - `h_{k+1,k} = ||v||`
   - 若 `h_{k+1,k}` 足够小则停止（breakdown）
   - 否则 `q_{k+1} = v / h_{k+1,k}`

最终得到 Arnoldi 关系 `A Q_m = Q_{m+1} Hbar_m`。

## R09

适用条件与局限：

适用：

- 需要若干主导特征值/特征向量近似；
- 线性系统/特征问题规模大，且可以高效执行 matvec。

局限：

- 未重启时，`Q` 列数随迭代增长，内存与正交化开销上升；
- 经典 GS 在有限精度下正交性会退化，需要重正交；
- 对谱分布不利的问题，Ritz 收敛可能慢。

## R10

正确性要点（工程可验证版）：

1. 每步都把 `A q_k` 对已得基做正交投影并剔除分量。  
2. 因而 `Q_m` 保持近似正交，且 `A q_k` 可在 `Q_{k+1}` 中表示。  
3. 组合各列得到矩阵关系 `A Q_m = Q_{m+1} Hbar_m`。  
4. `H_m = Q_m^T A Q_m`（投影矩阵）保留了 `A` 在 Krylov 子空间上的谱信息。  
5. `demo.py` 显式计算正交误差和关系残差验证上述性质。

## R11

数值稳定性：

- 本实现使用 Modified Gram-Schmidt（MGS）并可选二次重正交（reorth）；
- 通过 `h_{k+1,k} <= tol` 检测 breakdown；
- 在双精度下，正交误差常受 `O(eps)` 到 `O(k*eps)` 级别影响；
- 对高条件数问题，建议启用重正交或 Householder Arnoldi。

## R12

性能视角：

- 计算热点是 `A @ q_k` 与内积循环；
- 小 `m` 时 Arnoldi 非常轻量，适合作为外层方法子模块；
- 在工程中常配合重启（如 GMRES(m)、IRAM）控制内存与开销。

## R13

本目录 MVP 的可验证保证：

- 固定构造矩阵与初值，无交互、可复现；
- 自动输出并检查：
  - `||Q^TQ - I||`（正交性）
  - `||A Q - Qext Hbar||`（Arnoldi 关系残差）
  - Hessenberg 结构误差
  - Ritz 值与参考特征值的最大最小距离
- 全部通过后输出 `All checks passed.`。

## R14

常见失效模式：

- `b` 为零向量，无法归一化；
- `m <= 0` 或 `A` 非方阵；
- breakdown 提前出现（Krylov 子空间已封闭）；
- 不做重正交时，长迭代可能丢失正交性并污染 Ritz 值。

## R15

实现设计（`demo.py`）：

- `ArnoldiResult`：封装迭代结果与误差指标；
- `validate_inputs`：输入合法性检查；
- `arnoldi_iteration`：MGS + 可选重正交，显式构造 `Q/H`；
- `build_cyclic_shift_matrix`：构造可复现测试矩阵；
- `hessenberg_violation`、`max_min_eigen_distance`：结构与谱误差评估；
- `run_checks`：统一断言；
- `main`：组装实验、打印报告。

## R16

相关算法链路：

- 对称矩阵特化：Lanczos 迭代（`H` 退化为三对角）；
- 线性方程求解：GMRES 在 Arnoldi 基上做最小残差；
- 特征值求解：IRAM/ARPACK 在 Arnoldi 基础上做隐式重启；
- 预条件扩展：右/左预条件 GMRES 仍依赖 Arnoldi 子空间构造。

## R17

运行方式：

```bash
cd Algorithms/数学-线性代数-0059-Arnoldi迭代
python3 demo.py
```

依赖：

- `numpy`
- Python 标准库：`dataclasses`、`typing`

脚本无交互输入，运行后直接打印数值结果与检查结论。

## R18

`demo.py` 的源码级算法流程（9 步，非黑盒）如下：

1. `main` 构造 `n x n` 循环移位矩阵 `A` 与起始向量 `b=e1`，设定 `m=n`。  
2. `arnoldi_iteration` 调用 `validate_inputs` 检查维度、阈值、步数，并归一化 `b` 得到 `q1`。  
3. 每轮先做 `v = A @ q_k`，这是 Krylov 扩展的核心。  
4. 对 `j=1..k` 计算 `h_{j,k}=q_j^T v` 并执行 `v <- v - h_{j,k} q_j`，完成 MGS 正交化。  
5. 若开启 `reorthogonalize`，再做一次相同投影，抑制浮点误差下的正交退化。  
6. 计算 `h_{k+1,k}=||v||`；若小于 `tol` 判定 breakdown，说明子空间已闭合。  
7. 非 breakdown 时归一化得到新基向量 `q_{k+1}`，并推进下一轮。  
8. 迭代后裁剪出 `Q_m/H_m/Hbar_m`，显式计算 `orthogonality_error` 与 `arnoldi_relation_error`。  
9. 在 `main` 中对 `H_m` 做 `np.linalg.eigvals` 得到 Ritz 值，仅用于验算并与 `A` 的参考特征值做距离比对。

说明：第三方线性代数例程只用于“结果验算”，Arnoldi 主体迭代与正交化流程均在源码中逐步实现。

# Riccati方程求解

- UID: `MATH-0098`
- 学科: `数学`
- 分类: `线性代数`
- 源序号: `98`
- 目标目录: `Algorithms/数学-线性代数-0098-Riccati方程求解`

## R01

本条目聚焦连续时间代数 Riccati 方程（CARE）：

`A^T X + X A - X B R^{-1} B^T X + Q = 0`

其中：
- `A in R^{n*n}`，`B in R^{n*m}`；
- `Q=Q^T >= 0`，`R=R^T > 0`；
- 目标是求对称解 `X=X^T`，通常取稳定化解（stabilizing solution）。

在最优控制（LQR）中，`X` 直接决定状态反馈 `K = R^{-1} B^T X`。

## R02

Riccati 方程在控制和数值线性代数中是核心对象：

- 来源可追溯到变分法与二次型优化问题；
- 在线性二次调节器（LQR）中，CARE 是闭式最优反馈律的关键方程；
- 工程中常见算法包括 Schur/QZ 方法、Newton-Kleinman 迭代、结构保持迭代等。

本 MVP 选择 Newton-Kleinman 路线，强调可读性与源码可审计性。

## R03

本目录 MVP 解决的问题：

- 输入：固定的 `(A,B,Q,R)` 示例系统（脚本内置，非交互）；
- 计算：用 Newton-Kleinman 迭代求 CARE 的对称解 `X`；
- 输出：
  - 收敛标志、迭代步数；
  - 末次步长范数与 Riccati 残差范数；
  - 闭环矩阵 `A - B R^{-1} B^T X` 的特征值；
  - 解矩阵 `X`。

## R04

时间复杂度（本 MVP 的稠密实现）：

- 预处理：
  - `R^{-1}` 代价 `O(m^3)`；
  - `G = B R^{-1} B^T` 代价约 `O(n^2 m + n m^2)`。
- 每次 Newton 迭代主开销是求一个 Lyapunov 子问题：
  - 用 Kronecker 线性化后形成 `n^2 x n^2` 线性系统；
  - 稠密直接解约 `O((n^2)^3) = O(n^6)`。
- 总体约 `O(k * n^6)`（`k` 为迭代次数），适合小中规模演示。

## R05

空间复杂度：

- 存储 `A,B,Q,R,G,X` 级别为 `O(n^2 + nm + m^2)`；
- Kronecker 系数矩阵 `K` 大小为 `n^2 x n^2`，主导内存 `O(n^4)`；
- 因此本实现总空间复杂度以 `O(n^4)` 为主。

## R06

标量 Riccati 方程示例：

设 `a=-1, b=1, q=1, r=1`，则 CARE 退化为：

`2 a x - (b^2 / r) x^2 + q = 0`

即：

`-2x - x^2 + 1 = 0  =>  x^2 + 2x - 1 = 0`

两根为 `x = -1 +/- sqrt(2)`，稳定化解取正根：

`x* = -1 + sqrt(2) ~= 0.4142`。

这与矩阵情形中“选择稳定闭环对应解”的思想一致。

## R07

算法意义：

- CARE 把“最优控制律”转成“矩阵方程求解”；
- Riccati 解 `X` 同时编码控制增益与代价函数几何结构；
- Newton-Kleinman 将非线性矩阵方程分解为一系列线性 Lyapunov 方程，便于实现与调试。

## R08

本实现使用的核心数学关系：

1. 记 `G = B R^{-1} B^T`，CARE 写成 `A^T X + X A - X G X + Q = 0`。  
2. Newton-Kleinman 迭代（以 `X_k` 为当前点）：  
   `(A - G X_k)^T X_{k+1} + X_{k+1}(A - G X_k) + (X_k G X_k + Q) = 0`。  
3. Lyapunov 子问题 `M^T X + X M + C = 0` 通过向量化变成线性系统：  
   `[I ⊗ M^T + M^T ⊗ I] vec(X) = -vec(C)`。  
4. 求得 `X_{k+1}` 后计算残差 `||A^T X + X A - X G X + Q||_F` 判断收敛。

## R09

适用条件与边界：

适用：
- `Q` 对称半正定，`R` 对称正定；
- 需要 CARE 的稳定化解（如 LQR 设计）；
- 规模中小、追求实现透明性。

边界：
- Kronecker 线性化在大规模问题上内存与时间开销高；
- Newton-Kleinman 对初值与系统可稳定性有要求；
- 本 MVP 面向教学/验证，不是工业级大规模求解器。

## R10

正确性要点（工程可验）：

1. `validate_inputs` 检查维度、有限值、`Q/R` 对称性和 `R` 正定性。  
2. 每步显式计算 Riccati 残差，确保不是“仅看步长”假收敛。  
3. 最终断言 `X` 对称、近似半正定。  
4. 断言闭环矩阵 `A - B R^{-1} B^T X` 的特征值实部全负（稳定化解）。

## R11

数值稳定性与收敛注意事项：

- Kronecker 线性系统可能病态，矩阵缩放与条件数会影响精度；
- `R` 若接近奇异，会放大 `R^{-1}` 误差；
- 终止条件采用“步长 + 残差”双阈值更稳健；
- 每步对 `X` 做 `(X+X^T)/2` 对称化，可抑制浮点非对称漂移。

## R12

性能视角：

- 实际耗时热点在 `np.linalg.solve`（解 `n^2` 维线性系统）；
- 该实现强调“少依赖、全流程可见”，牺牲了大规模性能；
- 若面向大规模应用，通常改用 Schur 法或低秩迭代法，避免显式 `n^2 x n^2` 系统。

## R13

本目录的可验证保证（`demo.py`）：

- 固定系数矩阵，结果可复现；
- 自动检查：
  - `converged == True`；
  - 最终 Riccati 残差 `< 1e-8`；
  - `X` 对称且近似半正定；
  - 闭环矩阵稳定；
  - 最终残差优于初始残差。
- 任一条件失败会抛异常，避免静默通过。

## R14

常见失效模式：

- 输入维度不匹配或包含 NaN/Inf；
- `R` 非正定导致无法逆或数值不稳定；
- 初值不合适或系统条件不满足，导致 Newton 迭代停滞；
- 阈值设置不合理，出现“过早停止”或“无谓迭代”。

## R15

实现设计（`demo.py`）：

- `RiccatiResult`：封装求解结果与迭代诊断；
- `validate_inputs`：参数与矩阵合法性检查；
- `solve_continuous_lyapunov_via_kron`：显式 Kronecker 线性化求 Lyapunov 方程；
- `solve_care_newton_kleinman`：主迭代器，维护残差与步长历史；
- `run_checks`：统一正确性断言；
- `main`：构造样例、执行求解、打印报告。

## R16

相关算法链路：

- 同类方程：离散代数 Riccati 方程（DARE）；
- 同类目标：Lyapunov 方程、Sylvester 方程；
- 常见 CARE 求法：Schur/QZ、Newton-Kleinman、结构保持 doubling；
- 应用链路：LQR、H-infinity 控制、状态估计与滤波的矩阵方程子问题。

## R17

运行方式：

```bash
cd Algorithms/数学-线性代数-0098-Riccati方程求解
python3 demo.py
```

依赖：
- `numpy`
- Python 标准库：`dataclasses`、`typing`

脚本无交互输入，执行后直接打印迭代与校验结果。

## R18

`demo.py` 的源码级算法流程（9 步，非黑盒）如下：

1. `main` 固定构造 `A,B,Q,R`，明确 CARE 问题实例。  
2. `solve_care_newton_kleinman` 调用 `validate_inputs`，完成维度、对称性、正定性和有限值检查。  
3. 计算 `R^{-1}` 与 `G = B R^{-1} B^T`，把 CARE 统一写成 `A^T X + X A - X G X + Q = 0`。  
4. 用线性化方程 `A^T X + X A + Q = 0` 得到初值 `X0`（通过 Lyapunov 求解器）。  
5. 第 `k` 步构造 `A_cl = A - G Xk` 与 `Ck = Xk G Xk + Q`，形成 Lyapunov 子问题。  
6. `solve_continuous_lyapunov_via_kron` 将 `A_cl^T X + X A_cl + Ck = 0` 向量化为线性系统，并用 `np.linalg.solve` 求解 `vec(X_{k+1})`。  
7. 回填矩阵并对称化得到 `X_{k+1}`，计算步长范数与 Riccati 残差范数，写入历史。  
8. 满足“步长阈值 + 残差阈值”则收敛退出，否则继续迭代直到 `max_iter`。  
9. `run_checks` 断言解的对称性、半正定性、闭环稳定性和残差改善，全部通过后输出 `All checks passed.`。

说明：本 MVP 没有直接调用一行黑盒 CARE 接口，而是显式展开 Newton-Kleinman 与 Lyapunov 子问题求解流程。

# BiCGSTAB算法

- UID: `MATH-0065`
- 学科: `数学`
- 分类: `线性代数`
- 源序号: `65`
- 目标目录: `Algorithms/数学-线性代数-0065-BiCGSTAB算法`

## R01

BiCGSTAB（Biconjugate Gradient Stabilized）用于求解一般非对称线性方程组 `Ax=b`，属于 Krylov 子空间迭代法。

核心思想：

- 继承 BiCG 的双正交框架；
- 通过额外的“稳定化”步骤（`omega` 修正）抑制残差大幅震荡；
- 在不显式求逆、只需矩阵向量乘 `A @ v` 的前提下逼近解。

## R02

历史定位（简要）：

- 原始 BiCG 方法可处理非对称系统，但残差波动较明显；
- H. A. van der Vorst 于 1992 年提出 BiCGSTAB，目标是在保持低内存开销的同时改善收敛平滑性；
- 该方法成为工程仿真与稀疏线性系统中的常用迭代基线之一。

## R03

本条目 MVP 解决的问题：

- 输入：
  - 非对称实矩阵 `A`；
  - 右端项 `b`；
  - 初值 `x0`（可选）；
  - 迭代上限 `maxiter`、容差 `rtol/atol`；
  - 预条件器 `M^{-1}`（可选，MVP 提供 Jacobi 版本）。
- 输出：
  - 近似解 `x`；
  - 是否收敛、迭代次数、最终残差；
  - 残差历史（用于分析收敛过程）。

## R04

时间复杂度（每次迭代）：

- 2 次矩阵向量乘：`A @ p_hat` 与 `A @ s_hat`；
- 若用 Jacobi 预条件，预条件应用是 `O(n)`；
- 若 `A` 稠密：单次 `O(n^2)`，`k` 次约 `O(k n^2)`；
- 若 `A` 稀疏：单次约 `O(nnz(A))`，`k` 次约 `O(k * nnz(A))`。

## R05

空间复杂度：

- 需保存若干长度 `n` 的向量（`x,r,r_hat,p,v,s,t` 等），额外开销 `O(n)`；
- 输入矩阵若稠密存储为 `O(n^2)`，若稀疏则按稀疏格式计。

算法本体相比直接法的优势在于：不需要 `O(n^3)` 分解，也不需要完整逆矩阵。

## R06

微型直觉示例（概念层）：

设初始残差 `r0=b-Ax0`。

1. 先沿搜索方向 `p` 做一次 BiCG 风格更新，得到中间残差 `s`；
2. 若 `s` 已很小可提前收敛；
3. 否则再沿 `s` 对应方向做一次最小化修正，系数为 `omega`；
4. 新残差 `r = s - omega t` 往往比原始 BiCG 更平滑。

这就是 “STAB（stabilized）” 的核心直观。

## R07

算法意义：

- 面向非对称/非正定系统，弥补 CG（要求 SPD）的适用性限制；
- 内存占用低，适合大规模稀疏问题；
- 常与预条件器结合，作为工程 PDE 离散系统的重要求解器。

## R08

核心迭代公式（左预条件写法）如下：

- `r0 = b - A x0`，选固定阴影向量 `r~ = r0`
- 初始化 `rho0=alpha=omega=1`, `v0=0`, `p0=0`

对 `k = 1,2,...`：

1. `rho_k = <r~, r_{k-1}>`
2. `beta_k = (rho_k / rho_{k-1}) * (alpha_{k-1} / omega_{k-1})`
3. `p_k = r_{k-1} + beta_k (p_{k-1} - omega_{k-1} v_{k-1})`
4. `p_hat = M^{-1} p_k`, `v_k = A p_hat`
5. `alpha_k = rho_k / <r~, v_k>`
6. `s_k = r_{k-1} - alpha_k v_k`
7. `s_hat = M^{-1} s_k`, `t_k = A s_hat`
8. `omega_k = <t_k, s_k> / <t_k, t_k>`
9. `x_k = x_{k-1} + alpha_k p_hat + omega_k s_hat`
10. `r_k = s_k - omega_k t_k`

## R09

适用条件与局限：

适用：

- 非对称线性系统；
- 大规模稀疏矩阵，且可高效做 matvec；
- 需要低内存的迭代求解流程。

局限：

- 可能出现 breakdown（如 `rho`、`omega` 或分母接近 0）；
- 在病态问题上可能收敛慢或停滞；
- 对预条件质量较敏感。

## R10

正确性要点（工程可验证版）：

1. 每轮都在 Krylov 子空间上构造新方向并更新近似解；
2. `alpha` 控制第一阶段修正，`omega` 对尾部残差做稳定化最小二乘修正；
3. 若数值无 breakdown 且满足残差阈值，则得到 `||b-Ax||` 足够小的近似解；
4. `demo.py` 用已知真解构造 `b=A x_true`，可直接验证相对残差与相对解误差。

## R11

数值稳定性：

- MVP 对 `rho`、`<r~,v>`、`<t,t>`、`omega` 设置 breakdown 阈值保护；
- 同时采用 `max(rtol*||b||, atol)` 作为停止标准，避免 `b` 量级影响阈值解释；
- 使用 Jacobi 预条件器减少尺度差异带来的收敛压力；
- 仍需注意：BiCGSTAB 残差通常“非单调”，短时反弹并不一定表示失败。

## R12

性能视角：

- 主要耗时集中在两次 `A @ vector`；
- 向量内积和 AXPY 更新为次要成本；
- 对稀疏矩阵，若 matvec 高效，BiCGSTAB 常具备良好性价比；
- 预条件器越接近 `A^{-1}` 的作用，迭代次数通常越少（但构造成本更高）。

## R13

本目录 MVP 的可验证保证：

- 固定随机种子构造非对称对角占优系统，结果可复现；
- 自动断言：
  - `result.converged == True`
  - 最终相对残差 `< 1e-8`
  - 相对解误差 `< 1e-8`
- 可选对照 SciPy `bicgstab`，若环境安装 SciPy 则打印对照残差。

## R14

常见失效模式：

- `A` 非方阵或维度不匹配；
- `maxiter<=0`、容差非法；
- breakdown：`rho≈0`、`<r~,v>≈0`、`<t,t>≈0` 或 `omega≈0`；
- 预条件器过弱/不稳定，导致收敛缓慢或停滞。

## R15

实现设计（`demo.py`）：

- `BiCGSTABResult`：集中保存求解结果与诊断信息；
- `validate_inputs`：输入合法性与数值可用性检查；
- `bicgstab`：核心迭代实现（非黑盒）；
- `jacobi_preconditioner`：构造简单对角逆预条件器；
- `build_test_system`：生成可复现实验系统；
- `run_scipy_reference`：可选第三方对照；
- `main`：执行实验、断言、输出报告。

## R16

相关算法链路：

- BiCG：BiCGSTAB 的直接前身；
- CGS：由 BiCG 推导，但常出现更强振荡；
- GMRES：残差最小化更稳健，但内存与正交化成本更高；
- TFQMR、IDR(s)：同属非对称系统迭代求解家族。

## R17

运行方式：

```bash
cd Algorithms/数学-线性代数-0065-BiCGSTAB算法
python3 demo.py
```

依赖：

- 必需：`numpy`
- 可选：`scipy`（仅用于对照打印，不影响主流程）

脚本无交互输入，直接输出收敛报告与检查结果。

## R18

`demo.py` 的源码级算法流程（9 步）如下：

1. `main` 调用 `build_test_system` 构造固定种子的非对称方程组 `Ax=b`，并保留 `x_true` 用于验算。  
2. 构造 Jacobi 预条件器 `M^{-1}`，然后调用手写 `bicgstab`（不是第三方黑盒）。  
3. `bicgstab` 内先检查输入维度/阈值，初始化 `r, r_hat, p, v, alpha, omega, rho_prev`。  
4. 每次迭代先计算 `rho=<r_hat,r>`，再按 `beta` 公式更新搜索方向 `p`。  
5. 进行第一次 matvec：`p_hat=M^{-1}p`，`v=A@p_hat`，计算 `alpha` 并得到中间残差 `s`。  
6. 若 `||s||` 已达阈值，直接执行 `x <- x + alpha*p_hat` 并收敛返回。  
7. 否则进行第二次 matvec：`s_hat=M^{-1}s`，`t=A@s_hat`，计算稳定化系数 `omega`。  
8. 完成全量更新 `x <- x + alpha*p_hat + omega*s_hat`，`r <- s - omega*t`，记录残差历史并判断是否收敛。  
9. `main` 对返回结果做强校验（收敛、相对残差、相对解误差），并可选运行 `run_scipy_reference` 做外部对照。

说明：SciPy 在本 MVP 中只用于“结果交叉验证”；核心 BiCGSTAB 迭代、breakdown 处理和停止准则均在本地源码逐步实现。

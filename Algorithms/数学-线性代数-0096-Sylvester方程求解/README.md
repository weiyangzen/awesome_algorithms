# Sylvester方程求解

- UID: `MATH-0096`
- 学科: `数学`
- 分类: `线性代数`
- 源序号: `96`
- 目标目录: `Algorithms/数学-线性代数-0096-Sylvester方程求解`

## R01

Sylvester 方程的标准形式是：
\[
AX + XB = C
\]
其中：
- \(A\in\mathbb{R}^{m\times m}\)
- \(B\in\mathbb{R}^{n\times n}\)
- \(C\in\mathbb{R}^{m\times n}\)
- 待求 \(X\in\mathbb{R}^{m\times n}\)

它把“矩阵未知量”上的线性方程统一写成左乘与右乘的组合形式。

## R02

该方程在控制、信号处理与数值线性代数中非常常见：
- 连续/离散系统的坐标变换与模型降阶；
- Lyapunov 方程（其特例）求解；
- Kronecker 结构线性系统建模。

工程上常见求解路线有两类：
- 直接向量化为线性方程组（教学直观）；
- Bartels-Stewart（Schur 分解 + 三角 Sylvester 子问题，工业常用）。

## R03

输入：
- 方阵 `A (m x m)`；
- 方阵 `B (n x n)`；
- 矩阵 `C (m x n)`。

输出：
- `X (m x n)`，满足 `A @ X + X @ B ≈ C`；
- 误差指标：残差 `||A X + X B - C||_F`；
- 可选稳定性指标：谱分离度 `min_{i,j} |lambda_i(A)+lambda_j(B)|`。

## R04

唯一解存在条件（经典结论）：
- `spec(A)` 与 `-spec(B)` 不相交；
- 等价写法：对任意特征值对 `(lambda_i, mu_j)`，均有 `lambda_i + mu_j != 0`。

若该条件不满足，可能出现：
- 无解；
- 多解；
- 数值系统奇异或近奇异。

## R05

向量化后可得到显式线性系统：
\[
\mathrm{vec}(AX + XB)
= (I_n \otimes A + B^T \otimes I_m)\,\mathrm{vec}(X)
= \mathrm{vec}(C)
\]

因此可解：
\[
\mathrm{vec}(X) = K^{-1}\mathrm{vec}(C),\quad
K = I_n \otimes A + B^T \otimes I_m
\]

这正是本目录 MVP 的主实现思路。

## R06

伪代码（MVP 的 Kronecker 线性化版本）：

```text
输入 A, B, C
检查维度与数值有效性（方阵、有限值、C 形状匹配）
K <- kron(I_n, A) + kron(B^T, I_m)
rhs <- vec(C)  (按列展平)
求解 K * x = rhs
X <- unvec(x)  (按列恢复为 m x n)
返回 X
```

## R07

正确性要点：
1. `vec(AX)=(I⊗A)vec(X)` 与 `vec(XB)=(B^T⊗I)vec(X)` 是线性代数恒等式；
2. 两式相加即得到 `K vec(X)=vec(C)`；
3. 若 `K` 可逆，则解唯一；
4. 恢复矩阵形状后即得原方程解 `X`。

因此算法正确性可直接由向量化同构保证。

## R08

复杂度（稠密、小规模教学实现）：
- 构造 `K` 的尺寸是 `(mn) x (mn)`；
- 时间复杂度约 `O((mn)^3)`（主导于线性系统求解）；
- 空间复杂度约 `O((mn)^2)`。

结论：该法直观但不适合大规模问题。

## R09

数值性质：
- 当谱分离度很小（`lambda_i(A)+mu_j(B)` 接近 0）时，问题病态；
- 即便理论可解，解也可能对输入扰动高度敏感；
- 残差小不一定代表前向误差小，应结合条件性判断。

在高维工程中通常优先 Schur 路线，避免显式构造巨大 Kronecker 系统。

## R10

与相关方程的关系：
- Lyapunov：`A X + X A^T = -Q`，是 Sylvester 的特例；
- Riccati：非线性矩阵方程，常在迭代中反复求解 Sylvester/Lyapunov 子问题；
- 一般线性方程组：Sylvester 可被“打平”为普通线性系统，但会丢失结构优势。

## R11

常见坑：
- 把 `vec` 展平顺序写错（应与公式一致，通常按列展平）；
- 忽略唯一性条件，直接 `solve` 导致奇异矩阵异常；
- 误以为残差足够小就一定高精度，未检查谱分离；
- 在大规模上仍使用 Kronecker 显式构造，内存迅速爆炸。

## R12

本目录 MVP 设计：
- 主路径：手写 `solve_sylvester_kron`（`numpy`）实现完整可审计流程；
- 校验路径：若环境安装 `scipy`，再调用 `scipy.linalg.solve_sylvester` 对照；
- 输出关键指标：`谱分离度 / 残差 / 两实现差异`；
- 额外构造“不可解示例”验证异常路径。

## R13

运行方式：

```bash
python3 demo.py
```

脚本无交互输入，自动构造测试矩阵并输出结果。

## R14

预期输出要点：
- 打印 `A, B, C`；
- 打印 `X_kron` 与 `||A X + X B - C||_F`（应接近 0）；
- 打印谱分离度（用于判断是否接近病态）；
- 若本机有 SciPy，打印与 SciPy 解的差异；
- 打印一个“奇异/无解情形”的预期失败信息。

## R15

典型应用：
- 控制理论中的状态变换、稳定性分析与模型降阶；
- 连续系统 Gramian 相关计算（经常转化为 Sylvester/Lyapunov）；
- 信号处理中的耦合线性系统识别；
- PDE 半离散后的低秩/张量化线性代数子问题。

## R16

建议最小测试集：
- 小规模唯一可解样例（`m!=n` 也要覆盖）；
- 近奇异样例（谱分离度极小）；
- 明确不可解样例（如 `A=I, B=-I, C!=0`）；
- 含随机扰动样例，观察残差与解变化。

## R17

可扩展方向：
- 切换到 Bartels-Stewart（Schur + `TRSYL`）以提升稳定性与规模适应性；
- 利用稀疏 Kronecker 与迭代法降低内存；
- 引入低秩右端项 `C=UV^T` 的低秩 Sylvester 求解；
- 在批量场景中做分解复用与并行加速。

## R18

若使用第三方 `scipy.linalg.solve_sylvester`，其源码级主流程可拆为 7 步：
1. 在 Python 层检查 `A/B/C` 维度与形状合法性。  
2. 对 `A`、`B^T` 分别做实 Schur 分解，得到上准三角 `R`、`S` 和正交矩阵 `U`、`V`。  
3. 把原问题变换到 Schur 基：`F = U^T C V`。  
4. 将问题变为 `R Y + Y S^T = F` 的准三角 Sylvester 子问题。  
5. 调 LAPACK `?TRSYL`（如双精度 `dtrsyl`）按块回代求 `Y`，并返回可能的缩放因子 `scale`。  
6. 用缩放后的 `Y` 回到原坐标：`X = U (scale*Y) V^T`。  
7. 返回 `X`，若底层返回错误码则抛异常。

本目录 `demo.py` 没把第三方当黑箱：
- 主实现显式给出 `vec + Kronecker` 线性化；
- SciPy 路径只作为可选对照与工程参考。

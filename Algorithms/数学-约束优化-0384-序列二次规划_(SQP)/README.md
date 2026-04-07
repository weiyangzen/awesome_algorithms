# 序列二次规划 (SQP)

- UID: `MATH-0384`
- 学科: `数学`
- 分类: `约束优化`
- 源序号: `384`
- 目标目录: `Algorithms/数学-约束优化-0384-序列二次规划_(SQP)`

## R01

序列二次规划（Sequential Quadratic Programming, SQP）用于求解一般非线性约束优化问题：

\[
\begin{aligned}
\min_{x\in\mathbb{R}^n}\quad & f(x) \\
\text{s.t.}\quad & h_i(x)=0,\ i=1,\dots,m_e \\
& g_j(x)\ge 0,\ j=1,\dots,m_i
\end{aligned}
\]

SQP 的核心思想是：在当前点把目标函数做二次近似、把约束做一阶线性化，得到一个 QP 子问题；反复解 QP 并更新点，逐步逼近原问题 KKT 解。

## R02

适用场景：
- 中低维到中等维度的光滑非线性约束优化。
- 目标和约束都可导，且可以提供梯度/Jacobian。
- 需要比单纯罚函数法更稳健的约束可行性推进。
- 典型应用：参数估计中的结构约束、工程设计优化、控制和轨迹优化中的非线性约束问题。

## R03

标准 SQP 一步通常包含：
- 构造拉格朗日函数
\[
\mathcal{L}(x,\lambda,\mu)=f(x)-\lambda^\top h(x)-\mu^\top g(x)
\]
- 在当前迭代点 \(x_k\) 处构造 QP 子问题（变量为步长 \(d\)）
\[
\begin{aligned}
\min_d\quad & \nabla f(x_k)^\top d + \frac{1}{2} d^\top B_k d \\
\text{s.t.}\quad & h(x_k)+J_h(x_k)d=0 \\
& g(x_k)+J_g(x_k)d\ge 0
\end{aligned}
\]
其中 \(B_k\) 是 \(\nabla^2_{xx}\mathcal{L}\) 的近似（常用 BFGS 更新）。
- 用线搜索或信赖域接受步长，并更新乘子与 Hessian 近似，进入下一轮。

## R04

本目录 `demo.py` 的输入输出约定：
- 输入：
  - 固定初始点 `x0 = [0.2, 0.9]`。
  - 目标函数和梯度显式给出。
  - 3 个不等式约束（单位圆约束、线性和约束、`x0>=0` 约束）与对应 Jacobian。
- 输出：
  - `x*`、`f(x*)`、收敛标志 `success`。
  - 迭代次数 `nit`、函数/梯度评估次数。
  - 终点约束值与近似 KKT 驻点残差。
  - 迭代轨迹（`pandas.DataFrame` 打印 head/tail）。

## R05

SQP 伪代码（概念版）：

```text
given x0, multipliers (lambda0, mu0), Hessian approx B0
for k = 0,1,2,...
    build QP around xk:
        min_d  grad f(xk)^T d + 1/2 d^T Bk d
        s.t.   h(xk)+Jh(xk)d = 0
               g(xk)+Jg(xk)d >= 0

    solve QP -> dk, (lambda_qp, mu_qp)

    choose step length alpha_k (line search / trust region)
    x_{k+1} = x_k + alpha_k * d_k

    update multipliers from QP solution
    update Bk by quasi-Newton rule on Lagrangian gradient

    if KKT residual < tol:
        stop
```

## R06

正确性与收敛要点（工程视角）：
- QP 子问题是对原问题一阶（约束）+二阶（目标/拉格朗日）的局部模型，解 \(d_k\) 代表“局部最优修正方向”。
- 线搜索常配合罚函数/merit function 保证全局下降趋势，而不仅是局部牛顿步。
- 当 LICQ、二阶充分条件等常见假设成立，并且 Hessian 近似合理时，SQP 在解附近通常表现出快速局部收敛。
- 不等式约束通过活跃集（active set）自然进入 KKT 体系：活跃约束对应非零乘子，非活跃约束乘子趋近 0。

## R07

复杂度（单次迭代粗略量级）：
- 构建梯度/Jacobian：约 `O(C_f + C_c)`，取决于函数本身计算成本。
- 求解 QP 子问题（稠密情形）：通常主成本约 `O((n+m)^3)`，其中 `n` 为变量维数，`m` 为活跃约束规模。
- 存储：若显式维护稠密 Hessian 近似，空间约 `O(n^2)`。

实际耗时通常由“QP 求解 + 目标/约束评估次数”共同决定。

## R08

与常见约束优化方法对比：
- 内点法：把不等式放入 barrier 体系，适合大规模稀疏结构；SQP 在中等规模高精度场景常表现优良。
- 增广拉格朗日：外层乘子更新 + 内层无约束/弱约束优化，调参友好；SQP 对约束几何结构利用更直接。
- 投影梯度法：实现简单但步方向信息有限；SQP 使用二次模型，局部收敛通常更快。
- 纯罚函数法：实现成本低，但罚参数过大易病态；SQP 通常可行性与最优性平衡更好。

## R09

`demo.py` 的关键数据结构：
- `x`：`numpy.ndarray`，当前解向量（2 维）。
- `constraints`：SciPy 约束字典列表，含 `type/fun/jac`。
- `SQPDemoResult`：`dataclass` 封装最终结果与诊断信息。
- `history`：`pandas.DataFrame`，每轮记录 `x0/x1/f(x)/约束值`。

## R10

边界与异常处理：
- 约束全部按 SciPy 约定使用 `ineq >= 0`，避免符号写反导致“可行域颠倒”。
- 提供解析梯度与 Jacobian，减少有限差分噪声对收敛的影响。
- 设置 `maxiter` 和 `ftol` 防止无限迭代。
- 收敛后做二次校验：
  - `success` 必须为真；
  - 最小约束值不得低于容差；
  - 目标值相对初始点必须下降。

## R11

本目录 MVP 实现策略：
- 采用 `scipy.optimize.minimize(method="SLSQP")` 作为 SQP 求解器。
- 不把调用当黑箱：显式给出 `f`、`∇f`、`g`、`J_g`，并记录迭代轨迹与 KKT 风格残差。
- 依赖保持最小：
  - `numpy`：数值计算；
  - `scipy`：SLSQP 求解；
  - `pandas`：迭代日志结构化展示。

## R12

运行方式（仓库根目录）：

```bash
uv run python Algorithms/数学-约束优化-0384-序列二次规划_(SQP)/demo.py
```

或在当前目录直接：

```bash
uv run python demo.py
```

无需交互输入。

## R13

预期输出特征：
- 显示 `success=True` 与收敛消息（通常是 `Optimization terminated successfully`）。
- 打印 `x*` 和 `f(x*)`，结果应明显优于初始点。
- 终点各约束值应 `>= 0`（考虑数值容差）。
- 迭代轨迹前几步快速下降，后几步趋于稳定。

## R14

常见实现错误：
- 把不等式写成 `<=0` 却按 `>=0` 传给 SLSQP。
- 忘记提供 `jac`，导致数值梯度不稳、迭代次数激增。
- 约束函数返回向量/标量维度不一致。
- 仅检查 `success`，不检查最终约束残差，导致“名义收敛但不可行”。

## R15

最小测试清单：
- 功能测试：脚本可直接运行，且 `success=True`。
- 可行性测试：`min(constraint_values) >= -1e-7`。
- 改善性测试：`f(x*) < f(x0)`。
- 稳定性测试：修改初值（如 `[0.0, 0.8]`、`[0.5, 0.5]`）仍可收敛到可行解。

## R16

可扩展方向：
- 扩展到更多等式/不等式约束，并加入约束缩放（scaling）。
- 用自动微分（PyTorch/JAX）自动构造梯度与 Jacobian。
- 在大规模问题中切换到稀疏结构求解器（如 `trust-constr` + sparse Jacobian）。
- 对比 SQP 与内点法/增广拉格朗日法的收敛轨迹和鲁棒性。

## R17

局限与取舍：
- 本 MVP 只演示小规模 2 维问题，重点是“可运行 + 可解释”。
- 依赖 SciPy SLSQP 内核，未手写完整 QP 子问题求解器。
- 驻点残差是近似诊断，不等同于完整严谨 KKT 证明。
- 工业问题常需更强的缩放、正则化、warm-start 和失败重启策略。

## R18

`demo.py` 对应的源码级算法流（SciPy SLSQP，非一句黑箱）：
1. Python 侧先把目标函数、梯度、边界和每个约束函数/Jacobian 组装成 SLSQP 所需接口，并统一为 `ineq >= 0` 形式。  
2. 在当前迭代点，SLSQP 构造局部近似：目标采用二次模型（拉格朗日 Hessian 的近似），约束采用一阶线性化。  
3. 基于该局部模型形成一个 QP 子问题（或等价的最小二乘子问题变体）以求搜索方向。  
4. 使用活跃集思想处理不等式：候选活跃约束进入子问题，非活跃约束暂不绑定。  
5. 得到候选步后执行步长控制（line search / merit-function 风格接受准则），平衡目标下降与约束可行性改进。  
6. 更新原变量与约束乘子；并用拟牛顿思想更新拉格朗日 Hessian 近似矩阵。  
7. 反复执行“线性化约束 + 二次目标 + 子问题求解 + 更新”循环，直到满足终止准则（目标改变量、步长、约束残差等）。  
8. `demo.py` 在求解结束后再做外部校验：打印约束值、估计 KKT 驻点残差和迭代轨迹，确保求解过程可解释、可验证。

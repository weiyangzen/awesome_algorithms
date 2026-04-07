# 半定规划 (SDP)

- UID: `MATH-0389`
- 学科: `数学`
- 分类: `凸优化`
- 源序号: `389`
- 目标目录: `Algorithms/数学-凸优化-0389-半定规划_(SDP)`

## R01

半定规划（Semidefinite Programming, SDP）是凸优化中的核心模型之一。它把变量从向量扩展为对称矩阵 `X`，并要求 `X` 属于半正定锥（`X ⪰ 0`）。标准原始形式可写为：

\[
\min_X \langle C, X\rangle,\quad
\text{s.t. } \langle A_i, X\rangle=b_i,\ i=1,\dots,m,\ X\succeq 0
\]

其中 `⟨U,V⟩ = trace(U^T V)`。该模型可统一描述 Max-Cut 松弛、控制中的 LMI 约束、鲁棒优化等问题。

## R02

本目录实现的 MVP 选择了经典 **Max-Cut 的 SDP 松弛** 作为示例问题。  
对加权无向图（权重矩阵 `W`）：

\[
\max_X \frac14\langle L, X\rangle,\quad
\text{s.t. } \operatorname{diag}(X)=\mathbf{1},\ X\succeq 0
\]

`L = D - W` 是图拉普拉斯矩阵。脚本把它改写成最小化形式：

\[
\min_X \langle C,X\rangle,\quad C=-\frac14 L
\]

这样与标准 SDP 形式完全一致。

## R03

求解方法采用“**可行起点 + 对数障碍 + 牛顿法**”：

- 内层子问题（固定 `t`）：
\[
\min_X\ t\langle C,X\rangle - \log\det(X),\ \text{s.t. } \langle A_i,X\rangle=b_i
\]
- 外层 barrier 迭代逐步放大 `t`，逼近原始 SDP 边界最优解。

这是一种经典内点思想。MVP 中没有调用现成 SDP 黑盒求解器，而是手写关键 KKT 线性系统与回溯线搜索。

## R04

对障碍目标 `f_t(X)=t⟨C,X⟩-\log\det(X)`，有：

- 梯度：
\[
\nabla f_t(X)=tC-X^{-1}
\]
- Hessian 线性算子：
\[
\mathcal{H}_X[\Delta X]=X^{-1}\Delta X X^{-1}
\]

在等式约束下，牛顿方向 `ΔX` 与拉格朗日乘子方向 `w` 满足：

\[
\mathcal{H}_X[\Delta X] + \mathcal{A}^\*(w) = -\nabla f_t(X),\quad
\mathcal{A}(\Delta X)=0
\]

其中 `A(X)=[⟨A_1,X⟩,\dots,⟨A_m,X⟩]`，`A*(y)=Σ_i y_i A_i`。

## R05

`demo.py` 用 Schur 补把 KKT 系统降维到仅 `m` 维线性方程：

- 设 `h_i = ⟨A_i, X (∇f_t) X⟩`
- 设 `M_{ij} = ⟨A_i, X A_j X⟩`
- 解 `M w = -h`
- 再恢复牛顿步：
\[
\Delta X = -X\left(\nabla f_t(X)+\mathcal{A}^\*(w)\right)X
\]

该流程是源码中 `solve_barrier_subproblem_newton` 的核心，体现了 SDP 内点法最关键的“矩阵微分 + 线性代数”环节。

## R06

稳定性与可行性处理策略：

- 初始点固定为 `X0 = I`，天然满足 `diag(X)=1` 且严格正定。  
- 每次步长采用回溯线搜索，先确保 `X + αΔX` 仍是 SPD，再检查 Armijo 下降条件。  
- 每轮更新后做对称化 `0.5*(X+X^T)`，减少浮点误差。  
- 若 Schur 系统 `M` 数值退化，回退到最小二乘解 `lstsq`，提升鲁棒性。

## R07

外层 barrier 终止依据采用标准间隙估计：

\[
\text{gap}_{est}=\frac{n}{t}
\]

其中 `n` 是 `X` 的维度。每次中心化后若 `n/t <= gap_tol` 则认为 barrier 收敛。  
这与内点法中的“中心路径逼近误差界”一致，适合教学版 MVP。

## R08

复杂度（稠密实现）：

- 内层一次牛顿迭代主要开销：
  - `X^{-1}`：`O(n^3)`
  - 构造 `M (m×m)`：约 `O(m^2 n^3)`（本例 `m=n` 且很小）
  - 解 `M w = -h`：`O(m^3)`
- 外层 barrier 需要若干次中心化，每次包含多轮牛顿迭代。

因此总体更适合中小规模演示；大规模稀疏 SDP 通常需要专门结构化求解器。

## R09

本 MVP 的“最小但诚实”取舍：

- 只实现原始标准形 + 等式约束 + 对数障碍，不覆盖对偶形式和锥分解高级技巧。  
- 不依赖 CVXOPT/CVXPy 直接求解，核心迭代全手写，保证可追踪。  
- 图规模选 `n=5`，便于同时做随机 rounding 和 brute-force 精确解对照。  
- 保留迭代表格输出，能直观看到 `t`、间隙、最小特征值、可行性残差变化。

## R10

`demo.py` 的主要函数分工：

- `A_map / A_adjoint`：实现线性算子 `A` 与伴随算子 `A*`。  
- `solve_barrier_subproblem_newton`：固定 `t` 的可行牛顿中心化。  
- `solve_sdp_barrier`：外层 barrier 主循环（更新 `t`、记录 gap）。  
- `build_maxcut_sdp_instance`：构造 Max-Cut 松弛的 `C, A_i, b`。  
- `random_hyperplane_rounding`：从 SDP 解采样离散切割。  
- `exact_maxcut_bruteforce`：对 `n=5` 图做精确最优切割对照。  
- `main`：汇总求解结果并打印一致性检查。

## R11

运行方式（无交互输入）：

```bash
cd Algorithms/数学-凸优化-0389-半定规划_(SDP)
uv run python demo.py
```

脚本会自动：
1. 求解 SDP 松弛；
2. 输出 barrier 迭代历史；
3. 进行随机超平面 rounding；
4. 与 brute-force 精确 Max-Cut 比较。

## R12

关键输出字段说明：

- `SDP relaxation value`：松弛问题目标值（对 Max-Cut 是上界）。  
- `Best rounded cut`：随机 rounding 得到的离散可行割值。  
- `Exact Max-Cut`：穷举得到的真实最优割值（仅小图可做）。  
- `eq_residual`：`||A(X)-b||`，本问题即 `||diag(X)-1||`。  
- `min_eig(X)`：最小特征值，应非负且通常略大于 0（SPD 迭代点）。

理论上应满足：`rounded_cut <= exact_cut <= SDP_relaxation_value`（允许微小数值误差）。

## R13

正确性自检点（脚本已体现）：

- 检查 `diag(X)` 约束残差是否接近 0。  
- 检查 `X` 的最小特征值是否非负。  
- 检查 `rounded_cut <= SDP upper bound` 与 `exact_cut <= SDP upper bound`。  
- 若线搜索无法找到可行下降步，立即报错，避免静默输出伪结果。

## R14

可调参数建议：

- `gap_tol`：越小精度越高、迭代越多。  
- `mu`：外层 `t <- mu*t` 的放大倍率，过大可能导致中心化难度上升。  
- `max_outer_iters / max_newton_iters`：控制计算预算。  
- `rounds`：随机 rounding 轮数，越多越容易找到更好割值。  

实践中可先保持默认值，确认流程正确后再调精度和速度。

## R15

常见失败模式与处理：

- `Initial point must satisfy equality constraints`：起点不在可行仿射空间。  
- `Matrix is not SPD`：线搜索步长过大或数值误差导致越界。  
- `Line search failed`：当前方向不够下降，常见于病态问题或参数过激。  
- `Newton did not converge`：子问题过难或容差过严，可放宽 `newton_tol`/增加迭代上限。

## R16

与其他凸优化方法对比：

- 相比 ADMM：内点法通常在中小规模上精度高、收敛轮数少，但单步矩阵运算更重。  
- 相比一阶方法（投影梯度、Frank-Wolfe）：内点法更“二阶”，迭代稳定但不适合超大规模。  
- 相比通用建模器（CVXPy）：本实现更底层，教学和调试透明度高，但通用性较弱。

## R17

可扩展方向：

- 支持一般线性约束而不只 `diag(X)=1`。  
- 增加对偶变量恢复与 KKT 残差全面报告。  
- 使用稀疏矩阵结构优化 `M` 构造与线性求解。  
- 引入更系统的 predictor-corrector 原始-对偶内点法。  
- 将 rounding 扩展为 Goemans-Williamson 近似比实验。

## R18

`demo.py` 的源码级算法流程可拆为 8 步（非黑箱）：

1. `build_maxcut_sdp_instance` 构造 `W`、`L`、`C=-0.25L`，并把 `diag(X)=1` 写成 `A_i,b_i`。  
2. 在 `solve_sdp_barrier` 中设初值 `X0=I`、`t=1`，开始外层 barrier 循环。  
3. 每个外层迭代调用 `solve_barrier_subproblem_newton`，计算梯度 `tC-X^{-1}`。  
4. 组装 Schur 补矩阵 `M` 和向量 `h`，求解 `M w = -h` 得到约束乘子方向。  
5. 用 `ΔX = -X(grad + A*(w))X` 恢复牛顿步，并用回溯线搜索保证 SPD 与下降。  
6. 当牛顿减量足够小后返回中心点，记录 `<C,X>`、`gap_est=n/t`、`eq_residual`、`min_eig(X)`。  
7. 外层把 `t <- mu*t` 继续逼近原问题，直到 `n/t <= gap_tol` 或到达迭代上限。  
8. `main` 用最终 `X` 计算 SDP 上界，再做随机超平面 rounding 与 brute-force 精确解对比，完成端到端验证。  

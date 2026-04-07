# Sturm-Liouville理论

- UID: `MATH-0431`
- 学科: `数学`
- 分类: `常微分方程`
- 源序号: `431`
- 目标目录: `Algorithms/数学-常微分方程-0431-Sturm-Liouville理论`

## R01

Sturm-Liouville 理论研究如下特征值问题：

`-(p(x)y')' + q(x)y = \lambda w(x)y,\quad x\in[a,b]`

并配合边界条件（本条目采用 Dirichlet：`y(a)=y(b)=0`）。其核心性质包括：
- 特征值实且可按大小排序；
- 不同特征值对应的特征函数在加权内积 `⟨u,v⟩_w=∫u v w dx` 下正交；
- 在正则条件下特征函数构成完备基。

本目录给出一个可运行 MVP：用有限差分离散为三对角广义特征值问题，数值求前几个模态并验证谱与正交性。

## R02

本实现的任务定义：
- 输入：`a,b,p(x),q(x),w(x),n_interior,n_modes`。
- 输出：
  - 前 `n_modes` 个近似特征值 `lambda_1...lambda_k`；
  - 对应离散特征向量（内点值）；
  - 每个模态的相对残差 `||Au-\lambda Wu||/||Au||`；
  - 加权 Gram 矩阵 `G=U^T W U`（应接近单位阵）。

`demo.py` 内置两个确定性样例，不需要交互输入：
1. 经典可解析样例（与 `n^2` 对照）；
2. 变系数样例（检查稳定性与结构性质）。

## R03

数学离散基础（中心差分 + 通量形式）：

连续问题：
`-(p y')' + qy = \lambda w y`。

网格 `x_i=a+ih, i=0..N+1`，内点 `i=1..N`，边界 `y_0=y_{N+1}=0`。

对通量项采用
`-(p y')'(x_i) \approx -[p_{i+1/2}(y_{i+1}-y_i)-p_{i-1/2}(y_i-y_{i-1})]/h^2`。

得到三对角广义特征值系统：
`A u = \lambda W u`，其中
- `A_ii = (p_{i-1/2}+p_{i+1/2})/h^2 + q_i`
- `A_{i,i+1}=A_{i+1,i}=-p_{i+1/2}/h^2`
- `W = diag(w_i)`。

再通过相似变换 `z = W^{1/2}u` 转为标准对称问题：
`C z = \lambda z`, `C=W^{-1/2}AW^{-1/2}`。

## R04

算法流程（高层）：
1. 校验区间、网格规模、模态数。  
2. 构造均匀网格和中点网格。  
3. 评估 `p_mid, q_i, w_i` 并检查 `p>0, w>0`。  
4. 组装三对角 `A` 的主对角与副对角。  
5. 构造 `C=W^{-1/2}AW^{-1/2}` 的三对角系数。  
6. 调用 `scipy.linalg.eigh_tridiagonal` 求最小 `k` 个特征对。  
7. 回代 `u=W^{-1/2}z`，做加权归一化与符号固定。  
8. 计算残差和 Gram 误差，输出表格并按阈值判定 PASS/FAIL。

## R05

核心数据结构：
- `SturmLiouvilleProblem`（输入配置）：
  - `a,b,p_fn,q_fn,w_fn`
  - `n_interior,n_modes`
  - `name`
- `SturmLiouvilleResult`（求解结果）：
  - `x_interior,h`
  - `diag_a,off_a,weight`
  - `eigenvalues,eigenvectors`
  - `residual_rel,gram`
- `pandas.DataFrame`：展示每个模态的特征值、误差和残差。

## R06

正确性依据：
- 结构正确：离散矩阵来自 Sturm-Liouville 通量离散，保持对称性与权重结构。  
- 谱正确：经典样例 `-(y')'=\lambda y, y(0)=y(\pi)=0` 理论特征值为 `\lambda_n=n^2`，可直接对照。  
- 正交正确：数值特征向量满足 `U^T W U \approx I`。  
- 代数正确：每个模态都计算 `Au-\lambda Wu` 相对残差并限制上界。

## R07

复杂度分析（`N=n_interior`, `k=n_modes`）：
- 组装三对角系数：`O(N)`。
- 三对角特征分解（取前 `k` 个）：通常 `O(Nk)` 到 `O(N^2)`，依赖底层 LAPACK 路径。
- 残差与正交性检查：`O(Nk + Nk^2)`。
- 总体空间：`O(Nk)`（保存特征向量）与 `O(N)`（对角数据）。

对于 MVP 的 `N≈300, k≤6`，运行开销很小。

## R08

边界与异常处理：
- `b<=a`、`n_modes>=n_interior`、`n_interior<10`：抛 `ValueError`。  
- `p/q/w` 返回形状错误或含 `nan/inf`：抛 `ValueError`。  
- `min(p)<=0` 或 `min(w)<=0`：抛 `ValueError`（不满足正则 Sturm-Liouville 条件）。  
- 任一样例超过精度阈值：抛 `RuntimeError`，避免静默错误。

## R09

MVP 取舍：
- 保留：
  - 最核心的 Sturm-Liouville 离散谱求解链路；
  - 经典可解析验证 + 变系数验证；
  - 明确的残差与正交性检查。
- 不包含：
  - 非均匀网格与高阶差分；
  - Neumann/Robin 边界的完整参数化；
  - 自适应求谱区间与误差后验估计。

目标是“实现短小、理论关键点完整、结果可复核”。

## R10

`demo.py` 主要函数职责：
- `_validate_problem`：输入合法性检查。  
- `_build_discretization`：组装 `A` 的三对角系数与权重。  
- `_tridiag_matvec`：三对角矩阵乘向量（用于残差）。  
- `solve_sturm_liouville`：主求解流程（变换、求特征、归一化、诊断）。  
- `run_case_with_exact`：带解析谱对照的样例执行。  
- `run_case_without_exact`：无解析解样例执行。  
- `main`：构造两个固定样例、汇总验收。

## R11

运行方式：

```bash
cd Algorithms/数学-常微分方程-0431-Sturm-Liouville理论
uv run python demo.py
```

脚本是非交互式的，会直接打印每个样例的表格与总结果。

## R12

输出字段说明：
- 公共字段：
  - `mode`：模态编号（从 1 开始）；
  - `lambda_num`：数值特征值；
  - `rel_residual`：相对残差。
- 解析样例附加字段：
  - `lambda_exact`：理论值 `n^2`；
  - `abs_err`：绝对误差；
  - `rel_err`：相对误差。
- 样例汇总：
  - `max relative eigenvalue error`；
  - `max weighted orthogonality error`（`max|U^T W U - I|`）。
- 总结输出：`PASS: True/False`。

## R13

内置测试样例：
1. `Canonical`：
- 方程 `-(y')' = \lambda y`，区间 `[0,\pi]`，Dirichlet 边界；
- 解析特征值 `\lambda_n=n^2`；
- 用于检验谱精度。

2. `Variable coefficients`：
- 方程 `-(p y')' + q y = \lambda w y`，其中
  - `p(x)=1+0.5x`
  - `q(x)=1+x`
  - `w(x)=1+x`
  - 区间 `[0,1]`；
- 无解析谱对照，使用残差与正交性做一致性验证。

## R14

关键参数与经验：
- `n_interior`：内点数，越大通常谱误差越小（本 demo 约 280~320）。
- `n_modes`：求解前几个低频模态（本 demo 为 5~6）。
- 验收阈值（`main`）：
  - `max_rel_err_exact < 2e-3`
  - `stability_exact < 5e-10`
  - `stability_variable < 5e-10`

说明：这些阈值兼顾数值稳定与不同平台浮点差异。

## R15

与其它路径对比：
- 对比 shooting + root finding：
  - shooting 对每个模态都要做边值匹配；
  - 本实现一次性得到多个模态，且天然带正交结构。
- 对比通用稠密 `eigh(A,B)`：
  - 稠密法实现直观但存储和计算更重；
  - 三对角结构可显著降低成本。
- 对比黑盒 ODE 积分：
  - ODE 积分适合单轨迹；
  - Sturm-Liouville 关注的是“谱问题”，矩阵特征分解更直接。

## R16

典型应用：
- 振动与波动问题的模态分析（弦、梁、量子阱简化模型）。
- 分离变量法中的空间本征问题。
- 谱方法基函数构造与降维表示。
- 工程 PDE 离散前的 1D 基准验证。

## R17

可扩展方向：
- 增加 Neumann/Robin 边界条件离散模板。  
- 支持非均匀网格和更高阶差分。  
- 加入误差收敛阶实验（网格加密对比）。  
- 仅求部分谱区间（shift-invert / Lanczos）以处理更大规模问题。  
- 输出模态曲线与节点分布图用于教学可视化。

## R18

`demo.py` 源码级算法流（9 步）：
1. `main` 构造两个固定 `SturmLiouvilleProblem`，并设置网格规模与模态数。  
2. `solve_sturm_liouville` 调用 `_validate_problem` 做端点、规模和模态数量检查。  
3. `_build_discretization` 在内点与中点上评估 `p,q,w`，组装三对角 `A` 与权重向量 `w`。  
4. 按 `diag_c=diag_a/w`、`off_c=off_a/sqrt(w_i w_{i+1})` 构造 `C=W^{-1/2}AW^{-1/2}` 的三对角表示。  
5. 调用 `scipy.linalg.eigh_tridiagonal(diag_c, off_c, select="i")` 计算最小 `k` 个特征值与特征向量 `z`。  
6. 将 `z` 回代为原广义特征向量 `u=z/sqrt(w)`，并在加权内积下归一化。  
7. 计算 `gram = U^T W U`，检查正交归一性质；并用 `_tridiag_matvec` 计算每个模态的 `Au-\lambda Wu` 残差。  
8. 对解析样例，`run_case_with_exact` 逐模态对照 `\lambda_n=n^2`，统计绝对/相对误差并打印表格。  
9. `main` 汇总所有稳定性指标，与阈值比较后输出 `PASS`，若失败则抛 `RuntimeError`。

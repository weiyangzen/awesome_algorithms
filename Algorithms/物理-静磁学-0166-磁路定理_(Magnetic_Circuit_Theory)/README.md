# 磁路定理 (Magnetic Circuit Theory)

- UID: `PHYS-0165`
- 学科: `物理`
- 分类: `静磁学`
- 源序号: `166`
- 目标目录: `Algorithms/物理-静磁学-0166-磁路定理_(Magnetic_Circuit_Theory)`

## R01

磁路定理（Hopkinson's law）把磁场问题在工程近似下转写为“电路类比”问题：

\[
\mathcal{F} = NI = \Phi \mathcal{R},
\]

其中 `NI` 为磁动势（At），`Φ` 为磁通（Wb），`ℜ` 为磁阻（At/Wb）。

本条目实现一个最小可运行 MVP：

- 正向：求解两回路磁路（共享支路 + 两个并联分支，其中分支1含气隙）的分支磁通；
- 反向：根据带噪“测得总磁通”估计等效磁阻与气隙长度。

## R02

核心方程（SI）：

\[
\mathcal{R}=\frac{l}{\mu_0\mu_r A},\quad
\mathcal{R}_{gap}=\frac{g}{\mu_0 A_{eff}},\quad
A_{eff}=k_f A.
\]

两回路线性方程组：

\[
\begin{bmatrix}
R_{sh}+R_1 & R_{sh}\\
R_{sh} & R_{sh}+R_2
\end{bmatrix}
\begin{bmatrix}
\phi_1\\
\phi_2
\end{bmatrix}
=
\begin{bmatrix}
NI\\
NI
\end{bmatrix}
\]

- `R_sh`：共享支路磁阻；
- `R1`：分支1总磁阻（铁芯 + 气隙）；
- `R2`：分支2磁阻。

并且有闭式总磁通：

\[
\Phi_{total}=NI\cdot\frac{R_1+R_2}{R_1R_2+R_{sh}(R_1+R_2)}.
\]

## R03

MVP 问题定义：

- 给定几何参数与材料相对磁导率，计算不同电流下的 `φ1, φ2, Φtotal, B1, B2`；
- 合成带噪测量数据 `(I, Φ_measured)`；
- 用 `sklearn` 回归 `Φ = k·NI` 得到 `R_eq = 1/k`；
- 用 `scipy` / `torch` 从测量数据反演气隙 `g`。

这是磁路分析里“正向求解 + 参数辨识”的最小闭环。

## R04

算法选型：

- 正向求解：`numpy.linalg.solve`（2x2 线性系统）与闭式公式双通道；
- 线性估计：`LinearRegression(fit_intercept=False)` 拟合 `Φ-k·NI=0`；
- 非线性估计：`scipy.optimize.least_squares` 最小化 `Φ_model(g)-Φ_measured`；
- 交叉验证：可选 `PyTorch` 对 `g` 做梯度下降反演。

选择理由：

- 规模小（单参数 + 小矩阵），不需要大型有限元；
- 保持物理可解释性，避免黑箱；
- 易于验证和扩展。

## R05

物理与建模前提：

- 静磁、线性区近似；
- 忽略铁磁饱和、磁滞与涡流；
- 视各支路截面和材料参数为常量；
- 气隙边缘效应用固定修正系数 `k_f` 近似。

因此该 MVP 适合前期估算、教学与参数敏感性分析，不替代高保真电磁场仿真。

## R06

`demo.py` 内置合成数据：

- 电流范围：`0.2 ~ 2.0 A`，样本数 `25`；
- 匝数：`N=250`；
- 真实气隙：`g_true=1.2e-3 m`；
- 测量噪声：`σ_Φ = 3e-7 Wb`（加在总磁通上）。

脚本先用真实参数生成无噪声解，再叠加噪声构建“测量值”。

## R07

时间复杂度（样本数 `n`，优化迭代步数 `T`）：

- 两回路正向求解：`O(n)`（每点常数维 2x2 线性代数）；
- `sklearn` 线性拟合：`O(n)`；
- `scipy least_squares`：`O(T·n)`；
- `torch` 反演（可选）：`O(T·n)`。

空间复杂度整体为 `O(n)`。

## R08

核心数据结构：

- `MagneticCircuitConfig`：全部几何、材料和拟合参数；
- `dict[str, float]`：磁阻组件（共享支路、气隙、分支总磁阻）；
- `pandas.DataFrame`：
  - 两回路求解表（`phi1/phi2/B1/B2/环路残差`）
  - 合成测量表（`I/NI/Φ_true/Φ_measured`）
  - 拟合结果汇总表（`truth/scipy/torch`）。

## R09

伪代码：

```text
load config
build reluctances from geometry and gap
for each current:
    solve 2x2 loop equations -> phi1, phi2
    compute phi_total and B fields
create noisy measurements phi_measured = phi_true + noise
fit Phi = k * NI with sklearn -> R_eq = 1/k
estimate gap g with scipy least_squares on residual Phi_model(g)-Phi_measured
(optional) estimate gap with torch gradient descent
print tables + checks + PASS/FAIL
```

## R10

默认参数（见 `MagneticCircuitConfig`）：

- `turns=250`
- `n_samples=25`
- `current_min_a=0.2`, `current_max_a=2.0`
- `true_gap_m=1.2e-3`
- `flux_noise_std_wb=3e-7`
- `mur_shared=1200`, `mur_branch1=1200`, `mur_branch2=900`
- `gap_fringing_factor=1.08`
- `gap bounds=[2e-4, 3e-3] m`
- `torch_steps=2000`, `torch_lr=0.05`

## R11

脚本输出内容：

- 真实参数下的磁阻组件表；
- 两回路无噪声求解预览（含安培环路残差）；
- 前 5 行合成测量数据；
- `sklearn` 等效磁阻拟合质量（`R²`, `RMSE`）；
- `scipy/torch` 气隙估计对比与误差；
- 阈值检查与最终 `Validation: PASS/FAIL`。

## R12

`demo.py` 函数职责：

- `reluctance`：单段磁阻计算；
- `build_reluctances`：组装磁路各组件磁阻；
- `solve_two_loop_flux`：2x2 方程求 `φ1, φ2`；
- `simulate_flux_measurements`：生成带噪测量；
- `estimate_equivalent_reluctance`：线性拟合 `R_eq`；
- `estimate_gap_with_scipy`：最小二乘反演 `g`；
- `estimate_gap_with_torch`：梯度下降反演 `g`（可选）；
- `main`：组织流程与验证输出。

## R13

运行方式（当前算法目录下）：

```bash
uv run python demo.py
```

无需交互输入。

## R14

常见错误与规避：

- 单位错误（mm 与 m 混用）会导致磁阻和磁通量级错误；
- 忽略气隙边缘修正可能低估磁通；
- 用过大噪声做反演会导致 `g` 偏差明显；
- 把超出线性区（饱和区）数据硬套到线性磁路模型会失真。

## R15

最小验证建议：

1. 默认运行应得到 `Validation: PASS`；
2. 将 `flux_noise_std_wb` 提高 10 倍，观察 `g` 估计误差上升；
3. 修改 `true_gap_m`（例如 `0.8e-3` 或 `1.8e-3`），确认拟合可跟随；
4. 关闭/缺失 `torch` 时脚本仍应正常完成（仅少一个对照方法）。

## R16

适用范围与局限：

- 适用：磁路设计初算、参数回归、教学演示；
- 局限：
  - 不包含磁滞与饱和；
  - 不考虑三维漏磁、复杂边界效应；
  - 不替代有限元电磁场求解器。

## R17

可扩展方向：

- 用分段 `μ_r(B)` 或 `B-H` 曲线引入饱和非线性；
- 增加多气隙、多支路拓扑，改成稀疏矩阵通用求解；
- 引入不确定度传播（对 `μ_r`、尺寸、公差做蒙特卡洛）；
- 对真实实验数据加入鲁棒损失（Huber / Cauchy）。

## R18

`demo.py` 源码级流程追踪（8 步，含第三方库拆解）：

1. `main` 读取 `MagneticCircuitConfig`，调用 `build_reluctances` 生成真值磁阻。
2. `solve_two_loop_flux` 构造 2x2 系数矩阵，调用 `numpy.linalg.solve` 计算 `φ1, φ2`，并计算 `loop_residual` 验证安培环路平衡。
3. `simulate_flux_measurements` 在 `Φ_total_true` 上叠加高斯噪声，形成可回归的数据表。
4. `estimate_equivalent_reluctance` 调用 `LinearRegression.fit`：
   - 构造单特征输入 `X=NI`；
   - 最小化 `||Xk-Φ||²` 得到斜率 `k`；
   - 由 `R_eq=1/k` 回推等效磁阻。
5. `estimate_gap_with_scipy` 调用 `least_squares`：
   - 目标函数是残差 `r(g)=Φ_model(g)-Φ_measured`；
   - 在边界 `[g_min,g_max]` 内迭代更新参数；
   - 输出 `gap_m`、`nfev`、`cost`、`RMSE/R²`。
6. `model_total_flux_from_gap` 内部用闭式公式 `Φ_total(g)` 快速评估单参数模型，避免每次迭代重复构造大系统。
7. 若 `torch` 可用，`estimate_gap_with_torch` 使用 `softplus(raw)` 保证 `g>0`，以 Adam 最小化 MSE，作为独立反演通道。
8. `main` 汇总 `truth/scipy/torch` 结果并执行阈值检查，全部通过则输出 `Validation: PASS`。

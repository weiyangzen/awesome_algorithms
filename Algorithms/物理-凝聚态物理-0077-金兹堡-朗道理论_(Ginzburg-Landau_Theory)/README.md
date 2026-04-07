# 金兹堡-朗道理论 (Ginzburg-Landau Theory)

- UID: `PHYS-0077`
- 学科: `物理`
- 分类: `凝聚态物理`
- 源序号: `77`
- 目标目录: `Algorithms/物理-凝聚态物理-0077-金兹堡-朗道理论_(Ginzburg-Landau_Theory)`

## R01

金兹堡-朗道理论（Ginzburg-Landau, GL）是超导相变附近最常用的现象论框架之一。其核心思想是把超导序参量 `psi` 当作连续场，通过自由能泛函最小化描述正常态与超导态之间的转变。

本条目交付一个可运行 MVP：
- 在 1D 网格上数值求解无矢势版本的时间依赖 GL 弛豫方程；
- 扫描多个温度 `T`，得到稳态序参量幅值；
- 与解析平衡关系 `|psi|=sqrt(epsilon/beta)` 对照；
- 给出近 `Tc` 线性标度拟合与 `PyTorch` 参数反演（`beta`）。

## R02

本实现的计算任务定义如下。

输入：
- GL 参数：`Tc, beta, xi`；
- 空间离散：`length, n_grid`；
- 时间推进参数：`dt, max_steps, tol`；
- 温度样本：`temperatures`。

输出：
- 每个温度下的稳态指标：
  - `order_param_abs_mean = <|psi|>`
  - `order_param_sq_mean = <|psi|^2>`
  - `free_energy_density`
  - `converged_steps`
- 与解析解对照误差；
- 近 `Tc` 线性拟合结果（`slope, intercept, R^2`）；
- `torch` 反演 `beta_est` 及 `torch_mse`。

## R03

使用无磁场、实序参量的简化 GL 自由能密度：

`f = -epsilon*psi^2 + 0.5*beta*psi^4 + xi^2*(d psi/dx)^2`

其中：
- `epsilon = 1 - T/Tc`；
- `epsilon > 0` 时超导态稳定；
- `epsilon <= 0` 时正常态稳定。

采用弛豫动力学（TDGL 形式）：

`d psi / dt = epsilon*psi - beta*psi^3 + xi^2*d^2 psi/dx^2`

均匀平衡解（忽略梯度项）：
- `epsilon <= 0`: `psi = 0`
- `epsilon > 0`: `|psi| = sqrt(epsilon/beta)`

## R04

离散与算法设计：

1. 空间上用二阶中心差分离散 `d^2 psi/dx^2`。  
2. 边界条件取 Neumann（`d psi/dx = 0`），用 ghost-point 消元改写首尾差分系数。  
3. 时间推进用显式 Euler：`psi_{n+1} = psi_n + dt * RHS(psi_n)`。  
4. 对每个温度独立求稳态，收敛判据为 `max(|psi_next-psi|) < tol`（并要求步数超过初期过渡段）。

这里刻意保持“低复杂度 + 可读性优先”，方便把 GL 理论直接映射到源码。

## R05

复杂度估计（设网格点数 `N`、温度样本数 `M`、单温度迭代步数 `S`）：

- 单步更新：稀疏拉普拉斯矩阵向量乘法 `O(N)`；
- 单温度求解：`O(S*N)`；
- 全温度扫描：`O(M*S*N)`；
- 空间复杂度：`O(N)`（场变量 + 稀疏三对角矩阵）+ `O(M)`（汇总表）。

默认参数（`N=256, M=8, S<=6000`）在桌面环境秒级完成。

## R06

离散更新微例（示意）：

若某步有 `epsilon=0.1, beta=1, xi=0.2`，某网格点 `psi_i=0.3`，离散拉普拉斯值为 `lap_i=-0.5`，则

`RHS_i = epsilon*psi_i - beta*psi_i^3 + xi^2*lap_i`

`= 0.1*0.3 - 1*0.027 + 0.04*(-0.5)`

`= 0.03 - 0.027 - 0.02 = -0.017`

若 `dt=0.04`，则

`psi_i(next) = 0.3 + 0.04*(-0.017) = 0.29932`

这体现了“线性增益、非线性饱和、梯度平滑”三者竞争。

## R07

物理意义：

- `epsilon` 控制距离临界温度的“驱动力”；
- `beta` 决定非线性饱和强度，控制平衡幅值；
- `xi` 控制空间平滑尺度（相干长度角色）；
- 该模型可复现二级相变附近序参量从 0 连续长出的关键现象。

MVP 不追求材料级精度，而强调“理论公式 -> 数值步骤 -> 可验证输出”的闭环。

## R08

与 `demo.py` 一一对应的理论-代码映射：

- `build_neumann_laplacian`：构建 `d^2/dx^2` 离散算子；
- `relax_one_temperature`：执行 TDGL 显式弛豫；
- `analytic_uniform_order_parameter`：给出解析平衡幅值；
- `compute_free_energy_density`：计算自由能密度用于状态诊断；
- `fit_near_tc_scaling`：检验 `|psi|^2 ~ epsilon` 标度；
- `torch_fit_beta`：从模拟数据反推 `beta`，验证参数可辨识性。

## R09

适用范围：

- 超导临界区附近的现象论演示；
- 教学、算法验证、流程模板构建；
- 从解析规律到数值实验的快速闭环。

局限性：

- 未包含电磁矢势 `A` 与磁通涡旋结构；
- 未包含真实材料微观机制（如 BCS 细节、杂质散射、各向异性）；
- 仅 1D 且实序参量，无法覆盖完整涡旋动力学和相位拓扑。

## R10

正确性验证框架：

1. 对 `epsilon > 0` 的样本，比较数值 `|psi|` 与解析 `sqrt(epsilon/beta)`。  
2. 对 `epsilon <= 0` 的样本，验证 `|psi|` 接近 0。  
3. 近 `Tc` 区间拟合 `|psi|^2 = a*epsilon + b`，检查 `R^2`。  
4. 用 `torch` 最小化 `MSE` 反演 `beta`，检查与真值偏差。  
5. 对表格关键列做有限值检查，避免数值爆炸。

脚本中的硬阈值：
- `normal_max < 0.06`
- `sc_rel_err_mean < 0.08`
- `near-Tc R^2 > 0.95`
- `|beta_est - beta_true| < 0.15`

## R11

稳定性与误差来源：

- 显式时间推进受 `dt` 约束，过大可能震荡或发散；
- 空间离散误差为二阶差分误差（`O(dx^2)`）；
- 接近 `Tc` 时弛豫时间变长，收敛步数显著上升；
- 初值若随机符号混合，可能形成畴壁导致平均值偏低。

本实现通过“正偏置初值 + 收敛阈值 + 有限值断言”控制失败风险。

## R12

在本目录执行 `uv run python demo.py` 的一次实测结果（默认参数）：

- 低温点（`T=0.70`）：
  - 数值 `|psi|=0.54765028`
  - 解析 `0.54772256`
  - 相对误差 `1.32e-4`
- 临界附近（`T=0.98`）：
  - 数值 `|psi|=0.14024076`
  - 解析 `0.14142136`
  - 相对误差 `8.35e-3`
- 正常态（`T=1.02, 1.05`）：`|psi|` 分别约 `2.47e-3` 与 `9.69e-4`。

拟合与反演摘要：
- 近 `Tc` 拟合：`slope=1.002219`, `intercept=-0.000357`, `R^2=1.000000`
- `torch` 反演：`beta_est=1.000528`, `torch_mse=3.09e-08`

## R13

结果解释：

- `epsilon>0` 区域数值与解析高度一致，说明离散与收敛策略合理；
- 越靠近 `Tc`，误差略增且收敛步数增加，符合临界减速特征；
- `|psi|^2` 与 `epsilon` 近线性关系明显，拟合斜率接近 `1/beta`；
- `torch` 仅依据模拟数据即可准确回推 `beta`，说明该 MVP 中参数识别是可行的。

## R14

常见失败模式与修复建议：

- 失败：`dt` 太大导致发散（出现 `NaN/Inf`）。  
  修复：减小 `dt` 或减小 `xi`。  
- 失败：`max_steps` 太小导致未收敛。  
  修复：增大 `max_steps` 或放宽 `tol`。  
- 失败：临界点附近误差偏大。  
  修复：加密网格（增大 `n_grid`）并延长迭代。  
- 失败：初值随机符号导致畴壁残留。  
  修复：保持小正偏置初值，或改用更长时间弛豫。

## R15

`demo.py` 模块结构：

- `GLConfig`：参数集中管理与合法性检查；
- `SimulationResult`：单温度结果数据结构；
- `build_neumann_laplacian`：稀疏二阶导离散算子；
- `relax_one_temperature`：单温度 TDGL 主循环；
- `compute_free_energy_density`：自由能后处理；
- `fit_near_tc_scaling`：`scikit-learn` 线性回归；
- `torch_fit_beta`：`PyTorch` 反演 `beta`；
- `main`：温度扫描、汇总表打印与断言验收。

## R16

可扩展方向：

- 在方程中加入矢势 `A`，进入完整 GL 超导电磁耦合；
- 扩展到复序参量与 2D/3D 网格，研究涡旋成核与动力学；
- 引入随机势或空间非均匀参数，模拟缺陷与钉扎效应；
- 用半隐式或谱方法替代显式 Euler，提升稳定性与时间步长上限。

## R17

运行方式：

```bash
cd Algorithms/物理-凝聚态物理-0077-金兹堡-朗道理论_(Ginzburg-Landau_Theory)
uv run python demo.py
```

预期输出：
- 一张温度扫描汇总表（含解析对照误差）；
- 近 `Tc` 线性拟合结果；
- `torch` 参数反演结果；
- 最后一行 `All checks passed.`。

## R18

`demo.py` 源码级算法流程拆解（8 步，非黑盒）：

1. `GLConfig.validate()` 校验 `Tc/beta/xi` 与离散参数，阻断明显非法配置。  
2. 在 `main` 中构建 1D 网格 `x`，调用 `build_neumann_laplacian` 得到稀疏三对角离散算子。  
3. 逐个温度调用 `relax_one_temperature`：按 `epsilon=1-T/Tc` 建立动力学驱动。  
4. 在每个时间步显式计算 `epsilon*psi - beta*psi^3 + xi^2*Laplace(psi)`，并执行 Euler 更新。  
5. 用 `max_delta < tol` 判定收敛，输出稳态 `|psi|`、`|psi|^2` 与 `free_energy_density`。  
6. 调用 `analytic_uniform_order_parameter` 生成解析基准，计算数值-解析误差列。  
7. 调用 `fit_near_tc_scaling`（`LinearRegression`）与 `torch_fit_beta`（Adam）完成标度检验和参数反演。  
8. 在 `main` 中打印 `pandas` 表格并执行断言门槛，全部通过后输出 `All checks passed.`。

第三方库角色透明：`scipy` 仅用于数值积分/稀疏矩阵，`scikit-learn` 仅用于线性拟合，`PyTorch` 仅用于梯度优化；GL 方程离散、迭代和验收逻辑均在源码中显式实现。

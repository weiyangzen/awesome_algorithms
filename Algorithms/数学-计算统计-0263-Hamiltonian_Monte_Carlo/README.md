# Hamiltonian Monte Carlo

- UID: `MATH-0263`
- 学科: `数学`
- 分类: `计算统计`
- 源序号: `263`
- 目标目录: `Algorithms/数学-计算统计-0263-Hamiltonian_Monte_Carlo`

## R01

本条目实现 Hamiltonian Monte Carlo（HMC）的最小可运行版本，目标是：
- 从“目标密度 `pi(q)` + 梯度 `grad U(q)`”出发，完整走通 HMC 采样流程；
- 显式展示“动量扩展 -> leapfrog 积分 -> Metropolis 校正 -> 留样诊断”的链路；
- 在二维相关高斯目标分布上验证采样质量（均值、协方差、接受率、能量误差、ESS）。

## R02

问题定义（MVP 范围）：
- 输入：
  - 目标势能函数 `U(q) = -log pi(q) + C`；
  - 势能梯度 `grad U(q)`；
  - 初始位置 `q0`；
  - 采样配置 `(step_size, leapfrog_steps, burn_in, num_samples, thin, seed)`；
  - 质量矩阵 `M`（默认单位阵，可显式传入）。
- 输出：
  - 后验样本矩阵 `samples`；
  - 接受率 `acceptance_rate`；
  - 哈密顿量误差统计（`mean|max |delta H|`）；
  - 示例目标上的统计对照（均值/协方差误差与 ESS）。

## R03

数学基础：

1. 定义势能和动能：
`U(q) = -log pi(q) + C`，`K(p) = 1/2 * p^T M^{-1} p`。

2. 增广联合分布：
`pi(q, p) ∝ exp(-H(q, p))`，其中
`H(q,p) = U(q) + K(p)`。

3. 哈密顿动力系统：
`dq/dt = partial H / partial p = M^{-1}p`，
`dp/dt = -partial H / partial q = -grad U(q)`。

4. 用 leapfrog 做辛积分近似：
- 半步更新 `p`；
- 交替整步更新 `q, p`；
- 末尾再做半步更新 `p`。

5. 用 Metropolis 接受率修正离散化误差：
`alpha = min(1, exp(H_current - H_proposed))`。

这使得离散积分轨迹仍以目标分布为不变分布。

## R04

算法流程（MVP）：
1. 检查配置与矩阵合法性（步长、步数、样本数、`M` 对称正定等）。
2. 固定随机种子，设置当前状态 `q`。
3. 每次迭代先采样动量 `p0 ~ N(0, M)`。
4. 计算当前哈密顿量 `H(q, p0)`。
5. 用 leapfrog 执行 `L` 步离散动力学，得到 `(q', p')`。
6. 反转动量 `p' <- -p'` 保证可逆性。
7. 按 `exp(H_current - H_proposed)` 做接受拒绝，决定是否更新 `q`。
8. 经过 `burn_in` 后按 `thin` 间隔留样，直到获得 `num_samples`。
9. 汇总诊断并输出。

## R05

核心数据结构：
- `HMCConfig`（`dataclass`）：
  - `step_size`、`leapfrog_steps`、`num_samples`、`burn_in`、`thin`、`seed`。
- `HMCResult`（`dataclass`）：
  - `samples: ndarray[(num_samples, dim)]`；
  - `acceptance_rate`；
  - `mean_abs_energy_error`；
  - `max_abs_energy_error`。
- 数值对象：
  - `mass_matrix`, `mass_inv`, `mass_cholesky`；
  - `potential(q)` 与 `grad_potential(q)` 可调用对象。

## R06

正确性要点：
- 动量扩展后联合分布可分解为 `pi(q)*N(p|0,M)`；
- 理想哈密顿流保持体积且守恒 `H`，因此保测度；
- leapfrog 近似具有时间可逆与辛结构，离散误差主要体现在 `delta H`；
- Metropolis 校正确保马尔可夫链以 `pi(q)` 为不变分布；
- 丢弃 burn-in、按 thin 留样可减弱初值偏置和自相关影响（但 thin 不是必须，MVP 保留用于演示）。

## R07

复杂度分析：
- 记维度为 `d`，每次 leapfrog 步中主成本是一次 `M^{-1}p` 与一次 `grad U(q)`。
- 在当前密集矩阵实现下：
  - 单次 leapfrog 约 `O(L * d^2)`；
  - 总采样迭代 `T = burn_in + num_samples * thin`；
  - 总时间复杂度约 `O(T * L * d^2)`。
- 空间复杂度：
  - 留样主存储 `O(num_samples * d)`；
  - 其余工作内存约 `O(d^2)`（质量矩阵及其逆）。

## R08

边界与异常处理：
- `step_size <= 0`、`leapfrog_steps < 1`、`num_samples < 1`、`thin < 1` -> `ValueError`；
- `initial_position` 不是一维向量 -> `ValueError`；
- `mass_matrix` 维度不匹配、非对称或非正定 -> `ValueError`；
- 势能/梯度/动能出现非有限值（`nan/inf`）-> `ValueError`。

## R09

MVP 取舍：
- 仅实现固定超参数 `(epsilon, L)` 的基础 HMC，不做自适应步长；
- 不实现 NUTS、对角/稀疏质量矩阵在线学习等进阶机制；
- 目标分布选解析可微的相关高斯，优先保证算法透明和可复现；
- 诊断只做基础指标（接受率、能量误差、ESS、矩统计误差），不引入完整 MCMC 诊断框架。

## R10

`demo.py` 职责划分：
- `make_gaussian_target`：构建二维高斯的势能与梯度函数；
- `validate_hmc_inputs`：校验配置、位置向量与质量矩阵；
- `leapfrog`：执行一次哈密顿离散积分；
- `hmc_sample`：完整采样主循环（动量采样、提案、接受拒绝、留样）；
- `estimate_ess_per_dimension`：估算各维 ESS；
- `run_demo`：组织实验并输出诊断；
- `main`：非交互入口。

## R11

运行方式：

```bash
cd Algorithms/数学-计算统计-0263-Hamiltonian_Monte_Carlo
python3 demo.py
```

脚本无交互输入，会直接输出采样统计与通过状态。

## R12

输出字段解读：
- `Acceptance rate`：提案被接受的比例，过低通常意味着步长过大或轨迹太长；
- `Mean |delta H|`、`Max |delta H|`：leapfrog 离散误差强度，越小通常越稳定；
- `Empirical mean/covariance` 与目标参数对比：直观检查采样偏差；
- `L2(mean error)`、`Frobenius(cov error)`：量化统计误差；
- `ESS per dimension`：有效样本量估计，反映自相关影响后的“等效独立样本数”。

## R13

建议最小测试集：
- 正常场景：二维相关高斯（当前 demo 默认）；
- 配置异常：`step_size<=0`、`leapfrog_steps=0`、`thin=0`；
- 质量矩阵异常：非对称、奇异、非正定；
- 数值异常：故意返回 `nan` 的势能/梯度函数；
- 稳定性回归：固定 `seed` 检查接受率和矩误差在合理区间。

## R14

关键可调参数：
- `step_size`：每次 leapfrog 步长；
- `leapfrog_steps`：单次提案轨迹长度；
- `burn_in`：前期丢弃样本数；
- `num_samples`：保留样本规模；
- `thin`：留样间隔；
- `mass_matrix`：动量协方差，影响不同维度尺度匹配。

调参经验：
- 先固定 `L` 调 `step_size`，让接受率落在中高区间；
- 再调 `L` 平衡“探索距离 vs 计算成本”；
- 若各维尺度差异大，优先改进 `mass_matrix`。

## R15

方法对比：
- 相比 Random-Walk Metropolis：
  - HMC 利用梯度和动力学轨迹，通常在中高维混合更快；
  - RWM 步子局部、随机游走效应更明显。
- 相比 MALA：
  - MALA 每步只做一次局部梯度修正；
  - HMC 通过多步 leapfrog 在一次提案里走更长距离。
- 相比 NUTS：
  - NUTS 自动调节轨迹长度，工程上更省人工调参；
  - 本实现更小更透明，适合教学和源码级理解。

## R16

典型应用：
- 贝叶斯回归与层次模型参数后验采样；
- 概率图模型中的高维连续变量推断；
- 物理/统计模型中的配分函数相关抽样问题；
- 需要“可解释采样轨迹 + 梯度信息”的科研原型验证。

## R17

后续扩展方向：
- 对角或稀疏质量矩阵自适应（预条件）；
- 步长自适应（如 dual averaging）；
- NUTS 自动轨迹长度；
- 与 PyTorch/JAX 自动微分结合，支持复杂目标分布；
- 更系统的收敛诊断（`R-hat`、多链比较、trace/ACF 可视化）。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `run_demo` 定义目标高斯的 `mean/cov`，调用 `make_gaussian_target` 生成 `potential` 和 `grad_potential`。  
2. 组装 `HMCConfig` 与 `initial_position`，进入 `hmc_sample` 主过程。  
3. `hmc_sample` 内先调用 `validate_hmc_inputs`，并分解质量矩阵得到 `mass_cholesky`、`mass_inv`。  
4. 每次迭代先采样动量 `p0 = L @ N(0, I)`，计算当前哈密顿量 `H(q, p0)`。  
5. 调 `leapfrog` 执行“半步动量 + 多步位置/动量 + 半步动量”，得到提案 `(q_prop, p_prop)`。  
6. 反转提案动量并计算 `H(q_prop, p_prop)`，由 `log(u) < -(H_prop - H_cur)` 执行 Metropolis 接受拒绝。  
7. 若接受则更新 `q`，然后按 `burn_in` 和 `thin` 规则写入 `samples`。  
8. 采样结束后汇总接受率、能量误差；`run_demo` 再计算均值/协方差误差与 `ESS`，打印并执行轻量质量门限。  

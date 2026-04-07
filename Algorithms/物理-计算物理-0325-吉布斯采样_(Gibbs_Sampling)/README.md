# 吉布斯采样 (Gibbs Sampling)

- UID: `PHYS-0321`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `325`
- 目标目录: `Algorithms/物理-计算物理-0325-吉布斯采样_(Gibbs_Sampling)`

## R01

吉布斯采样（Gibbs Sampling）是一种基于条件分布的马尔可夫链蒙特卡罗（MCMC）方法。  
核心思想：在多变量分布中，每一步固定其余变量，仅从当前变量的条件分布采样，循环更新所有坐标，最终得到目标联合分布的样本。

## R02

本条目面向计算物理中的“从复杂平衡分布采样”问题，典型场景包括：
- 统计物理中的玻尔兹曼分布采样
- 晶格模型（如 Ising）中的自旋翻转更新
- 耦合自由度系统（多粒子、多模态）中的边缘量估计
- 难以直接采样但条件分布可写出的高维模型

## R03

MVP 采用二维相关高斯作为可验证目标分布：

\[
\mathbf{z}=(x,y)^\top \sim \mathcal{N}\left(
\mathbf{0},
\Sigma=\begin{bmatrix}1 & \rho\\ \rho & 1\end{bmatrix}
\right),\quad |\rho|<1
\]

该模型可解释为“两个线性耦合的连续自由度”，其负对数密度对应二次型能量。

## R04

对二维相关高斯，条件分布具有闭式：

\[
x \mid y \sim \mathcal{N}\left(\rho y,\;1-\rho^2\right),\qquad
y \mid x \sim \mathcal{N}\left(\rho x,\;1-\rho^2\right)
\]

因此每一轮 Gibbs 更新可写为：
1. 按 \(x \mid y\) 采样得到新 \(x\)
2. 按 \(y \mid x\) 采样得到新 \(y\)

## R05

伪代码（单链）：

```text
given rho, burn_in, thin, n_samples
initialize x, y
for t in 1 .. burn_in + n_samples * thin:
    x ~ Normal(rho * y, sqrt(1-rho^2))
    y ~ Normal(rho * x, sqrt(1-rho^2))
    if t > burn_in and keep_by_thin(t):
        save (x, y)
return saved samples
```

## R06

正确性直觉：
- 每个坐标更新都精确服从对应条件分布
- 由条件更新组成的马尔可夫核以目标联合分布为不变分布
- 在常见可约性/非周期条件下，链收敛到目标分布

实践中通常通过 burn-in、自相关、协方差误差等指标判断是否“足够接近平衡”。

## R07

复杂度（二维情形）：
- 时间复杂度：\(O(T)\)，其中 \(T = burn\_in + n\_samples \times thin\)
- 空间复杂度：\(O(n\_samples)\)（仅保存抽样结果）

推广到 \(d\) 维时，单 sweep 通常为 \(O(d)\) 到 \(O(d^2)\)，取决于条件分布计算成本。

## R08

实现要点：
- 约束 \(|\rho|<1\)，避免协方差矩阵退化
- 使用 `np.random.default_rng(seed)` 保证可复现
- 高相关（\(\rho\) 接近 1）会显著增大自相关，需更长链或更好参数化
- 通过 thin 减少样本存储相关性，但不会提升链本身混合速度

## R09

本 MVP 使用以下诊断量：
- `mean_l2_error`：样本均值与理论均值（0 向量）的 L2 偏差
- `cov_fro_error`：样本协方差与理论协方差的 Frobenius 误差
- `lag1_acf_x / lag1_acf_y`：一阶自相关
- `approx_ess_x / approx_ess_y`：基于 AR(1) 近似的有效样本量
- `avg_energy`：平均二次型能量，应接近二维系统的理论值 1

## R10

与常见 MCMC 方法对比：
- 相比 Metropolis-Hastings：Gibbs 无需接受-拒绝步骤（当条件分布可直接采样时）
- 相比 HMC：实现更简单，但在强相关方向可能混合较慢
- 相比独立采样：适合“联合分布难、条件分布易”的问题结构

## R11

计算物理落地示例：
- 晶格统计模型：逐点（或分块）条件更新自旋
- 贝叶斯反演中的物理参数估计：交替更新参数和隐变量
- 多体耦合近似模型：按子系统条件分布轮流采样

本目录的 `demo.py` 用“耦合高斯模态”展示了最小可运行链路。

## R12

`demo.py` 的 MVP 设计目标：
- 尽量少依赖，仅使用 `numpy + pandas`
- 显式写出条件分布采样，不调用黑盒 MCMC 框架
- 提供可验证的理论协方差对照与基础收敛诊断
- 非交互运行，适用于 `uv run python demo.py`

## R13

运行方式：

```bash
uv run python Algorithms/物理-计算物理-0325-吉布斯采样_(Gibbs_Sampling)/demo.py
```

默认参数（脚本内）：
- `seed = 20260407`
- `rho = 0.92`
- `n_samples = 12000`
- `burn_in = 2500`
- `thin = 2`

## R14

预期输出内容：
- 样本均值向量（应接近 `[0, 0]`）
- 样本协方差矩阵（应接近 `[[1, rho], [rho, 1]]`）
- 诊断表（误差、自相关、ESS、平均能量）

只要链长足够，`cov_fro_error` 通常会较小，`avg_energy` 通常接近 1。

## R15

常见失败模式：
- `rho` 取值超界（`|rho|>=1`）：协方差非法，脚本会抛错
- 样本数过少：统计误差大，协方差偏差明显
- 相关性过强但 burn-in 太短：链未充分混合
- 仅靠 thin 代替更长链：常导致 ESS 仍不足

## R16

可扩展方向：
- 从二维扩展到高维高斯（可配合精度矩阵分块更新）
- 引入块 Gibbs（block Gibbs）提升强耦合变量混合效率
- 在 Ising/Potts 等离散模型上实现逐点条件翻转
- 增加 Gelman-Rubin \(\hat{R}\)（多链）与更系统的 ESS 估计

## R17

参考资料（建议）：
- Casella, G. and George, E. I. (1992). *Explaining the Gibbs Sampler*. The American Statistician.
- Robert, C. P. and Casella, G. (2004). *Monte Carlo Statistical Methods*.
- Liu, J. S. (2008). *Monte Carlo Strategies in Scientific Computing*.
- 统计物理与 MCMC 相关课程讲义（关于玻尔兹曼分布和马尔可夫链平衡性质）。

## R18

`demo.py` 的源码级算法流程（非黑盒）：
1. 读取超参数：`rho, n_samples, burn_in, thin, seed`，并创建随机数生成器。  
2. 在 `gibbs_bivariate_gaussian` 中校验参数合法性（`|rho|<1` 等），初始化 `(x, y)=(0,0)`。  
3. 预计算条件标准差 `sqrt(1-rho^2)`，进入总迭代循环。  
4. 每轮先按 `x|y ~ N(rho*y, 1-rho^2)` 更新 `x`，再按 `y|x ~ N(rho*x, 1-rho^2)` 更新 `y`。  
5. 跳过 burn-in，并按 thin 规则保留样本到数组。  
6. 采样结束后，用 `pandas` 计算样本均值和协方差，并与理论协方差矩阵对比。  
7. 计算诊断量（lag-1 自相关、近似 ESS、平均能量）并打印结果，完成一次最小可验证 Gibbs 采样实验。  

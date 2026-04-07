# 梅特罗波利斯算法 (Metropolis Algorithm)

- UID: `PHYS-0314`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `317`
- 目标目录: `Algorithms/物理-计算物理-0317-梅特罗波利斯算法_(Metropolis_Algorithm)`

## R01

梅特罗波利斯算法是马尔可夫链蒙特卡洛（MCMC）的基础方法，用于从难以直接采样的目标分布
\(\pi(x)\) 中生成样本。它只需要目标分布的“相对密度”或未归一化形式 \(\tilde{\pi}(x)\)，不要求已知归一化常数。

## R02

典型问题场景：
- 目标分布归一化常数难算，但可计算 \(\log \tilde{\pi}(x)\)。
- 需要估计积分、期望或后验统计量：\(\mathbb{E}_{\pi}[f(X)]\)。
- 计算物理中常见于玻尔兹曼分布采样、自由能估计、统计平均值计算等。

## R03

核心思想：
1. 构造一个“提议分布” \(q(x'|x)\) 给出候选状态。
2. 依据接受概率决定是否接受候选状态。
3. 通过该接受机制保证平稳分布为目标分布 \(\pi\)。
4. 长时间运行后，链上的样本可近似视为来自 \(\pi\)。

## R04

当提议分布对称（随机游走常见：\(q(x'|x)=q(x|x')\)）时，接受率为：
\[
\alpha(x\to x') = \min\left(1,\frac{\tilde{\pi}(x')}{\tilde{\pi}(x)}\right).
\]
数值实现常用对数形式避免下溢：
\[
\log \alpha = \log \tilde{\pi}(x') - \log \tilde{\pi}(x).
\]
若 \(\log u < \log \alpha\)（\(u\sim U(0,1)\)），则接受，否则拒绝并停留在原状态。

## R05

伪代码（随机游走版本）：
```text
输入: log_target, x0, burn_in, n_samples, thin, proposal_std
x <- x0
for t in 1 .. burn_in + n_samples * thin:
    x_prop <- x + Normal(0, proposal_std^2)
    log_alpha <- log_target(x_prop) - log_target(x)
    if log(U(0,1)) < log_alpha:
        x <- x_prop
    if t > burn_in and (t - burn_in) mod thin == 0:
        保存 x
输出: 样本序列
```

## R06

正确性直觉（细致平衡）：
- 对称提议下，转移核满足
  \(\pi(x)P(x\to x') = \pi(x')P(x'\to x)\)。
- 细致平衡推出 \(\pi\) 是马尔可夫链平稳分布。
- 因而在不可约、非周期等常规条件下，链会收敛到目标分布。

## R07

收敛与混合：
- 理论上“足够长”后可逼近目标分布。
- 实践中需要 burn-in 去除初值影响。
- 样本自相关会降低有效样本量（ESS），可用 thinning 或更好提议缓解。
- 提议步长太小会“走得慢”，太大会“拒绝多”，需要折中。

## R08

复杂度：
- 单步时间复杂度：\(O(1)\)（假设一次 log-target 计算为常数时间）。
- 总时间复杂度：\(O(T)\)，\(T=\text{burn\_in}+n\_\text{samples}\times \text{thin}\)。
- 空间复杂度：\(O(n\_\text{samples})\)（仅存储保留样本）。

## R09

关键超参数：
- `proposal_std`：提议尺度，决定接受率与探索速度。
- `burn_in`：热身步数，去初值偏差。
- `thin`：抽稀间隔，减弱相邻样本相关性（会增加总计算量）。
- `n_samples`：最终保留样本数。
- `seed`：随机种子，保证可复现。

## R10

数值与工程注意：
- 统一使用 `log_target` 与 `log_alpha`，避免概率比直接相除导致下溢/上溢。
- 先缓存当前状态的 `log_target(x)`，避免重复计算。
- 参数合法性检查（样本数、步长、thin、burn-in）应在运行前完成。
- 结果诊断至少输出接受率与样本均值/方差，防止“看似运行成功但链质量差”。

## R11

本目录 MVP 设计：
- 目标分布：标准正态 \(N(0,1)\)（仅使用未归一化对数密度 \(-x^2/2\)）。
- 算法：一维随机游走 Metropolis。
- 输出：接受率、样本均值、样本方差、分位数、lag-1 自相关、AR(1) 近似 ESS。
- 依赖：仅 `numpy`，保证最小可运行实现。

## R12

运行方式（仓库根目录）：
```bash
uv run python "Algorithms/物理-计算物理-0317-梅特罗波利斯算法_(Metropolis_Algorithm)/demo.py"
```
或在当前目录：
```bash
cd "Algorithms/物理-计算物理-0317-梅特罗波利斯算法_(Metropolis_Algorithm)"
uv run python demo.py
```

## R13

预期输出特征（不同机器数值会有小波动）：
- 接受率通常在 0.2 到 0.7 之间（取决于 `proposal_std`）。
- 样本均值接近 0。
- 样本方差接近 1。
- 分位数与标准正态趋势一致（如中位数接近 0）。

## R14

边界条件与失败模式：
- `proposal_std <= 0`：提议无意义，应报错。
- `n_samples <= 0` 或 `thin <= 0`：无法形成有效输出，应报错。
- 常数目标分布会导致随机游走无约束扩散，不适合作为此演示目标。
- 若接受率极低（接近 0）或极高（接近 1），通常需要重新调节步长。

## R15

常见实现错误：
- 直接用概率比而非 log 比，导致浮点数溢出/下溢。
- 拒绝提议时错误地“跳过记录”，而不是记录原状态（破坏马氏链定义）。
- 忽略 burn-in 导致统计估计带明显初值偏差。
- 没有固定种子，导致验收结果不可复现。

## R16

可扩展方向：
- 升级为 Metropolis-Hastings（非对称提议）。
- 使用自适应步长或分块更新提升混合效率。
- 多维参数采样（向量状态）并接入真实物理能量函数。
- 与 Gibbs/HMC 对比在高维问题中的效率。

## R17

与相关算法对比：
- vs 直接采样：Metropolis 不需归一化常数，适合复杂分布。
- vs 拒绝采样：高维时拒绝采样常效率极差，Metropolis 更可行。
- vs Gibbs：Gibbs 依赖可条件采样结构；Metropolis 更通用但可能相关性更强。
- vs HMC：HMC 在连续高维常更高效，但实现复杂；Metropolis 更轻量、易入门。

## R18

`demo.py` 的源码级算法流程（非黑盒）：
1. 定义 `log_target_standard_normal(x) = -0.5*x*x`，提供未归一化对数密度。
2. 在 `metropolis_random_walk_1d` 中初始化当前状态 `x` 与 `log_p_x`，并准备样本数组。
3. 每步先用 `Normal(0, proposal_std)` 生成提议增量，得到候选 `x_proposal`。
4. 计算 `log_alpha = log_p_proposal - log_p_x`，作为对数接受比。
5. 抽取 `u~U(0,1)`，若 `log(u) < log_alpha` 则接受并更新当前状态，否则保持原状态。
6. 在 burn-in 之后按 `thin` 间隔保存当前状态，累计形成最终样本序列。
7. 统计接受率、样本均值方差、分位数、lag-1 自相关和 AR(1) 近似 ESS 作为最小诊断输出。

# 变分推断 (VI)

- UID: `MATH-0265`
- 学科: `数学`
- 分类: `贝叶斯统计`
- 源序号: `265`
- 目标目录: `Algorithms/数学-贝叶斯统计-0265-变分推断_(VI)`

## R01

变分推断（Variational Inference, VI）是把“后验分布求积分”转化为“优化问题”的方法：
- 选一个可计算的分布族 `q(z)` 近似真实后验 `p(z|x)`；
- 通过最大化 ELBO（等价于最小化 `KL(q(z)||p(z|x))`）来找最优近似。

本条目实现一个最小可复现实例：
- 一维高斯观测模型（均值与精度未知）
- 均值场分解 `q(mu)q(tau)`
- 用坐标上升变分推断（CAVI）迭代更新。

## R02

MVP 求解问题：

已知观测 `x_1, ..., x_n`，模型设定为
- `x_n | mu, tau ~ Normal(mu, tau^{-1})`
- `mu | tau ~ Normal(mu0, (lambda0 * tau)^{-1})`
- `tau ~ Gamma(a0, b0)`（shape-rate 参数化）

目标：
- 用 VI 近似 `p(mu, tau | x)`；
- 输出 `q(mu)` 与 `q(tau)` 的参数、ELBO 收敛轨迹；
- 与共轭精确后验做数值对照。

## R03

选择这个模型用于 VI 入门 MVP 的原因：
- 有闭式 CAVI 更新，核心逻辑清晰，避免黑盒优化器；
- 后验可精确求解，便于验证 VI 结果；
- 能完整展示 VI 的三个关键对象：分解假设、ELBO、迭代更新。

## R04

均值场假设与变分分布：
- `q(mu, tau) = q(mu) q(tau)`
- `q(mu) = Normal(m_n, var_mu)`
- `q(tau) = Gamma(a_n, b_n)`

其中：
- `E_q[tau] = a_n / b_n`
- `E_q[log tau] = digamma(a_n) - log(b_n)`
- `var_mu = 1 / ((lambda0 + n) * E_q[tau])`

`demo.py` 中按上述参数化显式实现，不调用任何现成 VI 黑盒接口。

## R05

ELBO 定义：

`ELBO(q) = E_q[log p(x, mu, tau)] - E_q[log q(mu)] - E_q[log q(tau)]`

本实现将其拆成 5 项计算：
1. `E_q[log p(x|mu,tau)]`
2. `E_q[log p(mu|tau)]`
3. `E_q[log p(tau)]`
4. `E_q[log q(mu)]`
5. `E_q[log q(tau)]`

每次迭代都记录 ELBO，用于检查“近似单调不下降”。

## R06

CAVI 更新式（与 `demo.py` 一致）：

记 `x_bar = mean(x)`，`n = len(x)`。

固定项：
- `lambda_n = lambda0 + n`
- `m_n = (lambda0 * mu0 + n * x_bar) / lambda_n`
- `a_n = a0 + (n + 1) / 2`

迭代更新：
1. `var_mu = 1 / (lambda_n * E_q[tau])`
2. `data_term = sum((x_i - m_n)^2) + n * var_mu`
3. `prior_term = lambda0 * ((m_n - mu0)^2 + var_mu)`
4. `b_n = b0 + 0.5 * (data_term + prior_term)`
5. `E_q[tau] = a_n / b_n`

当 `|E_q[tau]^{(t+1)} - E_q[tau]^{(t)}| < tol` 时停止。

## R07

算法流程：

1. 校验输入数据与超参数合法性。
2. 初始化 `E_q[tau] = a0 / b0`。
3. 根据当前 `E_q[tau]` 计算 `var_mu`。
4. 更新 `b_n`，再更新 `E_q[tau]`。
5. 计算当前 ELBO 并写入迭代轨迹。
6. 检查收敛条件，否则继续迭代。
7. 返回 VI 参数、收敛状态、轨迹表。
8. 额外计算精确共轭后验，输出近似误差对照。

## R08

正确性与收敛说明：
- CAVI 在每一步对一个变分因子做最优更新，理论上 ELBO 不下降；
- 本实现提供 ELBO 非下降检查（允许浮点级容差）；
- 该模型共轭结构简单，迭代通常在很少轮次收敛；
- 通过与精确后验矩（`E[mu|x]`、`E[tau|x]`）对照验证结果可信性。

## R09

复杂度（设样本数 `n`、迭代轮次 `T`）：
- 每轮主成本为 `sum((x_i - m_n)^2)`，复杂度 `O(n)`；
- 总时间复杂度 `O(T*n)`；
- 额外空间复杂度 `O(T)`（保存轨迹表）或 `O(1)`（若不保存轨迹）。

## R10

`demo.py` 主要函数职责：
- `validate_inputs`：输入与超参数校验。
- `compute_elbo`：按公式计算当前 ELBO。
- `cavi_normal_gamma`：核心 CAVI 循环。
- `exact_posterior_params`：计算共轭精确后验参数。
- `is_almost_monotone_non_decreasing`：ELBO 近似单调检查。
- `make_synthetic_data`：生成可复现实验数据。
- `main`：串联训练、对照、结果打印。

## R11

运行方式：

```bash
cd Algorithms/数学-贝叶斯统计-0265-变分推断_(VI)
uv run python demo.py
```

脚本无交互输入，直接输出结果。

## R12

输出字段说明：
- `n_samples`：样本数。
- `sample_mean` / `sample_variance`：观测统计量。
- `iterations`：CAVI 迭代轮次。
- `converged`：是否在最大迭代内收敛。
- `ELBO monotone nondecrease`：ELBO 是否近似非下降。
- `E_q[mu]`, `Var_q(mu)`, `E_q[tau]`：VI 后验矩。
- `E[mu|x]`, `E[tau|x]`：精确后验矩（对照基线）。
- `|E_q[...] - E[...]|`：VI 与精确解的误差。

## R13

最小测试覆盖（由 `main` 自动完成）：
- 固定随机种子生成合成数据，保证可复现；
- 执行 CAVI 并验证可收敛；
- 检查 ELBO 近似单调性；
- 与精确后验做数值对照，确认误差在浮点范围内。

## R14

关键参数建议：
- `lambda0`：先验均值强度，越大越偏向 `mu0`；
- `a0, b0`：精度 `tau` 的 Gamma 先验，决定先验均值 `a0/b0`；
- `tol`：收敛阈值，越小迭代越严格；
- `max_iter`：最大迭代上限。

经验设置：
- 若无强先验可用 `lambda0=1, a0=2, b0=2` 作为温和起点；
- 先用较宽松 `tol` 快速验证，再按需求收紧。

## R15

MVP 局限：
- 仅覆盖一维高斯-伽马共轭模型；
- 采用均值场分解，忽略 `mu` 与 `tau` 的后验相关性；
- 不包含黑盒 VI（随机梯度、重参数化）与大规模并行能力。

## R16

典型应用与迁移方向：
- 作为理解 VI 机制的教学样例；
- 扩展到多维高斯或线性回归的贝叶斯参数估计；
- 作为后续黑盒 VI、随机 VI、变分自编码器（VAE）等方法的前置基础。

## R17

可扩展改进：
- 从一维扩展到多维参数并引入矩阵形式更新；
- 使用随机小批量形成 SVI（Stochastic VI）；
- 改为黑盒 ELBO 梯度优化，支持非共轭模型；
- 增加先验敏感性分析与可视化（ELBO 曲线、后验区间）。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 通过 `make_synthetic_data` 生成一维高斯样本，并设定先验超参数。
2. 调用 `cavi_normal_gamma` 进入 VI 主循环，先用 `validate_inputs` 做维度与数值检查。
3. 在 `cavi_normal_gamma` 中预计算 `lambda_n, m_n, a_n`，初始化 `E_q[tau]`。
4. 每轮先由当前 `E_q[tau]` 得到 `var_mu`，再用数据项与先验项更新 `b_n`。
5. 用 `E_q[tau] = a_n / b_n` 完成 `q(tau)` 更新，并把 `iter/e_tau/var_mu/b_n/elbo` 记录进轨迹。
6. `compute_elbo` 按 `E_q[log p] - E_q[log q]` 的 5 项分解显式计算 ELBO。
7. 当相邻两轮 `E_q[tau]` 变化小于 `tol` 时停止，返回 `VIResult`（含收敛状态和轨迹表）。
8. `main` 再调用 `exact_posterior_params` 计算共轭真值，对比 `E_q[mu]` 与 `E_q[tau]` 并打印最后 5 轮迭代结果。

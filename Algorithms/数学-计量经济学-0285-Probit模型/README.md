# Probit模型

- UID: `MATH-0285`
- 学科: `数学`
- 分类: `计量经济学`
- 源序号: `285`
- 目标目录: `Algorithms/数学-计量经济学-0285-Probit模型`

## R01

Probit 模型是经典二元离散选择模型，用于刻画 \(y\in\{0,1\}\) 的概率：

\[
\Pr(y_i=1\mid x_i)=\Phi(x_i^\top\beta)
\]

其中 \(\Phi(\cdot)\) 是标准正态分布的 CDF。  
它可由潜变量视角解释：\(y_i^\*=x_i^\top\beta+\varepsilon_i,\ \varepsilon_i\sim\mathcal N(0,1)\)，且 \(y_i=\mathbf 1(y_i^\*>0)\)。

## R02

适用场景：
- 因变量为“是/否”“违约/不违约”“购买/不购买”等二元结果。
- 需要概率输出，且希望链接函数具有正态尾部性质（相较 Logit 尾部更轻）。
- 计量经济学中的政策效果、劳动力参与、信贷审批、医疗二元结局建模。

## R03

参数估计采用极大似然（MLE）：

\[
\ell(\beta)=\sum_{i=1}^{n}\left[y_i\log\Phi(z_i)+(1-y_i)\log(1-\Phi(z_i))\right],\ z_i=x_i^\top\beta
\]

在实现中最大化 \(\ell(\beta)\) 等价于最小化负对数似然 \(-\ell(\beta)\)。  
本目录 `demo.py` 显式实现了目标函数与梯度，再交给 `scipy.optimize.minimize(..., method="BFGS")` 迭代求解。

## R04

输入输出约定（对应 `demo.py`）：
- 输入：
  - 合成特征矩阵 `X`（自动加入截距列）。
  - 二元标签 `y`（0/1）。
  - 优化器初值（默认全零）与收敛参数（`gtol`, `maxiter`）。
- 输出：
  - `beta_hat`：估计系数。
  - `std_err`：由 BFGS 近似 Hessian 逆矩阵对角线给出的标准误近似。
  - 收敛状态、迭代次数、训练集对数似然。
  - 训练/测试集指标：Accuracy、AUC、Brier score、正类占比。

## R05

伪代码：

```text
given X_raw, y
X <- add_intercept(X_raw)
initialize beta <- 0

define nll(beta):
    p <- clip(Phi(X beta))
    return -sum(y log p + (1-y) log(1-p))

define grad(beta):
    z <- X beta
    p <- clip(Phi(z))
    phi <- normal_pdf(z)
    score_z <- phi * (y - p) / (p * (1-p))
    return -(X^T score_z)

beta_hat <- BFGS_minimize(nll, grad, beta)
prob <- clip(Phi(X beta_hat))
pred <- 1(prob >= 0.5)
report coefficients + metrics
```

## R06

正确性要点：
- Probit 链接保证预测概率在 \((0,1)\) 内。
- MLE 来自伯努利似然，目标函数与统计模型一致。
- 梯度由链式法则显式给出：

\[
\frac{\partial \ell}{\partial \beta}
=
X^\top\left[\phi(z)\odot\frac{y-\Phi(z)}{\Phi(z)(1-\Phi(z))}\right]
\]

实现中对 `p` 做 `clip`，避免 \(\log(0)\) 与除零，保证数值稳定。

## R07

复杂度（单次目标/梯度评估）：
- 计算 `z = Xβ`：`O(np)`
- 计算 CDF/PDF、逐样本 score：`O(n)`
- 梯度 `X^T score`：`O(np)`

若 BFGS 迭代 `T` 次，总体近似 `O(Tnp)`（忽略低阶项）。  
内存主开销为数据矩阵 `X`，约 `O(np)`。

## R08

与相邻模型对比：
- 线性概率模型（LPM）：训练简单，但概率可能超出 \([0,1]\)，且异方差明显。
- Logit：链接函数为 logistic CDF；与 Probit 通常方向一致，但系数尺度不同。
- Probit：正态链接，潜变量解释自然，经济学文献使用广泛。

## R09

本实现关键数据结构：
- `numpy.ndarray`：存储特征、标签、参数向量。
- `ProbitResult`（`dataclass`）：封装估计结果（系数、标准误、收敛信息）。
- `pandas.DataFrame`：系数表（真值/估计值/标准误）展示。
- `dict[str, float]`：分类与校准指标集合。

## R10

边界与异常处理：
- `X` 必须是二维矩阵，`y` 必须是一维且与样本数一致。
- `y` 仅允许 `0/1`，否则抛出 `ValueError`。
- `Phi(z)` 被裁剪到 `[1e-9, 1-1e-9]` 防止数值溢出。
- 若优化未收敛或关键指标过低，脚本抛 `RuntimeError`，便于自动验证流程判失败。

## R11

MVP 实现策略：
- 采用最小工具栈：`numpy + scipy + pandas + scikit-learn`。
- 不调用封装好的“整模型黑箱”（如现成 Probit 拟合器）；而是自行写 `nll + grad`。
- 只用 `scipy.optimize.minimize` 负责参数迭代，这样代码短且数学结构清晰。

## R12

运行方式（非交互）：

```bash
uv run python Algorithms/数学-计量经济学-0285-Probit模型/demo.py
```

或在仓库根目录直接：

```bash
uv run python demo.py
```

（当前目录切到 `Algorithms/数学-计量经济学-0285-Probit模型` 时）

## R13

预期输出特征：
- 显示是否收敛、迭代次数、最优对数似然。
- 打印系数表：`beta_true` 与 `beta_hat` 接近。
- 训练/测试集 AUC 较高（本合成数据一般 > 0.90），Accuracy 通常 > 0.80。
- 末尾出现 `Validation checks passed.`。

## R14

常见实现错误：
- 忘记加截距，导致系统性偏差。
- 把 CDF 与 PDF 混用，梯度公式写错。
- 未做概率裁剪，出现 `log(0)` 或分母为 0。
- 用分类阈值指标替代似然优化目标，导致“训练目标”和“估计理论”不一致。

## R15

最小测试清单：
- 功能测试：`uv run python demo.py` 可直接运行并收敛。
- 数值测试：优化返回 `success=True`，迭代次数有限。
- 质量测试：AUC 与 Accuracy 达到脚本内置阈值。
- 稳健性测试：更换随机种子后结果略波动但仍有合理预测性能。

## R16

可扩展方向：
- 加入稳健标准误（sandwich / Huber-White）。
- 扩展到多项 Probit（Multinomial Probit）或样本选择模型。
- 增加 L1/L2 正则化，处理高维特征。
- 与 Logit 并行拟合并做边际效应比较。

## R17

局限与取舍：
- Probit 系数解释不如线性模型直观，通常需转换为边际效应。
- 本 MVP 用 BFGS 逆 Hessian 近似标准误，未做更严格有限样本修正。
- 合成数据验证的是可运行性与方法链路，不代表真实业务数据复杂度（缺失、偏差、内生性等）。

## R18

源码级算法流程（`demo.py`，非黑箱拆解）：
1. `simulate_probit_data` 生成潜变量数据：先采样特征，再按 \(y^\*=X\beta+\epsilon\)（\(\epsilon\sim N(0,1)\)）阈值化得到 `y`。  
2. `add_intercept` 在特征前拼接常数列，形成估计所需设计矩阵。  
3. `neg_loglik_and_grad` 计算 `z=Xβ`，再用 `norm.cdf(z)` 得到 `p`、`norm.pdf(z)` 得到 `phi`，组成负对数似然。  
4. 同一函数继续计算解析梯度：`score_z = phi*(y-p)/(p*(1-p))`，再做 `X.T @ score_z` 得到 score 并取负号。  
5. `fit_probit_mle` 调用 `scipy.optimize.minimize(method="BFGS", jac=gradient)`；BFGS 在每轮用当前梯度构造近似逆 Hessian，并给出新参数。  
6. 优化结束后读取 `result.x` 作为 `beta_hat`，并从 `result.hess_inv` 提取近似标准误。  
7. `probit_prob` 用最终参数生成训练/测试概率；`evaluate_binary_metrics` 计算 Accuracy/AUC/Brier 等指标。  
8. `main` 汇总系数表与指标，执行阈值自检；若未收敛或性能异常则抛错，否则打印 `Validation checks passed.`。  

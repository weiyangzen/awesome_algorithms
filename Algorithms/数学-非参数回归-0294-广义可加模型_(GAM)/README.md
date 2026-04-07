# 广义可加模型 (GAM)

- UID: `MATH-0294`
- 学科: `数学`
- 分类: `非参数回归`
- 源序号: `294`
- 目标目录: `Algorithms/数学-非参数回归-0294-广义可加模型_(GAM)`

## R01

广义可加模型（Generalized Additive Model, GAM）是一类“可解释的非线性回归”方法。  
它保留了广义线性模型（GLM）的可解释结构，但把线性项 `beta_j x_j` 放宽为单变量平滑函数 `f_j(x_j)`：

\[
g(\mathbb{E}[Y|X])=\beta_0+\sum_{j=1}^{p}f_j(x_j)
\]

其中 `g` 是链接函数。  
本目录实现的是最常用的回归版本：高斯分布 + 恒等链接（identity link）。

## R02

本条目 MVP 的任务定义：

- 输入：
  - 训练特征 `X in R^(n x p)`（本示例 `p=3`）；
  - 训练标签 `y in R^n`。
- 输出：
  - 拟合后的加性函数模型 `y_hat = beta0 + sum_j f_j(x_j)`；
  - 在测试集上的 `MSE / MAE / R2`；
  - 每个分量函数恢复质量（与真分量的相关系数）。

为保证可审计，不依赖 `pyGAM` 等一键黑盒，而是显式实现：
- 每维 `B-spline` 基函数展开；
- 二阶差分平滑惩罚；
- 坐标式 backfitting 迭代。

## R03

本实现的数学模型（高斯族）：

\[
y_i = \beta_0 + \sum_{j=1}^{p} f_j(x_{ij}) + \epsilon_i,\quad \epsilon_i\sim \mathcal{N}(0,\sigma^2)
\]

每个 `f_j` 用样条基表示：

\[
f_j(x)=B_j(x)\theta_j
\]

其中 `B_j(x)` 是第 `j` 个特征的一维 B-spline 基向量，`theta_j` 为待估参数。

## R04

为防止过拟合，使用二阶差分粗糙度惩罚。  
第 `j` 个分量的目标子问题可写为：

\[
\min_{\theta_j}\ \|r_j-B_j\theta_j\|_2^2+\alpha\|D\theta_j\|_2^2
\]

- `r_j`：去掉其它分量后的部分残差；
- `D`：二阶差分矩阵（离散二阶导近似）；
- `alpha`：平滑强度，越大越平滑。

对应线性方程：

\[
(B_j^TB_j+\alpha D^TD)\theta_j=B_j^Tr_j
\]

此外使用“分量零均值约束”保证可辨识性：  
每次更新后令分量在训练集上的均值为 0，把均值并入截距 `beta_0`。

## R05

backfitting 主流程（概念级）：

1. 初始化 `beta0 = mean(y)`，各分量 `f_j = 0`。  
2. 迭代遍历每个特征 `j`。  
3. 计算部分残差 `r_j = y - (beta0 + sum_{k!=j} f_k)`。  
4. 解惩罚最小二乘得到 `theta_j`。  
5. 计算新分量 `f_j = B_j theta_j`，再做零均值中心化并修正截距。  
6. 记录本轮最大分量变化量，若小于阈值 `tol` 则收敛。  
7. 否则继续下一轮，直到 `max_iter`。

## R06

本实现的基函数与惩罚细节：

- 基函数：`sklearn.preprocessing.SplineTransformer`，`degree=3`（三次样条）、`n_knots=9`。  
- `include_bias=False`，避免与截距项强共线。  
- 惩罚矩阵：`P = D^T D`，其中 `D = diff(I, n=2, axis=0)`。  
- 数值稳定：求解时加小抖动项 `1e-8 * I`，减少病态矩阵风险。

## R07

复杂度分析（`n` 样本，`p` 特征，每维样条基数约 `m`，迭代轮数 `T`）：

- 每个分量一次更新：
  - 形成法方程约 `O(n m^2)`；
  - 求解 `m x m` 线性方程约 `O(m^3)`。
- 每轮遍历 `p` 个分量：
  - `O(p (n m^2 + m^3))`。
- 总复杂度：
  - `O(T p (n m^2 + m^3))`。
- 空间：
  - 设计矩阵存储约 `O(p n m)`；
  - 系数与中间向量约 `O(p m + n p)`。

## R08

边界处理与鲁棒性约束：

- 参数约束：
  - `n_knots >= 4`；
  - `degree >= 1`；
  - `alpha >= 0`；
  - `max_iter > 0`；
  - `tol > 0`。
- 数据校验：
  - `X` 必须二维、`y` 必须一维；
  - 样本数一致且不少于 4；
  - 输入必须是有限实数。
- 预测阶段：
  - 未 `fit` 先 `predict` 会抛错；
  - 特征维度不匹配会抛错。

## R09

`demo.py` 的核心对象与函数：

- `AdditiveSplineGAM`：
  - `fit`：执行 backfitting；
  - `predict`：输出总体预测；
  - `component_contributions`：返回各维分量贡献。
- `_second_order_difference_penalty`：构建 `D^T D` 平滑惩罚矩阵。  
- `make_synthetic_additive_data`：生成带真分量的可控数据集。  
- `search_alpha`：在验证集上搜索 `alpha`。  
- `_safe_corr`：稳健计算分量相关系数。

## R10

实验流程（主程序）：

1. 生成 3 特征加性非线性数据：`sin`、二次项、指数衰减项。  
2. 切分 `train/val/test`。  
3. 在 `alpha_grid=[1e-4, ..., 10]` 上做验证集搜索。  
4. 用最优 `alpha` 在 `train_full` 重训 GAM。  
5. 用线性回归做基线比较。  
6. 输出测试指标与分量恢复相关性。

## R11

运行方式：

```bash
cd Algorithms/数学-非参数回归-0294-广义可加模型_(GAM)
uv run python demo.py
```

也可使用：

```bash
python3 demo.py
```

无需交互输入。

## R12

输出字段说明：

- `Alpha search results`：每个 `alpha` 的验证集 `MSE/MAE` 与迭代轮数。  
- `Selected alpha`：验证集最优平滑强度。  
- `Backfitting iterations`：最终训练使用的迭代次数。  
- `Test metrics`：
  - `GAM` 与 `Linear` 的 `MSE / MAE / R2` 对比。  
- `Recovered component correlation`：
  - `corr(fj_hat, fj_true)` 越接近 1，表示分量恢复越准确。  
- `Sample predictions`：若干测试样本的真实值与预测值。

## R13

最小验证建议（脚本已覆盖）：

- 功能性：
  - 能完成 `alpha` 搜索；
  - 能收敛并输出测试指标；
  - 能输出分量相关系数。
- 数值合理性：
  - GAM 在该合成加性数据上通常优于线性基线（尤其 `R2`）。  
- 异常路径：
  - 非法参数（如 `tol<=0`）应触发 `ValueError`；
  - 未拟合直接预测应触发 `RuntimeError`。

## R14

关键超参数及调参建议：

- `alpha`（平滑强度）：
  - 小：更灵活，易过拟合；
  - 大：更平滑，易欠拟合。
- `n_knots`（结点数）：
  - 大：容量更高；
  - 小：形状更受限。
- `degree`（样条次数）：
  - 常用 `3`（三次样条）；
  - 更高次可能提升拟合但也增大不稳定性。
- `max_iter / tol`：
  - 控制 backfitting 收敛精度与时间。

## R15

方法对比：

- 对比线性回归：
  - 线性回归只能学全局线性关系；
  - GAM 在保持可解释分量结构的同时表达非线性。
- 对比核回归（NW）：
  - 核回归在查询时依赖全体样本、预测成本高；
  - GAM 训练后预测仅依赖基函数与系数，更适合部署。
- 对比全连接神经网络：
  - NN 更灵活但解释性弱；
  - GAM 的“每维单独函数”更利于业务解释与诊断。

## R16

典型应用场景：

- 风险评分与风控特征效应分析；
- 医疗/生物统计中的剂量-反应曲线建模；
- 营销与经济计量中的单因素边际效应解释；
- 需要“非线性 + 可解释”平衡的工业回归任务。

## R17

可扩展方向：

- 广义响应：
  - 逻辑回归型 GAM（Bernoulli + logit）；
  - 泊松计数型 GAM（Poisson + log）。
- 自动平滑参数：
  - GCV、AIC、K 折交叉验证。  
- 结构扩展：
  - 增加交互项（tensor-product spline）；
  - 单调/凸性等形状约束 GAM。
- 工程扩展：
  - 训练过程日志落盘；
  - 分量曲线可视化与模型诊断报告。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `make_synthetic_additive_data` 生成 3 维可加真函数与噪声观测，返回 `X, y` 与真分量。  
2. `main` 先做 `train/val/test` 切分，再调用 `search_alpha` 在候选 `alpha` 上训练多个 GAM。  
3. 每个 `AdditiveSplineGAM.fit` 首先对每个特征用 `SplineTransformer` 构建 `B_j`，并用 `_second_order_difference_penalty` 生成 `P_j=D^T D`。  
4. 初始化 `intercept = mean(y)`、各分量为 0，进入 backfitting 迭代。  
5. 对每个分量 `j`，计算部分残差 `r_j = y - (intercept + sum_{k!=j} f_k)`，再解线性系统 `(B_j^T B_j + alpha P_j) theta_j = B_j^T r_j`。  
6. 由 `theta_j` 得到分量值 `f_j=B_j theta_j`，做零均值中心化，并把均值并入 `intercept`，保持可辨识性。  
7. 每轮记录 `rmse` 与 `max_component_change`，当变化小于 `tol` 时停止；否则继续，直到 `max_iter`。  
8. 最终 `predict` 通过 `component_contributions` 逐维计算 `B_j(x_new)theta_j - offset_j` 并求和，加上截距得到预测值；主程序再打印指标和样例输出。

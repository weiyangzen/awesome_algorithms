# Logistic回归

- UID: `MATH-0283`
- 学科: `数学`
- 分类: `回归分析`
- 源序号: `283`
- 目标目录: `Algorithms/数学-回归分析-0283-Logistic回归`

## R01

Logistic 回归用于二分类任务：
将线性打分 `z = x^T w + b` 通过 Sigmoid 映射为概率 `p(y=1|x)`，再以对数似然（交叉熵）进行参数估计。

本目录的 MVP 目标：
- 从零实现可复现的 Logistic 回归训练器；
- 展示损失下降、分类指标与收敛行为；
- 避免把训练过程交给黑盒 API。

## R02

问题形式（带 `L2` 正则）：

- 数据：`X in R^(n*p)`，标签 `y in {0,1}^n`
- 概率模型：`p_i = sigma(x_i^T w + b)`
- 优化目标：

`min_{w,b}  J(w,b) = -(1/n) * sum_i [y_i log p_i + (1-y_i) log(1-p_i)] + (lambda/2)||w||_2^2`

输出包括：
- 训练过程：初始/最终损失、梯度范数、是否单调下降；
- 分类效果：Accuracy、Precision、Recall、F1、LogLoss；
- 与多数类基线比较；
- 学到的系数与标准化真值系数差异。

## R03

选择 Logistic 回归作为该条目的原因：
- 它是分类建模最基础、最可解释的概率模型之一；
- 与线性回归相比，输出概率天然适配分类决策；
- 目标函数是凸的，适合演示“数学建模 -> 数值优化 -> 评估验证”的完整闭环；
- 在小脚本里即可清晰展示工程可复现性与数值稳定性处理。

## R04

关键数学要点：

1. Sigmoid 函数
- `sigma(z) = 1 / (1 + exp(-z))`
- 在实现中采用分段稳定写法，避免指数溢出。

2. 损失函数（BCE + L2）
- `J = BCE + (lambda/2)||w||^2`
- `L2` 惩罚抑制过大的权重，提升泛化稳定性。

3. 梯度
- `dJ/dw = (1/n) X^T (p - y) + lambda w`
- `dJ/db = mean(p - y)`

4. 线搜索
- 用 Armijo 条件选择步长，确保每轮更新都能获得充分下降，提升训练稳定性。

## R05

整体流程：

1. 生成带相关特征的二分类合成数据。  
2. 划分训练/测试集（固定随机种子）。  
3. 使用训练集统计量标准化训练和测试特征。  
4. 初始化 `w=0, b=0`。  
5. 计算损失与梯度。  
6. 执行带回溯线搜索的梯度下降更新参数。  
7. 记录每轮 `(step, loss, grad_norm, lr)` 训练历史并判断收敛。  
8. 输出训练/测试指标、系数对比和全局检查结果。

## R06

正确性依据：
- 损失函数由 Bernoulli 似然最大化推导而来（取负号后最小化）；
- 梯度表达式是对 `w, b` 的解析求导结果；
- Logistic 回归 + `L2` 目标是凸问题，使用梯度法可收敛到全局最优附近；
- 回溯线搜索在实现层面保证每轮有可验证的下降（或最小步长回退）。

## R07

复杂度分析（`n` 样本数，`p` 特征数，`T` 迭代数）：

- 每轮前向与梯度计算约 `O(np)`；
- 回溯线搜索每次尝试也需一次损失计算，若平均尝试 `k` 次，则每轮约 `O(k*np)`；
- 总时间复杂度约 `O(T*k*np)`；
- 空间复杂度主要由 `X`、梯度与概率向量构成，为 `O(np + p + n)`。

## R08

边界与异常处理：
- 输入维度检查：`X` 必须二维，`y` 必须一维且样本数一致；
- 标签检查：`y` 必须仅包含 `0/1`；
- 数值检查：拒绝 `nan/inf`；
- 标准化检查：若某列方差近零则报错；
- 超参数检查：`lr_init > 0`、`max_iter > 0`、`tol > 0`、`l2 >= 0`；
- 对数损失计算使用 `eps` 截断概率，避免 `log(0)`。

## R09

MVP 设计取舍：
- 只用 `numpy` 完成训练核心，保证透明可审计；
- 不引入 mini-batch、动量、Adam 等额外机制，保持算法最小闭环；
- 使用合成数据避免外部文件依赖，确保 `uv run python demo.py` 直接可运行；
- 以可复现与可解释为优先，不追求大规模训练吞吐。

## R10

`demo.py` 关键函数职责：
- `stable_sigmoid`：稳定计算 Sigmoid。
- `validate_dataset`：输入合法性检查。
- `train_test_split`：固定随机种子划分训练/测试集。
- `standardize_from_train`：按训练集统计量标准化。
- `logistic_loss_and_grad`：计算损失与解析梯度。
- `fit_logistic_regression`：梯度下降 + Armijo 回溯线搜索。
- `predict_proba` / `predict_label`：概率与标签预测。
- `binary_metrics`：计算 Accuracy/Precision/Recall/F1/LogLoss。
- `is_monotone_nonincreasing`：审计损失序列是否非增。
- `main`：数据生成、训练、评估与全局检查总控。

## R11

运行方式：

```bash
cd Algorithms/数学-回归分析-0283-Logistic回归
uv run python demo.py
```

脚本不需要任何交互输入，直接打印训练日志和评估结果。

## R12

输出字段说明：
- `train_samples/test_samples/features`：数据规模信息；
- `positive_rate(train/test)`：类别分布；
- `converged`：是否达到停止条件；
- `iterations`：实际迭代轮数；
- `initial_loss/final_loss`：起始与结束目标值；
- `loss_monotone_nonincreasing`：损失是否单调非增；
- `final_grad_norm`：最终梯度范数；
- `accuracy/precision/recall/f1/log_loss`：分类性能指标；
- `l2_error_vs_true_standardized_weights`：估计系数与标准化真值系数的距离；
- `global_checks_pass`：全局质量检查汇总。

## R13

最小验证覆盖：
- 固定种子保证实验可复现；
- 检查损失从初始到最终是否下降；
- 检查测试集准确率是否优于多数类基线；
- 检查 F1 是否达到基础可用阈值（本脚本阈值 0.65）；
- 检查损失曲线非增性（实现级稳定性审计）。

## R14

关键超参数与调参建议：
- `l2=2e-2`：正则强度，越大越保守；
- `lr_init=1.0`：线搜索初始步长，较大可加速但需要回溯机制；
- `max_iter=400`：最大迭代轮数；
- `tol=1e-7`：收敛阈值；
- `armijo_c=1e-4`：线搜索充分下降常数；
- `min_lr=1e-8`：线搜索下界。

建议：
- 先固定数据和种子，仅调 `l2` 与 `max_iter`；
- 如果收敛慢，提高 `max_iter`；
- 如果过拟合，提高 `l2`；
- 如果下降不稳定，降低 `lr_init`。

## R15

方法对比与定位：
- 相比感知机：Logistic 回归提供概率输出，损失函数平滑可导；
- 相比线性回归做分类：Logistic 回归在概率建模上更合理；
- 相比黑盒库一行调用：本实现可见每一步梯度、步长与停止判据，便于审计与教学。

## R16

典型应用场景：
- 用户流失/转化预测（二分类）；
- 医疗风险分层（事件发生概率）；
- 金融风控中的违约二分类基线；
- 作为复杂模型前的可解释概率基线。

## R17

可扩展方向：
- 多分类扩展到 Softmax 回归；
- 使用 mini-batch 与动量/Adam 提升大样本效率；
- 加入 `L1` 或 Elastic Net 实现稀疏特征选择；
- 增加交叉验证、阈值搜索、ROC/AUC 分析；
- 对接真实业务数据并加入特征工程流程。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `make_correlated_binary_dataset` 生成相关性较强的特征矩阵与二分类标签，构造可复现训练样本。  
2. `validate_dataset` 校验维度、数值有效性以及标签是否为 `{0,1}`。  
3. `train_test_split` 固定种子划分数据，`standardize_from_train` 用训练集均值和方差标准化特征。  
4. `fit_logistic_regression` 初始化 `w,b`，循环调用 `logistic_loss_and_grad` 计算当前损失与解析梯度。  
5. 在每轮中执行 Armijo 回溯线搜索：从 `lr_init` 开始尝试步长，直到满足充分下降条件。  
6. 接受步长后更新 `w,b`，记录 `(step, loss, grad_norm, lr)` 到 `history`，并依据梯度范数/损失变化判断收敛。  
7. 训练后通过 `predict_proba` 输出概率，`binary_metrics` 计算训练与测试集的 Accuracy/Precision/Recall/F1/LogLoss。  
8. `main` 汇总优化状态、损失单调性、系数误差和全局检查结果，完成端到端可运行验证。

# 多项Logistic回归

- UID: `MATH-0284`
- 学科: `数学`
- 分类: `回归分析`
- 源序号: `284`
- 目标目录: `Algorithms/数学-回归分析-0284-多项Logistic回归`

## R01

多项 Logistic 回归（Multinomial Logistic Regression）用于多分类任务（类别数 `K >= 3`），
核心思想是对每个类别学习一组线性打分，再通过 `softmax` 转成概率：

- 线性打分：`z_k = w_k^T x + b_k`
- 概率输出：`p(y=k|x) = exp(z_k) / sum_j exp(z_j)`

本目录提供一个最小可运行 MVP，完整覆盖：数据生成、训练、预测、评估与结果打印。

## R02

问题定义：

- 输入：
  - 特征矩阵 `X in R^(n x d)`
  - 标签向量 `y in {0,1,...,K-1}^n`
  - 训练超参数：学习率、迭代数、L2 正则
- 输出：
  - 参数矩阵 `W in R^(K x (d+1))`（含截距）
  - 类别概率 `P in R^(n x K)`
  - 预测标签 `y_pred`
  - 评估指标：`accuracy`、`multiclass logloss`

目标函数（平均交叉熵 + L2 正则）：
`L(W) = -1/n * sum_i sum_k y_ik * log p_ik + (lambda/2) * ||W_no_bias||_F^2`

## R03

数学基础：

1. `softmax` 概率
- `p_ik = exp(z_ik) / sum_j exp(z_ij)`。

2. 交叉熵损失
- `CE = -1/n * sum_i sum_k y_ik * log p_ik`。

3. 梯度
- 设 `Y` 为 one-hot 标签矩阵，则
  `dL/dW = (P - Y)^T X_tilde / n + lambda * W_reg`，
  其中 `X_tilde` 含截距列，`W_reg` 不正则化截距列。

4. 预测
- `y_pred = argmax_k p_ik`。

## R04

算法总览（MVP）：

1. 构造可复现的多分类合成数据。
2. 校验输入维度、标签范围、有限值。
3. 加截距列，初始化参数矩阵 `W`。
4. 循环执行：`logits -> softmax -> loss -> gradient -> 参数更新`。
5. 判断收敛（参数增量范数）。
6. 在训练集与测试集上计算概率、类别与指标。
7. 打印参数形状、指标与样例预测。

## R05

核心数据结构：

- `numpy.ndarray`
  - `X`: `(n, d)`
  - `y`: `(n,)`
  - `W`: `(K, d+1)`
  - `proba`: `(n, K)`
- `MultinomialLogisticModel`（`dataclass`）
  - `weights`、`classes_`、`loss_history`、`n_iter_`、`converged_`

## R06

正确性要点：

- `softmax` 使每个样本的类别概率和为 1，且都在 `(0,1)`。
- 目标函数是凸函数（对线性参数化的多项 logistic），梯度下降可逼近全局最优。
- 使用数值稳定技巧（`logsumexp`）避免 `exp` 溢出。
- 训练后通过 `accuracy` 和 `logloss` 联合检查，避免单一指标误导。

## R07

复杂度分析：

设样本数 `n`、特征数 `d`、类别数 `K`、迭代轮数 `T`。

- 每次迭代：
  - 前向计算 `O(n*d*K)`
  - 梯度计算 `O(n*d*K)`
- 总时间复杂度：`O(T * n * d * K)`
- 空间复杂度：`O(n*K + d*K)`

## R08

边界与异常处理：

- `X` 必须是二维，`y` 必须是一维；
- `X` 与 `y` 样本数必须一致；
- 标签必须至少有 3 个类别，否则不属于多项 logistic 任务；
- 标签需要可映射到连续类别索引（如 `0..K-1`）；
- 发现 `NaN/Inf` 时直接报错；
- 学习率、迭代次数、正则项均做合法性检查。

## R09

MVP 取舍：

- 采用 `numpy` 手写 softmax + 交叉熵 + 梯度下降，避免完全黑盒调用。
- 使用合成数据，保证目录自包含、运行无外部依赖数据文件。
- 不引入复杂训练框架（如 mini-batch、早停回调、自动调参），保持算法主线清晰。

## R10

`demo.py` 函数职责：

- `make_synthetic_multiclass_data`：生成可复现多分类样本与真参数。
- `split_train_test`：固定随机种子切分训练/测试集。
- `validate_xy`：校验输入形状与数值。
- `add_intercept`：拼接截距列。
- `softmax`：稳定计算类别概率。
- `fit_multinomial_logistic_gd`：执行梯度下降训练。
- `multiclass_logloss`：计算多分类对数损失。
- `main`：组织端到端实验并打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-回归分析-0284-多项Logistic回归
uv run python demo.py
```

脚本无需交互输入。

## R12

输出解读：

- `train/test size`：训练与测试样本量。
- `classes`：模型识别到的类别集合。
- `converged / n_iter`：是否达到收敛以及迭代轮数。
- `train_accuracy/test_accuracy`：分类准确率。
- `train_logloss/test_logloss`：对数损失，越低越好。
- `sample predictions`：每个样本的真实标签、预测标签和各类概率。

## R13

建议最小测试集：

- 默认配置：`n=900, d=4, K=3`；
- 低噪声测试：降低随机扰动，准确率应提升；
- 高噪声测试：提升随机扰动，`logloss` 上升但流程应稳定；
- 异常测试：输入非法维度/包含 `NaN`/类别不足，需抛出明确错误。

## R14

可调参数：

- `n_samples`：样本规模；
- `seed`：随机种子（结果复现）；
- `learning_rate`：步长，过大可能震荡，过小收敛慢；
- `max_iter`：最大迭代轮数；
- `l2_reg`：L2 正则强度，控制过拟合与数值稳定；
- `tol`：参数更新收敛阈值。

## R15

方法对比：

- 多项 Logistic 回归
  - 优点：概率可解释、训练快、作为多分类基线可靠；
  - 缺点：决策边界线性，表达能力有限。
- One-vs-Rest 二分类组合
  - 优点：实现简单；
  - 缺点：各类概率不天然一致，概率解释不如 softmax 统一。
- 非线性模型（树模型/神经网络）
  - 优点：可学习复杂边界；
  - 缺点：可解释性和调参成本更高。

## R16

应用场景：

- 文本主题分类、意图识别；
- 图像或传感器多状态分类（线性可分或近似线性可分时）；
- 作为复杂模型前的可解释多分类 baseline。

## R17

后续扩展方向：

- 支持 mini-batch SGD / Adam；
- 增加标准化与特征工程流程；
- 增加混淆矩阵、宏平均 F1 等多分类指标；
- 与 `scikit-learn` 的 `LogisticRegression(multi_class="multinomial")` 结果对齐；
- 增加模型持久化与推理接口。

## R18

`demo.py` 源码级算法流（8 步）：

1. `main` 调用 `make_synthetic_multiclass_data` 生成 `X, y` 与真实参数，再通过 `split_train_test` 划分训练/测试集。  
2. `fit_multinomial_logistic_gd` 首先执行 `validate_xy`，并将标签映射到 `0..K-1` 的内部索引。  
3. `add_intercept` 给特征矩阵增加常数列，形成设计矩阵 `X_tilde`，初始化 `W` 为零矩阵。  
4. 每轮迭代先计算 `logits = X_tilde @ W^T`，再用稳定版 `softmax`（基于 `logsumexp`）得到 `P`。  
5. 用 one-hot 标签矩阵 `Y` 计算交叉熵损失与 L2 正则项，记录到 `loss_history`。  
6. 根据 `grad = (P - Y)^T X_tilde / n + lambda * W_reg` 计算梯度，并执行 `W <- W - lr * grad`。  
7. 通过 `||W_new - W||_F < tol` 判定收敛；若满足则提前停止，否则继续直到 `max_iter`。  
8. 训练结束后在 train/test 上调用 `predict_proba` 与 `predict`，计算 `accuracy`、`logloss` 并输出样例概率表。  

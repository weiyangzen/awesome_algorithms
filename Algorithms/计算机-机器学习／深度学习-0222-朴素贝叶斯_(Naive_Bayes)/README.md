# 朴素贝叶斯 (Naive Bayes)

- UID: `CS-0100`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `222`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0222-朴素贝叶斯_(Naive_Bayes)`

## R01

朴素贝叶斯（Naive Bayes）是一类基于贝叶斯定理的概率分类方法。核心假设是：在给定类别 `y` 的条件下，各个特征 `x_j` 条件独立。

这让高维联合概率从难估计的 `P(x_1,...,x_d|y)` 退化为可分解形式：

`P(x|y) = Π_j P(x_j|y)`

因此它训练速度快、参数量小，特别适合做文本分类和中小规模监督分类基线。

## R02

本条目聚焦 `Gaussian Naive Bayes`（高斯朴素贝叶斯）：
- 输入：连续特征矩阵 `X in R^(n x d)` 与类别标签 `y`；
- 输出：每个类别的先验 `P(y=k)`、每个特征在该类别下的高斯参数（均值、方差）；
- 预测：返回 `argmax_k P(y=k|x)` 及类别概率。

本目录 `demo.py` 同时给出：
- 手写 `GaussianNBFromScratch`（主实现）；
- `sklearn.naive_bayes.GaussianNB`（对照实现）。

## R03

贝叶斯决策规则：

`y* = argmax_k P(y=k|x) = argmax_k P(y=k) P(x|y=k)`

在条件独立与高斯似然下：

`P(x|y=k) = Π_j N(x_j ; mu_{k,j}, sigma_{k,j}^2)`

取对数后可避免下溢，并把乘法变加法：

`log P(y=k|x) = log P(y=k) - 1/2 Σ_j [log(2π sigma_{k,j}^2) + (x_j-mu_{k,j})^2 / sigma_{k,j}^2] + C`

其中常数 `C` 与类别无关，做 `argmax` 时可忽略。

## R04

本实现算法流程：
1. 用 `make_classification` 构造可复现三分类数据。  
2. `train_test_split(..., stratify=y)` 分层划分训练/测试集。  
3. 在训练集上按类别统计 `class_prior_`、`theta_`（均值）、`var_`（方差）。  
4. 计算 `epsilon = var_smoothing * max(var(X_train))`，并加到每类方差上。  
5. 推理时按类别计算联合对数似然（joint log likelihood）。  
6. 取 `argmax` 得类别预测，按 softmax 归一化得到概率。  
7. 对照 `sklearn` 输出精度、一致性与混淆矩阵。

## R05

`demo.py` 关键数据结构：
- `GaussianNBFromScratch`（dataclass）
  - `class_prior_`：类别先验；
  - `theta_`：每类每特征均值矩阵；
  - `var_`：每类每特征方差矩阵（含平滑项）；
  - `class_count_`：每类样本数。
- `numpy.ndarray`：训练、预测与概率计算主数据结构。
- `pandas.DataFrame`：指标表、混淆矩阵、预测预览表。

## R06

正确性与数值稳定要点：
- 必须在对数域计算似然，避免小概率连乘下溢；
- 方差不能为 0，需要 `var_smoothing` 添加 `epsilon`；
- 训练与预测使用同一类别顺序（`np.unique(y)`）；
- 概率输出需归一化，且每行和应接近 1；
- 用 `from-scratch` 与 `sklearn` 的参数/预测一致性做交叉验证。

## R07

复杂度分析（`n` 样本、`d` 特征、`K` 类）：
- 训练：按类统计均值方差，时间复杂度约 `O(n d)`，空间约 `O(K d)`；
- 预测：每个样本对每类计算一遍高斯 log-likelihood，时间复杂度约 `O(n_test K d)`；
- 与许多迭代优化模型相比，朴素贝叶斯训练几乎是一次统计汇总，速度优势明显。

## R08

边界与异常处理（脚本中已覆盖）：
- `x` 非二维或 `y` 非一维，抛 `ValueError`；
- 样本数不一致，抛 `ValueError`；
- 空训练集，抛 `ValueError`；
- 未 `fit` 调 `predict/predict_proba`，抛 `RuntimeError`。

## R09

MVP 取舍说明：
- 选择最核心的 `GaussianNB`，不扩展到文本分词流水线；
- 保留手写实现，避免只调用第三方黑盒；
- 用合成数据确保无下载依赖、可复现、可快速验证；
- 不引入 CLI 交互参数，保证 `uv run python demo.py` 一次运行完成。

## R10

`demo.py` 函数职责：
- `make_gaussian_classification_data`：生成可复现实验数据；
- `GaussianNBFromScratch.fit`：估计先验、均值、方差；
- `GaussianNBFromScratch._joint_log_likelihood`：计算按类别的联合对数似然；
- `GaussianNBFromScratch.predict/predict_proba`：输出类别与概率；
- `evaluate_metrics`：计算 `accuracy` 与 `macro_f1`；
- `main`：组织训练、对照、打印与质量门槛断言。

## R11

运行方式（无交互输入）：

```bash
cd Algorithms/计算机-机器学习／深度学习-0222-朴素贝叶斯_(Naive_Bayes)
uv run python demo.py
```

## R12

脚本输出字段说明：
- `train_size/test_size/n_features/classes`：数据规模与类别；
- `Metrics`：from-scratch 与 sklearn 的 `accuracy`、`macro_f1`；
- `prediction_agreement`：两实现预测标签一致率；
- `L2(theta/var/prior gap)`：参数差异；
- `RMSE(proba gap)`：概率输出差异均方根；
- `Confusion Matrix`：from-scratch 的分类混淆情况；
- `Prediction Preview`：前若干样本的真实标签、预测标签与最大概率。

## R13

建议验证项（本脚本已内置）：
1. 预测准确率不低于阈值（当前门槛 `accuracy >= 0.82`）。  
2. from-scratch 与 sklearn 一致率不低于阈值（当前门槛 `>= 0.98`）。  
3. `predict_proba` 每行概率和约等于 1（`atol=1e-9`）。  
4. 多次运行在固定随机种子下输出应稳定。

## R14

关键超参数：
- `var_smoothing`：方差平滑强度，过小可能数值不稳，过大可能欠拟合；
- `class_sep`：数据集可分性（主要用于 demo 数据构造）；
- `weights`：类别不平衡比例；
- `flip_y`：标签噪声比例。

调参建议：
- 先固定数据生成参数，扫描 `var_smoothing in [1e-12, 1e-3]`；
- 若模型过于自信且泛化差，可适当增大 `var_smoothing`；
- 不平衡场景需重点观察 `macro_f1` 而非只看准确率。

## R15

朴素贝叶斯常见变体对比：
- `GaussianNB`：连续特征，假设高斯分布；
- `MultinomialNB`：计数型特征（如词频）；
- `BernoulliNB`：二值特征（词出现/不出现）；
- `ComplementNB`：对类别不平衡文本分类更稳健。

选择原则通常由“特征分布形态”决定，而不是由任务名决定。

## R16

典型应用场景：
- 文本垃圾邮件过滤、主题初筛；
- 医疗或风控中的快速基线分类器；
- 边缘设备上的轻量分类；
- 需要高训练吞吐、可快速迭代的模型对照实验。

## R17

可扩展方向：
- 加入 `MultinomialNB` 文本示例（`CountVectorizer + NB`）；
- 加入先验平滑、类别代价敏感策略；
- 引入真实数据集并输出交叉验证报告；
- 在流式数据上使用 `partial_fit` 做在线更新实验。

## R18

`demo.py` + `scikit-learn` 的源码级流程拆解（8 步）：
1. `make_gaussian_classification_data` 生成三分类连续特征数据，`main` 做分层划分。  
2. `GaussianNBFromScratch.fit` 中按类别切片 `X_i`，统计 `class_count_`、`theta_`、`var_`。  
3. `fit` 按 `epsilon = var_smoothing * max(var(X))` 对每类方差做平滑，得到稳定的高斯参数。  
4. `GaussianNBFromScratch._joint_log_likelihood` 对每个类别计算 `log prior + log Gaussian likelihood`。  
5. `predict` 对联合对数似然做 `argmax`，`predict_proba` 用指数平移归一化输出概率。  
6. 对照实现中，`sklearn.naive_bayes.GaussianNB.fit`（位于 `site-packages/sklearn/naive_bayes.py`）会调用 `_partial_fit`。  
7. `_partial_fit` 内部调用 `_update_mean_variance` 更新每类均值/方差，并维护 `class_prior_`、`class_count_`。  
8. 推理阶段 `GaussianNB._joint_log_likelihood` 计算与手写版同构的对数似然，`predict/predict_proba` 基于该矩阵输出类别与概率；脚本最终比较两者指标和参数差异，形成可验证闭环。

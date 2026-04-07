# 支持向量机 (SVM)

- UID: `MATH-0301`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `301`
- 目标目录: `Algorithms/数学-机器学习-0301-支持向量机_(SVM)`

## R01

支持向量机（Support Vector Machine, SVM）是一个以“最大间隔分类”为核心思想的监督学习算法。在线性可分/近似线性可分场景下，SVM 通过寻找一个超平面 `w^T x + b = 0`，让两类样本在该超平面两侧且间隔尽可能大。

本条目实现的是二分类、线性软间隔 SVM 的最小可运行版本（MVP）：
- 损失函数采用 `hinge loss`；
- 正则项采用 `L2`；
- 训练方法采用全批次子梯度下降（subgradient descent）；
- 输出支持向量的近似集合（按 `margin <= 1 + tol` 判定）。

## R02

MVP 问题定义：

给定训练集 `D = {(x_i, y_i)}_{i=1}^n`，其中 `y_i ∈ {-1, +1}`，求解：

`min_{w,b} 1/2 ||w||^2 + C * (1/n) * sum_i max(0, 1 - y_i (w^T x_i + b))`

其中：
- `w, b` 为分类超平面参数；
- `C > 0` 控制“间隔最大化”和“训练误差惩罚”的权衡；
- `max(0, 1 - margin)` 是 hinge 损失。

预测规则：
- `score = w^T x + b`
- `score >= 0` 判为正类，否则判为负类。

## R03

为什么这个条目不直接把 `sklearn.svm.SVC` 当黑盒：
- 任务目标是“算法实现与机制可解释”，而不是仅完成 API 调用；
- 手写训练循环可以清楚看到梯度项由哪些样本贡献（`margin < 1` 的样本）；
- 便于学习“正则项 + hinge 损失 + 学习率衰减”如何共同影响收敛。

`demo.py` 中 `scikit-learn` 只作为可选 sanity check，不参与主训练流程。

## R04

核心数学关系（对应 `demo.py`）：

1. 函数间隔：
`m_i = y_i (w^T x_i + b)`

2. 单样本 hinge 损失：
`l_i = max(0, 1 - m_i)`

3. 总目标：
`J(w,b) = 1/2 ||w||^2 + C * mean(l_i)`

4. 子梯度（全批次）：
- 记 `M = {i | m_i < 1}`
- `∂J/∂w = w - (C/n) * sum_{i in M} y_i x_i`
- `∂J/∂b = -(C/n) * sum_{i in M} y_i`

5. 参数更新：
- `w <- w - eta_t * ∂J/∂w`
- `b <- b - eta_t * ∂J/∂b`
- `eta_t = lr0 / (1 + lr_decay * t)`

## R05

算法高层流程：

1. 校验输入维度与有限值，确保是二分类任务。
2. 标签映射到 `{-1, +1}`。
3. 对特征做标准化（零均值、单位方差）。
4. 初始化 `w=0, b=0`。
5. 迭代 `epochs` 次：计算 margin、抽取违反间隔样本、计算子梯度并更新参数。
6. 训练结束后计算最终目标值（objective）。
7. 根据 `margin <= 1 + tol` 记录支持向量近似集合。
8. 在测试集输出准确率、精确率、召回率、F1 及混淆矩阵。

## R06

正确性直觉：
- `L2` 正则项 `1/2||w||^2` 倾向于更小范数的 `w`，对应更大几何间隔；
- hinge 损失只惩罚“分类错误或落入间隔内”的样本；
- 因此，只有 `margin < 1` 的样本会进入更新项，这些样本正是决策边界附近的关键样本；
- 训练迭代把超平面逐步推向“间隔更大、经验风险更小”的方向。

这不是对偶 QP 的精确求解器（如 SMO），但作为线性软间隔 SVM 的最小实现，机制与目标函数是一致的。

## R07

复杂度分析（设样本数 `n`、特征数 `d`、训练轮数 `T`）：
- 每轮需要一次矩阵-向量乘 `X @ w`，时间约 `O(n*d)`；
- 共 `T` 轮，训练时间约 `O(T*n*d)`；
- 预测阶段每个样本 `O(d)`，整批 `O(n_test*d)`；
- 存储复杂度约 `O(n*d)`（数据）和 `O(d)`（参数）。

## R08

边界与异常处理：
- `X` 必须为二维，`y` 必须为一维，且样本数一致；
- 输入不得包含 `nan`/`inf`；
- `y` 必须恰好有两个类别；
- 超参数约束：`C > 0`、`lr0 > 0`、`lr_decay >= 0`、`epochs >= 1`；
- `predict`/`decision_function` 前必须先 `fit`；
- 预测特征维度必须与训练时一致。

## R09

MVP 范围与取舍：
- 已实现：二分类线性 SVM、标准化、子梯度训练、基础评估指标、支持向量近似统计。
- 未实现：核技巧（RBF/Poly）、多分类策略（OvR/OvO）、对偶优化器（SMO）、概率校准。
- 取舍原则：优先“短小可运行 + 可读可解释”，而不是覆盖所有工程特性。

## R10

`demo.py` 关键函数职责：
- `StandardScalerMVP`：手写标准化（`fit/transform`）。
- `LinearSVMMVP.fit`：训练入口，执行子梯度优化。
- `LinearSVMMVP._validate_xy`：输入校验。
- `LinearSVMMVP._to_signed_labels`：将原始标签映射为 `{-1,+1}`。
- `LinearSVMMVP.decision_function/predict`：打分与分类。
- `make_binary_dataset`：生成可复现的二分类合成数据。
- `train_test_split`：非交互、可复现的数据划分。
- `accuracy`、`precision_recall_f1`、`confusion_counts`：评估指标。
- `maybe_compare_with_sklearn`：可选外部对照（不影响主流程）。
- `main`：端到端运行和打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-机器学习-0301-支持向量机_(SVM)
python3 demo.py
```

脚本无需交互输入，直接打印训练与测试结果。

## R12

输出解读：
- `train/test size`：训练集和测试集规模；
- `hyperparameters`：`C`、学习率、衰减率、迭代轮次；
- `weight norm`：`||w||`，与间隔大小相关；
- `objective`：最终目标函数值（越低通常越好）；
- `approx support vectors`：支持向量近似数量及占比；
- `train/test metrics`：准确率、精确率、召回率、F1；
- `confusion matrix`：`TP/TN/FP/FN`；
- `sample predictions`：部分样本的真实标签、预测标签与决策分数。

## R13

最小实验设计（内置 `main`）：
- 构造 3 维二分类数据（两个高斯簇 + 噪声维度）；
- 按 70/30 划分训练集和测试集；
- 训练线性 SVM 并报告训练/测试指标；
- 与“多数类预测器”基线进行准确率对比；
- 若环境有 `scikit-learn`，额外给出 `SVC(kernel='linear')` 对照结果。

## R14

关键超参数与调参建议：
- `C`：越大越强调训练集拟合，越小正则越强。
- `lr0`：初始学习率，过大可能震荡，过小收敛慢。
- `lr_decay`：学习率衰减强度，控制后期步长。
- `epochs`：训练轮数，过少可能欠收敛。

推荐顺序：先固定 `lr0/lr_decay`，粗调 `C`；再根据 objective 曲线和测试指标调整 `epochs`。

## R15

与相关模型对比：
- 对比感知机：SVM 有显式间隔最大化与正则项，泛化更稳。
- 对比逻辑回归：二者都线性可分；逻辑回归优化对数损失，SVM 优化 hinge 损失。
- 对比核 SVM：核 SVM 可处理更强非线性，但计算与调参成本更高。
- 对比树模型：树擅长非线性划分，SVM 在线性边界明确时常更紧凑。

## R16

典型应用场景：
- 中小规模结构化数据的二分类任务；
- 文本分类（高维稀疏特征，常见线性核）；
- 需要“边界清晰、参数较少”的基线模型；
- 作为更复杂分类器前的快速 sanity baseline。

## R17

可扩展方向：
- 增加核函数版本（RBF、Polynomial）；
- 实现多分类（OvR/OvO）；
- 使用对偶优化（SMO/QP）获得更标准的支持向量集合；
- 增加早停、学习率自适应、交叉验证；
- 增加模型持久化与绘图诊断（margin 分布、损失曲线）。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 调用 `make_binary_dataset` 生成可复现实验数据，并通过 `train_test_split` 划分训练/测试集。
2. 初始化 `LinearSVMMVP`，设置 `C/lr0/lr_decay/epochs` 等超参数。
3. `fit` 先做输入校验与标签二值化映射，再调用 `StandardScalerMVP` 对训练特征标准化。
4. 在每个 epoch 内，计算 `margin = y*(Xw+b)`，定位 `margin < 1` 的样本集合（违反间隔样本）。
5. 基于该集合计算目标函数的子梯度：`grad_w = w - (C/n)Σ(yx)`、`grad_b = -(C/n)Σ(y)`，并按衰减学习率更新 `w,b`。
6. 训练结束后计算最终 objective，并用 `margin <= 1 + tol` 提取“支持向量近似索引与样本”。
7. `predict` 通过 `decision_function` 计算分数并按阈值 0 映射回原始类别标签。
8. `main` 汇总准确率、精确率、召回率、F1、混淆矩阵、基线对比，并可选调用 `maybe_compare_with_sklearn` 做外部 sanity check（不参与主训练流程）。

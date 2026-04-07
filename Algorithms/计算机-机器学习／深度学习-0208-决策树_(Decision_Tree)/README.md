# 决策树 (Decision Tree)

- UID: `CS-0095`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `208`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0208-决策树_(Decision_Tree)`

## R01

决策树（Decision Tree）是通过递归划分特征空间来进行监督学习的模型。每个内部节点表示一个“特征阈值判断”，每个叶节点输出类别（分类）或数值（回归）。

本条目实现一个最小可运行的 `CART` 风格分类树 MVP：
- 二叉划分（每次仅选择一个特征和一个阈值）；
- 使用 `Gini` 不纯度作为划分标准；
- 叶节点输出类别分布与预测类别；
- 核心训练/推理逻辑全部在 `demo.py` 中以源码展开。

## R02

MVP 问题定义（多分类）：

给定训练集 `D={(x_i, y_i)}_{i=1}^n`，其中 `x_i in R^d`，`y_i in {0,1,...,C-1}`，学习分类器 `f(x)`。

对任意节点样本集 `S`，其 Gini 不纯度：

`Gini(S) = 1 - sum_k p_k^2`，其中 `p_k` 是类别 `k` 在 `S` 中的频率。

若按 `(j, t)` 划分为左右子集 `L/R`，则子节点加权不纯度：

`G_child = (|L|/|S|) * Gini(L) + (|R|/|S|) * Gini(R)`

选择 `Gain = Gini(S) - G_child` 最大的切分。

## R03

为什么不直接把任务写成几行库调用：
- 决策树的关键并不在 API，而在“候选阈值扫描 + 不纯度比较 + 递归停止条件”；
- 手写实现可以直观看到每一步的数据流：排序、前缀计数、左右子集更新；
- 这能为随机森林、GBDT、XGBoost 等树模型家族打下可验证的机制理解。

`demo.py` 中 `scikit-learn` 仅作为可选 sanity check，不参与主训练流程。

## R04

单次最优切分的计算细节：

对某个特征 `j`，先按 `x[:, j]` 排序，得到有序样本。候选切分点在相邻且取值不同的位置之间。

扫描位置 `i`（表示左侧样本数为 `i`）时：
- 左计数向量 `count_L` 由前缀累加得到；
- 右计数向量 `count_R = count_total - count_L`；
- 计算 `Gini(L)`、`Gini(R)` 和 `G_child`；
- 用 `Gain = Gini(parent) - G_child` 更新最优解。

这种做法把“每个候选点都重新统计类别分布”的开销降为线性扫描。

## R05

算法高层流程：

1. 输入检查并把标签编码到连续整数（`np.unique(..., return_inverse=True)`）。
2. 在当前节点计算类别计数、节点 Gini、叶子预测类别与概率。
3. 若满足停止条件（深度上限、样本过少、节点纯），直接生成叶子。
4. 否则对每个特征排序并扫描候选阈值，找最大增益切分。
5. 若最优增益低于 `min_impurity_decrease`，停止分裂并建叶子。
6. 否则按最优阈值拆分左右子集，递归构建左右子树。
7. 预测时从根节点逐层比较阈值，落到叶子后输出类别或概率。

## R06

正确性直觉：
- 在当前节点，算法显式枚举该节点下的全部有效二叉阈值切分，并选择 Gini 降幅最大者，因此对“单步分裂”是贪心最优；
- 叶子输出的类别概率来自该叶样本经验分布，预测类别取最大概率；
- 递归重复后得到分段规则集合，实现非线性决策边界。

注意这是局部贪心策略，不保证全局最优树结构。

## R07

复杂度分析（节点样本数 `m`，特征数 `d`，类别数 `C`）：
- 单特征：排序 `O(m log m)`，扫描候选点 `O(m * C)`；
- 单节点：`O(d * (m log m + mC))`；
- 整树训练复杂度取决于各节点样本分布，一般是多节点开销之和。

空间复杂度：
- 模型存储约 `O(number_of_nodes)`；
- 节点内部排序索引和计数向量为局部临时数组。

## R08

边界与异常处理：
- `X` 必须二维，`y` 必须一维，样本数一致；
- `X` 中不允许 `NaN/Inf`；
- 标签类别数至少为 2；
- `max_depth >= 0`，`min_samples_split >= 2`，`min_samples_leaf >= 1`；
- `test_ratio` 需在 `(0,1)`；
- `predict/predict_proba` 前必须先 `fit`，且特征维度需匹配。

## R09

MVP 范围与取舍：
- 仅实现最核心的 CART 分类树机制（Gini + 二叉阈值划分）；
- 不实现剪枝、缺失值原生处理、类别特征特化策略、样本权重；
- 不做并行训练和工程级优化，优先保证逻辑透明、代码短小可审计；
- 数据采用固定种子的合成多分类任务，确保可复现实验。

## R10

`demo.py` 关键函数职责：
- `TreeNode`：树节点结构（叶标记、预测类别、概率、分裂条件、子节点）。
- `DecisionTreeClassifierMVP.fit`：训练入口，完成输入检查和递归建树。
- `DecisionTreeClassifierMVP._best_split`：排序 + 前缀类别计数扫描候选阈值。
- `DecisionTreeClassifierMVP._build_tree`：递归控制停止条件并创建子树。
- `DecisionTreeClassifierMVP.predict/predict_proba`：批量推理。
- `make_multiclass_data`：构造非线性三分类数据集。
- `train_test_split`：固定随机种子划分训练/测试集。
- `accuracy_score`、`macro_f1_score`、`confusion_matrix_int`：基础评估指标。
- `maybe_compare_with_sklearn`：可选地与 `DecisionTreeClassifier` 对照。
- `main`：串联训练、评估、输出与质量门槛断言。

## R11

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0208-决策树_(Decision_Tree)
uv run python demo.py
```

脚本无交互输入，执行后直接打印指标与检查结果。

## R12

输出字段解读：
- `train shape/test shape`：训练与测试数据维度；
- `hyperparameters`：树深、最小分裂样本、最小叶子样本、最小增益阈值；
- `tree stats`：节点总数、叶子数、树深；
- `train accuracy`：训练集准确率；
- `test accuracy`、`test macro_f1`：测试集泛化表现；
- `baseline majority ...`：多数类基线指标；
- `feature importances`：按累计增益归一化的重要性；
- `confusion matrix`：分类混淆矩阵；
- `sample predictions`：前若干测试样本的真值、预测和概率；
- `[optional sklearn] ...`：可选外部实现对照。

## R13

最小实验设计（已内置）：
1. 用固定种子生成 4 维特征的非线性三分类数据（共 480 条）。
2. 按 70/30 划分训练集和测试集。
3. 训练手写决策树并输出准确率、宏平均 F1。
4. 与“多数类常数预测”基线比较。
5. 若 `scikit-learn` 可用，额外输出同参数库模型指标作 sanity check。
6. 通过断言确保结果有限且达到最低可接受门槛。

## R14

关键超参数与调参建议：
- `max_depth`：控制模型容量，过大易过拟合；
- `min_samples_split`：节点继续分裂所需最小样本数；
- `min_samples_leaf`：叶子最小样本，提升鲁棒性；
- `min_impurity_decrease`：最小增益阈值，抑制无效细分。

调参可按“先控制复杂度，再放宽分裂条件”的顺序：先定 `max_depth/min_samples_leaf`，再微调 `min_samples_split/min_impurity_decrease`。

## R15

与相关方法对比：
- 对比线性模型：决策树可自动建模非线性和特征交互；
- 对比 kNN：决策树推理更快，规则路径可解释；
- 对比随机森林/GBDT：单树可解释性更强，但泛化稳定性通常不如集成方法。

本任务重点是“单棵树机制透明 + 可运行基线”。

## R16

典型应用场景：
- 结构化数据分类（风控、营销响应、用户分层）；
- 需要可解释规则路径的业务建模；
- 作为集成树模型（RF/GBDT）前置教学与调研基线。

## R17

可扩展方向：
- 代价复杂度剪枝（cost-complexity pruning）；
- 缺失值分裂策略与类别特征原生处理；
- 样本权重与类别不平衡处理（class weight）；
- 与袋装法结合为随机森林；
- 与梯度提升框架结合为 GBDT/XGBoost/LightGBM 风格模型。

## R18

`demo.py` 的源码级算法流程（9 步）：

1. `main` 调用 `make_multiclass_data` 生成固定随机种子的非线性三分类数据，并执行训练/测试划分。
2. 初始化 `DecisionTreeClassifierMVP`，设置 `max_depth/min_samples_split/min_samples_leaf/min_impurity_decrease`。
3. `fit` 先在 `_validate_and_encode` 中检查输入并将标签映射到连续整数区间，随后进入 `_build_tree`。
4. `_build_tree` 在当前节点计算类别计数、节点 Gini、叶概率和默认预测类别，并判断是否命中停止条件。
5. 若可继续分裂，调用 `_best_split`：对每个特征排序，线性扫描候选切分点并通过左右计数向量计算加权 Gini。
6. `_best_split` 返回增益最大的 `(feature, threshold, gain)`；若增益不足则当前节点退化为叶节点。
7. 若分裂有效，按阈值构造左右子集并递归调用 `_build_tree` 生成左右子树，同时累计特征增益用于重要性估计。
8. 预测时 `predict/predict_proba` 对每个样本调用 `_predict_one`，从根节点沿阈值比较一路下行到叶子并返回类别或概率。
9. `main` 汇总准确率、宏 F1、混淆矩阵、基线对比与可选 `scikit-learn` 对照，并执行质量断言确保 MVP 输出可靠。

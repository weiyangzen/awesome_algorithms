# 随机森林

- UID: `MATH-0296`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `296`
- 目标目录: `Algorithms/数学-机器学习-0296-随机森林`

## R01

随机森林（Random Forest）是基于 Bagging 的集成学习方法：
- 用自助采样（bootstrap）构造多份训练子集；
- 每棵树在随机子特征空间训练；
- 通过多数投票或平均概率融合，降低单树方差并提升泛化稳定性。

本目录实现一个可运行且可审计的 MVP：
- 森林层逻辑（采样、特征子集、投票、OOB 评估）手写实现；
- 单棵树采用手写 CART 风格二叉划分（Gini + 分位点候选阈值）；
- 脚本固定数据生成与质量门槛，运行后自动给出结果与校验。

## R02

本条目面向二分类任务，问题形式为：
- 输入：`X in R^(n*d)`、标签 `y in {0,1}^n`；
- 输出：分类器 `f(x)`，并在测试集给出 `accuracy/F1` 与 OOB 准确率。

`demo.py` 的数据是固定随机种子的合成样本，无需交互输入，直接执行即可复现。

## R03

核心机制与数学关系：

1. 自助采样：
   - 对每棵树 `t`，从 `n` 个样本中有放回采样 `n` 次得到索引集 `B_t`。
2. 随机子特征：
   - 在树的每个节点，从 `d` 个特征中随机抽取 `k` 个候选特征（常用 `k=floor(sqrt(d))`）。
3. 单树训练：
   - 在 bootstrap 样本上递归寻找最优二叉切分：最小化加权 Gini 不纯度。
4. 概率聚合：
   - `p(y=c|x) = (1/T) * sum_t p_t(y=c|x)`。
5. 最终预测：
   - `y_hat = argmax_c p(y=c|x)`。
6. OOB 估计：
   - 对每个样本，仅使用未包含该样本的树进行预测并聚合，得到近似无偏泛化估计。

## R04

算法流程（高层）：

1. 读取训练数据并做形状/数值合法性检查。  
2. 根据 `max_features` 规则确定每个树节点候选特征维度。  
3. 对 `t=1..T`：执行 bootstrap 抽样并训练一棵树。  
4. 若启用 bootstrap，则记录 OOB 样本并累加其预测概率。  
5. 全部树训练完成后，计算 OOB 覆盖率与 OOB 准确率。  
6. 推理时对每棵树概率输出直接平均。  
7. 用 `argmax` 得到分类标签，输出测试集指标并执行断言。

## R05

核心数据结构：
- `ForestFitSummary`：训练摘要，包含 `n_estimators/n_samples/n_features/max_features_used/oob_coverage/oob_accuracy`。  
- `TreeNode`：树节点结构（`is_leaf/proba/feature/threshold/left/right`）。  
- `SimpleDecisionTreeClassifier`：单树实现（Gini、分位点阈值搜索、递归建树、逐样本推理）。  
- `SimpleRandomForestClassifier`：主模型类。关键成员：
  - `trees_`：每棵 `SimpleDecisionTreeClassifier`；
  - `bootstrap_indices_`：每棵树的 bootstrap 样本索引；
  - `classes_`：全局类别顺序（标签到内部索引映射）。

## R06

正确性要点：
- 模型多样性来自“样本扰动 + 特征扰动”双随机化，降低单树相关性。  
- 每个叶子节点保存类别概率分布，森林端对概率做算术平均，避免单树硬投票抖动。  
- OOB 指标仅使用“未见过该样本”的树投票，避免训练内乐观偏差。  
- 通过与单棵树基线对比，验证森林集成在测试集上的有效性。  
- 运行末尾用门槛断言（测试精度、OOB 覆盖率、有限值检查）保障最小正确性。

## R07

复杂度分析（样本数 `n`，特征数 `d`，树数 `T`，候选阈值数 `q`）：
- 单树每个节点约需 `O(k*q*n_node)` 评估候选切分（`k` 为候选特征数）。  
- 在深度受限场景下，单树训练近似可写作 `O(k*q*n*log n)`。  
- 森林训练总计近似：`O(T*k*q*n*log n)`。  
- 预测复杂度：`O(T * depth)`（单样本）或 `O(m * T * depth)`（`m` 个样本）。  
- 空间复杂度：
  - 模型存储约为所有树节点总和；
  - 额外索引与中间概率数组约 `O(T*n + n*C)`（`C` 为类别数）。

## R08

边界与异常处理：
- `n_estimators<=0`、`max_depth<=0`、非法 `max_features` 会抛 `ValueError`。  
- 输入 `X` 不是二维、`y` 不是一维、样本数不匹配、含 `NaN/Inf` 会抛 `ValueError`。  
- 类别数小于 2 直接拒绝训练。  
- `predict/predict_proba` 在未 `fit` 时抛 `RuntimeError`。  
- 若 OOB 没有有效覆盖会给出 `nan`，并由质量门槛拦截。

## R09

MVP 取舍说明：
- 保持轻量：仅依赖 `numpy` + Python 标准库。  
- 不调用任何第三方黑盒随机森林接口，树内分裂与森林外层都在源码中展开。  
- 聚焦二分类与核心机制，不扩展到多任务、多输出、并行训练或特征重要性置信区间。  
- 采用合成数据保证可复现，不接入外部数据源或命令行配置系统。

## R10

`demo.py` 主要函数/类职责：
- `SimpleDecisionTreeClassifier._best_split`：在候选特征与候选阈值上搜索最小 Gini 切分。  
- `SimpleDecisionTreeClassifier._grow`：递归建树并构建叶子概率。  
- `SimpleRandomForestClassifier._resolve_max_features`：解析每个节点候选特征维度策略。  
- `SimpleRandomForestClassifier.fit`：训练主循环（采样、建树、OOB 累加）。  
- `SimpleRandomForestClassifier.predict_proba/predict`：集成推理与标签输出。  
- `stratified_train_test_split`：无第三方依赖的分层切分。  
- `build_dataset`：生成固定随机种子的二分类合成数据。  
- `binary_metrics`：输出准确率、精确率、召回率、F1 与混淆矩阵计数。  
- `run_baseline_tree`：训练单树基线用于对照。  
- `main`：串联训练、评估、打印报告与质量门槛断言。

## R11

运行方式：

```bash
cd Algorithms/数学-机器学习-0296-随机森林
python3 demo.py
```

脚本不读取命令行参数，不需要交互输入。

## R12

输出字段说明：
- `Train shape/Test shape`：训练集与测试集维度。  
- `Forest config`：森林树数、树深、每棵树子特征数。  
- `OOB coverage`：至少被一棵 OOB 树预测到的训练样本占比。  
- `OOB accuracy`：OOB 样本上的聚合准确率。  
- `Baseline single tree test accuracy`：单树基线性能。  
- `Random forest test accuracy/precision/recall/F1-score`：森林测试集性能。  
- `Confusion matrix counts`：`TP/TN/FP/FN`。  
- `All checks passed.`：达到内置质量门槛。

## R13

最小测试集与校验（已内置）：
1. 固定种子生成二分类数据，保证结果可复现。  
2. 训练随机森林并输出测试准确率与 F1。  
3. 训练单棵决策树作为对照基线。  
4. 自动断言：
   - 测试准确率必须达到最低阈值；
   - OOB 覆盖率必须足够；
   - OOB 与测试指标必须是有限值；
   - 森林相对单树不能出现明显退化。

## R14

关键参数与调参建议：
- `n_estimators`：树数量。增大通常提升稳定性但增加时间成本。  
- `max_depth`：树深上限。过深易过拟合，过浅易欠拟合。  
- `max_features`：每个节点候选子特征数。较小可增强多样性，较大可增强单树能力。  
- `bootstrap`：是否自助采样。启用后可做 OOB 评估。  
- 数据侧参数（信号幅度、标签翻转率）会显著影响任务难度与指标上限。

## R15

方法对比：
- 对比单棵决策树：
  - 单树偏差低但方差高，稳定性差；
  - 随机森林通过集成显著降方差。  
- 对比 Bagging（不做随机子特征）：
  - 随机森林额外引入特征随机化，进一步去相关。  
- 对比梯度提升树（GBDT/XGBoost）：
  - GBDT 偏序列化加法建模，常有更高上限；
  - 随机森林并行友好、调参更稳健。

## R16

典型应用场景：
- 表格数据分类（风控、营销转化、医疗辅助判断）。  
- 中等规模特征工程后任务，追求稳健基线。  
- 需要相对低调参成本且可输出特征重要性分析的场景。  
- 作为 AutoML 或模型选型流程中的强基线模型。

## R17

可扩展方向：
- 支持多分类与概率校准（Platt/Isotonic）。  
- 增加并行训练（`joblib`/多进程）与更系统的超参数搜索。  
- 引入不平衡学习策略（类权重、重采样）。  
- 增加 permutation importance、SHAP 等解释性模块。  
- 在大数据下接入 `ExtraTrees`、`HistGradientBoosting` 等对照基线。

## R18

`demo.py` 的源码级算法流程（9 步，非黑盒）如下：

1. `main` 调用 `build_dataset` 用固定随机种子生成可复现二分类数据并划分训练/测试集。  
2. 初始化 `SimpleRandomForestClassifier`，设定 `n_estimators/max_depth/max_features/bootstrap`。  
3. `fit` 内先做输入合法性检查，并通过 `_resolve_max_features` 得到每个节点候选特征维度 `k`。  
4. 对每棵树 `t`：执行 bootstrap 抽样得到 `sample_idx`，然后在该样本上训练 `SimpleDecisionTreeClassifier`。  
5. 单树训练时 `_grow` 递归建树；每个节点调用 `_best_split` 在随机 `k` 个特征与分位点阈值上搜索最小加权 Gini 切分。  
6. 若节点满足停止条件（纯节点、样本太少、深度上限），则生成叶子并保存类别概率。  
7. 对每棵树的 OOB 样本直接调用 `predict_proba`，将概率累加到 OOB 缓冲区并统计覆盖次数。  
8. 所有树训练结束后，基于 OOB 缓冲区计算 `oob_coverage` 与 `oob_accuracy`，形成 `ForestFitSummary`。  
9. 推理时森林对所有树概率按算术平均聚合并做 `argmax` 得到标签；`main` 再计算测试集 `accuracy/precision/recall/F1`、对比单树基线并执行质量门槛断言。

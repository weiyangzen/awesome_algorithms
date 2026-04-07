# 网格搜索CV

- UID: `MATH-0406`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `406`
- 目标目录: `Algorithms/数学-机器学习-0406-网格搜索CV`

## R01

网格搜索交叉验证（Grid Search with Cross-Validation, `GridSearchCV`）是一种超参数优化方法：
在离散参数网格上穷举候选组合，对每个组合做 K 折交叉验证，按给定评分函数选出最优组合，并在完整训练集上重新训练最优模型。

## R02

要解决的问题是：
给定模型族 `f(x; θ)` 与离散候选集 `Θ = {θ1, θ2, ..., θm}`，希望找到

`θ* = argmax_{θ in Θ} CVScore(θ)`（若是损失则取最小）。

其中 `CVScore(θ)` 是 K 折验证分数均值，常见指标有 `accuracy`、`f1`、`roc_auc`。

## R03

核心思想：

- 把“调参”转化为“有限候选集合上的系统搜索”。
- 用交叉验证替代单次切分，降低偶然划分带来的方差。
- 用统一评分标准比较不同超参数组合。
- 最后通过 `refit=True` 在全训练集重训，得到可直接部署的最优模型。

## R04

输入：

- 训练特征 `X`、标签 `y`
- 基学习器（如 `SVC`、`RandomForestClassifier`）
- 参数网格 `param_grid`
- 交叉验证方案（如 `StratifiedKFold(5)`）
- 评分函数 `scoring`

输出：

- `best_params_`：最优超参数
- `best_score_`：最优组合的平均 CV 分数
- `best_estimator_`：在完整训练集上重训后的模型
- `cv_results_`：所有候选组合的详细评估表

## R05

本任务 MVP 选择二分类 `F1` 作为主评分指标，原因：

- 适合类别可能不均衡的场景。
- 同时约束精确率和召回率，比单看准确率更稳健。
- 在工程上可直接对应“误报/漏报”的平衡。

## R06

标准流程：

1. 准备数据并划分训练/测试集。
2. 构建预处理 + 模型 `Pipeline`（避免数据泄漏）。
3. 定义离散参数网格。
4. 在训练集上运行 `GridSearchCV.fit(...)`。
5. 读取 `best_params_`、`best_score_`、`cv_results_`。
6. 用 `best_estimator_` 在测试集做最终评估。

## R07

复杂度（粗略）：

- 设参数组合数为 `M`，折数为 `K`，单次训练成本为 `C_train`。
- 总训练成本约为 `O(M * K * C_train)`。
- 若 `refit=True`，还要再加一次 `C_train`。

因此网格越密、模型越重、折数越高，计算量越大。

## R08

优点：

- 全面且可解释，结果复现性强。
- 对小到中等规模参数空间非常可靠。
- `cv_results_` 提供完整对比证据。

缺点：

- 计算成本可能很高。
- 连续参数只能离散化，最优点可能在网格间隙。
- 高维参数空间下组合爆炸明显。

## R09

适用场景：

- 需要稳健模型选择且参数维度不高。
- 需要审计或复盘调参过程（例如教学、科研、合规场景）。
- 基线模型构建阶段，需要先建立可解释的调参上限。

不太适用：

- 参数空间极大、时间预算严格的在线场景。

## R10

MVP 参数设计分两条路径：

- 主路径（环境可用 `scikit-learn` 时）：
- `SVC` 网格：`kernel in {linear, rbf}`、`C in [0.1, 1.0, 10.0, 30.0]`、`gamma in [0.01, 0.1, 1.0]`（仅 `rbf`）
- `cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`，`scoring = f1`
- 后备路径（无 `scikit-learn` 时）：
- 纯 `numpy` KNN 网格：`k in {1,3,5,7,11}`，`weighting in {uniform, distance}`
- 同样使用分层 5 折与 `F1` 评分，保证“网格搜索 + CV”核心机制不变

## R11

与随机搜索（Randomized Search）对比：

- 网格搜索：覆盖完整、对离散网格严格可复现。
- 随机搜索：在同等预算下更适合高维连续空间。
- 工程建议：先用随机搜索粗定位，再在局部用网格细化。

## R12

工程注意点：

- 预处理（标准化）必须放进 `Pipeline`，不能先在全数据上 `fit`。
- 训练/测试划分必须先做，测试集不能参与调参。
- 分类任务建议用分层 K 折，避免类别分布漂移。
- 记录随机种子和版本，保证可重现。

## R13

`demo.py` 的 MVP 特性：

- 使用纯 `numpy` 生成可控二分类数据（固定随机种子，可复现）。
- 自动检测 `scikit-learn`：可用时走 `GridSearchCV` 主路径，不可用时走纯 `numpy` 后备实现。
- 使用 `pandas` 承载结构化结果，便于排序和展示 Top-5 候选。
- 使用 `numpy` 做最优候选 CV 分数的 95% 近似置信区间估计（正态近似）。
- 输出最优参数、CV 分数区间、测试集指标，以及（若可用）`classification_report`。

## R14

运行方式：

```bash
cd Algorithms/数学-机器学习-0406-网格搜索CV
python3 demo.py
```

脚本无需交互输入，单次运行即可产出完整结果。

## R15

结果解读建议：

- 先看 `Best params` 与 `Best mean CV F1`，判断调参是否有效。
- 再看 `Top-5 candidates` 的分差是否显著，避免“偶然第一名”。
- 最后看测试集 `F1` 与默认模型 `F1` 的差值（`F1 gain`），确认泛化收益。

## R16

常见问题：

- 网格过大导致耗时过长：先缩小范围或减少折数做试跑。
- 分数波动大：检查样本量、类别分布、随机种子稳定性。
- 训练分数远高于验证分数：可能过拟合，应收缩模型复杂度。

## R17

可扩展方向：

- 换成回归任务（`GridSearchCV` + `neg_mean_squared_error`）。
- 增加多指标搜索（`scoring` 设为 dict，`refit` 指定主指标）。
- 在大规模场景改用 `HalvingGridSearchCV` 或贝叶斯优化。
- 将 `cv_results_` 持久化为 CSV，形成实验追踪记录。

## R18

`GridSearchCV` 的源码级执行流（非黑箱）可拆成 8 步：

1. 参数展开：`ParameterGrid` 将 `param_grid` 展开成有限候选列表 `[(θ1), (θ2), ...]`。
2. 候选遍历：对每个 `θi` 克隆基础估计器（`clone(estimator)`），保证不同候选互不污染。
3. 数据切分：按 `cv` 生成每一折的 `(train_idx, valid_idx)`。
4. 折内训练：在每折训练子集上 `fit` 克隆模型，并在验证子集上计算 `scoring`。
5. 分数聚合：对每个候选收集 `split*_test_score`，计算 `mean_test_score`、`std_test_score` 并排名。
6. 最优选择：根据主指标和排名规则确定 `best_index_` 与 `best_params_`。
7. 全量重训：若 `refit=True`，用最优参数在完整训练集上再次 `fit` 得到 `best_estimator_`。
8. 结果暴露：把全部统计写入 `cv_results_`，并提供 `predict/score` 直接用于下游评估。

以上 8 步对应了“搜索-验证-选择-重训”的完整闭环。

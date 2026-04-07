# 网格搜索

- UID: `MATH-0362`
- 学科: `数学`
- 分类: `优化`
- 源序号: `362`
- 目标目录: `Algorithms/数学-优化-0362-网格搜索`

## R01

网格搜索（Grid Search）是最常见的离散超参数优化方法之一：
给定每个超参数的候选值集合，先做笛卡尔积得到有限候选组合，再逐个评估并选择最优组合。

本目录提供一个“可运行、可审计”的最小 MVP：
- 不调用任何现成 `GridSearch` 黑盒；
- 手写参数组合枚举、分层 K 折评估与最优参数选择；
- 使用自实现 `KNNClassifier` 展示超参数 `k/p/weighted` 的搜索流程；
- 输出完整候选排名、最优参数和独立测试集表现。

## R02

本实现的问题定义：
- 输入：
  - 估计器 `estimator`（本例为手写 `KNNClassifier`）；
  - 参数网格 `param_grid: dict[str, list]`；
  - 训练特征 `X`、训练标签 `y`；
  - 交叉验证折数 `cv`。
- 目标：
  - 在离散候选集合 `G` 中，找到使交叉验证平均准确率最大的参数组合。
- 输出：
  - `best_params`：最优参数组合；
  - `best_cv_score`：最优组合的平均 CV 分数；
  - `result_df`：所有候选的排名表（均值/方差/区间）；
  - 独立测试集准确率（用于验证泛化效果）。

`demo.py` 内置数据生成与参数网格，无需交互输入。

## R03

数学形式化描述：

1. 设超参数搜索空间为离散集合
`G = V1 x V2 x ... x Vm`，其中 `Vi` 是第 `i` 个超参数候选值集合。

2. 对任意候选参数 `theta in G`，进行 `K` 折交叉验证，记第 `k` 折评分为 `s_k(theta)`。

3. 定义平均评分：
`S(theta) = (1/K) * sum_{k=1..K} s_k(theta)`。

4. 网格搜索目标：
`theta* = argmax_{theta in G} S(theta)`。

本 MVP 中评分函数为分类准确率 `accuracy`。

## R04

算法流程（高层）：
1. 校验 `param_grid` 结构与候选列表合法性。  
2. 通过笛卡尔积展开得到全部候选参数组合。  
3. 对每个候选组合：
   - 使用分层 K 折生成训练/验证索引；
   - 克隆模型并设置当前参数；
   - 在训练折拟合，在验证折计算准确率。  
4. 计算该候选的 `mean/std/min/max` CV 分数。  
5. 与当前最优分数比较并更新全局最优参数。  
6. 汇总全部候选结果并按均分降序排序。  
7. 用最优参数在完整训练集重训模型，并在测试集评估。

## R05

核心数据结构：
- `param_grid: dict[str, sequence]`：输入网格定义。  
- `candidates: list[dict[str, Any]]`：参数笛卡尔展开结果。  
- `fold_scores: list[float]`：单个候选在各折上的分数。  
- `records: list[dict]`：每个候选的统计记录。  
- `result_df: pandas.DataFrame`：最终排名表，字段包括：
  - `rank`、`candidate_id`、`params`、`mean_score`、`std_score`、`min_score`、`max_score`。

## R06

正确性要点：
- 穷举完备性：笛卡尔积确保离散搜索空间内所有候选都被评估。  
- 评估公平性：所有候选使用相同折数与同一随机种子。  
- 折叠分布一致性：分层划分保持各类别在每折中的比例相对稳定。  
- 选择准则清晰：以 `mean_score` 最大为主，`std_score` 仅作为稳定性参考。  
- 泛化核对：最优参数必须在独立测试集再次评估，而非只看 CV 分数。

## R07

复杂度分析：
- 设参数候选总数 `|G|`，交叉验证折数 `K`，单次训练+预测成本为 `C_fit`。  
- 时间复杂度约为：`O(|G| * K * C_fit)`。  
- 空间复杂度：
  - 模型参数与训练缓存（与估计器相关）；
  - 候选记录表 `O(|G|)`。

网格搜索的主要代价是候选数增加引发的组合爆炸。

## R08

边界与异常处理：
- `param_grid` 为空或不是字典：抛 `ValueError`。  
- 某参数候选为空列表：抛 `ValueError`。  
- `X` 不是二维、`y` 不是一维，或样本数不一致：抛 `ValueError`。  
- `X` 含 `nan/inf`：抛 `ValueError`。  
- `cv < 2`：抛 `ValueError`。  
- 任一类别样本数小于折数：抛 `ValueError`。  
- `KNN` 参数非法（如 `k<=0`、`p` 非 1/2）时抛 `ValueError`。

## R09

MVP 取舍说明：
- 使用 `numpy + pandas` 的最小依赖栈，保证脚本自包含可运行；
- 不依赖现成搜索器，核心逻辑完全可追踪；
- 仅实现单指标（accuracy）与单模型（KNN）示例，优先保证透明度；
- 不做并行、早停、分布式调度，保持实现短小且诚实。

## R10

`demo.py` 函数职责：
- `KNNClassifier`：手写 KNN 训练与预测。  
- `validate_param_grid`：检查网格结构合法。  
- `expand_param_grid`：参数网格笛卡尔展开。  
- `stratified_kfold_indices`：手写分层 K 折索引生成。  
- `cross_val_score_manual`：单参数组合的 K 折手动评估。  
- `manual_grid_search`：遍历全部候选并返回最优结果与完整表。  
- `generate_synthetic_multiclass_data`：构造可复现实验数据。  
- `stratified_train_test_split`：手写分层训练/测试切分。  
- `main`：组织数据、执行搜索、输出汇总结果。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0362-网格搜索
python3 demo.py
```

脚本无需命令行参数，不请求用户输入。

## R12

输出字段说明：
- `total samples / feature dim`：数据规模。  
- `train size / test size`：训练与测试样本量。  
- `candidate count`：网格展开后的候选总数。  
- 排名表字段：
  - `rank`：按 CV 均分排序后的名次；
  - `candidate_id`：原始遍历序号；
  - `params`：当前超参数组合；
  - `mean_score`：K 折平均准确率；
  - `std_score`：K 折标准差；
  - `min_score/max_score`：K 折最小/最大准确率。  
- `best params`：最优参数。  
- `best CV mean accuracy`：最优平均交叉验证准确率。  
- `test accuracy (best params)`：最优参数在独立测试集上的准确率。  
- `test accuracy (baseline/default KNN)`：默认参数基线表现。

## R13

建议最小测试集：
1. 正常路径测试（已内置）
- 三分类合成数据 + 20 组参数组合 + 5 折 CV。

2. 异常路径测试（建议补充）
- `param_grid={}`（应报错）。
- 某参数候选为空列表（应报错）。
- `cv=1`（应报错）。
- `X` 含 `np.nan`（应报错）。
- 某类样本少于折数（应报错）。

3. 一致性测试（建议补充）
- 固定随机种子，重复运行应得到一致最优参数与分数。

## R14

关键可调参数：
- `param_grid`：搜索范围来源，直接决定 `|G|`。  
- `cv`：折数，越大通常更稳但计算更慢。  
- `random_state`：控制划分可复现性。  
- `k`：近邻数量，过小高方差、过大高偏差。  
- `p`：距离度量（1 表示 L1，2 表示 L2）。  
- `weighted`：是否按距离加权投票。

调参建议：
- 先用粗网格定位可行区域，再做细网格；
- 当候选爆炸时，优先缩小参数空间或改随机搜索。

## R15

与常见方法对比：
- 对比随机搜索（Random Search）：
  - 网格搜索在低维小空间更可控；
  - 随机搜索在高维时通常更高效。  
- 对比贝叶斯优化：
  - 网格搜索实现最简单、可解释性强；
  - 贝叶斯优化样本效率更高，但实现复杂。  
- 对比手工调参：
  - 网格搜索可系统覆盖并减少主观偏差；
  - 但计算开销更大。

## R16

典型应用场景：
- 机器学习模型的基线超参数优化。  
- 教学场景中展示“离散搜索 + 交叉验证”的核心流程。  
- 小中型数据集上建立可复现实验基准。  
- 模型上线前的参数敏感性初筛。

## R17

可扩展方向：
- 支持多指标打分（如 `accuracy + f1`）与加权决策。  
- 增加并行评估以缩短搜索耗时。  
- 支持条件网格（参数依赖关系）。  
- 增加结果落盘（CSV/JSON）与可视化（热力图、参数曲线）。  
- 集成粗到细分层搜索策略。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 调用 `generate_synthetic_multiclass_data` 生成固定随机种子的三分类数据。  
2. `stratified_train_test_split` 按类别比例划分训练集与测试集。  
3. `manual_grid_search` 调用 `expand_param_grid`，对 `k/p/weighted` 做笛卡尔展开。  
4. 对每个候选，`cross_val_score_manual` 通过 `stratified_kfold_indices` 生成 K 折索引。  
5. 每折克隆 `KNNClassifier`，`set_params` 注入候选参数后执行 `fit` 与 `predict`。  
6. 用 `accuracy_score_np` 计算每折准确率，并汇总为 `mean/std/min/max` 记录到结果表。  
7. `manual_grid_search` 全量遍历后选出 `best_params`，并按评分排序生成 `result_df`。  
8. `main` 用 `best_params` 在完整训练集重训，计算测试集准确率并与默认 KNN 基线对比输出。

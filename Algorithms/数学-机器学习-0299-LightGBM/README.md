# LightGBM

- UID: `MATH-0299`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `299`
- 目标目录: `Algorithms/数学-机器学习-0299-LightGBM`

## R01

LightGBM（Light Gradient Boosting Machine）是面向表格数据的高效梯度提升树框架。其核心思想是：
- 用二阶梯度信息（梯度 + Hessian）驱动每轮树学习；
- 用直方图分箱代替逐点阈值搜索，显著降低分裂搜索成本；
- 采用叶子优先生长（leaf-wise, best-first）提升同等树规模下的拟合能力。

本条目给出一个可运行、可审计的 LightGBM 风格 MVP：不直接调用黑盒训练器，而是在 `demo.py` 中显式实现分箱、增益计算、叶子分裂与 boosting 主循环。

## R02

本条目求解二分类任务，定义为：
- 输入：`X in R^(n*d)`，`y in {0,1}^n`；
- 输出：分类函数 `f(x)`，并给出概率 `P(y=1|x)`。

`demo.py` 使用固定随机种子构造可复现实验数据，自动训练并输出 baseline、训练日志、测试指标和断言结果。

## R03

LightGBM 二分类（log-loss）核心关系：

1. 加法模型：
`F_t(x) = F_{t-1}(x) + eta * h_t(x)`

2. 概率映射：
`p(x) = sigmoid(F(x))`

3. 一阶/二阶信息：
- `g_i = p_i - y_i`
- `h_i = p_i * (1 - p_i)`

4. 叶子最优输出（带 L2 正则）：
`w_leaf = - sum(g_i) / (sum(h_i) + lambda)`

5. 分裂增益（近似目标下降）：
`gain = 0.5 * (G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)) - gamma`

其中 `G/H` 表示某节点梯度/Hessian 汇总，`gamma` 对应最小分裂收益门槛。

## R04

算法主流程（高层）：

1. 对训练特征做分位数分箱（`max_bin`），把连续特征映射到离散 bin。  
2. 初始化常数打分 `F0 = log(pos_rate/(1-pos_rate))`。  
3. 每轮根据当前 `F` 计算 `p`，再计算 `g/h`。  
4. 用 `g/h` 训练一棵直方图树：各叶节点搜索最佳特征+bin 切分。  
5. 采用 leaf-wise 方式，每次只分裂“全局增益最高”的叶子，直到达到 `num_leaves` 或无有效切分。  
6. 树训练后得到每个叶子的牛顿步输出值，整棵树可对样本给出增量。  
7. 更新 `F <- F + learning_rate * tree_output`。  
8. 重复迭代并在最后输出分类概率与标签。

## R05

`demo.py` 中的核心数据结构：

- `LightGBMConfig`：统一管理 `n_estimators/num_leaves/max_bin/lambda_l2` 等超参数。  
- `QuantileBinMapper`：保存每个特征的分箱边界并执行 `fit/transform`。  
- `TreeNode`：树节点结构（叶子值、分裂特征、分裂 bin、左右子节点）。  
- `SplitInfo`：一次候选分裂的结果容器（增益、阈值、左右样本索引）。  
- `LeafWiseHistogramTree`：单棵 LightGBM 风格树，负责直方图统计、增益搜索、leaf-wise 生长。  
- `SimpleLightGBMBinaryClassifier`：提升框架外层，负责多轮训练、推理、日志与指标。

## R06

正确性关键点：

- 使用 `grad = p - y` 与 `hess = p(1-p)`，对应二分类对数损失的二阶展开方向。  
- 叶子输出采用 `-G/(H+lambda)`，确保每叶是局部牛顿更新而非随意常数。  
- 分裂时强制 `min_data_in_leaf` 与 `min_sum_hessian_in_leaf`，避免极端小叶子造成不稳定。  
- 每轮树输出通过 `learning_rate` 缩放，防止步长过大导致训练震荡。  
- 最终通过 baseline 对比 + 断言门槛验证训练确实学到有效模式。

## R07

复杂度分析（`n` 样本、`d` 特征、`B` 分箱数、`T` 轮、`L` 最大叶子数）：

- 分箱阶段：每列分位数统计与映射约 `O(n*d)` 到 `O(n*d log n)`（依实现细节）。  
- 单树分裂搜索：每次分裂在候选特征上扫描直方图，近似 `O(d*B)`；总计约 `O(L*d*B)`。  
- 全部训练：约 `O(T*L*d*B + T*n)`（后者是每轮梯度与预测更新）。  
- 推理：单样本约 `O(T * tree_depth)`，通常远小于全特征线性扫描。

MVP 重点是算法路径透明，不追求极限工程优化。

## R08

边界与异常处理：

- `n_estimators<=0`、`num_leaves<2`、非法 `feature_fraction/bagging_fraction` 会抛 `ValueError`。  
- 输入 `x` 维度错误、含 `NaN/Inf`、标签非 `{0,1}` 会抛 `ValueError`。  
- 未 `fit` 时调用 `predict/predict_proba/decision_function` 会抛 `RuntimeError`。  
- 常数特征自动退化为单 bin，不参与有效切分。  
- Hessian 在实现中做下界裁剪（`>=1e-6`）以防数值不稳定。

## R09

MVP 范围与取舍：

- 已实现：
  - histogram 分箱；
  - leaf-wise 最优叶分裂；
  - 二阶增益与叶子牛顿值；
  - `feature_fraction` 与可选 `bagging_fraction`。

- 未实现（官方 LightGBM 的高级工程项）：
  - GOSS（Gradient-based One-Side Sampling）；
  - EFB（Exclusive Feature Bundling）；
  - 多线程直方图复用、GPU 训练、类别特征专门编码等。

因此该实现是“LightGBM 风格教学 MVP”，而非官方完整工业版本。

## R10

`demo.py` 关键函数职责：

- `QuantileBinMapper.fit/transform`：构建并应用分位数分箱。  
- `LeafWiseHistogramTree._find_best_split`：在候选特征上按 bin 前缀累计梯度/Hessian，搜索最佳切分增益。  
- `LeafWiseHistogramTree.fit`：执行 leaf-wise best-first 分裂并确定最终叶子值。  
- `SimpleLightGBMBinaryClassifier.fit`：多轮 boosting 训练主循环。  
- `decision_function/predict_proba/predict`：完成 raw score 到概率到标签的推理链路。  
- `main`：数据生成、baseline 对比、训练、评估、断言闭环。

## R11

运行方式：

```bash
cd Algorithms/数学-机器学习-0299-LightGBM
python3 demo.py
```

脚本不需要命令行参数与交互输入，运行结束会打印 `All checks passed.`。

## R12

本次实现的典型输出解释：

- `baseline_*`：常数模型（仅使用训练正例率）在 train/test 上的对数损失与精度。  
- `iter=... train_logloss=...`：每若干轮打印一次训练集 log-loss，观察是否稳定下降。  
- `train/test_logloss`：最终损失。  
- `train/test_accuracy`：最终精度。  
- `test_precision/recall/f1`：分类质量细分。  
- `confusion_counts`：`TP/TN/FP/FN`。  
- `All checks passed.`：内置质量门槛全部通过。

## R13

最小验证闭环（已内置）：

1. 固定随机种子生成非线性二分类数据。  
2. 计算常数 baseline 作为对照。  
3. 训练 LightGBM 风格模型并输出迭代损失。  
4. 在测试集计算 `accuracy/precision/recall/f1`。  
5. 自动断言：
   - 训练/测试 loss 均显著优于 baseline；
   - 测试精度较 baseline 有明显提升；
   - 概率有限值、F1 达标。

## R14

调参建议：

- `n_estimators`：轮数增大通常可降偏差，但会提升训练时间与过拟合风险。  
- `learning_rate`：减小可提高稳定性，通常要配合更多轮数。  
- `num_leaves`：增大提升单树表达能力，但也更易过拟合。  
- `max_bin`：分箱更细可提升分裂精度，但计算与内存代价上升。  
- `min_data_in_leaf`：增大可抑制噪声叶子，提升泛化稳健性。  
- `feature_fraction/bagging_fraction`：可降低相关性与过拟合，兼顾速度。

## R15

与相关方法对比：

- 对比经典 GBDT（level-wise）：
  - LightGBM 常用 leaf-wise，往往在相同叶子预算下拟合更强；
  - 但 leaf-wise 若约束不足，过拟合风险也更高。

- 对比 XGBoost：
  - 两者都用二阶信息与正则化；
  - LightGBM 以 histogram + leaf-wise + 多项工程优化著称，常在大规模表格任务中训练更快。

- 对比随机森林：
  - 随机森林偏并行 Bagging 降方差；
  - LightGBM 偏串行 Boosting 降偏差，通常精度上限更高但调参更敏感。

## R16

适用场景：

- 中大型表格数据（二分类/多分类/回归）基线与强模型。  
- 特征非线性、交互关系明显且希望较高性能的业务任务。  
- 需要较快迭代训练并支持后续特征重要性/解释分析的场景。  
- 风控、营销转化、CTR 预估、运营评分、工业质量判定等结构化数据问题。

## R17

可扩展方向：

- 增加 GOSS 与 EFB，进一步贴近官方 LightGBM。  
- 支持多分类目标和自定义损失函数。  
- 引入早停、验证集监控与超参数搜索。  
- 增加类别特征原生处理与缺失值专门策略。  
- 在工程层面加入并行化、模型持久化和特征重要性导出。

## R18

`demo.py` 的源码级流程（9 步，非黑盒）如下：

1. `main` 调用 `make_dataset` 生成可复现二分类样本，并计算常数 baseline 指标。  
2. `SimpleLightGBMBinaryClassifier.fit` 先用 `QuantileBinMapper.fit_transform` 把连续特征映射为离散 bin。  
3. 按标签正例率初始化全体样本 raw score：`F0 = log(pos/(1-pos))`。  
4. 每轮迭代先计算 `p=sigmoid(F)`，再得到 `grad=p-y` 与 `hess=p(1-p)`。  
5. `LeafWiseHistogramTree.fit` 从根叶开始，对每个候选叶调用 `_find_best_split`：通过 `np.bincount` 汇总每个 bin 的梯度/Hessian/计数。  
6. 对每个特征遍历阈值 bin，按增益公式计算 `gain`，并选择当前全局增益最高的叶子执行分裂（leaf-wise best-first）。  
7. 达到 `num_leaves` 或无正增益分裂后，按每个叶子的 `-G/(H+lambda_l2)` 写入叶子输出值。  
8. 树训练完成后对全体训练样本给出增量，执行 `F <- F + learning_rate * tree_output`，并记录训练 log-loss。  
9. 全部树训练结束后，`predict_proba` 复用同一分箱与树遍历逻辑输出概率，`main` 计算测试集指标并执行断言验收。

说明：该实现没有调用第三方一行式 LightGBM 训练接口，核心流程（分箱、增益、leaf-wise、生长、牛顿叶值、boosting 更新）均在源码中显式展开。

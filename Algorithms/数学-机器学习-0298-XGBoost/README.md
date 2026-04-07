# XGBoost

- UID: `MATH-0298`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `298`
- 目标目录: `Algorithms/数学-机器学习-0298-XGBoost`

## R01

XGBoost（Extreme Gradient Boosting）是梯度提升树（GBDT）的工程化与正则化增强版本。核心思想是：
- 仍然采用加法模型 `F(x)=sum_t f_t(x)`；
- 每一轮新树 `f_t` 用当前损失的一阶/二阶信息（梯度与 Hessian）训练；
- 在目标函数中显式加入树结构复杂度与叶子权重惩罚，从而得到更稳健的泛化效果。

本目录 `demo.py` 给出一个“可运行、非黑盒、最小但诚实”的 XGBoost 风格二分类实现。

## R02

算法背景与定位：
- GBDT 的思想来自 Friedman（2001）的 gradient boosting 框架；
- XGBoost 由 Tianqi Chen 等提出并工程化（2014 起，系统论文 2016），关键贡献是二阶优化、正则化目标、高效分裂搜索与系统级优化；
- 工业中它是表格数据（structured/tabular）任务的强基线之一，尤其在中小规模到中大型数据上表现稳定。

## R03

本条目解决的问题（以二分类为例）：
- 输入：`X in R^(n*d)`，标签 `y in {0,1}^n`；
- 输出：由多棵 CART 树组成的加法模型，最终输出概率 `p(y=1|x)`；
- 目标：最小化带正则项的 log-loss，并在训练速度、表达能力、泛化能力之间取得平衡。

## R04

XGBoost 的目标函数（第 `t` 轮）可写为：

`Obj_t = sum_i l(y_i, yhat_i^{(t-1)} + f_t(x_i)) + Omega(f_t)`

其中树复杂度惩罚：

`Omega(f) = gamma * T + (lambda / 2) * sum_{j=1..T} w_j^2`

- `T`：叶子节点数；
- `w_j`：第 `j` 个叶子的分值；
- `gamma`：新增叶子的结构惩罚；
- `lambda`：叶子权重 L2 正则。

通过对损失做二阶泰勒展开，训练时使用：
- 一阶项 `g_i = d l / d yhat_i`；
- 二阶项 `h_i = d^2 l / d yhat_i^2`。

## R05

对某个叶子 `j`，记 `G_j = sum g_i`、`H_j = sum h_i`，有：

- 最优叶子权重：`w_j* = -G_j / (H_j + lambda)`
- 分裂增益（父节点拆成左右子节点）：

`Gain = 0.5 * ( G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda) ) - gamma`

这就是 XGBoost 风格“按 gain 选最优分裂”的核心公式。`demo.py` 按该公式实现了分裂搜索。

## R06

复杂度（本 MVP，朴素阈值搜索）粗略为：
- 设树数 `M`、样本数 `n`、特征数 `d`、每特征候选阈值数 `q`、树深上限 `D`；
- 单棵树约 `O(n * d * q * D)`（未使用直方图/预排序缓存优化）；
- 总训练约 `O(M * n * d * q * D)`；
- 推理单样本约 `O(M * D)`。

空间上主要是数据矩阵与树结构存储。

## R07

优点：
- 对非线性和特征交互建模能力强；
- 相比普通一阶 GBDT，二阶信息让更新方向更“曲率感知”；
- 正则项（`lambda/gamma`）可显式抑制过拟合。

局限：
- 超参数较多，调参成本高于线性模型；
- 树模型在外推（extrapolation）任务上通常不如参数化函数；
- 高噪声或极小样本场景下容易出现不稳定分裂。

## R08

前置知识与依赖链：
1. 梯度提升（加法模型逐轮优化）；
2. logistic 损失与 sigmoid 映射；
3. 一阶/二阶导数在优化中的意义；
4. 决策树二叉切分与叶子常数预测；
5. 正则化思想（复杂度惩罚与权重惩罚）。

本 MVP 依赖：
- `numpy`（必需）；
- Python 3.10+（类型注解使用 `|` 联合类型）。

## R09

适用场景：
- 结构化表格数据分类/回归；
- 特征间存在非线性与交互关系；
- 需要强基线模型并兼顾可解释的树路径结构。

不适用或需谨慎：
- 极端高维稀疏文本（线性/专门方法可能更优）；
- 强时间外推任务；
- 标注噪声极大且缺少足够样本时。

## R10

实现正确性的关键检查点：
1. 每轮梯度/二阶梯度计算是否与损失一致（本实现 `g=p-y`, `h=p*(1-p)`）。
2. 分裂增益是否严格按 `Gain` 公式计算。
3. 叶子权重是否使用 `-G/(H+lambda)`。
4. 模型更新是否执行 `raw += learning_rate * tree_pred`。
5. 训练损失是否总体下降，且优于常数先验 baseline。

`demo.py` 末尾包含自动断言，作为最小验收门槛。

## R11

数值稳定性处理：
- `sigmoid` 前对 raw score 做 `[-50, 50]` 裁剪，防止 `exp` 溢出；
- `log-loss` 概率做 `clip(eps, 1-eps)`，防止 `log(0)`；
- Hessian `h` 下界裁剪到 `1e-8`，避免分母接近 0；
- `pos_rate` 裁剪后再取 logit，避免初始化出现无穷值。

## R12

真实成本与调参经验：
- `n_estimators` 增大通常提升拟合能力，但训练更慢且过拟合风险更高；
- `learning_rate` 降低可提高稳定性，但常需更多树；
- `max_depth` 决定单树复杂度，上去太快容易记忆噪声；
- `min_child_weight`、`min_samples_leaf` 提升可抑制碎片化分裂；
- `subsample` 与 `colsample_bytree` 可降低树间相关性并提升泛化。

实践常用策略：较小 `learning_rate` + 适中树深 + 足够轮数，再配合早停或验证集监控。

## R13

理论保证说明：
- 近似比保证：N/A（XGBoost 不是组合优化近似算法）；
- 概率成功保证：N/A（不是随机化近似算法那类成功概率模型）；
- 可给出的工程保证是：在同一数据与参数下，目标优化过程可重复，且训练损失通常单调下降或总体下降。

## R14

常见失效模式：
1. `max_depth` 过大 + 树数过多，测试集性能下降（过拟合）。
2. `learning_rate` 过大，训练后期震荡或不稳定。
3. `min_child_weight` 太小，产生大量低质量叶子。
4. `gamma` 太大导致几乎不分裂，出现欠拟合。
5. 样本分布偏移（train/test drift）时，离线指标高但线上效果退化。

## R15

本目录实现策略（`demo.py`）：
- 不调用任何外部 XGBoost 包，核心流程全部手写；
- 每轮基于当前 `raw_scores` 计算 `g/h`；
- 树分裂使用分位点阈值候选（简化版 exact/approx split）；
- 支持行采样（`subsample`）与列采样（`colsample_bytree`）；
- 使用固定随机种子生成数据，保证可复现。

这是教学 MVP，不包含直方图加速、并行块存储、缺失值方向学习等生产级优化。

## R16

谱系与应用位置：
- 谱系：`GBDT -> XGBoost -> LightGBM/CatBoost`（后两者在分裂策略与类别特征处理上继续演进）。
- 与随机森林关系：随机森林偏并行 Bagging 降方差，XGBoost 偏串行 Boosting 降偏差。
- 应用：风控评分、广告点击率、营销转化预测、信用违约评估、医疗风险分层等表格任务。

## R17

运行方式：

```bash
cd Algorithms/数学-机器学习-0298-XGBoost
python3 demo.py
```

运行边界与输出：
- 无交互输入；
- 自动生成可复现二分类数据并训练；
- 输出 baseline、训练过程 logloss、最终 train/test 指标；
- 内置断言通过后打印 `All checks passed.`。

依赖：
- `numpy`
- `python3`

## R18

`demo.py` 的源码级算法流程（8 步，非黑盒）如下：

1. `main` 调用 `make_dataset` 生成固定随机种子的二分类样本，并做分层训练/测试切分。  
2. `SimpleXGBoostBinary.fit` 用训练集正例比例初始化 `base_score = logit(mean(y))`，得到初始 raw score 向量。  
3. 每一轮先计算 `p=sigmoid(raw)`，再得到 `g=p-y` 与 `h=p*(1-p)`，这是 logistic 目标的一阶/二阶信息。  
4. 进行行采样与列采样，构造本轮训练树使用的样本索引与特征子集。  
5. `XGBTree._best_split` 在候选特征与分位点阈值上枚举，按 `Gain` 公式选择增益最大的有效分裂。  
6. `XGBTree._build` 递归建树；若触发停止条件则用 `w*=-G/(H+lambda)` 生成叶子权重。  
7. 本轮树训练完成后，对全体样本预测增量并执行 `raw += learning_rate * update`，随后记录训练 logloss。  
8. 所有轮次结束后，`predict_proba` 通过 `sigmoid(base + sum eta*tree(x))` 输出概率，`main` 对比 baseline 并执行质量断言。

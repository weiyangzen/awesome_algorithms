# AdaBoost

- UID: `CS-0097`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `217`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0217-AdaBoost`

## R01

AdaBoost（Adaptive Boosting）是一种经典集成学习算法，核心思想是把许多“略好于随机猜测”的弱分类器按权重加权，组合成一个强分类器。它通过迭代地调整训练样本权重，让后续弱分类器重点关注前一轮分错的样本。

本条目给出一个可运行的最小 MVP：

- 用 `numpy` 从零实现二分类 AdaBoost；
- 弱学习器使用“单特征阈值 + 极性”的决策树桩（decision stump）；
- 用 `scikit-learn` 内置乳腺癌数据集做离线训练与评估；
- 输出每轮权重更新轨迹和最终性能指标。

## R02

二分类 AdaBoost（标签 `y_i in {-1, +1}`）的目标是构建加性模型：

`F(x) = sum_{t=1..T} alpha_t * h_t(x)`

最终分类器：

`H(x) = sign(F(x))`

其中 `h_t(x)` 是第 `t` 轮弱分类器，`alpha_t` 是其投票权重，`T` 是弱分类器数量。

## R03

每轮训练过程（离散 AdaBoost）包含三件事：

- 在当前样本权重分布 `w_t` 下，训练一个弱分类器 `h_t`；
- 计算加权错误率 `epsilon_t = sum_i w_t(i) * I(h_t(x_i) != y_i)`；
- 根据错误率计算分类器权重并更新样本权重。

其中样本权重更新为：

`w_{t+1}(i) = w_t(i) * exp(-alpha_t * y_i * h_t(x_i)) / Z_t`

`Z_t` 是归一化常数，保证权重和为 1。

## R04

弱分类器权重的闭式解：

`alpha_t = 0.5 * eta * ln((1 - epsilon_t) / epsilon_t)`

这里 `eta` 是学习率（`learning_rate`）。当 `epsilon_t < 0.5` 时，`alpha_t > 0`，该弱分类器对最终模型有正贡献；错误率越低，`alpha_t` 越大。

实践中会对 `epsilon_t` 做裁剪（如 `clip` 到 `[1e-12, 1-1e-12]`）以避免数值溢出。

## R05

本实现的弱分类器是决策树桩：

- 选择一个特征 `j`；
- 选择一个阈值 `theta`；
- 选择一个极性 `p in {+1, -1}`。

预测规则可写为：

- 若 `p = +1`：`x_j < theta` 预测为 `-1`，否则 `+1`；
- 若 `p = -1`：`x_j >= theta` 预测为 `-1`，否则 `+1`。

通过穷举特征、阈值、极性，最小化加权分类误差。

## R06

为何 AdaBoost 有效：

- 每轮都在当前“困难样本”上再学习，形成逐步纠错机制；
- 最终模型是多个简单边界的加权组合，可拟合更复杂边界；
- 若弱学习器稳定优于随机，训练误差通常会快速下降。

需要注意：AdaBoost 对噪声和离群点较敏感，因为这些样本会被持续放大权重。

## R07

`demo.py` 的数据与实验设置：

- 数据集：`sklearn.datasets.load_breast_cancer`（二分类）；
- 划分：`train_test_split(..., test_size=0.25, stratify=y, random_state=42)`；
- 标签映射：把 `{0,1}` 映射到 `{-1,+1}`；
- 训练轮数：默认 `n_estimators=30`；
- 评估指标：训练/测试准确率、`classification_report`。

该脚本离线运行，不依赖网络下载与交互输入。

## R08

实现中的正确性检查点：

- 每轮更新后强制 `sum(weights) = 1`；
- 裁剪加权误差避免 `log(0)`；
- 预测输出统一为 `{-1,+1}`，与指数损失公式一致；
- 保存每轮 `weighted_error`、`alpha`、`weight_min/max` 便于诊断；
- 与单个树桩基线对比，确认 Boosting 确实带来增益。

## R09

复杂度分析（训练集大小 `n`，特征维度 `d`，迭代轮数 `T`）：

- 单个树桩训练大约是 `O(d * n * U)`，`U` 为每个特征候选阈值数；
- 若阈值取自唯一值中点，最坏 `U = O(n)`，单轮约 `O(d * n^2)`；
- 总训练复杂度约 `O(T * d * n^2)`（本 MVP 追求可解释性而非极致优化）。

推理复杂度是 `O(T * d_stump)`，其中每个树桩仅访问一个特征，实践中很快。

## R10

`demo.py` 关键模块映射：

- `DecisionStump`：加权树桩训练与预测；
- `AdaBoostBinary`：集成训练、权重更新、累积决策函数；
- `to_pm_one` / `from_pm_one`：标签空间转换；
- `build_rounds_dataframe`：整理训练轮次日志（`pandas.DataFrame`）；
- `main`：数据准备、训练评估、断言检查、结果打印。

## R11

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0217-AdaBoost
uv run python demo.py
```

脚本应直接结束并打印指标，不需要人工输入。

## R12

典型输出解读：

- `Dataset shape`：样本数与特征数；
- `Train/Test size`：训练与测试划分规模；
- `Top boosting rounds`：每轮 `weighted_error`、`alpha`、样本权重范围；
- `Single stump baseline accuracy`：单弱分类器基线；
- `AdaBoost train/test accuracy`：集成模型效果；
- `All checks passed.`：最小有效性断言通过。

## R13

内置最小实验目标：

1. 在真实二分类数据上完成从零 AdaBoost 训练；
2. 验证训练过程中的权重动态与 `alpha` 权重是合理数值；
3. 证明集成效果不低于单树桩基线；
4. 保证测试准确率达到实用下限（示例中设为 `>= 0.90`）。

## R14

关键超参数与影响：

- `n_estimators`：轮数越多，拟合能力更强，但过大可能过拟合噪声；
- `learning_rate`：缩放每轮 `alpha`，减小后训练更稳但收敛更慢；
- `epsilon clip`：过小会数值不稳定，过大则削弱高质量弱分类器权重；
- 阈值搜索策略：全量阈值更精确，分位数阈值更快。

## R15

与相关方法对比：

- 对比 Bagging/随机森林：Bagging 主要降方差，样本权重通常固定；AdaBoost 主动重加权，偏向降偏差；
- 对比 GBDT：GBDT 在函数空间做梯度下降，通常拟合残差；AdaBoost 可看作指数损失下的前向分步加法模型；
- 对比单模型：AdaBoost 通过加权投票显著增强弱学习器表达能力。

## R16

适用场景：

- 中小规模结构化二分类任务；
- 需要快速可解释基线（可查看每个树桩对应特征与阈值）；
- 资源受限但希望比单一浅模型更强的场景。

不太适合：高噪声标签、超大规模特征搜索未优化的场景。

## R17

可扩展方向：

- 将树桩替换为浅层决策树（提高单轮表达力）；
- 增加样本权重裁剪或早停策略抑制噪声影响；
- 支持多分类版本（SAMME / SAMME.R）；
- 用更高效的阈值扫描（排序后线性扫描）降低单轮复杂度；
- 增加交叉验证与超参数搜索。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 加载乳腺癌数据并拆分训练/测试集，把标签从 `{0,1}` 转成 `{-1,+1}`。
2. `AdaBoostBinary.fit` 初始化样本权重为均匀分布 `w_i = 1/n`。
3. 每一轮调用 `DecisionStump.fit`：枚举特征、阈值和极性，选择当前加权错误率最小的树桩。
4. 用该树桩预测训练集，计算加权错误率 `epsilon_t`，并做数值裁剪。
5. 根据 `epsilon_t` 计算弱分类器系数 `alpha_t = 0.5 * eta * ln((1-epsilon_t)/epsilon_t)`。
6. 按 `w_i <- w_i * exp(-alpha_t * y_i * h_t(x_i))` 更新样本权重，再归一化使其和为 1。
7. 保存每轮日志（`weighted_error`、`alpha`、`weight_min/max`），并把树桩与系数加入集成器。
8. 训练结束后用 `predict` 聚合 `sum(alpha_t * h_t(x))` 并取符号，输出准确率、分类报告和断言检查结果。

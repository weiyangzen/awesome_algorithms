# 多层感知机 (MLP)

- UID: `CS-0109`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `238`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0238-多层感知机_(MLP)`

## R01

多层感知机（MLP, Multi-Layer Perceptron）是最基础的前馈神经网络之一：
- 输入层接收特征；
- 若干隐藏层通过线性变换 + 非线性激活提取表征；
- 输出层给出类别概率或回归值。

本目录给出一个可运行、可审计的最小实现：
- 用 `numpy` 手写前向传播、反向传播、参数更新；
- 用 `scikit-learn` 仅做数据生成、划分和指标计算；
- 不调用深度学习框架中的黑盒 MLP 训练接口。

## R02

本条目解决二分类任务：
- 输入：`X ∈ R^(n×d)`，标签 `y ∈ {0,1}^n`；
- 输出：训练后的分类器 `f(x)`，能给出 `predict_proba` 和 `predict`。

`demo.py` 使用固定随机种子的合成双月数据集，运行无需任何交互输入。

## R03

核心数学关系：

1. 线性层：`z^(l) = a^(l-1) W^(l) + b^(l)`。  
2. 隐藏层激活：`a^(l) = ReLU(z^(l)) = max(0, z^(l))`。  
3. 输出层概率：`p = softmax(z^(L))`。  
4. 交叉熵损失：`L_ce = - (1/m) * Σ_i Σ_c y_ic log p_ic`。  
5. L2 正则：`L = L_ce + (λ/2) * Σ_l ||W^(l)||_2^2`。  
6. 输出层梯度：`δ^(L) = (p - y_onehot) / m`。  
7. 隐层反传：`δ^(l) = (δ^(l+1) (W^(l+1))^T) ⊙ ReLU'(z^(l))`。  
8. 参数梯度：`∂L/∂W^(l) = (a^(l-1))^T δ^(l) + λW^(l)`，`∂L/∂b^(l) = Σ δ^(l)`。

## R04

算法主流程：

1. 构造数据并做训练/测试划分。  
2. 用训练集均值方差做标准化。  
3. 按层宽初始化权重（He 初始化）与偏置。  
4. 每个 epoch 打乱样本并按 mini-batch 训练。  
5. 每个 batch 先前向传播得到概率与损失。  
6. 再反向传播得到每层梯度。  
7. 使用学习率做梯度下降更新参数。  
8. 训练结束后在测试集计算准确率与 F1，并执行质量门槛断言。

## R05

核心数据结构：
- `TrainSummary`：记录 `epochs/batch_size/learning_rate/l2/final_loss`。  
- `NumpyMLPClassifier.weights_`：每层权重矩阵列表。  
- `NumpyMLPClassifier.biases_`：每层偏置向量列表。  
- `NumpyMLPClassifier.classes_`：标签集合（用于 index→原标签映射）。  
- `NumpyMLPClassifier.loss_curve_`：每个 epoch 的平均训练损失。

## R06

正确性关键点：
- `softmax` 使用减去每行最大值的数值稳定写法，避免指数溢出。  
- 损失函数与输出层梯度匹配（softmax + cross-entropy），梯度形式简洁且稳定。  
- 反向传播严格按层逆序链式求导，隐藏层使用 `ReLU'` 门控。  
- L2 正则同时作用于损失和权重梯度，约束参数规模。  
- 训练后通过 `accuracy/F1/有限值检查` 三重门槛进行最小正确性验证。

## R07

设样本数为 `n`，批大小为 `b`，网络层宽序列为 `d0,d1,...,dL`（`d0` 为输入维度，`dL` 为类别数）。

- 单个 batch 前向+反向复杂度约为：`O(Σ_l b * d_(l-1) * d_l)`。  
- 单个 epoch 复杂度约为：`O((n/b) * Σ_l b * d_(l-1) * d_l) = O(n * Σ_l d_(l-1) * d_l)`。  
- 总训练复杂度约为：`O(epochs * n * Σ_l d_(l-1) * d_l)`。  
- 参数空间复杂度约为：`O(Σ_l d_(l-1) * d_l + Σ_l d_l)`。

## R08

异常与边界处理：
- 超参数非法（如负学习率、空隐藏层、非正批大小）直接抛 `ValueError`。  
- 输入维度不匹配、`NaN/Inf`、类别数不足会抛 `ValueError`。  
- 未训练即调用 `predict/predict_proba` 会抛 `RuntimeError`。  
- 标准化时对极小标准差特征使用 `1.0` 防止除零。  
- 末尾断言检查最终损失与概率是否有限，并约束最低测试指标。

## R09

MVP 取舍：
- 目标是“可运行 + 可解释”，不追求工业级训练框架。  
- 仅覆盖二分类和全连接前馈网络，不扩展卷积、循环、注意力结构。  
- 优先源码透明：训练逻辑完全展开，不调用 `sklearn.neural_network.MLPClassifier` 或 `torch` 黑盒训练。  
- 只保留最必要工程能力：固定种子、可复现数据、自动质量门槛。

## R10

`demo.py` 代码映射：
- `relu/relu_grad/softmax`：基础激活与输出函数。  
- `NumpyMLPClassifier._init_parameters`：按层创建权重和偏置。  
- `NumpyMLPClassifier._forward`：完成多层前向传播。  
- `NumpyMLPClassifier._compute_loss`：交叉熵 + L2。  
- `NumpyMLPClassifier._backward`：逐层反向求梯度。  
- `NumpyMLPClassifier._update`：梯度下降更新。  
- `NumpyMLPClassifier.fit`：训练主循环。  
- `build_dataset`：生成并标准化双月数据。  
- `main`：训练、评估、打印结果与断言。

## R11

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0238-多层感知机_(MLP)
uv run python demo.py
```

脚本不读取命令行参数，不需要交互输入。

## R12

输出字段说明：
- `Train shape/Test shape`：训练集与测试集样本规模。  
- `Config`：隐藏层结构、学习率、正则项、epoch、批大小。  
- `Final training loss`：最后一个 epoch 的平均训练损失。  
- `Test accuracy`：测试集准确率。  
- `Test F1 (binary)`：测试集二分类 F1。  
- `Sample probabilities`：前 3 个测试样本的类别概率。  
- `All checks passed.`：全部内置质量门槛达标。

## R13

内置最小验证：
1. 固定随机种子生成数据，保证可复现。  
2. 训练完成后输出 `accuracy` 和 `F1`。  
3. 检查 `final_loss` 与预测概率是否有限值。  
4. 设置性能门槛：`accuracy >= 0.88` 且 `F1 >= 0.88`。

## R14

关键超参数与影响：
- `hidden_sizes`：控制模型容量；过小欠拟合，过大更易过拟合。  
- `learning_rate`：收敛速度与稳定性权衡。  
- `l2`：权重衰减强度，抑制过大参数。  
- `epochs`：训练轮数，影响拟合程度与耗时。  
- `batch_size`：梯度噪声与吞吐折中。

## R15

与常见方法对比：
- 对比逻辑回归：MLP 通过隐藏层可学习非线性决策边界。  
- 对比树模型：MLP 更依赖特征缩放与超参数，但在平滑连续边界上表现通常更好。  
- 对比深层网络：本实现层数浅、结构简洁，适合作为教学和算法最小原型，而非 SOTA 方案。

## R16

典型应用场景：
- 结构化数据上的非线性分类基线。  
- 小中规模数据集的快速神经网络原型验证。  
- 深度学习入门中的前向/反向传播教学样例。  
- 需要可读、可审计源码训练流程的研究/课程作业场景。

## R17

可扩展方向：
- 支持多分类以外的回归任务（修改输出层与损失）。  
- 加入 `Adam/RMSProp` 等优化器。  
- 引入 `dropout`、`batch norm` 提升泛化与训练稳定性。  
- 增加早停、学习率调度与验证集监控。  
- 提供 `PyTorch` 对照实现用于速度与工程可维护性比较。

## R18

`demo.py` 的源码级流程（8 步，非黑盒）如下：

1. `build_dataset` 调用 `make_moons` 生成固定随机种子的二分类数据，并做分层切分与标准化。  
2. `main` 初始化 `NumpyMLPClassifier`，设置隐藏层结构、学习率、L2、epoch 和 batch 大小。  
3. `fit` 中先校验输入合法性，构建 `classes_` 映射，再通过 `_init_parameters` 按层初始化 `weights_/biases_`。  
4. 每个 epoch 内打乱训练样本并按 mini-batch 迭代。  
5. 每个 batch 在 `_forward` 中逐层执行 `线性 -> ReLU`，最后经 `softmax` 得到类别概率。  
6. `_compute_loss` 计算 `交叉熵 + L2`，`_backward` 根据链式法则从输出层到输入层计算每层梯度。  
7. `_update` 用梯度下降更新全部权重和偏置，`fit` 记录每个 epoch 的平均损失到 `loss_curve_`。  
8. 训练完成后 `main` 调用 `predict/predict_proba` 计算测试 `accuracy/F1`，并执行有限值与性能阈值断言。

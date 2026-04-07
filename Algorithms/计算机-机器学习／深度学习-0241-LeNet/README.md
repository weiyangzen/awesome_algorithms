# LeNet

- UID: `CS-0111`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `241`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0241-LeNet`

## R01

LeNet（通常指 LeNet-5）是经典卷积神经网络，用于手写数字识别等图像分类任务。其核心设计思想是：

- 用卷积层提取局部空间特征；
- 用池化层做下采样并增强平移鲁棒性；
- 用全连接层完成最终分类。

它是后续 AlexNet/VGG/ResNet 等现代 CNN 的结构起点之一。

## R02

对于多分类任务，本实现优化目标是交叉熵损失：

`L = - (1/N) * Σ_i log softmax(z_i)[y_i]`

其中：

- `z_i` 是模型输出 logits；
- `y_i` 是真实类别；
- `N` 是 batch 大小。

训练过程使用反向传播与 Adam 优化器更新参数。

## R03

本目录 MVP 的输入与输出：

- 输入数据：`sklearn.datasets.load_digits()`（8x8 灰度图，10 类）；
- 预处理：用 `scipy.ndimage.zoom` 将 8x8 上采样到 32x32，以匹配经典 LeNet 输入；
- 输出结果：训练日志、测试集准确率、分类报告、混淆矩阵。

实现目标是“小而完整可复现”，而非大规模 SOTA 训练。

## R04

网络结构（`demo.py` 中 `LeNet`）：

- `Conv2d(1, 6, kernel=5)` + `Tanh` + `AvgPool2d(2)`
- `Conv2d(6, 16, kernel=5)` + `Tanh` + `AvgPool2d(2)`
- 展平后 `Linear(16*5*5, 120)` + `Tanh`
- `Linear(120, 84)` + `Tanh`
- `Linear(84, 10)` 输出 logits

尺寸流（单样本）：

- `1x32x32 -> 6x28x28 -> 6x14x14 -> 16x10x10 -> 16x5x5 -> 400 -> 120 -> 84 -> 10`

## R05

训练策略：

- 设备固定 `CPU`，避免环境差异；
- 使用 `Adam(lr=1e-3)`；
- mini-batch 训练，默认 `batch_size=64`；
- 每轮记录 `train_loss/train_acc/test_loss/test_acc`；
- 训练结束做一次门槛检查：`final_test_acc >= 0.90`。

## R06

数据处理流程：

1. 载入 digits 数据，像素从 `[0,16]` 缩放到 `[0,1]`；
2. 用 `ndimage.zoom(..., zoom=(1,4,4), order=1)` 变为 `32x32`；
3. 使用 `train_test_split(..., stratify=labels)` 划分训练/测试集；
4. 转成 `torch.Tensor` 并增加通道维，得到形状 `N x 1 x 32 x 32`；
5. 打包为 `DataLoader`。

## R07

算法主流程：

1. 固定随机种子；
2. 读取并预处理数据；
3. 构建 LeNet、损失函数、优化器；
4. 循环训练多个 epoch；
5. 每个 epoch 后在测试集评估；
6. 汇总历史曲线（DataFrame）；
7. 输出分类报告与混淆矩阵；
8. 进行最终准确率门槛校验。

## R08

复杂度（粗略）：

- 设样本数为 `N`，epoch 数为 `E`，单样本前后向卷积+全连接开销为 `C`；
- 总训练复杂度约为 `O(E * N * C)`；
- 推理复杂度约为 `O(N * C)`；
- 内存开销主要来自模型参数与 batch 激活，MVP 网络较小，CPU 可运行。

## R09

常见风险与应对：

- 风险：上采样后的 digits 仍较简单，可能高分但泛化有限。
  - 应对：在 README 中明确这是教学级 MVP，不夸大结论。
- 风险：小数据集导致训练波动。
  - 应对：固定 seed、分层抽样、输出完整指标而非只看单个 loss。
- 风险：结构实现错误（尺寸不匹配）。
  - 应对：采用经典 32x32 LeNet 尺寸链路并在代码中显式定义线性层输入维度。

## R10

默认超参数（`LeNetConfig`）：

- `seed=42`
- `test_size=0.2`
- `batch_size=64`
- `epochs=12`
- `lr=1e-3`
- `device="cpu"`

这些参数平衡了运行速度与可见效果，通常数十秒内可完成。

## R11

评估指标：

- `train_acc/test_acc`：总体分类准确率；
- `classification_report`：每个类别的 `precision/recall/f1/support`；
- `confusion_matrix`：类别间混淆情况；
- `train_loss/test_loss`：优化过程稳定性参考。

相比只看准确率，这些指标更有助于发现偏科类别。

## R12

实现注意点：

- LeNet 输出的是 logits，损失函数应使用 `CrossEntropyLoss`（内部含 softmax）；
- `model.eval()` + `@torch.no_grad()` 用于评估阶段，避免梯度开销；
- `TensorDataset` 输入需要显式加通道维 `unsqueeze(1)`；
- 训练和评估需在同一设备（本实现为 CPU）上执行。

## R13

`demo.py` 的 MVP 特性：

- 依赖仅使用 `numpy/pandas/scipy/scikit-learn/torch`；
- 无外部下载数据，离线可跑；
- 无交互输入，`uv run python demo.py` 直接完成；
- 输出包含训练曲线摘要与分类层面的可解释结果。

## R14

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0241-LeNet
uv run python demo.py
```

或在仓库根目录执行：

```bash
uv run python Algorithms/计算机-机器学习／深度学习-0241-LeNet/demo.py
```

## R15

结果解读建议：

- `test_acc` 若稳定高于 0.90，说明 LeNet 在该 toy 任务上已有效收敛；
- 查看混淆矩阵可识别易混类别（例如某些相似笔画数字）；
- 若训练准确率明显高于测试准确率，可视为过拟合信号，应减少 epoch 或加正则。

## R16

本目录验收标准：

- `README.md` 与 `demo.py` 不含未填充占位符；
- `uv run python demo.py` 可非交互运行结束；
- 输出包含训练日志、分类报告、混淆矩阵；
- 最终门槛检查通过（默认 `test_acc >= 0.90`）。

## R17

可扩展方向：

- 将 `Tanh` 改为 `ReLU` 并比较收敛速度；
- 增加数据增强（旋转、平移、噪声）测试鲁棒性；
- 替换为更复杂数据集（FashionMNIST、MNIST）；
- 对比 LeNet 与 MLP/CNN 轻量变体的参数量与精度；
- 加入学习率调度和早停策略。

## R18

`demo.py` 的源码级算法流可拆为 9 步：

1. `main()` 构建 `LeNetConfig`，调用 `set_seed` 固定 `random/numpy/torch` 随机性。
2. `make_loaders()` 调用 `load_digits_32x32()`，读取 `load_digits` 并用 `ndimage.zoom` 将 8x8 放大到 32x32。
3. `train_test_split(..., stratify=labels)` 生成可复现实验划分，随后转换为 `TensorDataset + DataLoader`。
4. 初始化 `LeNet`（2 个卷积层 + 3 个全连接层）以及 `CrossEntropyLoss` 和 `Adam`。
5. 在每个 epoch 中，`run_epoch()` 对每个 batch 执行：前向计算 logits -> 计算交叉熵 -> 反向传播 -> `optimizer.step()`。
6. 同一 epoch 结束后，`evaluate()` 在 `model.eval()` 下计算测试损失、准确率，并收集 `y_true/y_pred`。
7. 训练循环把每轮统计写入 `history`，最终转换为 `pandas.DataFrame` 并打印尾部结果。
8. 使用 `classification_report` 生成各类别 `precision/recall/f1`，再用 `confusion_matrix` 输出混淆矩阵。
9. 读取最后一轮 `test_acc` 做门槛校验（<0.90 则抛错），保证 MVP 不是“只跑通但无效果”。

上述流程直接在代码中显式展开 LeNet 的数据处理、前向、反向、评估与验收逻辑，没有把核心训练机制委托给黑盒封装。

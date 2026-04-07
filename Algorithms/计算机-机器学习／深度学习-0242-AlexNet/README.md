# AlexNet

- UID: `CS-0112`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `242`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0242-AlexNet`

## R01

AlexNet 是深度卷积神经网络（CNN）发展史上的关键里程碑模型（ImageNet 2012 冠军结构）。它证明了“更深的卷积网络 + ReLU + GPU 训练 + 正则化”在大规模视觉任务上的显著优势。

本条目给出一个离线可运行的最小 MVP：使用 PyTorch 实现 AlexNet 风格网络，并在 `sklearn digits` 数据集（8x8 手写数字）上完成端到端训练与评估。

## R02

目标问题是多分类图像识别。给定样本 `(x, y)`，其中 `x` 为图像、`y in {0,...,9}` 为类别，模型学习映射 `f_theta(x)` 并最小化交叉熵损失：

`L = - sum_{k=0}^{9} 1[y=k] * log softmax(f_theta(x))_k`

预测阶段取 `argmax_k softmax(f_theta(x))_k` 作为类别输出。

## R03

AlexNet 的核心设计要点：

- 深层卷积堆叠（5 个卷积层）提取分层视觉特征；
- ReLU 激活缓解饱和梯度、加速收敛；
- 最大池化增强平移鲁棒性并压缩空间分辨率；
- 分类头使用全连接层与 Dropout 抑制过拟合；
- 早期版本还使用了 LRN（Local Response Normalization）。

本实现保留这些机制，并对输入尺寸和分类头宽度做了轻量化缩放，以便在 CPU 上快速运行。

## R04

`demo.py` 中的 AlexNet 结构（64x64 输入）如下：

1. `Conv(3->64, 11x11, stride=4) + ReLU + LRN + MaxPool`
2. `Conv(64->192, 5x5) + ReLU + LRN + MaxPool`
3. `Conv(192->384, 3x3) + ReLU`
4. `Conv(384->256, 3x3) + ReLU`
5. `Conv(256->256, 3x3) + ReLU + MaxPool`
6. `AdaptiveAvgPool(1x1)`
7. `FC(256->512)->ReLU->Dropout->FC(512->256)->ReLU->Dropout->FC(256->10)`

这保持了“5 卷积 + 3 全连接”的 AlexNet 主干范式。

## R05

卷积层的基本计算可写为：

`Y[c_out, i, j] = sum_{c_in, u, v} W[c_out, c_in, u, v] * X[c_in, i+u, j+v] + b[c_out]`

其中卷积核共享参数，使模型对局部模式（边缘、角点、纹理）具有高效建模能力。随着层数增加，特征从低级局部结构逐步抽象到类别相关语义模式。

## R06

训练目标采用标准监督学习：

- 前向：`logits = model(images)`
- 损失：`CrossEntropyLoss(logits, labels)`
- 反向：`loss.backward()`
- 更新：`optimizer.step()`

优化器使用 `Adam`，并配合 `weight_decay` 进行 L2 正则化。Dropout 仅在训练模式下生效。

## R07

`demo.py` 的端到端流程：

1. 读取 `sklearn.datasets.load_digits()`；
2. 使用 `scipy.ndimage.zoom` 将 8x8 放大到 64x64；
3. 使用 `gaussian_filter` 做轻量平滑，再复制为 3 通道；
4. 用 `train_test_split(..., stratify=labels)` 构造分层训练/测试集；
5. 基于训练集统计量做标准化；
6. 训练 AlexNet 并按 epoch 输出 train/test 指标；
7. 输出分类报告、混淆矩阵、样例预测；
8. 进行最小精度阈值断言。

## R08

正确性保障点：

- 固定随机种子（`random/numpy/torch`）减少结果波动；
- 使用分层抽样，防止类别分布偏斜导致评估失真；
- 训练集统计量用于 train/test 同步标准化，避免数据泄漏；
- `model.train(True/False)` 区分训练与推理行为（特别是 Dropout）；
- 以 `best_test_acc`（各轮最佳测试精度）进行阈值校验，避免最后一轮波动造成误判。

## R09

复杂度（单样本、单层近似）：

- 卷积层时间复杂度约为 `O(H * W * C_in * C_out * K^2)`；
- 全连接层约为 `O(D_in * D_out)`；
- 整体瓶颈通常在前几层高分辨率卷积。

空间复杂度主要来自：

- 参数存储（卷积核 + 全连接权重）；
- 中间特征图激活（训练时还需保存反向传播缓存）。

## R10

代码模块对应：

- `TrainConfig`：训练超参数与阈值；
- `DigitsDataset`：将 numpy 数据封装为 `torch.utils.data.Dataset`；
- `AlexNet`：AlexNet 风格网络定义；
- `resize_digit_to_rgb`：离线图像放大与平滑预处理；
- `prepare_dataset`：数据拆分与标准化；
- `run_epoch`：训练/评估统一循环与指标汇总；
- `main`：串联全流程并输出报告。

## R11

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0242-AlexNet
uv run python demo.py
```

脚本无交互输入，不依赖在线下载。

## R12

预期输出字段：

- `device`：当前运行设备（CPU/GPU）；
- `train label distribution`：训练集标签分布（pandas 统计）；
- `trainable parameters`：可训练参数量；
- `epoch ... train_loss/train_acc/test_loss/test_acc`：逐轮训练日志；
- `classification report`：每类 precision/recall/f1；
- `confusion matrix`：类别混淆明细；
- `sample predictions`：随机抽样预测示例；
- `All checks passed.`：阈值验证通过。

## R13

最小实验配置：

- 数据：`sklearn digits`（1797 张样本，10 类）；
- 输入：8x8 灰度图上采样到 64x64，并复制为 RGB；
- 划分：`test_ratio=0.25`，分层抽样；
- 训练：`batch_size=64`，`epochs=14`；
- 验证门槛：`best_test_acc >= 0.94`。

这是面向“算法机制可复现”的 MVP，不追求大规模 SOTA。

## R14

关键超参数与影响：

- `image_size`：输入分辨率，越大越耗时但特征更细；
- `learning_rate`：过大易震荡，过小收敛慢；
- `epochs`：轮数不足会欠拟合，过高可能过拟合；
- `weight_decay`：控制模型复杂度，提升泛化；
- `batch_size`：影响梯度噪声与吞吐效率；
- `min_test_accuracy`：最小可接受性能门槛。

## R15

与其他经典 CNN 的关系：

- 对比 LeNet：AlexNet 更深、更宽，适合更复杂视觉任务；
- 对比 VGG：VGG 更统一小卷积核堆叠，参数量更大；
- 对比 ResNet：ResNet 通过残差连接缓解深层训练退化，通常可堆叠得更深。

AlexNet 依然是理解现代 CNN 设计演进的关键“中间站”。

## R16

适用场景：

- 入门级图像分类教学与 CNN 机制验证；
- 中小规模视觉数据上的快速基线构建；
- 需要解释“卷积特征提取 + 全连接分类头”流程的实验项目。

## R17

局限与扩展方向：

- 局限：对当前任务而言参数偏多，训练成本高于轻量网络；
- 可扩展：加入数据增强（裁剪/旋转/颜色扰动）和学习率调度；
- 可扩展：改为 SGD+momentum 以贴近原始 AlexNet 训练范式；
- 可扩展：替换为真实彩色数据集（如 CIFAR-10）检验迁移表现；
- 可扩展：加入模型保存与可视化（loss/acc 曲线）。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `main` 读取 `TrainConfig` 并调用 `set_global_seed`，固定实验随机性。
2. `prepare_dataset` 加载 `load_digits()`，将原始 8x8 图像与标签读入内存。
3. `resize_digit_to_rgb` 对每张图执行 `zoom` 上采样与 `gaussian_filter` 平滑，再复制为 3 通道。
4. `train_test_split(..., stratify=labels)` 生成训练集/测试集，并用训练集均值方差标准化两者。
5. `DigitsDataset + DataLoader` 组装 mini-batch 输入流水线。
6. `AlexNet.forward` 依次经过 5 个卷积块（含 ReLU/池化/LRN）提取特征，再经 3 层全连接输出 logits。
7. `run_epoch` 计算交叉熵损失；若为训练阶段则执行 `zero_grad -> backward -> step`。
8. 每轮训练后调用 `run_epoch(..., optimizer=None)` 做评估，累计 `loss/accuracy` 并打印日志。
9. 训练结束后输出最佳轮次的分类报告、混淆矩阵和样例预测；若 `best_test_acc` 低于阈值则抛错，否则打印 `All checks passed.`。

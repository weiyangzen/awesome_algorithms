# VGGNet

- UID: `CS-0113`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `243`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0243-VGGNet`

## R01

VGGNet（Visual Geometry Group Network）是经典卷积神经网络架构，核心特征是重复堆叠 `3x3` 小卷积核并在阶段末尾做池化。它通过“多层小卷积近似大感受野”的方式，在保持参数可控的同时提升特征表达能力。

本条目提供一个可运行 MVP：
- 用 PyTorch 显式实现 VGG 风格卷积块 `VGGBlock`；
- 在 `sklearn digits`（8x8 灰度手写数字）上训练 `TinyVGGNet`；
- 输出训练日志、测试准确率、基线对比和样例预测。

## R02

监督分类形式化为：

给定训练集 `D = {(x_i, y_i)}_{i=1}^N`，其中 `x_i in R^{1x8x8}`，`y_i in {0,...,9}`，学习参数 `theta` 以最小化交叉熵损失：

`min_theta (1/N) * sum_i CE(f_theta(x_i), y_i)`。

VGG 风格映射写作：

`f_theta = Classifier(Pool(Block_3(Pool(Block_2(Pool(Block_1(x)))))))`

其中每个 `Block_k` 都由若干 `3x3 Conv + BN + ReLU` 组成。

## R03

VGGNet 的关键思想：
- 使用统一的小卷积核（典型为 `3x3`）反复堆叠，增加非线性深度；
- 用多层小卷积替代单层大卷积，在相近感受野下通常拥有更好的参数效率；
- 通过分阶段池化逐步压缩空间尺寸，扩大有效感受野并聚合高层语义特征。

在本 MVP 中，这一思想被压缩为适配 `8x8` 小图像的三段式结构。

## R04

`demo.py` 的网络结构对应关系：
- `VGGBlock`：`(Conv-BN-ReLU) * N -> MaxPool2d(2)`；
- `TinyVGGNet.features`：
  - Block1: `1 -> 16`（2 个卷积），`8x8 -> 4x4`；
  - Block2: `16 -> 32`（2 个卷积），`4x4 -> 2x2`；
  - Block3: `32 -> 64`（2 个卷积），`2x2 -> 1x1`；
- `classifier`：`Flatten -> Linear(64,64) -> ReLU -> Dropout -> Linear(64,10)`。

这保持了 VGG 的“卷积堆叠 + 池化 + 全连接分类头”主干范式。

## R05

MVP 端到端流程：

1. 固定随机种子（`random`、`numpy`、`torch`）。
2. 读取 `digits` 数据并做归一化（像素除以 16）。
3. 分层划分训练/测试集并转换成 `NCHW` 张量。
4. 构建 `DataLoader`（训练打乱、测试不打乱）。
5. 初始化 `TinyVGGNet + CrossEntropyLoss + Adam`。
6. 循环多个 epoch 执行训练与测试评估。
7. 计算多数类基线并打印样例预测。
8. 执行精度门槛断言，确保结果有效。

## R06

正确性依据：
- 交叉熵损失对应多分类最大似然目标；
- `Conv-BN-ReLU` 组合提升训练稳定性并增强非线性表达；
- `MaxPool` 实现逐级空间压缩，符合 VGG 分层抽象思路；
- `train/eval` 模式显式切换，保证 BN/Dropout 在训练与评估阶段行为正确；
- 测试精度与基线差值双重断言，避免“看似能跑但无有效学习”。

## R07

复杂度分析（单次前向，忽略常数）：
- 卷积主导计算：`O(sum_l H_l * W_l * C_in_l * C_out_l * k^2)`；
- 反向传播同量级，训练总体通常是前向的约 2-3 倍常数；
- 参数主要来自卷积层和两层全连接层，本 MVP 为小规模参数网络，可在 CPU 上快速收敛。

由于输入是 `8x8` 小图，训练成本较低，适合最小可运行验证。

## R08

边界与异常处理：
- `test_ratio` 必须位于 `(0,1)`，否则抛出 `ValueError`；
- 使用固定随机种子与分层抽样，减小评估波动；
- 对最终测试准确率执行有限值检查（防 NaN/Inf）；
- 若准确率低于阈值或未显著优于基线，抛出 `RuntimeError`。

## R09

MVP 范围与取舍：
- 保留 VGG 核心：重复小卷积、阶段池化、全连接分类头；
- 不追求完整 VGG-16/VGG-19 的大规模通道配置与 ImageNet 级训练策略；
- 不引入复杂数据增强、学习率调度、迁移学习等工程扩展；
- 目标是“可运行、可读、可验证”的最小教学实现。

## R10

`demo.py` 主要函数职责：
- `VGGBlock`：构造单个 VGG 风格卷积块；
- `TinyVGGNet`：拼接三段卷积特征提取器和分类器；
- `load_dataset`：加载 `digits`、归一化并切分数据；
- `build_dataloaders`：封装训练/测试迭代器；
- `run_epoch`：统一训练或评估一个 epoch；
- `majority_class_baseline`：计算多数类基线精度；
- `main`：串联数据、训练、评估、打印和质量检查。

## R11

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0243-VGGNet
uv run python demo.py
```

脚本无交互输入，直接输出训练和评估结果。

## R12

输出字段说明：
- `device`：实际运行设备（CPU/GPU）；
- `train/test shape`：输入与标签张量形状；
- `model ... params`：模型名与参数量；
- `epoch ... train_loss/train_acc/test_loss/test_acc`：每轮关键指标；
- `majority baseline`：多数类常数预测基线；
- `sample predictions`：测试集前 8 条样例预测及置信度；
- `All checks passed.`：所有质量门槛通过。

## R13

最小实验配置（内置在脚本中）：

1. 数据集：`sklearn.datasets.load_digits()`，共 1797 条样本。
2. 划分：按 `75/25` 分层拆分训练集和测试集。
3. 优化：`Adam(lr=1e-3, weight_decay=1e-4)`。
4. 轮次：`14` 个 epoch。
5. 评估：测试准确率 + 多数类基线比较 + 样例预测检查。
6. 断言：要求测试精度达到阈值并显著超过基线。

## R14

关键超参数与影响：
- `epochs`：过小会欠拟合，过大可能过拟合；
- `lr`：控制收敛速度与稳定性；
- `weight_decay`：限制权重过大，缓解过拟合；
- `batch_size`：影响梯度噪声与吞吐；
- 通道数（16/32/64）和块深度（每块卷积数）决定模型容量。

建议调参顺序：先调 `lr` 与 `epochs`，再调通道数和正则强度。

## R15

与相关方法对比：
- 对比 MLP：VGG 能利用二维局部结构，更适合图像任务；
- 对比浅层 CNN：VGG 通过更深的卷积堆叠表达更强的层级特征；
- 对比 ResNet：VGG结构更直观，但没有残差捷径，深层训练稳定性通常弱于 ResNet。

本 MVP 的定位是“结构清晰的 VGG 入门实现”。

## R16

典型应用场景：
- 基础图像分类教学和卷积网络入门实验；
- 作为视觉任务中的可解释基线模型；
- 小数据集上的快速原型验证；
- 用于对比不同 CNN 架构（VGG / ResNet / MobileNet）训练行为。

## R17

可扩展方向：
- 扩展为 VGG-11/13/16/19 风格配置；
- 加入更丰富数据增强（随机裁剪、翻转、噪声扰动）；
- 增加学习率调度（StepLR、CosineAnnealingLR）；
- 引入早停、混合精度和实验日志系统；
- 在更大图像数据集上做迁移学习或蒸馏实验。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 调用 `set_global_seed(42)`，固定 Python/Numpy/PyTorch 随机性。
2. `load_dataset` 读取 `digits`，完成归一化、分层切分，并将输入扩展到 `NCHW`。
3. `build_dataloaders` 将训练/测试张量封装成批迭代器。
4. 初始化 `TinyVGGNet`；其 `features` 由 3 个 `VGGBlock` 串联，每个块内部显式执行多次 `Conv-BN-ReLU` 后接 `MaxPool`。
5. 在每个 epoch 中，`run_epoch(..., optimizer=Adam)` 执行前向计算、交叉熵损失、反向传播与参数更新。
6. 同一轮后调用 `run_epoch(..., optimizer=None)` 在测试集仅做前向评估，得到 `test_loss/test_acc`。
7. 训练完成后，`majority_class_baseline` 计算常数预测基线，再对前 8 个测试样本输出 softmax 概率与预测类别。
8. 执行质量断言（精度阈值与基线差距）；通过后打印 `All checks passed.`。

# DenseNet

- UID: `CS-0116`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `246`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0246-DenseNet`

## R01

DenseNet（Densely Connected Convolutional Network）是一种卷积神经网络结构。其核心定义是：在同一个 Dense Block 内，第 `l` 层会接收前面 `0..l-1` 层的特征拼接结果作为输入，而不是只接收上一层输出。

## R02

该结构针对深层网络中的梯度传播困难与特征复用不足问题。通过“密集连接”，浅层纹理特征可直接传到深层，减少信息丢失，并在参数量受控的情况下提升表达能力。

## R03

DenseNet 的典型组件有三类：
- Dense Layer：`BN -> ReLU -> Conv(3x3)`，输出新特征并与输入拼接。
- Dense Block：堆叠多个 Dense Layer，通道数按增长率 `growth_rate` 线性增长。
- Transition Layer：`BN -> ReLU -> Conv(1x1) -> AvgPool(2x2)`，用于通道压缩和下采样。

## R04

关键公式：
- 第 `l` 层输入：`x_l_in = [x_0, x_1, ..., x_{l-1}]`（按通道拼接）
- 第 `l` 层输出：`x_l = H_l(x_l_in)`，其中 `H_l` 为 `BN-ReLU-Conv`
- Block 通道变化：若初始通道 `C0`，每层增长率 `k`，层数 `L`，则输出通道约为 `C0 + L*k`

## R05

训练目标与常规分类网络一致：最小化交叉熵损失。
在本任务 MVP 中，输入是 `8x8` 灰度数字图像（sklearn digits），输出是 `0-9` 共 10 类概率分布。

## R06

复杂度（以单个 Dense Block 粗略估计）：
- 时间复杂度随层数增加近似二次增长，因为后层卷积输入通道变大。
- 空间复杂度较高，需保存多层特征用于拼接。
- DenseNet 用更强特征复用换取更大的中间特征内存开销。

## R07

优点：
- 梯度流更顺畅，训练深网络更稳定。
- 特征复用充分，参数效率通常优于同深度普通 CNN。
- 对小样本或中等规模任务常有较好泛化。

## R08

局限：
- 特征拼接带来显存/内存压力。
- 实现复杂度高于顺序堆叠网络。
- 在超大分辨率任务中，需仔细设计 `growth_rate` 和压缩率。

## R09

本目录 MVP 的实现边界：
- 采用 PyTorch 手写 TinyDenseNet，不调用现成 DenseNet 黑盒模型。
- 数据集使用 `sklearn.datasets.load_digits`，避免外网下载。
- 只做单文件训练/评估流程，目标是可运行、可验证、可读。

## R10

`demo.py` 关键超参数：
- `growth_rate=12`
- `block_config=(3, 3, 3)`
- `init_channels=16`
- `compression=0.5`
- `epochs=10`
- `batch_size=64`
- `learning_rate=3e-3`

## R11

运行方式（无需交互输入）：

```bash
uv run python demo.py
```

脚本将自动完成：数据准备、训练、测试集精度评估、样例预测打印。

## R12

预期输出形态：
- 设备信息（CPU/CUDA）
- 每若干 epoch 的训练损失与训练精度
- 最终 `Test accuracy`
- 若干条 `(pred, true)` 样例预测对

## R13

正确性检查建议：
- 能完整跑通且无异常退出。
- 训练损失总体下降。
- 测试精度明显高于随机猜测（10 分类随机约 0.10）。
- 样例预测大部分与真实标签一致。

## R14

常见失败模式：
- 学习率过大导致损失震荡。
- `growth_rate` 过大导致显存或内存占用过高。
- 训练轮数过少导致欠拟合。
- 未固定随机种子造成结果波动较大。

## R15

与 ResNet 的简要对比：
- ResNet 是“逐层残差相加”，DenseNet 是“跨层特征拼接”。
- DenseNet 更强调特征复用；ResNet 更强调残差学习与训练深度。
- DenseNet 往往参数更省，但中间特征占用更大。

## R16

可扩展方向：
- 加入 Bottleneck（`1x1` + `3x3`）以降低计算量。
- 调整 block 数与每块层数，构建更深版本。
- 引入数据增强或学习率调度提升精度。
- 切换到 CIFAR-10 等更复杂数据集做对照实验。

## R17

工程实现要点：
- 使用 `set_seed` 固定随机性，提升复现实验稳定性。
- 将 `DenseLayer`、`DenseBlock`、`TransitionLayer` 明确拆分，便于单元测试与扩展。
- 通过 `DataLoader` + 张量化处理，保持训练循环简洁。
- `main()` 无输入参数即可运行，满足批处理验证场景。

## R18

`demo.py` 的源码级算法流程（8 步）：
1. `main()` 初始化 `TrainConfig`，调用 `set_seed()` 固定随机种子，选择 `cpu/cuda` 设备。
2. `build_dataloaders()` 加载 `load_digits()`，做归一化与 `train_test_split`，封装成训练/测试 `DataLoader`。
3. 构建 `TinyDenseNet`：`stem conv` 后串联多个 `DenseBlock`，块间插入 `TransitionLayer` 做压缩与下采样。
4. 在每个 `DenseBlock` 内，`DenseLayer.forward()` 执行 `BN-ReLU-Conv`，再 `torch.cat([x, new_features], dim=1)` 实现密集连接。
5. 分类头 `head + classifier` 对最终特征做 `AdaptiveAvgPool2d(1)` 与全连接输出 logits。
6. `train_one_epoch()` 中执行标准训练闭环：前向、`CrossEntropyLoss`、反向传播、`Adam` 更新，并统计 loss/acc。
7. `evaluate()` 在 `no_grad` 模式下聚合测试集预测，使用 `accuracy_score` 计算最终准确率。
8. `collect_examples()` 抽取若干 `(pred, true)` 对并打印，形成可读的最小验证证据。

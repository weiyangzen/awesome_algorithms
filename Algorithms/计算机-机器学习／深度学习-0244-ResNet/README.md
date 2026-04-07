# ResNet

- UID: `CS-0114`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `244`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0244-ResNet`

## R01

ResNet（Residual Network，残差网络）是通过“跳跃连接（skip connection）”缓解深层网络退化问题的卷积神经网络架构。核心思想是让网络学习残差映射 `F(x)`，再输出 `y = F(x) + x`，而不是直接拟合 `H(x)`。

本条目提供一个可运行 MVP：
- 用 PyTorch 手写 `BasicBlock`（含恒等/投影 shortcut）；
- 在 `sklearn digits`（8x8 灰度手写数字）上训练 `TinyResNet`；
- 输出训练过程、测试准确率、基线对比与样例预测。

## R02

监督分类任务形式化：

给定训练集 `D = {(x_i, y_i)}_{i=1}^N`，其中 `x_i in R^{1x8x8}`，`y_i in {0,...,9}`，学习参数 `theta` 使分类损失最小：

`min_theta (1/N) * sum_i CE(f_theta(x_i), y_i)`。

ResNet 基本残差单元写作：

`y = sigma(F(x, W) + S(x))`

其中：
- `F` 为两层卷积堆叠；
- `S` 为 shortcut（恒等映射或 `1x1` 投影）；
- `sigma` 为 ReLU。

## R03

为什么 ResNet 有效：
- 深层普通网络中，梯度传播路径过长，优化难度上升；
- 残差连接给出更短的梯度通路，降低“深度增加但训练误差反而变高”的退化风险；
- 当最优映射接近恒等时，学习 `F(x)=0` 比直接学习 `H(x)=x` 更容易。

因此 ResNet 在相同深度预算下通常更稳定、更易收敛。

## R04

`demo.py` 中残差块细节：
- 主分支：`3x3 Conv -> BN -> ReLU -> 3x3 Conv -> BN`；
- shortcut 分支：
  - 若输入/输出通道和步幅一致，使用 `Identity`；
  - 否则使用 `1x1 Conv(stride) + BN` 做投影匹配；
- 两分支相加后再过 ReLU。

这对应经典 BasicBlock 的最小实现。

## R05

MVP 高层流程：

1. 固定随机种子（`random`/`numpy`/`torch`）。
2. 载入 `digits` 数据并归一化到 `[0,1]`。
3. 划分训练/测试集并封装 `DataLoader`。
4. 构建 `TinyResNet`（stem + 2 个残差 stage + GAP + 全连接层）。
5. 用 `CrossEntropyLoss + Adam` 训练多个 epoch。
6. 在测试集评估准确率并与多数类基线比较。
7. 打印样例预测并执行质量断言。

## R06

正确性直觉：
- `CrossEntropyLoss` 直接优化多分类对数似然；
- 残差块中的 `out + identity` 保留原始特征并叠加增量特征；
- 下采样时通过投影 shortcut 保证张量形状可加；
- 训练和评估分离（`model.train(True/False)`），保证 BN 行为正确。

整体上，模型结构、损失函数和优化过程与标准 ResNet 分类训练范式一致。

## R07

复杂度（单次前向，忽略常数）：
- 卷积层主导开销，约为 `O(sum_l H_l * W_l * C_l^2 * k^2)`；
- 反向传播同量级，训练约为前向的 2-3 倍常数；
- 参数量由卷积核和全连接层决定，本 MVP 参数量约 4 万级。

在 `digits` 小数据集上，CPU 即可快速完成训练。

## R08

边界与异常处理：
- `test_ratio` 必须在 `(0,1)`；
- 数据按固定随机种子和分层抽样拆分，减少偶然波动；
- 若测试精度非有限值或低于阈值，脚本抛出 `RuntimeError`；
- 仅使用本地可得数据集，不依赖网络下载。

## R09

MVP 范围与取舍：
- 保留 ResNet 最核心机制：残差分支、投影 shortcut、端到端分类训练；
- 不实现完整 ResNet-18/34 的大规模配置、复杂数据增强、学习率调度；
- 目标是“最小可运行 + 源码可读”，而非 SOTA 精度。

## R10

`demo.py` 关键组件职责：
- `BasicBlock`：定义残差块与 shortcut 逻辑；
- `TinyResNet`：组装 stem、残差层、池化和分类头；
- `load_dataset`：加载并预处理 digits 数据；
- `build_dataloaders`：构建批训练/评估迭代器；
- `run_epoch`：统一训练或评估一个 epoch；
- `majority_class_baseline`：计算多数类基线；
- `main`：串联训练、评估、打印和断言。

## R11

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0244-ResNet
uv run python demo.py
```

脚本无交互输入，执行后直接输出训练日志和评估结果。

## R12

输出字段解释：
- `device`：运行设备（CPU/GPU）；
- `train/test shape`：数据张量维度；
- `model ... params`：模型规模；
- `epoch ... train_loss/train_acc/test_loss/test_acc`：每轮训练与验证指标；
- `majority baseline`：常数预测基线；
- `sample predictions`：前 8 个测试样本的真值、预测和置信度；
- `All checks passed.`：质量门槛验证通过。

## R13

最小实验设计（已内置）：

1. 使用 `sklearn.datasets.load_digits()`（1797 样本）。
2. 按 75/25 分层划分训练集和测试集。
3. 训练 `TinyResNet` 共 12 轮。
4. 记录训练/测试准确率曲线。
5. 与多数类基线比较，并要求显著超越。
6. 输出样例预测做人工 sanity check。

## R14

关键超参数与建议：
- `epochs`：训练轮次，不足会欠拟合；
- `lr`：学习率，过大会震荡、过小收敛慢；
- `weight_decay`：控制过拟合；
- `batch_size`：影响梯度噪声与速度；
- 残差 stage 通道数（16/32）：决定模型容量。

推荐调参顺序：先调 `lr` 与 `epochs`，再调通道数和正则强度。

## R15

与相关方法对比：
- 对比普通 CNN：ResNet 多了 shortcut，深层训练更稳定；
- 对比 MLP：ResNet 更适合图像局部结构建模；
- 对比更大视觉模型（如 ViT）：本 MVP 更轻量、数据需求更小、可解释性更直接。

## R16

典型应用场景：
- 图像分类、检测、分割中的 backbone；
- 医学影像与工业质检的视觉特征提取；
- 迁移学习场景中的预训练骨干网络。

## R17

可扩展方向：
- 从 `TinyResNet` 扩展到 ResNet-18/34/50；
- 增加数据增强（随机裁剪、翻转、颜色扰动）；
- 引入学习率调度（Cosine、StepLR）；
- 加入混合精度和更严格的实验追踪；
- 把分类头替换为检测/分割任务头。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 调用 `set_global_seed` 固定随机性，确保实验可复现。
2. `load_dataset` 读取 digits 图像，做归一化、分层切分，并转为 `NCHW` 张量。
3. `build_dataloaders` 将训练/测试张量封装为批迭代器。
4. 初始化 `TinyResNet`；其中每个 `BasicBlock` 显式实现主分支卷积与 shortcut（恒等或投影）。
5. 每个 epoch 内，`run_epoch(..., optimizer=Adam)` 执行前向、交叉熵、反向传播和参数更新。
6. 同一 epoch 后，`run_epoch(..., optimizer=None)` 在测试集仅前向评估，得到 `test_acc`。
7. 训练结束后，计算多数类基线并对前 8 个测试样本输出 softmax 预测结果。
8. 执行质量断言（精度阈值和基线差距），全部通过则打印 `All checks passed.`。

# EfficientNet

- UID: `CS-0117`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `247`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0247-EfficientNet`

## R01

EfficientNet 是一类高参数效率的卷积神经网络家族，核心思想是复合缩放（compound scaling）：在给定计算预算下，不只加深网络，还同时按比例调整网络宽度与输入分辨率。

本条目提供一个可运行 MVP：
- 用 PyTorch 手写 EfficientNet 的关键模块：`MBConv`（倒残差 + 深度可分离卷积）与 `SE` 注意力；
- 在 `sklearn digits`（`8x8` 灰度数字）上训练 `TinyEfficientNet`；
- 输出训练日志、测试精度、基线对比与样例预测。

## R02

监督分类目标：

给定训练集 `D = {(x_i, y_i)}_{i=1}^N`，其中 `x_i in R^{1x8x8}`，`y_i in {0,...,9}`，学习参数 `theta` 使

`min_theta (1/N) * sum_i CE(f_theta(x_i), y_i)`。

EfficientNet 的复合缩放思想可写为：
- 深度缩放：`d = alpha^phi`
- 宽度缩放：`w = beta^phi`
- 分辨率缩放：`r = gamma^phi`
- 约束：`alpha * beta^2 * gamma^2 ~= 2`。

本 MVP 因数据分辨率固定（`8x8`）仅实现深度和宽度缩放（`depth_mult`, `width_mult`）。

## R03

EfficientNet 的关键构件：
- `MBConv`：先 `1x1` 扩展通道，再深度卷积，再 `SE` 通道重标定，最后 `1x1` 投影；
- 残差连接：当步幅为 1 且输入输出通道一致时使用 skip connection；
- 激活函数：`SiLU/Swish`，在移动端 CNN 中常优于纯 ReLU；
- 复合缩放：通过统一规则控制模型容量，而非任意堆层。

## R04

`demo.py` 中网络结构：
- `stem`：`ConvBNAct(1 -> C_stem, k=3, s=1)`；
- `blocks`：按 stage 配置串联 MBConv，基础配置为：
  - `(expand=1, out=16, repeat=1, stride=1, k=3)`
  - `(expand=6, out=24, repeat=2, stride=2, k=3)`
  - `(expand=6, out=40, repeat=2, stride=1, k=5)`
  - `(expand=6, out=64, repeat=3, stride=2, k=3)`
- `head`：`1x1 Conv` 聚合通道；
- `classifier`：`GlobalAvgPool -> Dropout -> Linear(num_classes)`。

其中通道数和重复次数分别经过 `round_filters` 与 `round_repeats` 缩放。

## R05

端到端流程：

1. 固定随机种子（`random/numpy/torch`）。
2. 读取 `digits` 数据集并归一化到 `[0,1]`。
3. 分层划分训练/测试集，转换为 `NCHW` 张量。
4. 构建 `DataLoader`。
5. 初始化 `TinyEfficientNet + CrossEntropyLoss + AdamW`。
6. 训练多个 epoch，并在每轮后评估测试集。
7. 计算多数类基线，输出样例预测与置信度。
8. 通过精度和基线差距断言后打印 `All checks passed.`。

## R06

正确性依据：
- 交叉熵是多分类最大似然的标准目标；
- MBConv 的深度可分离卷积降低参数与 FLOPs，同时保留空间建模能力；
- SE 模块通过全局池化后重标定通道权重，增强有效特征；
- `model.train()` 与 `model.eval()` 显式切换，确保 BN/Dropout 行为正确；
- 通过阈值断言防止“可运行但未学习到有效模式”。

## R07

复杂度（单次前向，忽略常数）：
- 标准卷积约为 `O(H*W*C_in*C_out*k^2)`；
- 深度卷积约为 `O(H*W*C*k^2)`，逐点卷积约为 `O(H*W*C_in*C_out)`；
- MBConv 相对同通道标准卷积通常更省计算；
- 总体训练成本与各层 `H_l, W_l, C_l, repeat_l` 成正相关。

本任务输入仅 `8x8`，因此训练可在 CPU 上快速完成。

## R08

边界与异常处理：
- `load_dataset` 对 `test_size` 做 `(0,1)` 合法性检查；
- `MBConv` 仅允许 `stride` 为 `1` 或 `2`，否则抛 `ValueError`；
- 训练后检查 `test_acc` 是否有限值（防 NaN/Inf）；
- 若精度低于阈值或未充分超过基线，抛出 `RuntimeError`。

## R09

MVP 取舍：
- 保留 EfficientNet 核心机制（MBConv、SE、复合缩放）；
- 不复刻完整 ImageNet 版 B0-B7 超大配置；
- 不引入复杂数据增强、学习率调度和多机训练；
- 聚焦“结构可读、运行稳定、指标可验证”的教学级最小实现。

## R10

`demo.py` 主要组件职责：
- `EfficientNetConfig`：收拢宽度/深度/分类数等配置；
- `round_filters` / `round_repeats`：实现复合缩放的离散化规则；
- `SqueezeExcitation`：实现 SE 通道注意力；
- `MBConv`：实现 EfficientNet 核心倒残差块；
- `TinyEfficientNet`：组装 stem、多 stage MBConv、head 与分类器；
- `run_epoch`：统一训练与评估循环；
- `main`：执行数据加载、训练、评估、断言和输出。

## R11

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0247-EfficientNet
uv run python demo.py
```

脚本无交互输入，直接输出训练与评估日志。

## R12

输出字段说明：
- `device`：运行设备（CPU/GPU）；
- `train/test shape`：数据张量形状；
- `model ... params`：模型参数规模；
- `epoch ... train_loss/train_acc/test_loss/test_acc`：每轮指标；
- `majority baseline`：多数类常数预测基线；
- `sample predictions`：样例的真实标签、预测标签与置信度；
- `All checks passed.`：所有质量检查通过。

## R13

最小实验配置（内置）：

1. 数据：`sklearn.datasets.load_digits()`（1797 条样本）。
2. 划分：`train/test = 75/25` 且分层采样。
3. 模型：`TinyEfficientNet(width_mult=1.0, depth_mult=1.0)`。
4. 优化器：`AdamW(lr=2e-3, weight_decay=1e-4)`。
5. 轮次：`18`。
6. 检查：`test_acc >= 0.90` 且显著优于基线。

## R14

关键超参数与影响：
- `width_mult`：影响每层通道数，提升容量但增加计算；
- `depth_mult`：影响每个 stage 的重复次数，提升表达深度；
- `dropout`：分类头正则化强度；
- `lr` 与 `weight_decay`：收敛速度和泛化稳定性；
- `epochs`：训练充分程度。

建议调参顺序：先 `lr/epochs`，再 `width_mult/depth_mult`，最后调 `dropout`。

## R15

与相关架构对比：
- 对比普通 CNN：EfficientNet 在相近精度下通常更省参数；
- 对比 VGG：EfficientNet 更轻量，结构更现代（DWConv + SE）；
- 对比 ResNet：ResNet 强调残差主干稳定训练，EfficientNet 更强调计算效率与复合缩放。

本 MVP 的定位是“EfficientNet 核心思想的可运行最小示例”。

## R16

典型应用场景：
- 资源受限设备上的图像分类基线；
- 需要较好精度/算力比的视觉任务原型；
- 教学场景下理解 MBConv、SE 与复合缩放；
- 作为更大视觉系统中的轻量 backbone 原型。

## R17

可扩展方向：
- 扩展到更接近 B0 的完整 stage 配置；
- 加入数据增强与学习率调度；
- 支持混合精度和早停；
- 在 CIFAR-10/Imagenette 等更大数据集上验证；
- 增加与 ResNet/MobileNet 的系统对比实验。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 调用 `set_global_seed(42)`，固定 Python、NumPy、PyTorch 随机性并限制线程数。  
2. `load_dataset` 读取 `digits`，完成归一化、扩维到 `NCHW`、分层切分训练/测试集。  
3. `round_filters` 和 `round_repeats` 根据 `width_mult/depth_mult` 计算每个 stage 的通道数和重复次数。  
4. `TinyEfficientNet.__init__` 按配置构建 `stem -> MBConv stages -> head -> classifier`。  
5. 每个 `MBConv.forward` 依次执行：扩展 `1x1` 卷积（可选）-> 深度卷积 -> `SqueezeExcitation` -> 投影 `1x1` 卷积；满足条件时再加残差。  
6. `run_epoch(..., optimizer=AdamW)` 在训练阶段执行前向、交叉熵、反向传播与参数更新；评估阶段仅前向累积指标。  
7. 训练完成后，`majority_class_baseline` 计算常数预测基线，`predict_samples` 输出样例预测置信度用于快速 sanity check。  
8. `main` 对最终精度执行阈值和基线差距断言，通过后打印 `All checks passed.`，形成完整闭环。

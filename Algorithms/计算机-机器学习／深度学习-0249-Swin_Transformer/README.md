# Swin Transformer

- UID: `CS-0119`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `249`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0249-Swin_Transformer`

## R01

Swin Transformer（Shifted Window Transformer）是面向视觉任务的层次化 Transformer。它把全局自注意力改成局部窗口注意力（W-MSA），并通过“窗口平移”（SW-MSA）在相邻窗口间建立信息交互，从而在保持表达能力的同时显著降低计算量。

本条目给出可运行最小 MVP：
- 用 PyTorch 手写 `PatchEmbedding`、`WindowAttention`、`SwinBlock`；
- 显式实现 shifted-window 的 attention mask，而非调用黑盒模型；
- 在 `sklearn digits`（8x8 灰度数字）上完成训练、评估与断言。

## R02

监督分类形式化：

给定 `D = {(x_i, y_i)}_{i=1}^N`，其中 `x_i in R^{1x8x8}`，`y_i in {0,...,9}`，优化参数 `theta`：

`min_theta (1/N) * sum_i CE(f_theta(x_i), y_i)`。

在 Swin 块中，令 token 序列为 `X in R^{B x L x C}`，局部窗口注意力对每个窗口独立计算：

`Attention(Q,K,V) = Softmax(QK^T / sqrt(d) + B_rel + M_shift) V`

其中：
- `B_rel` 是相对位置偏置；
- `M_shift` 是 shifted-window 下的掩码（同窗口可见，跨窗口大负值屏蔽）。

## R03

Swin Transformer 相比标准 ViT 的关键优势：
- 复杂度更可控：全局注意力 `O((HW)^2)`，窗口注意力约 `O(HW * ws^2)`；
- 局部归纳偏置更强，适合视觉局部结构；
- 通过交替使用 W-MSA 和 SW-MSA，让信息跨窗口传播，不会长期割裂。

本 MVP 采用“两块结构”：第 1 块不平移，第 2 块平移半窗，最小化实现 Swin 的核心机制。

## R04

`demo.py` 的最小实现结构：
- `PatchEmbedding`：`Conv2d(kernel=stride=patch_size)` 生成 patch token；
- `window_partition/window_reverse`：在 `(B,H,W,C)` 与窗口序列之间转换；
- `WindowAttention`：手写 `qkv`、多头缩放点积、相对位置偏置、投影；
- `build_shifted_window_mask`：构造 SW-MSA 所需的窗口可见性掩码；
- `SwinBlock`：`LN -> (W-MSA/SW-MSA) -> Residual -> LN -> MLP -> Residual`；
- `TinySwinClassifier`：`PatchEmbed -> Block1 -> Block2 -> LN -> Token均值池化 -> FC`。

## R05

MVP 运行流程：

1. 固定随机种子（`random/numpy/torch`）。
2. 读取 `digits`，归一化到 `[0,1]`，分层划分训练/测试集。
3. 构建 `DataLoader`。
4. 初始化 `TinySwinClassifier`（2 个 Swin block）。
5. 使用 `CrossEntropyLoss + AdamW` 训练若干 epoch。
6. 每轮在测试集评估准确率。
7. 输出多数类基线、样例预测、最终指标。
8. 通过阈值断言后输出 `All checks passed.`。

## R06

正确性直觉：
- `window_partition` 与 `window_reverse` 互为逆操作，保证窗口计算后可还原空间布局；
- SW-MSA 使用循环平移 + mask，确保仅在合法窗口内做注意力，同时引入跨窗口通信路径；
- 残差连接维持训练稳定性，MLP 增加通道混合能力；
- 交叉熵直接对应多分类最大似然目标。

因此，该实现虽小，但保留了 Swin 的关键数学与工程机制。

## R07

复杂度（单个 Swin block 前向，忽略常数）：
- 令图像 token 总数 `L = H*W`，窗口内 token 数 `M = ws^2`，通道 `C`；
- 注意力主项约为 `O(L * M * C)`（每个 token 仅与同窗 `M` 个 token 交互）；
- MLP 主项约为 `O(L * C^2)`；
- 相比全局注意力 `O(L^2 * C)`，窗口化在高分辨率下更节省。

本例输入极小（8x8），CPU 即可快速训练。

## R08

边界与异常处理：
- `img_size` 必须能被 `patch_size` 整除；
- `H/W` 必须能被 `window_size` 整除；
- `dim % num_heads == 0`；
- `shift_size < window_size`；
- `test_ratio` 必须在 `(0,1)`；
- 最终测试精度需为有限值且超过设定阈值，否则抛出 `RuntimeError`。

## R09

MVP 范围与取舍：
- 保留：窗口注意力、平移窗口掩码、相对位置偏置、端到端训练；
- 省略：多 stage 层次缩减（Patch Merging）、DropPath、复杂数据增强、预训练权重加载；
- 目标是“源码透明、可运行、可验证”，不是复刻论文全尺寸模型。

## R10

`demo.py` 关键函数职责：
- `window_partition / window_reverse`：窗口切分与恢复；
- `build_shifted_window_mask`：SW-MSA 的可见性掩码；
- `WindowAttention.forward`：多头注意力完整计算（含相对位置偏置）；
- `SwinBlock.forward`：平移、分窗注意力、逆平移、残差与 MLP；
- `load_dataset / build_dataloaders`：数据准备；
- `run_epoch`：统一训练/评估逻辑；
- `main`：实验编排与质量门控。

## R11

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0249-Swin_Transformer
uv run python demo.py
```

脚本无需任何交互输入，运行结束后直接打印训练和评估结果。

## R12

输出字段说明：
- `device`：执行设备（CPU/GPU）；
- `train/test shape`：数据维度；
- `model ... params`：参数量；
- `epoch ... train_loss/train_acc/test_loss/test_acc`：每轮指标；
- `majority baseline`：多数类常数预测基线；
- `sample predictions`：前 8 个测试样本的真值、预测与最大置信度；
- `All checks passed.`：精度与稳定性断言全部通过。

## R13

最小实验设计（内置）：

1. 使用 `load_digits()`（1797 条样本，10 类）。
2. 分层切分 75% 训练、25% 测试。
3. 训练 `TinySwinClassifier` 20 个 epoch。
4. 记录训练/测试损失与准确率。
5. 与多数类基线比较，要求有显著提升。
6. 输出样例预测做可解释性 sanity check。

## R14

关键超参数：
- `patch_size=2`：决定 token 数（8x8 -> 4x4 token 网格）；
- `window_size=2`：每窗 4 token，控制注意力局部范围；
- `embed_dim=48`、`num_heads=4`：表示容量与并行头数；
- `epochs=20`、`lr=2e-3`、`weight_decay=1e-4`：影响收敛与泛化。

调参建议：优先调 `lr` 和 `epochs`，其次调 `embed_dim` 与 `window_size`。

## R15

与相关模型对比：
- 对比 CNN：Swin 用注意力显式建模 token 关系，但保留局部窗口偏置；
- 对比 ViT：Swin 不做全局注意力，计算更省，适配高分辨率更友好；
- 对比本仓库的 ResNet MVP：Swin 更强调 token 间动态关系，ResNet 更强调固定卷积先验。

## R16

典型应用场景：
- 图像分类（主干或轻量模型）；
- 目标检测/实例分割/语义分割中的 backbone；
- 医学影像、遥感图像等高分辨率视觉任务。

## R17

可扩展方向：
- 加入 `Patch Merging` 形成真正多层次 Swin stage；
- 增加 block 深度并引入 DropPath；
- 使用更高分辨率数据与更强增强策略；
- 引入学习率调度、混合精度和更完整实验追踪。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `main` 调用 `set_global_seed` 固定随机性，确保可复现。
2. `load_dataset` 读取并归一化 digits，再做分层切分，得到张量数据。
3. `PatchEmbedding.forward` 用步长卷积把图像转为 patch token 序列 `X`。
4. 第 1 个 `SwinBlock` 走 W-MSA：`window_partition -> WindowAttention -> window_reverse`，不平移窗口。
5. 第 2 个 `SwinBlock` 走 SW-MSA：先 `torch.roll` 平移，再分窗注意力，并叠加 `build_shifted_window_mask` 生成的掩码，最后逆平移。
6. `WindowAttention.forward` 内部显式执行 `qkv` 线性映射、缩放点积、相对位置偏置注入、softmax、加权求和与输出投影。
7. 每个 block 都执行两次残差：一次包裹注意力子层，一次包裹 `MLP(LN(x))`。
8. `TinySwinClassifier` 对 token 做 `LayerNorm + token均值池化 + Linear` 得到类别 logits；`run_epoch` 完成训练或评估。
9. 训练结束后计算多数类基线与样例预测，并做精度断言；通过后打印 `All checks passed.`。

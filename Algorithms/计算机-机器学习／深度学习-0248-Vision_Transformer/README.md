# Vision Transformer

- UID: `CS-0118`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `248`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0248-Vision_Transformer`

## R01

Vision Transformer（ViT）把图像分类问题转写为“序列建模”问题：先把图像切成固定大小 patch，再把 patch 当作 token 输入 Transformer 编码器，最后用 `class token` 的表示做分类。

核心思想是把 NLP 中已验证有效的自注意力机制直接迁移到视觉任务，而不是依赖卷积核的局部平移先验。

## R02

ViT 的关键里程碑是 2020 年 Google Research 提出的论文《An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale》。其主结论是：

1. 当数据规模足够大时，纯 Transformer 视觉模型可以达到或超过主流 CNN。
2. ViT 的扩展规律更接近 NLP Transformer 的“随模型/数据增大而持续收益”。
3. 下游任务中可通过预训练再微调获得稳定收益。

## R03

ViT 主要解决的问题：

1. CNN 的局部感受野在早期层捕获全局关系较慢；
2. 深层卷积网络结构设计高度依赖人工经验；
3. 在大规模预训练时代，希望统一使用 Transformer 框架处理文本/视觉。

ViT 通过全局自注意力在每层直接建模 patch 之间的关系，使远距离依赖更直接。

## R04

ViT 的基本计算链路（分类任务）：

1. 图像 `x ∈ R^{H×W×C}` 切分为 `N` 个 patch，每个 patch 展平为向量；
2. 线性映射得到 patch embedding：`z_i = E * patch_i`；
3. 拼接可学习 `cls` token，并加位置编码：`Z_0 = [cls; z_1; ...; z_N] + P`；
4. 通过 `L` 层 Transformer Encoder：`Z_l = Encoder_l(Z_{l-1})`；
5. 取最终 `cls` 表示做分类：`y_hat = Head(LN(Z_L^0))`；
6. 使用交叉熵损失训练。

## R05

设 patch 数量为 `N`、隐藏维度为 `d`、层数为 `L`。

- 自注意力主复杂度约为 `O(L * N^2 * d)`；
- MLP 主复杂度约为 `O(L * N * d^2)`；
- 空间开销中，注意力矩阵约为 `O(N^2)`。

因此 ViT 的性能瓶颈通常来自 patch 数增大导致的二次复杂度。

## R06

一个最小直观例子（与本目录 demo 一致）：

1. 输入图像大小 `8x8`（digits 数据集）；
2. patch 大小取 `2x2`，得到 `4x4=16` 个 patch；
3. 每个 patch 展平后维度是 `4`；
4. 线性投影到 `emb_dim=64`；
5. 加上 `cls` 后序列长度是 `17`；
6. 经过 2 层 Transformer Encoder，输出 10 类分类 logits。

## R07

优点：

1. 全局依赖建模直接；
2. 架构统一，易与语言多模态框架对齐；
3. 扩展到大数据/大模型时潜力高。

局限：

1. 小数据场景下通常不如强先验 CNN 稳定；
2. 计算复杂度对 token 数敏感；
3. 对训练策略和超参数较敏感。

## R08

实现 ViT MVP 的前置知识：

1. Transformer 编码器结构（MHA、MLP、残差、LayerNorm）；
2. 图像 patch 切分与线性 embedding；
3. 分类损失（cross entropy）与优化器（AdamW）；
4. PyTorch 训练/评估循环；
5. `sklearn` 数据切分与指标计算。

## R09

适用场景：

1. 中大规模视觉分类/检索任务；
2. 需要建模全局关系的视觉任务；
3. 与多模态 Transformer 共用技术栈的项目。

不适用或需谨慎：

1. 数据极少且算力受限；
2. 强实时低延迟场景（超小设备端）；
3. 需要非常强局部归纳偏置且无预训练条件的任务。

## R10

实现正确性的关键检查：

1. patch 切分与重排维度必须一致；
2. `cls token` 与位置编码长度必须等于 `num_patches + 1`；
3. 分类头输入必须是 `x[:, 0, :]`（cls 位）；
4. `CrossEntropyLoss` 输入输出维度对齐：`[B, K]` vs `[B]`；
5. `model.train()` / `model.eval()` 切换正确。

## R11

数值稳定与训练稳定要点：

1. 使用 LayerNorm 与残差连接减小梯度不稳定；
2. 使用 AdamW 并配合合理学习率；
3. 对梯度做 `clip_grad_norm_` 限制异常梯度；
4. 固定随机种子提高复现性；
5. 输入像素先缩放到 `[0,1]`，减少数值范围问题。

## R12

本 MVP 的核心超参数：

1. `patch_size=2`：token 数与局部信息粒度平衡；
2. `emb_dim=64`、`num_heads=4`、`depth=2`：足够小且可学习；
3. `epochs=18`、`lr=3e-3`：在 digits 上可稳定收敛；
4. `dropout=0.1`：抑制过拟合。

调参优先顺序通常是：先学习率，再深度/宽度，再 patch 大小。

## R13

理论性质说明：

1. ViT 训练是非凸优化问题，不保证全局最优；
2. 本任务是监督学习分类，不涉及近似比（approximation ratio）定义；
3. 泛化效果依赖数据规模、增强策略和正则化；
4. 当数据量不足时，模型可能需要更强先验或预训练迁移。

## R14

常见失效模式与处理：

1. 失效：准确率长时间停在随机水平。
   处理：检查 patch 切分维度、标签是否错位、学习率是否过大。
2. 失效：训练集高、测试集低（过拟合）。
   处理：增加 dropout、减小模型、缩短训练轮数。
3. 失效：loss 出现 NaN。
   处理：降低学习率，检查输入是否归一化，启用梯度裁剪。
4. 失效：训练过慢。
   处理：减小 `emb_dim/depth` 或 patch 数量。

## R15

工程实践建议：

1. 先用最小配置验证算法闭环，再扩展规模；
2. 保持实验可复现（seed、数据切分、关键超参固定）；
3. 在日志中同时记录 `train_acc` 和 `test_acc`；
4. 为 MVP 设置“最低质量门槛”（本 demo 设为 `final_test_acc >= 0.90`）。

## R16

与相关视觉模型的关系：

1. CNN：局部卷积归纳偏置强，小数据友好；
2. ViT：全局注意力更直接，规模化潜力更强；
3. DeiT：在数据与蒸馏策略上改进 ViT 的小数据训练；
4. Swin Transformer：通过层级窗口注意力降低复杂度，兼顾效率与精度。

ViT 是“纯 Transformer 视觉主干”的基线代表。

## R17

本目录 `demo.py` 的 MVP 特性：

1. 仅依赖 `numpy/pandas/scipy/scikit-learn/torch`；
2. 手工实现 Tiny ViT（patch embedding + cls token + pos embedding + encoder + head）；
3. 使用 `sklearn` 自带 digits 数据集，离线可运行；
4. 自动输出每轮训练指标、最终指标和样例预测；
5. 无需交互输入。

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0248-Vision_Transformer
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程拆解（8 步）：

1. `build_dataloaders` 调用 `load_digits` 读取 `8x8` 灰度图，并在 `extract_patches` 中按 `2x2` 切分成 16 个 patch 序列。  
2. `PatchDataset` 把 patch 张量与标签封装为 `torch.utils.data.Dataset`，再构建 train/test `DataLoader`。  
3. `TinyViTClassifier.__init__` 初始化 `patch_embed`、可学习 `cls_token`、`pos_embed`、`TransformerEncoder` 与分类头。  
4. 前向时先把每个 patch 线性投影到 `emb_dim`，再在序列首位拼接 `cls token` 并加位置编码。  
5. 编码序列进入多层 `TransformerEncoder`，每层内部执行多头注意力与前馈网络的残差更新。  
6. 取输出序列第 0 位 `cls` 表示，经 `LayerNorm + Linear` 得到类别 logits。  
7. `train` 中按批执行 `forward -> cross entropy -> backward -> gradient clipping -> AdamW.step`，并在每轮后调用 `evaluate` 统计 train/test 指标。  
8. 训练结束后，`show_prediction_examples` 用 `scipy.special.softmax` 把 logits 转成概率，打印样例 `true/pred/confidence` 作为可解释输出。  

该实现把 ViT 的关键组件逐层展开，避免把核心流程封装成不可见黑盒。

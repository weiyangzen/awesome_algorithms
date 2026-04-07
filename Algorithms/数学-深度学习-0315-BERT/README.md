# BERT

- UID: `MATH-0315`
- 学科: `数学`
- 分类: `深度学习`
- 源序号: `315`
- 目标目录: `Algorithms/数学-深度学习-0315-BERT`

## R01

BERT（Bidirectional Encoder Representations from Transformers）是基于 Transformer Encoder 的双向预训练语言模型。核心思想是：先在大规模无标注语料上做自监督预训练，再在下游任务上做轻量微调。

经典预训练目标包含两部分：
- MLM（Masked Language Modeling）：随机遮盖部分 token，让模型预测原词；
- NSP（Next Sentence Prediction）：判断句子 B 是否是句子 A 的真实后继句。

## R02

BERT 的背景是“预训练 + 微调”范式在 NLP 中的落地。相比仅单向建模（如早期左到右语言模型），BERT 用双向自注意力在每层同时融合左右文信息，显著提高了文本分类、问答、自然语言推理等任务的效果。

它把 Transformer Encoder 从“序列到序列的部件”升级为“通用文本表示器”，成为后续 RoBERTa、ALBERT、DeBERTa 等模型的重要基线。

## R03

BERT 解决的核心问题是：
1. 仅靠有监督小数据训练，泛化能力弱；
2. 单向上下文表示不足，难以表达复杂语义关系；
3. 下游任务各自训练成本高，难共享语言知识。

BERT 的改进路径是先学习“语言共性表示”，再把表示迁移到具体任务，从而降低标注依赖并提升收敛速度与精度。

## R04

本目录 MVP 实现了一个可运行的 Tiny-BERT 预训练流程（教育版，不追求工业规模）：

- 输入表示：`token embedding + position embedding + segment embedding`；
- 编码器：`nn.TransformerEncoder`（双向注意力）；
- 双头输出：MLM 词预测头 + NSP 二分类头；
- 总损失：
  - `L = L_mlm + L_nsp`
  - `L_mlm = CE(mlm_logits, mlm_labels, ignore_index=-100)`
  - `L_nsp = CE(nsp_logits, nsp_labels)`

`demo.py` 中显式实现了语料构造、NSP 正负样本构造、15% MLM 掩码策略、padding mask、训练循环与评估，不依赖外部黑盒训练框架。

## R05

设：
- 批大小 `B`，序列长度 `L`，隐藏维度 `d`，层数 `N`，头数 `h`，训练轮数 `T`。

单层自注意力主要开销约为 `O(B * L^2 * d)`，前馈层约为 `O(B * L * d^2)`；总训练复杂度可粗略写为：
- `O(T * N * (B * L^2 * d + B * L * d^2))`。

本 MVP 数据量很小（toy corpus），因此可在 CPU 上快速跑完并观察损失下降。

## R06

一个最小输入示意（未展开全部 token id）：

- 句子 A: `bert uses bidirectional attention`
- 句子 B: `the model is pretrained`
- 拼接后：`[CLS] A [SEP] B [SEP]`
- segment id：A 段（含 `[CLS]` 和第一个 `[SEP]`）为 0，B 段为 1。

MLM 会在非特殊 token 中抽取约 15% 位置：
- 80% 替换为 `[MASK]`，10% 随机词，10% 保持原词；
- 仅被选中的位置参与 `L_mlm`，其余位置标签为 `-100`。

## R07

优点：
- 双向上下文建模能力强；
- 预训练表示可迁移到多任务；
- 通过自监督可利用大量无标注文本。

局限：
- 预训练成本较高（算力与数据）；
- 经典 BERT 的 NSP 目标在部分场景收益有限；
- 序列长度平方复杂度使长文本成本较高。

## R08

理解和实现本算法建议具备：
- Transformer 编码器基础（多头注意力、前馈层、残差与层归一化）；
- 交叉熵损失与 softmax；
- 词表、padding、attention mask、segment id 的数据管线；
- PyTorch 的张量形状管理与训练循环（前向、反向、优化器）。

这些前置知识与 R16 中和 RoBERTa/ALBERT/GPT 的对比是对应的：先理解 BERT 组件，再看变体如何删改目标或共享参数。

## R09

适用场景：
- 文本分类、匹配、抽取、问答等 NLP 任务；
- 标注数据有限但有较多无标注文本；
- 希望通过统一预训练底座支持多任务。

不适用或需谨慎：
- 极长序列且算力受限（需考虑稀疏注意力或长序列变体）；
- 纯生成任务更偏向自回归解码器模型；
- 数据域与语言差异过大时，需领域继续预训练。

## R10

实现正确性的关键检查点：
1. 输入是否按 `[CLS] A [SEP] B [SEP]` 组织；
2. token/position/segment 三种 embedding 是否正确相加；
3. MLM 是否只在被选中位置计算损失（`ignore_index=-100`）；
4. attention mask 是否阻断 PAD 对注意力的干扰；
5. NSP 标签是否正负样本平衡、定义一致（1=真后继，0=随机句）。

## R11

数值与训练稳定性处理（MVP 已实现）：
- MLM 用 `ignore_index=-100`，避免未遮盖位对梯度造成噪声；
- `src_key_padding_mask = (attention_mask == 0)`，避免 PAD 污染上下文；
- 使用 `AdamW` + `clip_grad_norm_`（`max_norm=1.0`）降低梯度爆炸风险；
- 固定随机种子，减小小数据场景下结果抖动。

## R12

与 R04 结构对应的主要调参点：
- `max_len`：控制截断与显存开销；
- `hidden_size / num_layers / num_heads`：控制模型容量；
- `lr / batch_size / epochs`：控制优化收敛速度；
- `mask_ratio`（默认 15%）：控制 MLM 学习难度；
- `L = L_mlm + L_nsp` 的任务权重（本 MVP 使用 1:1）。

本实现默认配置为 `hidden_size=64, layers=2, heads=4, epochs=80`，目标是快速、可复现地看到损失下降与 NSP 可学习性。

## R13

理论视角（对应本实现）：
- 训练目标是深度网络上的非凸优化问题，不保证全局最优；
- 但在固定数据与随机种子下，梯度法通常可找到有效局部解；
- 本任务不是近似算法问题，因此不存在“近似比”表述；
- 成功性依赖数据覆盖、模型容量与优化稳定性（见 R11、R14 的工程约束）。

## R14

常见失效模式与防护：
- 失效：loss 不降或震荡。  
  防护：降低学习率、增加训练轮数、检查梯度裁剪是否生效。
- 失效：NSP 精度接近随机。  
  防护：检查正负样本构造逻辑与标签定义是否一致。
- 失效：MLM 预测几乎不学习。  
  防护：检查掩码比例、`ignore_index=-100`、词表映射与 label 对齐。
- 失效：padding 位置干扰注意力。  
  防护：核对 `attention_mask` 与 `src_key_padding_mask` 方向。

## R15

工程落地建议：
- 先用 Tiny 版本验证数据管线与损失定义，再扩展到大语料；
- 按阶段保存 checkpoint，并记录词表与配置文件保证复现；
- 线上任务中优先复用成熟预训练权重，领域差异大时做继续预训练；
- 在微调阶段监控任务指标与过拟合，不盲目追求更大模型。

## R16

与相关模型的关系（与 R08 前置知识对应）：
- Transformer Encoder：BERT 的骨干网络来源；
- RoBERTa：常见做法是去除 NSP、加强训练策略与数据规模；
- ALBERT：通过参数共享和分解 embedding 降低参数量；
- GPT 系列：主要是自回归（单向）目标，偏生成；
- DeBERTa 等：在位置编码和注意力建模上继续改进。

因此学习路径通常是：先掌握 BERT 的 MLM/NSP 与输入构造，再理解各变体删改了哪些部件。

## R17

本目录 `demo.py` 的 MVP 功能：
- 构造 toy 文档语料并建立词表；
- 生成 NSP 正负样本，执行 BERT 风格 15% MLM 掩码；
- 训练 TinyBERT（双头联合损失）；
- 输出初始/最终损失、最终 MLM/NSP 损失、NSP 准确率与若干 MLM 预测样例；
- 含基本断言：`loss` 必须下降，`NSP accuracy` 需达到可学习水平。

运行方式（无交互输入）：

```bash
cd Algorithms/数学-深度学习-0315-BERT
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程拆解（8 步）：
1. `build_toy_documents` 提供按“文档-句子”组织的小语料，`build_vocab` 生成包含 `[PAD]/[CLS]/[SEP]/[MASK]/[UNK]` 的词表。  
2. `encode_documents` 将句子分词并映射为 token id 序列。  
3. `make_nsp_pairs` 为每个相邻句对生成一个正样本，并从其他文档抽取负样本，形成 NSP 训练对。  
4. `build_pretrain_batch` 把样本组装为 `[CLS] A [SEP] B [SEP]`，生成 `token_type_ids`、`attention_mask`，并统一 padding 到 `max_len`。  
5. `apply_mlm_mask` 在非特殊 token 上按 15% 采样，执行 80/10/10 替换策略，同时构造 `mlm_labels`（未选位置置 `-100`）。  
6. `TinyBERT.forward` 计算 token/position/segment embedding 之和，经 TransformerEncoder 得到上下文表示，再分别输出 `mlm_logits` 与 `nsp_logits`。  
7. `train_tiny_bert` 计算 `L_mlm + L_nsp`，执行反向传播、梯度裁剪和 AdamW 更新，记录每轮损失曲线。  
8. `main` 调用 `evaluate_nsp_accuracy` 与 `inspect_mlm_predictions` 输出可解释结果，并用断言验证训练有效。

该实现没有把 BERT 作为第三方“黑盒一键调用”，而是把预训练数据流、目标函数与优化过程完整展开到源码级。

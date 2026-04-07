# ELECTRA

- UID: `CS-0125`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `262`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0262-ELECTRA`

## R01

ELECTRA（Efficiently Learning an Encoder that Classifies Token Replacements）是一种预训练范式：不再只预测被 `MASK` 的词，而是让判别器对每个位置判断“该 token 是否被替换”。相比传统 MLM 只在少量 mask 位产生监督信号，ELECTRA 在所有位置都有监督，样本效率更高。

## R02

核心思想由两个网络组成：
1. 生成器（Generator）：小模型，执行 MLM，在 mask 位预测原词。
2. 判别器（Discriminator）：主模型，输入“被生成器替换后”的序列，输出逐 token 的二分类（原始/替换）。

最终常用于下游任务的是判别器编码器参数。

## R03

记原始序列为 `x`，mask 集合为 `M`。  
生成器在 `M` 上学习：
`L_G = - sum_{i in M} log p_G(x_i | x_masked)`  
用生成器采样得到替换序列 `x_tilde` 后，判别器在所有位置学习：
`L_D = - sum_i [ y_i log D(x_tilde, i) + (1-y_i) log(1-D(x_tilde, i)) ]`  
其中 `y_i=1` 表示该位置是“被替换”的 token。  
总损失常写为 `L = L_D + lambda * L_G`。

## R04

与 BERT-MLM 的关键差异：
1. BERT 只在 mask 位预测词表分类；ELECTRA 在全部位置进行替换检测。
2. ELECTRA 用较小生成器 + 较大判别器，计算预算更高效。
3. ELECTRA 预训练后直接保留判别器编码器用于微调。

## R05

本目录 MVP 的任务定义：
1. 预训练阶段：在合成 token 序列上做 RTD（Replaced Token Detection）。
2. 迁移阶段：把判别器编码器迁移到一个序列二分类任务（前半段 token 和是否大于后半段）。

输入：离散 token 序列（`[batch, seq_len]`）。  
输出：RTD 指标（accuracy/precision/recall/F1）与下游分类准确率。

## R06

`demo.py` 使用 PyTorch 实现了最小可运行组件：
1. `TinyEncoder`：Embedding + 双向 GRU。
2. `TinyGenerator`：编码器 + 词表投影头（MLM）。
3. `TinyDiscriminator`：编码器 + 二分类头（逐 token）。
4. `SequenceClassifier`：编码器 + 序列级分类头（mean pooling）。

## R07

时间复杂度（单步近似）：
1. 编码器主耗时约为 `O(B * L * H^2)`（GRU）。
2. 生成器 MLM 头约为 `O(B * L * H * V)`，但监督仅来自 mask 位。
3. 判别器头约为 `O(B * L * H)`。

MVP 采用小维度（`emb_dim=48`，`hidden_dim=48`）控制运行成本，可在 CPU 快速完成。

## R08

关键超参数（见 `Config`）：
1. `mask_ratio=0.15`：mask 比例。
2. `pretrain_steps=140`：预训练步数。
3. `gen_loss_weight=1.0`：生成器损失权重。
4. `vocab_size=64`、`seq_len=20`：玩具语料规模。

训练稳定性措施：
1. `set_seed(42)` 固定随机性。
2. `build_mask` 保证每个样本至少一个 mask 位，避免空监督。

## R09

可复现实验流程：
1. 构造随机 clean 序列。
2. 打 mask 并训练生成器预测被遮挡 token。
3. 采样生成器输出，构造 corrupted 序列。
4. 判别器学习识别替换位置。
5. 评估 RTD 指标。
6. 迁移编码器到下游分类，比较“预训练初始化 vs 随机初始化”。

## R10

运行环境：
1. Python 3.11+（项目通过 `uv` 管理）。
2. 依赖：`torch`、`numpy`（其余标准库）。
3. 无需交互输入，无需外部数据下载。

## R11

运行命令（仓库根目录）：

```bash
uv run python Algorithms/计算机-机器学习／深度学习-0262-ELECTRA/demo.py
```

或在当前目录下：

```bash
uv run python demo.py
```

## R12

预期输出包含三部分：
1. `[pretrain] ...`：预训练过程中生成器/判别器损失与 token-level 准确率。
2. `[rtd-eval] ...`：RTD 的准确率、精确率、召回率、F1。
3. `[downstream] ...`：下游任务中预训练初始化与随机初始化的准确率对比及增益。

## R13

常见失败模式与解释：
1. 若 `replaced_rate` 很低，判别器可能靠“全预测未替换”取得虚高准确率，应结合 precision/recall/F1。
2. 生成器过强会让替换词过于接近原词，RTD 难度上升；过弱则任务过易，迁移收益下降。
3. 合成数据过简单时，下游增益可能不稳定，这是玩具任务的正常现象。

## R14

为何 ELECTRA 通常更高效：
1. MLM 的监督信号稀疏（仅 mask 位）。
2. RTD 的监督信号密集（全 token 位）。
3. 同等算力下，判别器往往学到更强表示，尤其在中小训练预算下更明显。

## R15

从 MVP 走向工程版可扩展点：
1. 编码器替换为 Transformer（BERT 风格 block）。
2. 生成器/判别器采用不同宽度（小 G、大 D）。
3. 采样策略从 multinomial 扩展到 Gumbel、top-k/top-p。
4. 接入真实语料与标准下游基准（GLUE 等）。

## R16

边界与限制：
1. 本实现是教学级最小版，不追求 SOTA 结果。
2. 数据为随机合成，结论仅用于验证算法流程可运行。
3. 未实现混合精度、分布式、学习率 warmup 等工程优化。

## R17

验收清单（本目录）：
1. `README.md` 的 R01-R18 已全部填充。
2. `demo.py` 不含任何占位符，且 `main()` 可直接执行。
3. `meta.json` 与任务元数据一致：UID=`CS-0125`，Name=`ELECTRA`，Source=`262`，目录一致。
4. `uv run python demo.py` 可在无交互情况下完成一次端到端运行。

## R18

源码级算法流程拆解（对应 `demo.py`）：
1. `sample_clean_tokens` 生成原始 token 序列，`build_mask` 采样替换位置并保证每个样本至少一个 mask。
2. 在 `pretrain_electra` 中把 mask 位替换成 `mask_token_id`，送入 `TinyGenerator`，通过 `cross_entropy(gen_logits[mask], clean[mask])` 计算 MLM 损失。
3. 仍在 `pretrain_electra` 中，对生成器在 mask 位的 softmax 分布做 `torch.multinomial` 采样；若采样与原词相同则重新采样不同 token，再构造 `corrupted` 序列。
4. 在该实现中，由于 mask 位被强制替换为不同 token，逐 token 的二值标签可直接写为 `replaced_labels = mask.float()`。
5. `TinyDiscriminator` 对 `corrupted` 输出逐 token logit，用 `binary_cross_entropy_with_logits` 计算 RTD 损失。
6. 组合总损失 `total_loss = disc_loss + gen_loss_weight * gen_loss`，联合反向传播更新生成器与判别器。
7. `evaluate_rtd` 在新批次上累计 TP/FP/FN，给出 accuracy/precision/recall/F1，验证 RTD 学习是否有效。
8. `main` 中把判别器编码器权重拷贝到 `SequenceClassifier`，并与随机初始化基线比较下游准确率，验证 ELECTRA 预训练迁移价值。

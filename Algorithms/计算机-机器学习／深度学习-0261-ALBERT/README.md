# ALBERT

- UID: `CS-0124`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `261`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0261-ALBERT`

## R01

ALBERT（A Lite BERT）是 BERT 的轻量化改造版本，核心目标是在保持 Transformer 表达能力的同时显著减少参数量和内存占用。它主要通过两点达成：

- 因子分解嵌入（Factorized Embedding Parameterization）；
- 跨层参数共享（Cross-layer Parameter Sharing）。

本条目给出可运行最小 MVP：用 PyTorch 从零实现一个 `TinyALBERT`，并在离线小语料上联合训练 `MLM + SOP` 两个预训练任务。

## R02

预训练目标定义如下。给定输入序列 `x=(x_1,...,x_T)`：

- MLM（Masked Language Modeling）：随机掩码部分 token，最小化
  `L_mlm = - sum_{t in M} log p_theta(x_t | x_{\setminus M})`；
- SOP（Sentence Order Prediction）：给句对 `(A, B)` 判断是否为正确顺序，最小化
  `L_sop = - log p_theta(y_sop | [CLS] A [SEP] B [SEP])`。

总损失：`L = L_mlm + L_sop`。

## R03

ALBERT 相对 BERT 的关键改进：

- 嵌入矩阵分解：将 `vocab_size x hidden_size` 拆成 `vocab_size x embedding_size` 与 `embedding_size -> hidden_size` 投影；
- 参数共享：多个 Transformer 编码层共享同一组参数，而不是每层一套独立参数；
- 句子级任务使用 SOP（同文档相邻句顺序）替代 NSP（是否下一句）。

这三点让模型在参数规模和泛化之间达到更高性价比。

## R04

因子分解嵌入在 `demo.py` 中由 `FactorizedEmbedding` 实现：

1. `token_embeddings: Embedding(vocab_size, embedding_size)`；
2. `embedding_projection: Linear(embedding_size, hidden_size)`；
3. 再叠加 `position_embeddings` 和 `token_type_embeddings`（均在 `hidden_size` 空间）。

参数量对比直觉：

- 传统做法约为 `O(V*H)`；
- 分解后约为 `O(V*E + E*H)`，当 `E << H` 时显著更小。

## R05

跨层参数共享在 `TinyALBERT` 中由单个 `self.shared_layer` 完成，前向时重复调用：

`for _ in range(num_hidden_layers): hidden = shared_layer(hidden, mask)`

这意味着：

- 深度仍由循环次数提供；
- 但每层不再新增一套注意力和前馈参数；
- 训练与推理内存更友好，尤其适合资源受限场景。

## R06

SOP 与 NSP 的差别：

- NSP 更偏“主题相关性”判断，容易被语义相似性投机；
- SOP 固定同一文档相邻句，负样本来自调换/替换顺序，更关注篇章连贯和顺序建模。

本 MVP 中：

- `label=0`：`B` 是 `A` 的真实后继句；
- `label=1`：`B` 来自其他文档句子（错误顺序）。

## R07

`demo.py` 的整体流程：

1. 构造 20 个小文档语料并分词建表；
2. 生成 SOP 正负句对；
3. 编码为 `[CLS] A [SEP] B [SEP]`，并构建 `token_type_ids`、`attention_mask`；
4. 进行 15% MLM 掩码，生成 `mlm_labels`；
5. 划分训练/测试集，组装 DataLoader；
6. 训练 `TinyALBERT`，优化 `L_mlm + L_sop`；
7. 输出训练曲线、最终指标和样例预测；
8. 执行阈值断言，保证最小有效性。

## R08

正确性要点：

- MLM 使用 `CrossEntropyLoss(ignore_index=-100)`，确保只在被掩码位置反传；
- SOP 基于 `[CLS]` 向量做二分类，符合 ALBERT 句级判别设计；
- `attention_mask` 转换为 `key_padding_mask`，避免 PAD 位置污染注意力；
- 训练/评估模式分离（`model.train(True/False)`）；
- 固定随机种子，减少一次性波动。

## R09

复杂度（单 batch，序列长度 `T`，隐藏维 `H`）：

- 自注意力约 `O(T^2 * H)`；
- 前馈网络约 `O(T * H * I)`（`I` 为中间层维度）；
- 共享参数不降低单次前向 FLOPs，但显著降低参数存储和优化器状态开销。

因此 ALBERT 的主要收益是“参数效率”，不是“每步计算复杂度阶数变化”。

## R10

代码模块对应关系：

- `FactorizedEmbedding`：实现低维词嵌入 + 投影；
- `SharedTransformerLayer`：单层注意力与 FFN；
- `TinyALBERT`：重复调用共享层，挂接 MLM/SOP 两个 head；
- `build_sop_pairs`：构造 SOP 监督信号；
- `apply_mlm_mask`：生成 MLM 输入与标签；
- `run_epoch`：统一训练或评估并统计 `mlm_acc/sop_acc`。

## R11

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0261-ALBERT
uv run python demo.py
```

脚本无需任何交互输入，也不依赖在线下载数据。

## R12

输出字段说明：

- `documents/sop_pairs/vocab_size`：语料规模；
- `ALBERT config`：核心结构与参数量；
- `epoch ... train_* / test_*`：每轮损失与两项任务准确率；
- `sample predictions`：若干测试样本的 SOP 与 MLM 预测摘要；
- `final metrics`：最终测试集 `MLM/SOP` 指标；
- `All checks passed.`：阈值断言通过。

## R13

内置最小实验：

1. 20 个文档，每个文档 4 句；
2. 每个相邻句构造 1 个正样本 + 1 个负样本；
3. 25% 作为测试集（分层抽样）；
4. 模型训练 25 轮；
5. 验证 `test_sop_acc` 超过随机基线并检查 `test_mlm_acc` 下限。

目标不是追求 SOTA，而是验证 ALBERT 关键机制可运行且可解释。

## R14

关键超参数与影响：

- `embedding_size`：越小参数越省，但表达能力会下降；
- `hidden_size` / `intermediate_size`：控制主干容量与计算量；
- `num_hidden_layers`：共享层重复次数，增加深度；
- `num_attention_heads`：注意力子空间粒度；
- `mlm_probability`：掩码比例，过低信号不足，过高语义破坏严重；
- `lr` 与 `epochs`：决定收敛速度与稳定性。

建议先固定结构，优先调 `lr`、`epochs`、`mlm_probability`。

## R15

与常见模型对比：

- 对比 BERT：ALBERT 通过分解与共享大幅减参；
- 对比 DistilBERT：DistilBERT主要做蒸馏与层数压缩，ALBERT强调参数复用；
- 对比 RoBERTa：RoBERTa通常更大规模训练优化，ALBERT更强调参数效率。

三者都基于 Transformer 编码器，但优化目标和工程侧重点不同。

## R16

典型应用场景：

- 资源受限部署（移动端、低显存推理）；
- 需要快速迭代的垂直领域文本建模；
- 参数预算严格但仍需保留深层语义建模能力的任务。

## R17

可扩展方向：

- 将离线小语料替换为真实领域语料并增大词表；
- 加入动态 masking（每轮重新采样掩码）；
- 补充下游微调任务（分类、匹配、抽取）；
- 替换为更完整的 ALBERT 配置（更多层/更大宽度/更长序列）；
- 加入学习率调度、早停与日志追踪。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `build_tiny_documents` 构造多文档句子序列，`build_vocab` 建立词表与特殊 token。
2. `build_sop_pairs` 以相邻句生成正样本，并从其他文档采样负样本，形成 SOP 监督标签。
3. `encode_pair` 将每个样本编码成 `[CLS] A [SEP] B [SEP]`，并同步生成 `token_type_ids` 与 `attention_mask`。
4. `apply_mlm_mask` 对非特殊 token 进行 15% 掩码，输出训练输入 `input_ids_masked` 与 `mlm_labels`。
5. `FactorizedEmbedding` 先查低维 token embedding，再投影到 hidden 维并叠加位置/句段嵌入。
6. `TinyALBERT.forward` 中把同一个 `SharedTransformerLayer` 循环调用 `num_hidden_layers` 次，实现跨层参数共享编码。
7. 编码输出分别进入 `MLM head`（并与词嵌入矩阵绑定投影）和 `SOP head`（取 `[CLS]` 做二分类），得到两路 logits。
8. `run_epoch` 计算 `L_mlm + L_sop`，训练模式下执行反向传播与参数更新，并累计 `mlm_acc/sop_acc`。
9. `main` 汇总训练日志、打印样例预测、执行阈值断言，确保模型确实学到非随机信号。

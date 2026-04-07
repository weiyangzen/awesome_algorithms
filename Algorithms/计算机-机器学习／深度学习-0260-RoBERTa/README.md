# RoBERTa

- UID: `CS-0123`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `260`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0260-RoBERTa`

## R01

RoBERTa（Robustly optimized BERT approach）是对 BERT 预训练流程的系统性改进版本，核心思想不是发明新的 Transformer 结构，而是把预训练策略做得更充分、更稳健：

- 只保留 MLM（去掉 NSP）；
- 使用动态 masking（每次看到样本都可产生不同 mask）；
- 配合更长训练、更大 batch、更大语料（工程论文设置）。

本条目的 `demo.py` 用一个离线可运行的 `TinyRoBERTaForMLM` 复现上述关键机制。

## R02

MLM 目标可写为：给定序列 `x=(x_1,...,x_T)`，随机采样掩码集合 `M`，优化

`L_mlm = - sum_{t in M} log p_theta(x_t | x_{\setminus M})`

实现细节沿用 BERT 风格 80/10/10 规则：

- 80% 替换成 `[MASK]`；
- 10% 替换成随机 token；
- 10% 保留原 token。

`demo.py` 中损失通过 `CrossEntropyLoss(ignore_index=-100)` 计算，仅在被监督位置反传。

## R03

RoBERTa 与原始 BERT 的关键差异：

- 去掉 NSP：不再做“是否下一句”二分类；
- 动态 masking：同一文本在不同 step 可 mask 不同位置；
- 更强调训练配方（数据、batch、迭代）而非结构创新。

本 MVP 对应实现：

- 无 NSP head；
- `apply_dynamic_masking` 在每个 batch 现场生成掩码；
- 训练日志输出 `loss / masked_acc / perplexity` 以监控收敛。

## R04

为什么动态 masking重要：

- 静态 masking 会让模型反复看到同一批 mask 位，监督信号利用率下降；
- 动态 masking 让同一句在多轮训练中提供更多监督组合；
- 在同等语料规模下，通常更有利于泛化。

`demo.py` 中动态逻辑由 `run_mlm_epoch -> apply_dynamic_masking` 完成，训练和验证都走该路径。

## R05

为什么去掉 NSP：

- NSP 任务容易被主题相关性等捷径“投机”；
- 对编码器 token-level 表示提升有限；
- RoBERTa 经验上表明只做 MLM 也能得到更强表示。

本条目因此将模型头部收敛为单任务 MLM，聚焦 token 恢复质量。

## R06

`demo.py` 的最小模型结构：

- `TinyRoBERTaForMLM`
- 词嵌入 + 位置嵌入 + LayerNorm + Dropout
- `nn.TransformerEncoder`（2 层）
- MLM head：`Linear -> GELU -> LayerNorm -> decoder`
- decoder 与 token embedding 权重绑定（weight tying）

这是教学级最小实现，不依赖 HuggingFace 预训练权重。

## R07

数据与流程设计：

- 预训练数据：`build_pretrain_dataframe` 生成 6 个主题的结构化文本；
- 分词：正则分词 `simple_tokenize`；
- 词表：`build_vocab` 自动构建并保留特殊 token；
- 编码：`encode_text / encode_many` 统一到固定长度；
- 训练集划分：`train_test_split(..., stratify=topic)`。

全部数据在脚本内构造，无下载、无交互。

## R08

正确性关键点：

- `attention_mask` 同步传入 Transformer，避免 PAD 污染；
- `apply_dynamic_masking` 保证每个样本至少一个监督位，避免空损失；
- `ignore_index=-100` 确保只在 mask 位置训练；
- 固定随机种子便于复现；
- 验证阶段统计 `masked_entropy`，监控输出分布是否异常。

## R09

复杂度分析（单 batch，序列长度 `T`，隐藏维 `H`）：

- 自注意力主项约 `O(T^2 * H)`；
- 前馈网络约 `O(T * H * I)`（`I` 为 FFN 中间维）；
- MLM 词表投影约 `O(T * H * V)`（`V` 为词表大小）。

本 MVP 通过较小 `H=64`、`L=2` 控制运行成本，保证 CPU 也可快速完成。

## R10

代码模块映射：

- `Config`：集中管理超参数；
- `TinyRoBERTaForMLM`：编码器与 MLM 头；
- `apply_dynamic_masking`：RoBERTa 风格动态 mask；
- `run_mlm_epoch`：单轮训练/验证与指标统计；
- `build_role_topic_probe_dataframe`：下游角色-主题匹配分类数据；
- `extract_cls_embeddings + run_linear_probe`：线性探针迁移评估。

## R11

运行方式（当前目录）：

```bash
uv run python demo.py
```

无需命令行参数、无需手工输入、无需联网。

## R12

预期输出字段说明：

- `pretrain_samples/train/val/vocab_size`：数据规模；
- `config=...`：模型与训练配置；
- `epoch=... train_loss/train_acc val_loss/val_acc val_ppl val_entropy`：逐轮收敛情况；
- `[linear-probe] ...`：预训练编码器与随机编码器在线性探针下的对比；
- `accuracy_gain_pretrained_minus_random`：迁移增益；
- `All checks passed.`：最小验收断言通过。

## R13

内置实验设计：

1. 先在结构化文本上做纯 MLM 预训练；
2. 再构建“角色-主题是否匹配”的二分类任务（词汇相近、配对关系不同）；
3. 提取 `[CLS]` 表征并训练 `LogisticRegression`；
4. 对比“预训练编码器特征”与“随机编码器特征”的分类效果。

该实验用于观察预训练表示的迁移效果，并与随机初始化编码器做同口径对照。

## R14

关键超参数与影响：

- `mlm_probability=0.15`：监督稀疏度；
- `hidden_size/num_layers/num_heads`：容量与计算成本；
- `max_len`：上下文窗口；
- `lr` 与 `pretrain_epochs`：收敛速度和最终质量；
- `dropout`：正则化强度。

调参建议：先固定结构，优先调 `lr`、`epochs`、`mlm_probability`。

## R15

与相邻模型对比（本仓条目维度）：

- 对比 BERT：RoBERTa 重点在训练策略优化（动态 mask、去 NSP）；
- 对比 ALBERT：ALBERT重点是参数压缩（共享层、分解嵌入）；
- 对比 ELECTRA：ELECTRA用 RTD 判别任务提高样本效率。

三者都可视作 Transformer 编码器预训练家族，但优化目标不同。

## R16

适用场景：

- 文本分类、匹配、检索等通用语言表示需求；
- 需要稳定强基座、再做任务微调的 NLP 工程；
- 先做无监督预训练，再迁移到小样本监督任务。

本 MVP 偏教学验证，不是生产版大模型训练脚本。

## R17

验收清单（本目录）：

- `README.md` 的 R01-R18 全部已填充；
- `demo.py` 无占位符，`main()` 可直接运行；
- `meta.json` 与任务元数据一致（UID=`CS-0123`，Name=`RoBERTa`，Source=`260`）；
- `uv run python demo.py` 全流程执行并输出最终检查通过。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `build_pretrain_dataframe` 生成多主题结构化文本，`build_vocab` 建词表并保留 `[PAD]/[CLS]/[SEP]/[MASK]/[UNK]`。
2. `encode_many` 把文本编码成定长 `input_ids + attention_mask`，再用 `train_test_split` 构建训练/验证集。
3. `TinyRoBERTaForMLM` 初始化 embedding、TransformerEncoder 和 MLM head，并进行 embedding-decoder 权重绑定。
4. `run_mlm_epoch` 每个 batch 调用 `apply_dynamic_masking` 现场采样 mask 位，执行 80/10/10 替换策略。
5. 前向阶段把 `masked_input_ids` 与 `attention_mask` 输入编码器，输出 token 级 logits。
6. 用 `CrossEntropyLoss(ignore_index=-100)` 只在被 mask 位置计算 MLM 损失；训练模式下反向传播更新参数。
7. 同步统计 `masked_acc`，验证模式下基于 `softmax` 概率计算 `masked_entropy` 与 `perplexity`。
8. 预训练结束后，`build_role_topic_probe_dataframe` 构造角色-主题匹配下游任务；`extract_cls_embeddings` 提取 `[CLS]` 表征。
9. `run_linear_probe` 训练逻辑回归并对比“预训练编码器 vs 随机编码器”，最后执行阈值断言并输出 `All checks passed.`。

# GPT系列

- UID: `MATH-0316`
- 学科: `数学`
- 分类: `深度学习`
- 源序号: `316`
- 目标目录: `Algorithms/数学-深度学习-0316-GPT系列`

## R01

GPT 系列（Generative Pre-trained Transformer）是基于 Transformer Decoder 的自回归语言模型家族。其核心目标是学习条件概率：

`p(x_1, ..., x_T) = Π_t p(x_t | x_{<t})`

训练时采用 next-token prediction（下一个 token 预测），推理时逐步生成：每次根据已有上下文采样一个新 token，再拼接回上下文继续生成。

## R02

GPT 系列的发展脉络可概括为：
- GPT-1：验证“预训练 + 微调”在 NLP 的有效性；
- GPT-2：扩大模型和数据规模，展示强生成能力；
- GPT-3：通过大规模参数实现 few-shot/in-context learning；
- 后续 GPT 系列：在对齐、推理、多模态与工具调用方面持续演进。

其工程主线是“Decoder-only 架构 + 大规模自监督预训练 + 指令/对齐阶段”。

## R03

GPT 主要解决的问题：
1. 传统任务特定模型迁移能力弱，泛化不足；
2. 规则系统难覆盖开放文本场景；
3. 端到端生成需要统一理解与生成能力。

通过大规模语料上的统一语言建模目标，GPT 把“语言统计规律”内化为参数，从而在多任务上以同一模型执行生成、改写、问答、摘要等任务。

## R04

GPT 的核心计算流程（单层视角）：
1. token/position embedding 相加得到输入表示 `h`；
2. 计算 `Q,K,V`，并做缩放点积注意力；
3. 使用因果掩码（causal mask）屏蔽未来位；
4. softmax 后加权求和得到注意力输出；
5. 残差连接 + MLP 前馈 + 残差连接；
6. 最后线性映射到词表 logits，使用交叉熵训练。

训练目标：
`L = - Σ_t log p_theta(x_t | x_{<t})`

## R05

设批大小 `B`、序列长度 `L`、隐藏维 `d`、层数 `N`。

- 自注意力主耗时约 `O(B * N * L^2 * d)`；
- 前馈层主耗时约 `O(B * N * L * d^2)`；
- 推理阶段若逐 token 生成且不做 KV cache，复杂度会随已生成长度增长。

因此 GPT 对长序列计算开销敏感，实践中常通过 cache、并行优化和硬件加速降低成本。

## R06

一个最小自回归示例：
- 上下文：`"gpt is"`
- 模型输出下一个 token 分布（示意）：
  - `a: 0.55`
  - `the: 0.15`
  - `decoder: 0.12`
  - 其他：`0.18`

若采样到 `a`，新上下文变为 `"gpt is a"`，再预测下一 token。重复该过程即可生成完整文本。

## R07

优点：
- 架构统一，任务迁移能力强；
- 自回归目标简单稳定，易扩展到大数据；
- 生成质量高，适配对话与文本创作。

局限：
- 训练与推理成本高；
- 存在幻觉、偏见和事实不确定性问题；
- 超长上下文时计算/显存压力大。

## R08

实现 GPT MVP 的前置知识：
- Transformer 基础：多头注意力、残差、层归一化、前馈网络；
- 交叉熵损失与 softmax；
- 自回归数据构造（输入右移一位作为标签）；
- PyTorch 训练循环（前向、反向、优化器、梯度裁剪）。

## R09

适用场景：
- 语言生成、改写、摘要、对话；
- 大规模文本模式学习；
- 统一底座模型的多任务迁移。

不适用或需谨慎：
- 需要严格可验证结论的高风险决策；
- 数据极少但要求高可靠推理的场景；
- 受限硬件下的长上下文低延迟需求。

## R10

实现正确性的关键检查项：
1. 因果掩码必须屏蔽未来位置；
2. 训练标签必须是输入序列的右移版本；
3. logits 与 target 的 shape 对齐到 `(B*L, vocab_size)` vs `(B*L,)`；
4. 推理阶段每步只用已有上下文，不能泄露未来 token；
5. `eval()` 与 `train()` 模式切换要正确。

## R11

数值稳定性与训练稳定性要点：
- 注意力分数按 `sqrt(d_head)` 缩放，避免 softmax 过饱和；
- 使用 LayerNorm 减小深层网络梯度不稳定；
- 使用 `clip_grad_norm_` 控制梯度爆炸；
- 采用 AdamW + 合理学习率提升收敛稳定性；
- 在 demo 中固定随机种子，减少结果抖动。

## R12

本 MVP 的主要调参点：
- `block_size`：上下文窗口；
- `n_embd / n_head / n_layer`：模型容量；
- `lr / max_steps / batch_size`：优化过程；
- `temperature`：生成时的随机性。

经验上，先保证损失下降，再逐步增加模型规模与训练步数。

## R13

理论性质说明：
- GPT 训练对应非凸优化问题，不保证全局最优；
- 本任务不属于近似算法问题，因此无“近似比”结论；
- 实际效果依赖数据质量、规模、优化稳定性和评估方案。

## R14

常见失效模式与防护：
- 失效：loss 不下降。
  - 防护：检查标签右移与掩码逻辑，降低学习率。
- 失效：生成文本重复或退化。
  - 防护：调整 temperature、增加训练步数或数据多样性。
- 失效：训练出现 NaN。
  - 防护：梯度裁剪、降低学习率、检查输入值范围。
- 失效：显存不足。
  - 防护：减小 batch size / block size / 模型维度。

## R15

工程实践建议：
- 先用 tiny 配置验证算法链路，再做规模扩展；
- 保留可复现配置（seed、超参数、词表）；
- 把“训练有效性断言”写入脚本，避免静默失败；
- 生成质量评估应结合自动指标与人工抽样。

## R16

与相关模型关系：
- BERT：双向编码器，偏表示学习与理解任务；
- GPT：单向解码器，偏生成与补全；
- T5/Encoder-Decoder：统一 seq2seq 表达；
- LLaMA 等开源大模型：沿用 GPT 式 decoder-only 主线并做工程改进。

因此 GPT 系列可以视为“大规模自回归 Transformer”路线的核心代表。

## R17

本目录 `demo.py` 的 MVP 特性：
- 用 PyTorch 手写 Tiny GPT（未调用 `nn.Transformer` 黑盒）；
- 显式实现：因果多头注意力、残差块、前馈层、next-token 训练；
- 自动输出初始/最终 train+val loss，并断言 `val loss` 改善；
- 训练后从提示词 `"gpt "` 生成样本文本。

运行方式（无交互）：

```bash
cd Algorithms/数学-深度学习-0316-GPT系列
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程拆解（8 步）：
1. `build_toy_corpus` 构造小语料，`build_vocab`/`encode` 建立字符级词表与 token 序列。  
2. `main` 中按 9:1 切分 `train_data` 与 `val_data`，创建 `GPTConfig` 并实例化 `TinyGPT`。  
3. `TinyGPT` 初始化 token/position embedding、若干 `TransformerBlock`、最终 `lm_head`。  
4. `TransformerBlock` 内先做 LayerNorm，再走 `CausalSelfAttention`；注意力里由 `qkv_proj` 得到 `Q/K/V`，并用下三角 `causal_mask` 屏蔽未来位。  
5. 注意力结果经输出投影后与残差相加，再经第二个 LayerNorm + `FeedForward`，形成标准 GPT block 更新。  
6. `get_batch` 采样长度为 `block_size` 的输入片段 `x`，并构造右移标签 `y`，`forward` 用交叉熵计算 next-token loss。  
7. 训练循环执行 `loss.backward()`、`clip_grad_norm_`、`AdamW.step()`，并通过 `estimate_loss` 比较初始与最终 train/val 损失。  
8. 收敛后 `generate` 从最后一个位置 logits 采样下一个 token，循环追加，输出自回归生成文本样例。  

该实现把 GPT 的关键机制在源码中逐层展开，而不是把训练和推理交给单一高层黑盒接口。

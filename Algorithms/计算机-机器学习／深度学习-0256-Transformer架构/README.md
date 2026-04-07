# Transformer架构

- UID: `CS-0122`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `256`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0256-Transformer架构`

## R01

Transformer 架构是一种以注意力机制为核心的序列建模框架。它用“多头自注意力 + 前馈网络 + 残差连接 + 层归一化”替代传统 RNN 的逐步递归，使模型能够在训练时并行处理整段序列，并更直接地建模长距离依赖。

标准形式是编码器-解码器：
- 编码器把源序列映射为上下文表示；
- 解码器在因果掩码约束下逐 token 生成目标序列，并通过交叉注意力读取编码器信息。

## R02

该架构由 Vaswani 等人在 2017 年提出（Attention Is All You Need），其关键改进是把序列关系表达为注意力矩阵，而非仅靠隐藏状态链式传递。  
相较 RNN/LSTM：
- 训练阶段并行度更高；
- 长程依赖路径更短（任意位置可直接交互）；
- 便于通过堆叠层数和宽度扩展规模。

## R03

本任务关注“Transformer 架构本身”的最小可运行验证，而不是某个下游大模型变体。  
目录中的 MVP 选择了一个可控任务：**数字序列反转**，用它验证以下核心点：
- 编码器-解码器数据流正确；
- 解码器因果掩码正确；
- 训练（teacher forcing）与推理（自回归）一致；
- 模型确实学到了序列变换而非偶然记忆。

## R04

`demo.py` 实现了源码透明的 Tiny Transformer（未调用 `torch.nn.Transformer`）：
- 手写 `MultiHeadAttention`（`Q/K/V` 投影、分头、缩放点积、mask、拼接）；
- 手写 `EncoderLayer` 与 `DecoderLayer`；
- 正弦位置编码 `PositionalEncoding`；
- 训练循环（交叉熵 + Adam + 梯度裁剪）；
- 贪心自回归解码与 exact-match 评估。

## R05

核心数学公式：

1. 缩放点积注意力  
`Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + M) V`

2. 多头注意力  
`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`  
`MHA(Q,K,V) = Concat(head_1, ..., head_h) W^O`

3. 前馈层  
`FFN(x) = W_2 * GELU(W_1 x + b_1) + b_2`

4. 子层结构  
`y = LayerNorm(x + SubLayer(x))`

## R06

MVP 的监督任务定义：
- 源序列：长度 3~7 的随机数字（1..9）；
- 目标序列：源序列反转；
- 特殊 token：`PAD=0, BOS=10, EOS=11`；
- 训练输入：
  - `src = source + EOS`
  - `tgt_in = BOS + target`
  - `tgt_out = target + EOS`
- 损失函数：`CrossEntropyLoss(ignore_index=PAD)`。

## R07

复杂度（单层，忽略常数）：
- 注意力主项：`O(B * L^2 * d)`；
- 前馈主项：`O(B * L * d^2)`；
- 总体随层数 `N` 约为 `O(N * (B*L^2*d + B*L*d^2))`。

对于本目录的短序列任务（`L<=8`），CPU 即可在较短时间内完成训练。

## R08

实现中的关键正确性约束：
1. `d_model % num_heads == 0`，保证每头维度整数化；
2. 解码器必须使用下三角因果 mask，禁止“看未来”；
3. `PAD` token 在注意力中不可见；
4. `tgt_in` 与 `tgt_out` 必须严格右移对齐；
5. 推理遇到 `EOS` 立即停止。

## R09

数值稳定性与可复现策略：
- 注意力分数按 `1/sqrt(d_k)` 缩放；
- mask 位填充 `-1e9` 防止 softmax 泄漏；
- 每个子层使用 `Residual + LayerNorm`；
- 训练时 `clip_grad_norm_(..., 1.0)`；
- 固定 `numpy/torch` 随机种子。

## R10

`demo.py` 模块划分：
- `ModelConfig`：超参数容器；
- `PositionalEncoding`：位置编码；
- `MultiHeadAttention`：多头注意力核心计算；
- `EncoderLayer` / `DecoderLayer`：Transformer 子层堆叠单元；
- `TransformerSeq2Seq`：完整编码器-解码器模型；
- `build_reverse_dataset` / `tensorize_pairs`：数据构造；
- `train_model`：训练循环；
- `greedy_decode` / `evaluate_exact_match`：推理与评估。

## R11

运行方式（无交互）：

```bash
cd Algorithms/计算机-机器学习／深度学习-0256-Transformer架构
uv run python demo.py
```

脚本会直接打印训练前后 loss、测试 exact-match 和若干样例预测。

## R12

默认超参数（可在 `main` 中看到）：
- `d_model=64`
- `num_heads=4`
- `num_layers=2`
- `d_ff=128`
- `epochs=70`
- `batch_size=32`
- `lr=2e-3`
- 训练样本 `480`，测试样本 `120`

这些配置目标是“可运行 + 可观察收敛”，不是追求大规模 SOTA。

## R13

判定训练有效的最小标准：
1. `initial_loss > final_loss`；
2. `exact_match` 明显高于随机水平；
3. 样例预测与目标序列在多数案例上对齐。

脚本中包含断言：
- `loss` 必须下降；
- `exact_match >= 0.75`。

## R14

常见问题与排查：
- 现象：loss 不下降  
  原因：标签对齐错误、学习率过高、mask 逻辑错误。
- 现象：预测过早结束  
  原因：`EOS` 学习偏置过强，可增加训练轮次或调整数据。
- 现象：输出重复 token  
  原因：训练不足或模型容量不足，可提高 `epochs/d_model`。
- 现象：训练不稳定  
  原因：梯度过大，检查学习率与梯度裁剪设置。

## R15

与相关模型关系：
- 相比 LSTM/GRU：Transformer 训练并行性更强，长依赖路径更短；
- 相比 BERT：BERT 是基于 Transformer 编码器的双向预训练模型；
- 相比 GPT：GPT 主要使用 Transformer 解码器做自回归生成；
- 本条目实现的是基础 Encoder-Decoder Transformer 骨架。

## R16

适用场景：
- 机器翻译、摘要、文本改写等序列到序列任务；
- 需要显式建模全局 token 关系的任务；
- 训练阶段可利用并行算力的场景。

不适合直接使用标准全注意力的场景：
- 极长上下文且显存紧张（需稀疏/线性/分块注意力改造）。

## R17

MVP 取舍说明：
- 保留 Transformer 的核心结构和关键数学机制；
- 省略大规模训练技巧（学习率调度、混合精度、分布式训练等）；
- 任务使用可控合成数据，避免外部数据依赖；
- 重点在“原理可追踪、实现可运行、结果可验证”。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `build_reverse_dataset` 生成随机数字序列，并构造其反转标签。  
2. `tensorize_pairs` 将样本转成 `src / tgt_in / tgt_out` 三类张量并做 padding。  
3. `PositionalEncoding` 预计算正弦/余弦编码并注入 token 表示。  
4. `MultiHeadAttention` 执行 `Q/K/V` 线性投影、分头、`QK^T/sqrt(d_k)`、mask、softmax、加权求和和输出投影。  
5. `EncoderLayer` 堆叠“自注意力 + FFN”，每个子层都走残差与层归一化。  
6. `DecoderLayer` 先做带因果 mask 的自注意力，再做对编码器输出的交叉注意力，最后 FFN。  
7. `train_model` 用交叉熵训练，执行反向传播、梯度裁剪和 Adam 更新，记录每轮 loss。  
8. `greedy_decode` 自回归逐 token 生成，`evaluate_exact_match` 计算精确匹配率，`main` 完成断言与样例输出。

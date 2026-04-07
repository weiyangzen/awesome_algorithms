# 自注意力机制

- UID: `MATH-0314`
- 学科: `数学`
- 分类: `深度学习`
- 源序号: `314`
- 目标目录: `Algorithms/数学-深度学习-0314-自注意力机制`

## R01

自注意力机制（Self-Attention）的目标是：让序列中每个位置都能根据“与其它位置的相关性”动态聚合信息。

本条目的 MVP 聚焦单头缩放点积自注意力：
- 输入：序列嵌入矩阵 `X ∈ R^{n×d_model}`；
- 投影：`Q = XW_Q`、`K = XW_K`、`V = XW_V`；
- 相关性：`S = QK^T / sqrt(d_k)`；
- 归一化：`P = softmax(S)`（可叠加因果 mask）；
- 输出：`O = PV`。

## R02

背景定位（简要）：
- 自注意力是 Transformer 的核心运算单元；
- 相比循环结构，它允许并行计算并直接建模长程依赖；
- 工程中常见版本为“多头注意力 + 前馈层 + 残差 + 归一化”；
- 本目录先实现“可解释、可验证”的单头最小闭环，强调数学流程透明。

## R03

MVP 输入输出约定（脚本内部固定构造，无交互）：
- 输入：
1. 固定随机种子下构造的 `x`，形状 `(seq_len, d_model)`；
2. 投影矩阵 `W_Q/W_K/W_V`；
3. 可选因果 mask（上三角屏蔽未来位置信息）。
- 输出：
1. `Q/K/V`、分数矩阵、注意力概率矩阵与最终输出张量；
2. 形状信息与四舍五入后的数值；
3. 自动检查通过标记 `All checks passed.`。

## R04

典型应用：
- NLP：语言建模、机器翻译、问答；
- 视觉：ViT 中 patch token 的全局关系建模；
- 时间序列：多变量序列中跨时刻依赖捕获；
- 推荐系统：用户行为序列权重聚合。

## R05

核心数学关系：
- 线性投影：`Q = XW_Q`、`K = XW_K`、`V = XW_V`；
- 打分：`s_{ij} = <q_i, k_j> / sqrt(d_k)`；
- 概率：`p_{ij} = exp(s_{ij}) / Σ_j exp(s_{ij})`；
- 聚合：`o_i = Σ_j p_{ij} v_j`。

缩放因子 `sqrt(d_k)` 的作用：避免 `d_k` 大时点积值过大，使 softmax 过度饱和导致梯度不稳定。

## R06

单头自注意力计算流程（向量化视角）：
1. 构造输入嵌入 `X`；
2. 计算 `Q/K/V` 三个投影；
3. 计算相关性矩阵 `S = QK^T / sqrt(d_k)`；
4. 若开启因果约束，对 `S` 的上三角做 mask；
5. 对每一行做 softmax 得到 `P`；
6. 计算输出 `O = PV`。

## R07

本实现的工程选择：
1. 使用 PyTorch 张量实现核心矩阵运算；
2. 启用因果 mask（`i` 位置不能看 `j>i`）；
3. 添加循环版 reference 实现，与向量化实现做数值对齐校验。

这样既有“真实可跑”的张量实现，也保留“源码可审计”的逐步计算对照。

## R08

时间复杂度（单头，序列长度 `n`）：
- `Q/K/V` 投影：约 `O(n d_model d_k)` 与 `O(n d_model d_v)`；
- 相关性矩阵 `QK^T`：`O(n^2 d_k)`；
- softmax：`O(n^2)`；
- 聚合 `PV`：`O(n^2 d_v)`。

当 `n` 很大时，瓶颈通常来自 `n^2` 项。

## R09

空间复杂度：
- `Q/K/V`：`O(n d_k + n d_k + n d_v)`；
- 分数与概率矩阵：`O(n^2)`；
- 输出：`O(n d_v)`。

因此标准全连接自注意力的内存瓶颈也主要在 `O(n^2)` 级别的注意力矩阵。

## R10

技术栈：
- Python 3
- `torch`
- 标准库：`math`、`dataclasses`

说明：未调用 `torch.nn.MultiheadAttention` 直接黑盒求值，核心公式在 `demo.py` 内显式展开。

## R11

运行方式：

```bash
cd Algorithms/数学-深度学习-0314-自注意力机制
uv run python demo.py
```

脚本不接收命令行参数，不会请求交互输入。

## R12

输出字段说明：
- `Input embedding shape`：输入 `X` 的形状；
- `Q/K/V shape`：三个投影结果形状；
- `Attention score/prob shape`：相关性与概率矩阵形状；
- `Output shape`：聚合后输出形状；
- `Attention probabilities`：每行是当前位置对全序列的注意力分布；
- `All checks passed.`：行和、mask 与 reference 对齐检查均通过。

## R13

鲁棒性与校验策略：
- 固定随机种子保证可复现；
- 检查每行注意力概率和是否接近 1；
- 检查因果 mask 后上三角概率是否接近 0；
- 用循环版 reference 对比向量化输出，限制最大误差阈值。

## R14

当前局限：
- 仅实现单头注意力；
- 未包含训练过程、损失函数与反向传播演示；
- 未实现相对位置编码、RoPE、KV cache 等推理优化；
- 未覆盖稀疏注意力、线性注意力等长序列优化变体。

## R15

可扩展方向：
- 从单头扩展到多头（Multi-Head Attention）；
- 加入残差连接与 LayerNorm，贴近 Transformer Block；
- 增加批次维度 `(batch, seq, d_model)`；
- 增加 padding mask 与 cross-attention；
- 对比 Flash Attention / xFormers 等高性能实现。

## R16

最小测试建议：
- 形状测试：不同 `seq_len`、`d_model` 是否保持维度一致；
- 数值测试：向量化与循环 reference 的最大误差；
- 约束测试：因果 mask 上三角概率近 0；
- 稳定性测试：不同随机种子是否均能稳定通过检查。

## R17

方案对比（简要）：
- `torch.nn.MultiheadAttention`：工程调用最方便，但源码级教学透明度较低；
- 本实现（显式 Q/K/V + 缩放点积）：可直接看到每一步数学映射；
- RNN/LSTM：按时间步串行，长程依赖路径更长；
- 自注意力：并行性好，长程依赖更直接，但 `n^2` 成本在长序列上较高。

## R18

`demo.py` 源码级流程拆解（8 步）：
1. `AttentionConfig` 固定 `seq_len/d_model/d_k/d_v` 与随机种子，形成可复现实验配置。  
2. `build_input_embeddings()` 生成输入矩阵 `X`，并叠加轻量位置偏置，模拟序列位置信息差异。  
3. `build_projection_matrices()` 构造 `W_Q/W_K/W_V`，随后在 `self_attention()` 中得到 `Q/K/V`。  
4. `scaled_dot_product_attention()` 计算 `S = QK^T / sqrt(d_k)`，若开启因果模式则用 `mask` 屏蔽上三角。  
5. 对 `S` 做按行 `softmax` 得到注意力概率 `P`，再计算输出 `O = PV`。  
6. `self_attention_reference_loops()` 用双层循环逐项重算同一机制，构建可审计 reference。  
7. `run_checks()` 执行三项检查：行和为 1、mask 无泄漏、向量化与循环版输出误差在阈值内。  
8. `main()` 打印形状与关键矩阵（四舍五入展示），最后输出 `All checks passed.` 作为成功标记。  

说明：此实现把“投影、打分、归一化、加权求和”四个核心动作全部显式写在源码中，不依赖高层注意力黑盒。

# 注意力机制

- UID: `CS-0121`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `255`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0255-注意力机制`

## R01

注意力机制（Attention Mechanism）的核心目标是：让模型在处理当前查询（query）时，动态选择并加权最相关的信息，而不是对全部信息一视同仁。

在本目录 MVP 中，采用的是最常见的“缩放点积自注意力”（Scaled Dot-Product Self-Attention）：

1. `Q = XW_Q, K = XW_K, V = XW_V`；
2. `S = QK^T / sqrt(d_k)`；
3. `P = softmax(S)`；
4. `O = PV`。

## R02

历史脉络（简要）：

1. 早期在机器翻译中，Bahdanau Attention 解决编码器-解码器的“固定长度瓶颈”；
2. Luong Attention 推动点积注意力在序列任务上的工程化；
3. Transformer 把注意力提升为主干计算单元，形成现代深度学习中最重要的模块之一。

## R03

注意力机制主要解决的问题：

1. 长距离依赖难建模：远处信息对当前决策影响难以显式表达；
2. 信息选择缺失：传统均匀池化无法区分关键信号与噪声；
3. 串行瓶颈：相比逐步递归结构，注意力更便于并行计算。

## R04

本条目采用的单头自注意力计算链路（与 `demo.py` 一致）：

1. 输入序列 `X ∈ R^{B×N×d_model}`；
2. 线性投影得到 `Q/K/V`；
3. 计算相关性分数 `S = QK^T / sqrt(d_k)`；
4. 对最后一维做 softmax 得到权重 `P`；
5. 用 `P` 对 `V` 做加权求和得到上下文 `O`；
6. 与残差、LayerNorm、FFN 组合成注意力块；
7. 用 `cls token` 表示做分类头输出 logits。

## R05

复杂度（单头自注意力，序列长度 `N`，通道维度 `d`）：

1. `Q/K/V` 投影约 `O(B * N * d^2)`；
2. 分数矩阵 `QK^T` 约 `O(B * N^2 * d)`；
3. 注意力加权 `PV` 约 `O(B * N^2 * d)`；
4. 空间开销中，注意力矩阵是 `O(B * N^2)`。

因此当 `N` 变大时，`N^2` 项通常是性能与显存瓶颈。

## R06

一个直观例子：

1. 数字图像 `8x8` 可看成长度为 8 的序列（每行是一个 token，维度 8）；
2. 模型在每一层中计算“行与行之间”的相关性；
3. `cls token` 会学习关注对分类最有贡献的行；
4. 最后用 `cls` 的表征预测 0~9 类别。

这比“简单平均所有行”更灵活，因为权重是样本相关的。

## R07

优点：

1. 动态加权，信息利用更精细；
2. 长程关系建模直接；
3. 并行友好，适合现代硬件。

局限：

1. 序列长时 `N^2` 成本高；
2. 对超参数较敏感；
3. 数据较小时可能不如强归纳偏置模型稳定。

## R08

实现本 MVP 的前置知识：

1. 线性代数：矩阵乘法、转置、softmax；
2. PyTorch：张量、模块、自动求导；
3. 监督学习分类流程（loss、优化器、评估指标）；
4. 基本数据切分与可复现控制（随机种子）。

## R09

适用场景：

1. NLP/视觉/多模态中的序列建模；
2. 需要按样本动态聚焦关键信息的任务；
3. 希望统一在 Transformer 技术栈下开发。

不太适用：

1. 极长序列且资源受限；
2. 数据量非常小且无预训练可迁移；
3. 对时延极端敏感的边缘设备场景。

## R10

实现正确性的关键检查：

1. 输入维度必须是 `[batch, seq_len, input_dim]`；
2. 分数矩阵形状应是 `[batch, seq, seq]`；
3. softmax 维度必须沿 `key` 维（最后一维）；
4. `cls token` 与位置编码长度要匹配 `seq_len + 1`；
5. 分类头输入应来自 `cls` 位表示。

## R11

数值稳定与训练稳定要点：

1. 使用 `sqrt(d_k)` 缩放点积，避免 softmax 饱和；
2. 残差 + LayerNorm 缓解梯度不稳定；
3. 使用 `AdamW` 并做梯度裁剪；
4. 输入归一化到 `[0,1]`；
5. 固定随机种子保证复现实验。

## R12

本 demo 超参数：

1. `d_model=64`, `d_k=64`, `depth=2`：规模小但足够展示机制；
2. `dropout=0.10`：降低过拟合；
3. `epochs=18`, `lr=2e-3`：在 digits 上可稳定收敛；
4. `batch_size=64`：兼顾速度与梯度稳定性；
5. 质量门槛：`final_test_acc >= 0.90`。

## R13

理论性质说明：

1. 注意力层本质是可学习的内容相关加权映射；
2. 训练问题是非凸优化，不保证全局最优；
3. 该分类任务不涉及近似比（approximation ratio）；
4. 泛化表现受数据规模、正则化与模型容量共同影响。

## R14

常见失效模式与处理：

1. 现象：准确率接近随机。  
   处理：检查 softmax 维度、标签对齐、学习率设置。
2. 现象：训练高、测试低。  
   处理：提高 dropout、降低模型宽度、提前停止。
3. 现象：loss 出现 NaN。  
   处理：降低学习率，检查输入范围，开启梯度裁剪。
4. 现象：训练慢。  
   处理：降低 `d_model/depth`，减少 epoch。

## R15

工程实践建议：

1. 先做最小闭环（可训练、可评估、可解释）再扩展；
2. 保持日志结构化输出，便于回归比较；
3. 同时监控 `train_acc` 与 `test_acc`；
4. 对注意力权重做抽样打印，验证模型是否学到可解释焦点。

## R16

与相关方法关系：

1. Additive Attention：通过小网络打分，早期 seq2seq 常用；
2. Dot-Product Attention：计算更高效，易向量化；
3. Multi-Head Attention：多个子空间并行关注，是 Transformer 标配；
4. Cross-Attention：`Q` 与 `K/V` 来自不同序列，常用于编码器-解码器或多模态。

本 MVP 专注“单头缩放点积自注意力”，突出核心机制透明性。

## R17

本目录 `demo.py` 的 MVP 特性：

1. 显式实现 `Q/K/V -> score -> softmax -> weighted sum`，不调用黑盒 `MultiheadAttention`；
2. 使用 `sklearn` digits 离线数据集（`8x8`），无需外部下载；
3. 实际使用 `numpy/pandas/scipy/scikit-learn/torch`；
4. 输出训练过程、最终指标、样例预测与 `cls` 注意力焦点；
5. 无需交互输入。

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0255-注意力机制
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程拆解（8 步）：

1. `build_dataloaders` 读取 `load_digits`，把每张 `8x8` 图像视为长度 8 的序列并完成 train/test 切分。  
2. `TinyAttentionClassifier.__init__` 初始化输入投影、`cls token`、位置编码、注意力块堆叠与分类头。  
3. `ScaledDotSelfAttention.forward` 显式计算 `Q/K/V` 三个投影矩阵。  
4. 同函数内计算 `scores = QK^T / sqrt(d_k)`，对最后一维做 `softmax` 得到注意力概率。  
5. 以 `context = probs @ V` 完成加权聚合，再经输出投影回到 `d_model`。  
6. `AttentionBlock.forward` 执行“注意力残差 + FFN 残差 + 两次 LayerNorm”得到稳定表征。  
7. `train` 中按批次执行 `forward -> cross entropy -> backward -> clip_grad_norm -> AdamW.step`，每轮调用 `evaluate` 记录 train/test 指标。  
8. `show_predictions_with_attention` 用 `scipy.special.softmax` 转概率并打印样例预测，同时输出 `cls -> row` 的注意力焦点，提供机制可解释性。

该流程完整展开注意力核心计算，不依赖第三方高层黑盒封装。

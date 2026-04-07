# GRU

- UID: `MATH-0312`
- 学科: `数学`
- 分类: `深度学习`
- 源序号: `312`
- 目标目录: `Algorithms/数学-深度学习-0312-GRU`

## R01

GRU（Gated Recurrent Unit）是循环神经网络的一种门控变体，核心目标是在保留时序建模能力的同时，
用更少参数缓解普通 RNN 的梯度消失问题。

本目录实现一个最小可运行 MVP：  
- 任务：判断序列中 `token A(0)` 是否先于 `token B(1)` 首次出现；  
- 模型：`Embedding + 单层 GRU + Linear`；  
- 特点：不依赖外部数据文件、无交互输入，直接运行可得到训练和评估结果。

## R02

问题定义（合成二分类）：
- 输入：`x in {0,1,...,V-1}^T`，默认 `V=10, T=30`。
- 标签：`y in {0,1}`。
  - `y=1`：`A=0` 的首次位置早于 `B=1`；
  - `y=0`：`B=1` 的首次位置早于 `A=0`。

数据构造原则：
- 背景 token 从 `2..V-1` 随机采样，避免与 `A/B` 混淆；
- 每个样本强制仅放置一个 `A` 和一个 `B`；
- 用标签控制 `A/B` 先后顺序，保证监督信号清晰可验证。

## R03

GRU 核心数学形式（与 PyTorch 文档一致）：

1. 重置门  
`r_t = sigma(W_ir x_t + b_ir + W_hr h_(t-1) + b_hr)`

2. 更新门  
`z_t = sigma(W_iz x_t + b_iz + W_hz h_(t-1) + b_hz)`

3. 候选状态  
`n_t = tanh(W_in x_t + b_in + r_t ⊙ (W_hn h_(t-1) + b_hn))`

4. 新隐藏状态  
`h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_(t-1)`

5. 分类头  
`logits = W_o h_T + b_o`，损失函数为交叉熵 `CrossEntropyLoss`。

## R04

算法流程（高层）：
1. 固定随机种子并检查配置合法性。  
2. 生成含 `A/B` 顺序标签的序列数据。  
3. 分层切分训练/测试集并构建 `DataLoader`。  
4. 初始化 `GRUOrderClassifier`。  
5. 按 epoch 执行训练（前向、损失、反向传播、参数更新）。  
6. 每个 epoch 在测试集计算 `loss/acc`。  
7. 训练后输出 `accuracy/f1/confusion_matrix`。  
8. 抽样打印 `pos_A/pos_B/true/pred` 做可解释审计。

## R05

核心数据结构：
- `Config`：统一管理样本规模、模型宽度、训练超参数。  
- `sequences: np.ndarray[int64]`：形状 `(n_samples, seq_len)` 的 token 序列。  
- `labels: np.ndarray[int64]`：二分类标签。  
- `pos_a / pos_b`：`A/B` 在序列中的位置，便于审计。  
- `TensorDataset + DataLoader`：批训练和批评估。  
- `GRUOrderClassifier`：`Embedding -> GRU -> Linear` 最小网络。

## R06

正确性要点：
- 标签正确性：`labels` 与 `pos_a/pos_b` 顺序关系严格一致。  
- 建模正确性：GRU 通过门控状态更新累计历史信息。  
- 目标正确性：二分类使用交叉熵，输出 logits 维度为 2。  
- 评估正确性：独立测试集上报告准确率、F1 与混淆矩阵。  
- 可复现性：固定 `numpy/torch` 随机种子并启用确定性选项。

## R07

复杂度分析（`N` 样本数，`T` 序列长度，`d` 嵌入维度，`h` 隐状态维度）：
- GRU 单步代价近似 `O(3*(d*h + h^2))`（三组门控/候选线性变换）。  
- 单样本前向：`O(T*(d*h + h^2))`。  
- 单 epoch 训练（含反向传播同阶）：`O(N*T*(d*h + h^2))`。  
- 空间复杂度：
  - 参数量约 `O(V*d + 3*(d*h + h^2) + h*C)`；  
  - 激活缓存约 `O(batch_size*T*h)`。

## R08

边界与异常处理：
- `n_samples <= 0`、`seq_len < 2`、`learning_rate <= 0` 会抛 `ValueError`；  
- `token_a == token_b` 直接报错，避免标签歧义；  
- `vocab_size < 4` 或无填充 token 时报错；  
- `test_size` 不在 `(0,1)` 报错；  
- `batch_size/epochs` 非正整数报错。

## R09

MVP 取舍说明：
- 聚焦单层 GRU，避免堆叠网络、注意力、可变长 pack 等工程复杂度；  
- 使用合成数据保证可复现实验与快速迭代；  
- 仅保留最关键指标与样本审计输出，减少样板代码；  
- 引入 `pandas` 记录 epoch 指标、`scipy` 做二项检验，增强结果解释性。

## R10

`demo.py` 模块分工：
- `set_global_seed`：固定随机行为。  
- `validate_config`：参数合法性校验。  
- `generate_order_dataset`：构造带顺序标签的样本。  
- `GRUOrderClassifier`：定义 GRU 分类网络。  
- `build_dataloaders`：划分训练/测试并封装 DataLoader。  
- `run_epoch`：统一训练与评估循环。  
- `predict_numpy`：批量推理。  
- `analyze_classification`：计算准确率/F1/混淆矩阵/二项检验。  
- `print_sample_audit`：打印样本级可解释审计。  
- `main`：串联完整流程。

## R11

运行方式：

```bash
cd Algorithms/数学-深度学习-0312-GRU
uv run python demo.py
```

脚本不需要命令行参数，不读取交互输入。

## R12

输出字段说明：
- `Config`：样本规模、序列长度、模型维度、训练轮数等。  
- `epoch/train_loss/train_acc/test_loss/test_acc`：逐轮训练日志。  
- `accuracy`：最终测试准确率。  
- `f1`：二分类 F1 分数。  
- `confusion_matrix`：混淆矩阵（行真值、列预测）。  
- `binom_pvalue` 与 `acc_ci95`：相对随机猜测（0.5）的显著性与置信区间。  
- `Sample audit`：展示 `pos_A/pos_B/true/pred` 与 token 片段。

## R13

最小验证建议（本脚本可直接观察）：
- 收敛性：`train_loss` 随 epoch 下降。  
- 任务可学性：`test_acc` 明显高于 0.5。  
- 规则一致性：审计样本中预测应与 `pos_A/pos_B` 顺序匹配。  
- 稳定性：同样随机种子可复现近似结果。

可补充异常测试：
- 设置 `token_a == token_b` 应立即报错；  
- 设置 `seq_len = 1` 应报错；  
- 设置 `test_size = 1.2` 应报错。

## R14

关键超参数（默认值）：
- `n_samples=2800`：样本规模。  
- `seq_len=30`：序列长度。  
- `embed_dim=24`：嵌入维度。  
- `hidden_dim=40`：GRU 隐状态维度。  
- `epochs=12`、`batch_size=64`、`learning_rate=8e-3`。  
- `grad_clip=1.0`：梯度裁剪上限。

调参建议：
- 欠拟合：增大 `hidden_dim` 或 `epochs`；  
- 训练震荡：降低学习率至 `3e-3` 附近；  
- 训练不稳定：适当减小 `grad_clip`。

## R15

方法对比：
- 对比普通 RNN：GRU 通过更新门/重置门显式控制历史信息保留。  
- 对比 LSTM：GRU 参数更少、实现更简洁，常用于轻量序列建模。  
- 对比 Transformer：Transformer 并行性强，但最小教学 MVP 的实现和算力成本更高。

第三方库说明：
- 使用 PyTorch 的 `nn.GRU` 负责门控递推数值计算；  
- 数据生成、训练循环、评估、审计均在 `demo.py` 显式实现，不是黑盒一行调用。

## R16

典型应用场景：
- 文本序列分类基线（情感/意图）；  
- 用户行为序列判别；  
- 事件流短期预测；  
- 时序 anomaly 检测的前置编码器。

## R17

可扩展方向：
- 增加多层 GRU 或双向 GRU；  
- 支持可变长序列与 padding mask；  
- 引入验证集、早停和学习率调度；  
- 把合成任务替换为真实数据集（如字符级文本分类）；  
- 增加模型保存与加载逻辑，形成可复用基线。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 创建 `Config`，`validate_config` 检查样本规模、序列长度、学习率等边界约束。  
2. `set_global_seed` 固定 `numpy/torch` 随机性，启用确定性算法保证复现。  
3. `generate_order_dataset` 先采样背景 token，再按标签强制 `A/B` 的先后顺序，输出 `sequences/labels/pos_a/pos_b`。  
4. `build_dataloaders` 通过 `train_test_split(..., stratify=labels)` 分层切分，并构造 `TensorDataset + DataLoader`。  
5. `GRUOrderClassifier.forward` 中，`Embedding` 将 token 映射到向量序列，`nn.GRU` 按时间递推门控状态，取最终隐藏状态送入线性层得到 logits。  
6. `run_epoch` 执行批量前向、交叉熵计算、反向传播与 Adam 更新；训练阶段加 `clip_grad_norm_` 抑制梯度爆炸。  
7. `predict_numpy` 在测试集推理，`analyze_classification` 用 `sklearn` 计算 `accuracy/f1/confusion_matrix`，再用 `scipy.stats.binomtest` 估计显著性与准确率区间。  
8. `print_sample_audit` 打印 `pos_A/pos_B/true/pred` 与 token 片段，人工核验模型确实学到“顺序关系”而非偶然相关性。

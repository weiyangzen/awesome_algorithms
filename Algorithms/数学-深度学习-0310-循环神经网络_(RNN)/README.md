# 循环神经网络 (RNN)

- UID: `MATH-0310`
- 学科: `数学`
- 分类: `深度学习`
- 源序号: `310`
- 目标目录: `Algorithms/数学-深度学习-0310-循环神经网络_(RNN)`

## R01

循环神经网络（Recurrent Neural Network, RNN）是一类面向序列数据的神经网络：
它在时间维度上复用同一组参数，并通过隐藏状态 `h_t` 累积历史信息。

本目录的 MVP 用一个基础 `tanh-RNN` 完成“顺序判别”任务：
- 输入：离散 token 序列；
- 输出：`token A(0)` 是否先于 `token B(1)` 首次出现；
- 特点：无需外部数据文件、无需交互输入，直接可运行并输出训练与评估结果。

## R02

本实现的问题定义：
- 样本 `x in {0,...,V-1}^T`，其中 `V=8, T=20`；
- 标签 `y in {0,1}`：
  - `y=1`：`A` 在序列中首次出现位置早于 `B`；
  - `y=0`：`B` 早于 `A`。

数据生成规则：
- 先用填充 token（`2..V-1`）构造随机背景序列；
- 再把 `A=0` 与 `B=1` 放到两个不同位置，并按标签控制先后顺序；
- 保证每个样本都恰有一个 `A` 和一个 `B`，避免标签歧义。

## R03

核心数学形式：

1. 嵌入层：
   `e_t = Emb(x_t)`, `e_t in R^d`。

2. 单层 RNN 递推（`tanh`）：
   `h_t = tanh(W_x e_t + W_h h_(t-1) + b_h)`。

3. 分类头：
   `z = W_o h_T + b_o`, `p = softmax(z)`。

4. 训练目标：
   最小化交叉熵
   `L = -sum_i log p(y_i | x_i)`。

5. 参数更新：
   使用 Adam 对 `Emb/RNN/Linear` 参数执行反向传播更新。

## R04

算法流程（高层）：
1. 固定随机种子并校验配置参数。  
2. 生成顺序分类数据集（含 `A/B` 位置审计信息）。  
3. 按分层抽样划分训练/测试集并构建 DataLoader。  
4. 初始化 `Embedding + RNN + Linear` 模型。  
5. 逐 epoch 训练：前向、交叉熵、反向传播、Adam 更新。  
6. 每个 epoch 在测试集评估 `loss/accuracy`。  
7. 训练结束后输出最终准确率与混淆矩阵。  
8. 打印若干样本的 `pos_A/pos_B/true/pred` 进行可解释审计。

## R05

核心数据结构：
- `Config`（dataclass）：统一管理样本规模、序列长度、模型维度、训练超参数。  
- `sequences: np.ndarray[int64]`：`(n_samples, seq_len)` 的 token 序列矩阵。  
- `labels: np.ndarray[int64]`：`(n_samples,)` 二分类标签。  
- `pos_a / pos_b`：记录 `A/B` 在每条序列中的位置，便于核对学习到的顺序规则。  
- `TensorDataset + DataLoader`：批处理训练与评估。  
- `RNNOrderClassifier`：`Embedding -> RNN -> Linear` 的最小分类网络。

## R06

正确性要点：
- 标签构造正确：同一样本中 `A/B` 位置与 `y` 一一对应。  
- 时序建模正确：RNN 以递推隐藏状态压缩历史信息，不是静态特征拼接。  
- 目标函数正确：二分类使用 `CrossEntropyLoss`。  
- 评估口径正确：测试集上独立计算准确率与混淆矩阵。  
- 可复现实验：固定随机种子与确定性算法开关。

## R07

复杂度分析（`N` 样本数，`T` 序列长度，`d` 嵌入维度，`h` 隐藏维度）：
- RNN 单步近似代价：`O(d*h + h^2)`；
- 单样本前向（长度 `T`）：`O(T*(d*h + h^2))`；
- 单 epoch 训练（含反向同阶）：`O(N*T*(d*h + h^2))`；
- 空间复杂度：
  - 参数量约 `O(V*d + d*h + h^2 + h*C)`；
  - 激活缓存约 `O(batch_size*T*h)`。

## R08

边界与异常处理：
- `n_samples<=0`、`seq_len<2`、`learning_rate<=0` 等配置非法会抛 `ValueError`；
- `vocab_size<4` 时无法同时容纳 `A/B` 与填充 token，会抛错；
- `token_a == token_b` 会抛错；
- `test_size` 不在 `(0,1)` 会抛错；
- 数据生成阶段若填充 token 为空也会抛错。

## R09

MVP 取舍：
- 采用最基础 `nn.RNN`，不直接上 LSTM/GRU，优先突出“循环状态”核心机制；
- 任务选用可控合成数据，避免外部数据依赖；
- 仅做单标签二分类，不引入 attention、mask、packed sequence 等工程复杂度；
- 保留位置审计输出（`pos_A/pos_B`），增强结果可解释性。

## R10

`demo.py` 主要函数职责：
- `set_global_seed`：固定随机种子与确定性行为。  
- `validate_config`：参数合法性检查。  
- `generate_order_dataset`：生成顺序判别样本与位置标注。  
- `RNNOrderClassifier`：定义 RNN 分类网络。  
- `build_dataloaders`：训练/测试划分并封装 DataLoader。  
- `run_epoch`：统一训练与评估循环。  
- `predict_numpy`：批量推理输出类别。  
- `print_samples`：打印样本级审计信息。  
- `main`：串联全流程并输出结果。

## R11

运行方式：

```bash
cd Algorithms/数学-深度学习-0310-循环神经网络_(RNN)
uv run python demo.py
```

脚本不读取命令行参数，不要求交互输入。

## R12

输出字段说明：
- `Config`：实验配置与设备信息；
- `epoch/train_loss/train_acc/test_loss/test_acc`：每轮训练与测试指标；
- `test_accuracy`：最终测试准确率；
- `confusion_matrix`：混淆矩阵（行是真值，列是预测）；
- `Sample predictions`：抽样显示 `pos_A/pos_B/true/pred` 与前 10 个 token，用于人工审计。

## R13

最小验证覆盖建议（本脚本已覆盖核心项）：
- 功能正确：`test_accuracy` 应明显高于随机猜测（`>0.5`）；
- 收敛行为：随 epoch 推进，`train_loss` 应下降、`train_acc` 提升；
- 泛化能力：`test_acc` 与 `train_acc` 同步提升且不过度偏离；
- 规则一致性：抽样样本中预测结果应与 `pos_A/pos_B` 顺序一致。

可补充的异常测试：
- 人为设置 `token_a == token_b`；
- 人为设置 `seq_len = 1`；
- 人为设置 `test_size = 1.2`。

## R14

关键超参数：
- `embed_dim=16`：离散 token 的向量表示维度；
- `hidden_dim=32`：RNN 状态容量；
- `epochs=14`：训练轮数；
- `learning_rate=1e-2`：Adam 学习率；
- `batch_size=64`：批大小；
- `n_samples=2400`：样本规模。

调参建议：
- 若欠拟合：增大 `hidden_dim` 或 `epochs`；
- 若训练震荡：减小学习率到 `3e-3` 左右；
- 若过拟合：减少模型容量或增加样本量。

## R15

方法对比：
- 对比 MLP（忽略顺序）：RNN 利用递推状态建模时序依赖；
- 对比 LSTM/GRU：基础 RNN 参数更少、教学更直接，但长依赖稳定性较弱；
- 对比 Transformer：Transformer 并行能力强但实现和算力开销更高，基础 RNN 更适合最小化演示。

第三方库使用说明：
- 使用 PyTorch 的 `nn.RNN` 与自动求导做数值计算；
- 但数据构造、训练循环、评估与审计逻辑均在源码显式实现，非“一行黑盒调用”。

## R16

典型应用场景：
- 文本序列分类（情感/意图初版基线）；
- 事件序列预测（简单日志模式识别）；
- 时间序列离散化后的短期模式判别；
- 作为 LSTM/GRU/Transformer 前的轻量对照基线。

## R17

可扩展方向：
- 将 `nn.RNN` 替换为 `nn.GRU/nn.LSTM` 比较长依赖性能；
- 引入可变长度序列和 `pack_padded_sequence`；
- 增加验证集、早停和学习率调度；
- 输出训练曲线到 CSV 便于可视化；
- 将任务扩展为多分类或真实文本数据集。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 创建 `Config`，调用 `validate_config` 校验样本规模、序列长度、学习率等边界。  
2. `set_global_seed` 固定 `numpy/torch` 随机性，并启用确定性算法以保证可复现。  
3. `generate_order_dataset` 先用填充 token 采样背景序列，再按标签把 `A/B` 放到不同位置，输出 `sequences/labels/pos_a/pos_b`。  
4. `build_dataloaders` 通过 `train_test_split(..., stratify=labels)` 分层切分数据，并封装成 `TensorDataset + DataLoader`。  
5. 构建 `RNNOrderClassifier`：`Embedding` 把 token id 映射为向量，`nn.RNN` 按时间递推隐藏状态，`Linear` 用最终隐藏状态给出二分类 logits。  
6. 每个 epoch 在 `run_epoch` 中执行批量前向 `logits = model(xb)`、交叉熵 `loss`、反向传播 `loss.backward()` 与 `optimizer.step()`；评估阶段复用同函数但不更新参数。  
7. 训练结束后，`predict_numpy` 对测试集分批推理，`accuracy_score` 和 `confusion_matrix` 计算最终指标。  
8. `print_samples` 打印样本级 `pos_A/pos_B/true/pred` 与 token 片段，验证模型是否真正学到“先后顺序”而非偶然统计。

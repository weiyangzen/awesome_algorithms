# Dropout

- UID: `MATH-0308`
- 学科: `数学`
- 分类: `深度学习`
- 源序号: `308`
- 目标目录: `Algorithms/数学-深度学习-0308-Dropout`

## R01

Dropout 是一种深度学习正则化方法，用于缓解神经网络过拟合。核心做法是在训练阶段对某层激活随机“失活”一部分神经元，从而打破神经元之间的共适应（co-adaptation），提升泛化能力。

常见形式是“反向缩放（inverted dropout）”：
- 训练时：对激活 `a` 采样掩码 `m ~ Bernoulli(1-p)`，并使用 `a_drop = a * m / (1-p)`；
- 推理时：不采样、不失活，直接使用完整激活 `a`。

其中 `p` 是失活概率（dropout rate）。

## R02

要解决的问题是：模型容量较大时，训练集误差持续降低但测试集误差升高，即过拟合。

Dropout 通过在每个 step 随机屏蔽子网络，可理解为对大量共享参数子模型进行近似集成，达到以下效果：
- 降低单一特征路径依赖；
- 增强特征冗余与鲁棒性；
- 使训练目标更偏向可泛化结构而非训练集噪声。

## R03

数学形式（以一层隐藏层为例）：

1. 线性层与激活：
   - `z1 = XW1 + b1`
   - `h = ReLU(z1)`
2. Dropout（训练阶段）：
   - `m_ij ~ Bernoulli(1-p)`
   - `h_drop = h * m / (1-p)`
3. 输出层：
   - `logits = h_drop W2 + b2`
   - `y_hat = sigmoid(logits)`
4. 损失（BCE）：
   - `L = -mean(y*log(y_hat+eps) + (1-y)*log(1-y_hat+eps))`

梯度传播时，dropout 掩码同样作用于反向梯度，使被屏蔽单元在该步不更新。

## R04

高层算法流程：

1. 初始化网络参数（两层 MLP）与随机种子。  
2. 对每个 epoch：执行前向传播。  
3. 若是训练模式，在隐藏层采样 dropout 掩码并做反向缩放。  
4. 计算 BCE 损失和输出层梯度。  
5. 反传到隐藏层，并对梯度乘相同掩码。  
6. 按学习率更新参数。  
7. 定期切到评估模式（无 dropout）统计训练/测试准确率。  
8. 输出对照实验（`p=0` vs `p>0`）结果与质量断言。

## R05

本目录 `demo.py` 的数据与模型设定：
- 数据：固定随机种子的二分类合成数据（训练集小、特征维度较高，易过拟合）。
- 模型：`input -> hidden(ReLU) -> dropout -> sigmoid` 的最小 MLP。
- 对照：同一数据上训练两次
  - `dropout_rate = 0.0`（无 dropout）
  - `dropout_rate = 0.5`（有 dropout）
- 指标：训练/测试准确率、训练-测试泛化差距（generalization gap）。

## R06

正确性关键点：

1. 训练与推理模式必须严格分离：
   - `training=True` 才采样掩码；
   - `training=False` 禁止掩码。
2. 使用反向缩放 `/(1-p)`，保证训练与推理激活期望一致。  
3. 反向传播必须复用同一掩码，否则梯度路径不一致。  
4. `p` 需满足 `0 <= p < 1`，避免除零与无意义配置。  
5. 随机种子固定，保证示例可复现。

## R07

复杂度分析（样本数 `n`，输入维度 `d`，隐藏维度 `h`）：
- 单次前向主要成本：`O(n*d*h + n*h)`。  
- 单次反向主要成本同量级：`O(n*d*h + n*h)`。  
- 每 epoch 训练复杂度约：`O(n*d*h)`。  
- 空间复杂度：参数 `O(d*h + h)`，缓存（`z1/h/mask/logits`）约 `O(n*h)`。

Dropout 的额外开销主要是采样掩码与逐元素乘法，通常相对矩阵乘法成本较小。

## R08

边界与异常处理：
- `dropout_rate` 不在 `[0,1)` 直接抛 `ValueError`。  
- 输入维度与参数维度不一致时报错。  
- 训练过程中检测损失是否为有限值，若出现 `NaN/Inf` 立即中止。  
- 对 `sigmoid` 输入做裁剪，降低指数溢出风险。  
- BCE 中加入 `eps`，避免 `log(0)`。

## R09

MVP 取舍说明：
- 仅用 `numpy` + 标准库实现，避免依赖大型框架，确保环境适配。  
- 手写前向与反向，Dropout 机制完全可审计，不调用黑盒训练 API。  
- 聚焦二分类与单隐藏层，不扩展到 BatchNorm、多层堆叠、学习率调度等复杂工程组件。  
- 用固定随机种子和合成数据，强调“可运行 + 可复现 + 可解释”。

## R10

`demo.py` 主要函数职责：
- `make_dataset`：构造可复现二分类数据并划分 train/test。  
- `sigmoid`、`bce_loss`：数值稳定的激活与损失。  
- `BinaryMLPWithDropout.forward`：执行前向传播与 dropout 掩码采样。  
- `BinaryMLPWithDropout.backward`：基于缓存执行手写反向传播。  
- `BinaryMLPWithDropout.step`：参数梯度下降更新。  
- `train_model`：训练循环、日志记录与阶段评估。  
- `run_experiment`：运行单次配置并汇总指标。  
- `main`：跑无 dropout/有 dropout 对照、打印结果与断言。

## R11

运行方式：

```bash
cd Algorithms/数学-深度学习-0308-Dropout
uv run python demo.py
```

脚本无需命令行参数，也不需要交互输入。

## R12

输出字段说明：
- `Dataset`：训练集与测试集尺寸。  
- `Baseline(no dropout)`：`p=0` 时的最终损失、训练准确率、测试准确率、泛化差距。  
- `Dropout(p=0.5)`：`p=0.5` 时同口径指标。  
- `Gap improvement`：启用 dropout 后泛化差距的变化量（越大越好）。  
- `All checks passed.`：通过最小质量门槛。

## R13

内置最小测试与质量门槛：
1. 两组实验都需完成训练，且最终损失为有限值。  
2. 两组测试准确率都需高于随机猜测（阈值设为 `> 0.60`）。  
3. Dropout 方案的泛化差距不应显著恶化（`gap_dropout <= gap_baseline + 0.03`）。  
4. 训练全程无需外部输入，单次运行即可得到完整报告。

## R14

关键参数建议：
- `dropout_rate`：常见取值 `0.1 ~ 0.5`，越大正则越强但欠拟合风险增大。  
- `hidden_dim`：隐藏层越大越容易过拟合，dropout 收益通常更明显。  
- `lr` 与 `epochs`：学习率过大会不稳定；epoch 过少会看不到泛化差异。  
- `train_size`：训练样本越少，dropout 对抗过拟合的价值通常越高。

## R15

与相近正则方法对比：
- 对比 L2 权重衰减：
  - L2 约束参数幅度；
  - Dropout 在表示层面引入随机结构扰动。  
- 对比早停（Early Stopping）：
  - 早停是训练过程控制；
  - Dropout 是模型内部随机正则。  
- 工程实践中常把 Dropout 与 L2、早停联合使用。

## R16

典型应用场景：
- 小数据/中等规模深度模型训练，容易过拟合的任务。  
- 全连接层参数占比较高的分类器。  
- 需要简单、低侵入式正则化方案的教学或原型阶段。  
- 在大模型里常见于分类头（head）而非所有层。

## R17

可扩展方向：
- 增加 mini-batch 训练、动量或 Adam 优化。  
- 扩展到多分类（softmax + cross-entropy）。  
- 对比不同 dropout rate 曲线，画出 bias-variance 变化。  
- 扩展为 Monte Carlo Dropout，用多次随机前向估计预测不确定性。  
- 与 BatchNorm 组合研究顺序与位置影响。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `main` 调用 `make_dataset` 生成固定随机种子的高维二分类数据，并拆分训练/测试集。  
2. `run_experiment` 分别构建两套模型：`dropout_rate=0.0` 与 `dropout_rate=0.5`，其余超参数一致。  
3. `train_model` 在每个 epoch 调 `forward(training=True)`：先算 `z1`、`ReLU`，再按 `Bernoulli(1-p)` 采样掩码并执行 `/(1-p)` 反向缩放。  
4. 前向继续计算 `logits` 与 `sigmoid` 概率，用 `bce_loss` 得到当前损失。  
5. `backward` 从 `dlogits=(pred-y)/n` 开始回传；先算输出层梯度，再把梯度传回隐藏层。  
6. 若训练阶段启用了 dropout，则隐藏层反向梯度乘同一掩码，保证仅未屏蔽单元参与该步更新。  
7. `step` 用学习率对 `W1/b1/W2/b2` 做梯度下降更新；周期性在 `training=False` 下评估准确率（此时不采样掩码）。  
8. 两组实验结束后，`main` 汇总训练/测试准确率与泛化差距，执行断言并打印 `All checks passed.`。

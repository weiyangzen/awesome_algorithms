# Nadam/AdamW

- UID: `MATH-0401`
- 学科: `数学`
- 分类: `深度学习`
- 源序号: `401`
- 目标目录: `Algorithms/数学-深度学习-0401-Nadam／AdamW`

## R01

`Nadam` 和 `AdamW` 都属于一阶自适应随机优化器，常用于训练深度神经网络参数 `theta`。

- Nadam（Nesterov-accelerated Adam）：在 Adam 的一阶/二阶矩估计基础上引入 Nesterov 风格前瞻动量项。
- AdamW（Adam with decoupled weight decay）：把权重衰减从梯度项中解耦，避免把正则项混入自适应二阶矩。

两者都维护：
- 一阶矩：`m_t = beta1 * m_{t-1} + (1-beta1) * g_t`
- 二阶矩：`v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2`

## R02

发展脉络：
- Adam 在 2014 年提出，成为深度学习默认优化器之一。
- Nadam 随后被提出，目标是结合 Adam 的自适应缩放和 Nesterov 动量的前瞻性。
- AdamW 在 2017 年左右成为主流实践，核心是“解耦权重衰减”，解决 Adam 中 L2 正则与自适应缩放耦合导致的泛化问题。

工程上常见经验：
- 想要更快前期收敛，可尝试 Nadam。
- 想要更稳定正则化和更可控泛化，优先 AdamW。

## R03

它们要解决的核心问题是：
1. 纯 SGD 在病态曲率和噪声梯度下收敛慢、抖动大。
2. 不同参数尺度差异大时，统一学习率不稳定。
3. Adam 系算法中，传统 L2 正则会被自适应分母扭曲，权重衰减效果不直观。

对应改进：
- Nadam：在 Adam 的基础上增强动量前瞻，常使训练初中期更积极。
- AdamW：把衰减作为独立“参数收缩”项，正则强度更接近超参本意。

## R04

两者的单步流程（向量形式）如下。

Nadam（本目录 `demo.py` 使用的 Dozat 风格）：
1. `m_t = beta1 * m_{t-1} + (1-beta1) * g_t`
2. `v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2`
3. `m_hat_t = m_t / (1-beta1^t)`，`v_hat_t = v_t / (1-beta2^t)`
4. `g_hat_t = g_t / (1-beta1^t)`
5. `n_t = beta1 * m_hat_t + (1-beta1) * g_hat_t`
6. `theta_t = theta_{t-1} - lr * n_t / (sqrt(v_hat_t) + eps)`

AdamW：
1. 若参数属于衰减集合，先做 `theta <- theta * (1 - lr * weight_decay)`
2. 更新 `m_t, v_t`
3. 偏差修正得到 `m_hat_t, v_hat_t`
4. 梯度步：`theta <- theta - lr * m_hat_t / (sqrt(v_hat_t) + eps)`

## R05

设模型参数总量为 `P`。

- 单次迭代时间复杂度：`O(P)`（不含前向/反向本身）。
- 额外状态空间：`O(P)`（每个参数都要存 `m` 和 `v`）。
- 训练 `T` 步时优化器开销约 `O(T*P)`。

在深度网络中，主要耗时通常仍是前向与反向传播；优化器更新是线性附加成本。

## R06

单维简化示例（仅展示更新差异）：

假设当前：
- `theta=1.0, g=0.2, lr=0.1`
- `beta1=0.9, beta2=0.999, eps=1e-8`
- 初始 `m=v=0`

首步后有：
- `m_hat≈0.2, v_hat≈0.04`

则：
- AdamW 梯度步约 `0.1 * 0.2 / 0.2 = 0.1`，若 `weight_decay=0.01`，还会额外做 `theta <- theta * (1-0.001)`。
- Nadam 用 `n_t = 0.9*0.2 + 0.1*0.2 = 0.2`，首步梯度部分与此例中 Adam 接近，但在连续多步时前瞻动量会带来不同轨迹。

## R07

优点：
- Nadam：在非凸深度模型中常有更快的早期收敛。
- AdamW：正则化语义清晰，通常比“Adam+L2”更稳定。
- 两者都具备逐参数自适应步长，对特征尺度不均匀更鲁棒。

局限：
- 仍需调学习率与 `beta`，并非免调参。
- 在某些任务，最终泛化未必优于精调的 SGD+Momentum。
- 自适应方法若学习率过大，仍会振荡或发散。

## R08

前置知识：
- 链式法则与反向传播。
- 指数滑动平均（EMA）与偏差修正。
- 动量法、Nesterov 动量的直觉。
- 正则化（L2 / weight decay）区别：耦合 vs 解耦。
- 数值稳定技巧（`eps`、梯度裁剪、输入标准化）。

## R09

适用场景：
- 大规模神经网络训练。
- 噪声梯度、非平稳目标、参数尺度差异大。
- 希望快速得到可用结果（Nadam）或更稳定控制权重衰减（AdamW）。

谨慎场景：
- 极小样本且追求极限泛化时，需与 SGD 系对比。
- 损失面极端尖锐时，需要更小学习率或 warmup。
- 若梯度计算本身错误，优化器再好也无法收敛。

## R10

实现正确性的关键检查点：
1. `m`、`v` 张量形状必须与参数完全一致。
2. 迭代计数 `t` 必须从 1 开始递增，用于偏差修正。
3. Nadam 必须构造前瞻项 `n_t`，不能直接退化成 Adam。
4. AdamW 的 `weight_decay` 必须与梯度更新解耦。
5. 分母需使用 `sqrt(v_hat) + eps` 防止除零。

本目录 `demo.py` 还包含有限差分梯度检查，先验证反向传播再进入训练。

## R11

数值稳定性策略（本 MVP 已使用）：
- `sigmoid` 输入裁剪到 `[-50, 50]`，防止指数溢出。
- BCE 中用 `log(p+eps)`，避免 `log(0)`。
- 使用梯度全局范数裁剪（`max_norm=5`）降低极端更新。
- 数据标准化减少病态尺度。
- 偏差修正减少前期 `m/v` 零初始化带来的偏差。

## R12

调参建议（经验起点）：
- Nadam：`lr` 常比 Adam 略小或相当，常从 `1e-3 ~ 1e-2` 网格开始。
- AdamW：先定 `lr`，再调 `weight_decay`（如 `1e-4 ~ 1e-1` 任务相关）。
- `beta1=0.9, beta2=0.999, eps=1e-8` 是稳妥默认值。

真实成本：
- 每步需要额外维护两套矩估计，内存约为参数量的 2 倍（不含参数本体）。
- 若模型很大，优化器状态的显存/内存占用不可忽略。

## R13

- 近似比保证：N/A。
- 概率成功保证：N/A（除数据采样等外部随机性外，更新本身是确定性公式）。

Nadam/AdamW 属于深度学习训练中的实用优化器，重点在经验收敛与泛化表现，不是有近似比界的组合优化算法。

## R14

常见失效模式与防护：
- 学习率过大，loss 震荡或 nan。
  - 防护：降低 `lr`，增加梯度裁剪，必要时加 warmup。
- AdamW 衰减过大，模型欠拟合。
  - 防护：减小 `weight_decay`，并排查是否把偏置也衰减了。
- Nadam 训练过快但后期抖动。
  - 防护：后期降学习率（scheduler）或切换到更保守设置。
- 反向传播写错导致“看似训练其实无效”。
  - 防护：有限差分梯度检查。

## R15

工程实践建议：
- 固定随机种子，保留每次实验超参与关键指标。
- 同一模型初始化下对比不同优化器，减少比较偏差。
- 先在小数据集做“实现正确性验证”，再扩展到大训练。
- 监控 loss、梯度范数、参数范数三类指标，定位收敛问题更快。

## R16

关联关系：
- Adam 是基线；Nadam 在其上加入 Nesterov 风格前瞻。
- AdamW 主要修改正则项处理方式（解耦 weight decay）。
- 常见近亲：AMSGrad、RAdam、AdaBelief、Lion 等。

实践选择：
- 优先快速迭代模型：Nadam 往往手感更“快”。
- 优先泛化与正则控制：AdamW 常是默认首选。

## R17

本目录 MVP（`demo.py`）内容：
- 用 NumPy 从零实现两层 MLP、前向、BCE 损失、反向梯度。
- 实现 `nadam_step` 与 `adamw_step`，不调用任何黑盒优化器。
- 在同一 moons-like 二分类数据上，分别训练 Nadam 与 AdamW 并输出对比指标。
- 运行前做有限差分梯度检查，避免反向传播实现错误。

运行方式：

```bash
cd Algorithms/数学-深度学习-0401-Nadam／AdamW
uv run python demo.py
```

脚本无交互输入，自动打印 loss/accuracy 并做阈值断言。

## R18

源码级流程拆解（对应 `demo.py`，9 步）：
1. `make_moons_like_dataset` 生成非线性二分类样本，`standardize_features` 做标准化。
2. `split_train_test` 固定种子切分训练/测试，保证可复现。
3. `init_mlp_params` 初始化 `W1,b1,W2,b2`，`mlp_forward` 实现 `tanh` 隐层 + `sigmoid` 输出。
4. `mlp_loss_and_grads` 计算 BCE 损失并手写反向传播，得到每个参数梯度。
5. `finite_diff_gradient_check` 对若干坐标做中心差分，与解析梯度对比，先验证导数正确性。
6. `train_model(..., optimizer_name="nadam")` 中每轮执行：算梯度 -> `clip_gradients` -> `nadam_step`（含 `n_t` 前瞻项）-> 记录 loss。
7. `train_model(..., optimizer_name="adamw")` 中每轮执行：先对权重做解耦衰减，再做 Adam 偏差修正梯度步。
8. `summarize_run` 统一评估 train/test loss 与 accuracy，打印 Nadam 与 AdamW 的收敛结果。
9. `main` 中设置断言（loss 显著下降、测试精度达标），确保脚本在 CI/批处理环境可自动判定成功。

以上实现把 Nadam/AdamW 的状态变量与更新方程全部显式展开，可直接逐行追踪。 

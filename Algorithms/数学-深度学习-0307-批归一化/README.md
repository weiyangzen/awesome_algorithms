# 批归一化

- UID: `MATH-0307`
- 学科: `数学`
- 分类: `深度学习`
- 源序号: `307`
- 目标目录: `Algorithms/数学-深度学习-0307-批归一化`

## R01

批归一化（Batch Normalization, BN）是在神经网络层内对小批量数据做标准化的技术。它在训练时将每个特征维度变为近似零均值、单位方差，再通过可学习参数恢复表达能力：

- 标准化：`x_hat = (x - mean_batch) / sqrt(var_batch + eps)`
- 仿射变换：`y = gamma * x_hat + beta`

其中 `gamma`、`beta` 为可学习参数，`eps` 是数值稳定项。

## R02

BN 主要解决两个工程痛点：

- 深层网络训练不稳定：中间层激活分布剧烈波动，梯度传播变差。
- 学习率受限：若激活尺度漂移，优化器需要更小步长避免发散。

BN 的直接收益通常包括：

- 更快收敛；
- 对初始化不那么敏感；
- 在很多任务上带来一定正则化效果（由 batch 统计噪声引入）。

## R03

以二维输入 `X in R^(N x C)`（`N` 为 batch 大小，`C` 为通道/特征维）为例：

训练模式（`training=True`）：

1. `mu_c = (1/N) * sum_i X_ic`
2. `sigma2_c = (1/N) * sum_i (X_ic - mu_c)^2`
3. `X_hat_ic = (X_ic - mu_c) / sqrt(sigma2_c + eps)`
4. `Y_ic = gamma_c * X_hat_ic + beta_c`
5. 运行统计（指数滑动平均）：
   - `running_mean = (1-m) * running_mean + m * mu`
   - `running_var  = (1-m) * running_var  + m * sigma2_unbiased`
   - 其中 `sigma2_unbiased = (N/(N-1)) * sigma2`（当 `N>1`）

推理模式（`training=False`）：

- 使用 `running_mean/running_var` 替代当前 batch 统计：
  `Y = gamma * (X - running_mean) / sqrt(running_var + eps) + beta`

## R04

高层算法流程：

1. 初始化 `gamma=1`、`beta=0`、`running_mean=0`、`running_var=1`。  
2. 进入训练步时，读取一批输入 `X`。  
3. 计算该批次的均值与方差。  
4. 按 `eps` 做标准化得到 `X_hat`。  
5. 应用可学习仿射参数得到输出 `Y`。  
6. 更新运行均值/方差供推理阶段使用。  
7. 反向传播时计算 `dX`、`dGamma`、`dBeta`。  
8. 在推理阶段切换为运行统计路径，不再依赖当前批次统计。

## R05

本目录 `demo.py` 的 MVP 设计：

- 手写 `NumpyBatchNorm1D`，完整实现训练/推理前向和反向。  
- 与 `torch.nn.BatchNorm1d` 在相同输入、参数下做数值对照。  
- 验证项：
  - 训练态前向输出误差；
  - 推理态前向输出误差；
  - 反向梯度（`dX/dGamma/dBeta`）误差；
  - 训练态输出每通道均值约 0、方差约 1。

## R06

正确性关键点：

1. 方差定义需与实现对齐：训练态归一化使用 `ddof=0`（总体方差）；`running_var` 更新使用无偏方差（`N/(N-1)` 修正），与 PyTorch 行为一致。  
2. `eps` 必须加在方差开方前，避免小方差下除零。  
3. 训练与推理分支严格分开：推理不能用当前 batch 统计。  
4. `momentum` 语义要一致：`new = (1-m)*old + m*batch_stat`。  
5. 反向传播需基于训练缓存（`x, mean, var, x_hat`），否则梯度错误。

## R07

复杂度分析（输入形状 `N x C`）：

- 前向时间复杂度：`O(NC)`。  
- 反向时间复杂度：`O(NC)`。  
- 额外空间复杂度：`O(C)`（参数+运行统计）与 `O(NC)`（训练缓存）。

BN 的计算主要是逐通道统计和逐元素仿射，通常低于同层线性/卷积主计算成本。

## R08

边界与异常处理：

- 输入必须是二维 `N x C`，否则抛出 `ValueError`。  
- `C` 必须与初始化 `num_features` 一致。  
- `momentum` 需在 `(0, 1]`，`eps` 必须为正。  
- `backward` 只能在训练态 `forward` 之后调用。  
- 若出现 `NaN/Inf`，应优先排查输入尺度与 `eps`。

## R09

MVP 取舍：

- 不实现完整网络训练器，只聚焦 BN 本身。  
- 不覆盖 4D 张量（`BatchNorm2d`）与分布式同步 BN。  
- 不引入自动求导框架来“偷算”梯度，保持算法透明。  
- 使用 PyTorch 仅作对照基准，核心逻辑在 NumPy 中可审计。

## R10

`demo.py` 主要结构：

- `NumpyBatchNorm1D`：BN 参数、运行统计、前向/反向。  
- `run_forward_alignment()`：训练态与推理态前向对齐测试。  
- `run_backward_alignment()`：反向梯度对齐测试。  
- `main()`：统一执行测试并打印关键指标与断言。

## R11

运行方式：

```bash
cd Algorithms/数学-深度学习-0307-批归一化
uv run python demo.py
```

脚本无需任何交互输入。

## R12

输出字段说明：

- `Forward(train)`：NumPy 与 PyTorch 在训练态输出的最大绝对误差。  
- `Forward(eval)`：NumPy 与 PyTorch 在推理态输出的最大绝对误差。  
- `running_mean/running_var`：更新后运行统计误差。  
- `Backward(dx/dgamma/dbeta)`：三组梯度最大绝对误差。  
- `mean(abs)` / `var(abs)`：训练态归一化后，每通道均值与方差偏离目标的量级。

## R13

预期结果（不同机器会有极小浮点差异）：

- 前向与反向误差通常在 `1e-10 ~ 1e-12` 量级（`float64`）。
- 训练态标准化后，通道均值接近 `0`，方差接近 `1`。
- 程序末尾输出 `All BatchNorm checks passed.` 表示 MVP 校验通过。

## R14

关键超参数建议：

- `eps`：常用 `1e-5`；数值不稳定时可增大至 `1e-4`。  
- `momentum`：常用 `0.1`；batch 很小时可适度降低更新速度。  
- `batch size`：过小会使统计噪声变大，训练抖动明显。  
- `gamma/beta` 初始化：通常 `gamma=1`、`beta=0`。

## R15

常见错误：

1. 把推理态误写成继续用当前 batch 均值/方差。  
2. 方差用 `N-1`（无偏估计）但对照框架用 `N`，导致误差看似“实现错”。  
3. 忘记更新 `running_mean/running_var` 或更新公式方向写反。  
4. 在反向中漏掉 `dvar/dmean` 链条，导致 `dx` 不正确。  
5. 通道维处理错误（把样本维和特征维混用）。

## R16

工程扩展方向：

- `BatchNorm2d/3d`：对卷积特征图按通道聚合统计（跨 `N,H,W` 或 `N,D,H,W`）。  
- 与残差网络结合：常见顺序是 `Conv -> BN -> ReLU`。  
- 小 batch 场景可考虑 `GroupNorm` 或 `LayerNorm`。  
- 多卡训练可用 `SyncBatchNorm` 同步统计。

## R17

与相近归一化方法比较：

- `BatchNorm`：跨 batch 统计，训练快，但对 batch size 敏感。  
- `LayerNorm`：按样本内特征归一化，不依赖 batch，常用于 Transformer。  
- `GroupNorm`：在通道分组内归一化，兼顾卷积场景与小 batch 稳定性。  
- `InstanceNorm`：每样本每通道独立归一化，风格迁移中常见。

## R18

下面给出“从框架调用到算法实质”的源码级流程拆解（以 PyTorch 为参考，非黑箱描述）：

1. 用户层调用 `torch.nn.BatchNorm1d.forward`，它来自 `_BatchNorm.forward`。  
2. `_BatchNorm.forward` 根据 `self.training` 与 `track_running_stats` 决定是否使用 batch 统计，并准备 `running_mean/running_var`。  
3. 随后进入 `torch.nn.functional.batch_norm`，这里会把 `input/weight/bias/running stats/momentum/eps` 统一传给底层算子。  
4. 底层 `torch.batch_norm`（ATen）执行核心计算：求每通道均值、方差，完成标准化与仿射变换。  
5. 若处于训练态且跟踪统计，则按 `new=(1-m)*old+m*batch_stat` 更新运行均值/方差。  
6. 反向阶段由自动求导图回传到 BN backward 内核，计算 `dX`、`dWeight(dGamma)`、`dBias(dBeta)`。  
7. 在 `demo.py` 中，`NumpyBatchNorm1D` 显式复现了同一数学链路，便于逐项校验。  
8. 通过前向/反向误差对照，可验证“手写实现 == 框架核心算法”的一致性。

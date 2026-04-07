# LSTM

- UID: `MATH-0311`
- 学科: `数学`
- 分类: `深度学习`
- 源序号: `311`
- 目标目录: `Algorithms/数学-深度学习-0311-LSTM`

## R01

LSTM（Long Short-Term Memory）是循环神经网络（RNN）的改进结构，专门用于缓解长序列训练中的梯度消失问题。

它通过“门控机制 + 细胞状态”来控制信息保留和遗忘：

- 遗忘门 `f_t` 决定旧记忆保留多少；
- 输入门 `i_t` 决定新信息写入多少；
- 输出门 `o_t` 决定当前时刻向外暴露多少隐状态。

## R02

本条目要解决的问题：

给定长度为 `L` 的历史时间窗 `x_{t-L+1}, ..., x_t`，预测下一时刻值 `x_{t+1}`。

形式化目标可写为：

`min_theta E[(x_{t+1} - f_theta(x_{t-L+1:t}))^2]`

其中 `f_theta` 为 LSTM 回归模型。本 MVP 使用单变量合成时序做一步预测（one-step forecasting）。

## R03

LSTM 单元在时刻 `t` 的核心计算（简写）为：

- `f_t = sigma(W_f [h_{t-1}, x_t] + b_f)`
- `i_t = sigma(W_i [h_{t-1}, x_t] + b_i)`
- `g_t = tanh(W_g [h_{t-1}, x_t] + b_g)`
- `o_t = sigma(W_o [h_{t-1}, x_t] + b_o)`
- `c_t = f_t * c_{t-1} + i_t * g_t`
- `h_t = o_t * tanh(c_t)`

相对普通 RNN，`c_t` 的加法路径使长期依赖更容易传递。

## R04

输入与输出定义（对应 `demo.py`）：

- 输入：
  - 原始单变量时间序列 `series`；
  - 滑窗长度 `window_size=32`；
  - 训练配置（`epochs/batch_size/lr/hidden_size`）。
- 输出：
  - 训练后的 LSTM 参数；
  - 测试集预测值；
  - 评估指标（RMSE、MAE、Pearson 相关、基线对比）。

## R05

本 MVP 采用 `LSTM + 线性回归头 + MSELoss`，原因：

- 与时序回归目标直接匹配；
- 比堆叠复杂注意力结构更小、更容易验证；
- 能清晰体现“门控记忆”在序列预测中的作用。

为了减少噪声干扰，先用 `scipy.signal.savgol_filter` 做轻量平滑，再进入监督样本构造。

## R06

标准流程：

1. 生成带长滞后依赖、季节项和噪声的合成时序；
2. 用滑窗把序列改写为 `(X_window, y_next)` 监督样本；
3. 按时间顺序切分训练/测试（`shuffle=False`）；
4. 用 `StandardScaler` 标准化输入和目标；
5. 训练 `nn.LSTM` 回归器（反向传播 + Adam）；
6. 反标准化得到真实尺度预测值；
7. 与朴素基线（最后一个观测值）对比。

## R07

复杂度（粗略）：

设样本数 `N`、序列长度 `L`、隐藏维 `H`、输入维 `D=1`、训练轮数 `E`。

- 单步 LSTM 计算约 `O(H*(H + D))`；
- 单样本单轮约 `O(L * H*(H + D))`；
- 总训练复杂度约 `O(E * N * L * H*(H + D))`。

空间开销主要来自参数和中间激活，近似 `O(H*(H+D) + batch*L*H)`。

## R08

优点：

- 能建模中长期依赖，通常优于基础 RNN；
- 对时序预测、文本建模等序列任务适配性强；
- 结构成熟、实现稳定。

局限：

- 长序列仍有并行效率问题（时间维递归）；
- 参数量与训练成本高于简单线性模型；
- 在超长依赖场景中常被 Transformer 类模型替代。

## R09

适用场景：

- 电力负荷、传感器、交易指标等单/多变量时序预测；
- 语音、文本等序列建模；
- 需要“窗口上下文 -> 下一步预测”的任务。

不太适用：

- 极长上下文且要求高并行吞吐的场景；
- 关系主要是全局注意而非局部时间连续性的任务。

## R10

本 MVP 参数设置：

- 数据点数：`n_points=2800`
- 滑窗长度：`window_size=32`
- 切分比例：`test_size=0.2`（按时间顺序）
- 模型：`LSTM(input_size=1, hidden_size=40, num_layers=1) + Linear(40->1)`
- 训练：`epochs=24`, `batch_size=64`, `lr=2e-3`, `optimizer=Adam`
- 损失：`MSELoss`

## R11

与相关方法对比（简述）：

- 对比普通 RNN：LSTM 通过门控显著缓解长期依赖退化。
- 对比 GRU：LSTM 参数更多，表达更强；GRU 更轻量。
- 对比 AR/线性回归：LSTM 能拟合非线性时序模式。
- 对比 Transformer：LSTM 在小数据、低复杂度场景更容易快速落地。

## R12

工程注意点：

- 时序任务切分要避免“未来信息泄漏”，本实现使用 `shuffle=False`。
- 标准化必须只用训练集拟合，再作用于测试集。
- `window_size` 太小会欠拟合，太大则训练慢且易过拟合。
- 评估应包含朴素基线，避免“看似有数值但无增益”。

## R13

`demo.py` 的最小栈与功能：

- `numpy`：生成合成时序与窗口构造；
- `scipy`：`savgol_filter` 平滑，`pearsonr` 相关性评估；
- `scikit-learn`：`train_test_split`、`StandardScaler`、误差指标；
- `PyTorch`：LSTM 模型、训练循环、推理；
- `pandas`：最终指标表与预测样例表输出。

脚本无交互输入，运行一次即可完成训练与评估。

## R14

运行方式：

```bash
cd Algorithms/数学-深度学习-0311-LSTM
uv run python demo.py
```

输出内容包括：

- 分阶段 epoch 训练/测试 MSE；
- 最终指标表（RMSE、MAE、基线 RMSE、Pearson r/p）；
- 前 10 条预测样例；
- 校验通过提示 `All checks passed.`。

## R15

结果解读建议：

- 先看 `train_mse/test_mse` 是否整体下降并趋稳；
- 再看 `rmse` 是否低于 `baseline_rmse`；
- `pearson_r` 越接近 1，说明趋势跟随越好；
- 若误差下降但相关性低，通常代表幅值拟合了但时序相位没对齐。

## R16

常见问题：

- 损失不降：学习率过高、窗口过长或输入未标准化；
- 过拟合：训练误差低、测试误差高，可降隐藏维或减 epoch；
- 结果抖动：随机种子不固定，或数据噪声过大；
- 模型劣于基线：说明任务/特征设计不足，需检查窗口与数据生成。

## R17

可扩展方向：

- 从一步预测扩展到多步滚动预测；
- 加入外生变量（多维输入）形成多变量 LSTM；
- 堆叠多层 LSTM 或加注意力层；
- 用早停、学习率调度、正则化提升泛化。

## R18

本实现在源码层面的算法流（8 步）如下：

1. `generate_series` 生成带趋势/季节/噪声的原始序列，并用 `savgol_filter` 平滑，得到可学习信号。  
2. `build_supervised_dataset` 用滑窗把时序重排为监督样本 `(X_t_window, y_{t+1})`。  
3. `split_and_scale` 先按时间切分训练/测试，再用 `StandardScaler` 在训练集拟合并变换两侧数据，避免泄漏。  
4. `make_dataloaders` 把 `numpy` 样本封装为 `[batch, seq_len, 1]` 张量批次，供 LSTM 按时间维读取。  
5. `LSTMForecaster.forward` 内部执行 LSTM 门控更新（遗忘/输入/候选/输出门），取最后时刻隐状态经线性层得到一步预测。  
6. `run_epoch` 执行 `forward -> MSELoss -> backward -> Adam.step`（训练）或纯前向（评估），形成标准监督学习闭环。  
7. `predict_scaled` 生成测试预测后，经 `y_scaler.inverse_transform` 还原到真实量纲，并构造朴素基线（上一时刻值）对照。  
8. `summarize_metrics` 计算 `RMSE/MAE/Pearson` 并输出 `pandas` 表格，`main` 进行数值健全性与基线比较校验。  

这 8 步覆盖了“数据构造 -> 门控建模 -> 优化训练 -> 反标准化评估”的完整 LSTM 最小实现链路。

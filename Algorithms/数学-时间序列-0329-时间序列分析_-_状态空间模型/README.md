# 时间序列分析 - 状态空间模型

- UID: `MATH-0329`
- 学科: `数学`
- 分类: `时间序列`
- 源序号: `329`
- 目标目录: `Algorithms/数学-时间序列-0329-时间序列分析_-_状态空间模型`

## R01

状态空间模型（State Space Model, SSM）把时间序列拆成两层：
- 隐状态层（系统真实但不可直接观测）
- 观测层（带噪声的测量值）

在高斯线性条件下，核心推断工具是卡尔曼滤波与RTS平滑。这个 MVP 选择最小可解释版本: 本地水平模型（Local Level Model），用于展示“估计隐藏状态 + 预测未来 + 参数拟合”的完整闭环。

## R02

本地水平模型定义为：

- 状态方程：`x_t = x_{t-1} + w_t`, `w_t ~ N(0, q)`
- 观测方程：`y_t = x_t + v_t`, `v_t ~ N(0, r)`

其中：
- `x_t` 是时刻 `t` 的潜在真实水平
- `y_t` 是观测值
- `q` 为过程噪声方差，控制状态变化快慢
- `r` 为观测噪声方差，控制测量误差

## R03

线性高斯状态空间的矩阵形式是：

- `x_t = A x_{t-1} + w_t`, `w_t ~ N(0, Q)`
- `y_t = H x_t + v_t`, `v_t ~ N(0, R)`

本题取：
- `A = 1`
- `H = 1`
- `Q = q`
- `R = r`

即一维最简模型，便于把每一步数值流程写清楚。

## R04

卡尔曼滤波递推（对每个 `t`）：

1. 预测步：
- `a_t = m_{t-1}`
- `P_t = C_{t-1} + q`

2. 更新步（若 `y_t` 可观测）：
- 创新：`v_t = y_t - a_t`
- 创新方差：`F_t = P_t + r`
- 卡尔曼增益：`K_t = P_t / F_t`
- 后验均值：`m_t = a_t + K_t v_t`
- 后验方差：`C_t = (1 - K_t) P_t`

3. 若 `y_t` 缺失：仅传播，`m_t = a_t`, `C_t = P_t`。

## R05

对数似然（高斯）可在滤波时累加：

`log p(y_{1:T}) = Σ_t -0.5 * [log(2π) + log(F_t) + v_t^2 / F_t]`

这允许在不显式积分隐藏状态的前提下做参数估计。demo.py 直接用该似然进行 MLE。

## R06

RTS 平滑（Rauch-Tung-Striebel）在滤波完成后反向修正状态：

- 初始化：`s_T = m_T`, `S_T = C_T`
- 反向递推：
  - `J_t = C_t / (C_t + q)`（本地水平模型中等价于 `C_t / P_{t+1}`）
  - `s_t = m_t + J_t (s_{t+1} - a_{t+1})`
  - `S_t = C_t + J_t^2 (S_{t+1} - P_{t+1})`

平滑结果通常比单向滤波更接近真实隐藏状态。

## R07

参数估计采用极大似然（MLE）：

- 目标：最小化负对数似然 `NLL(q, r)`
- 约束：`q > 0`, `r > 0`
- 实现策略：优化 `log(q), log(r)`，再通过 `exp` 映射回正数域
- 优化器：`scipy.optimize.minimize(method="L-BFGS-B")`

这种做法稳健且易于迁移到更高维状态空间模型。

## R08

时间复杂度（本地水平一维）：

- 滤波：`O(T)`
- 平滑：`O(T)`
- 单次似然评估：`O(T)`
- MLE 总体：`O(T * N_iter)`

空间复杂度为 `O(T)`（保存轨迹用于平滑）。

## R09

数值稳定与工程细节：

- 方差参数统一用对数参数化，避免优化过程中出现负方差。
- 对缺失值 `NaN` 使用“跳过更新、仅预测”策略。
- 初始化协方差 `c0` 取较大值（如10），表示先验不确定较高。
- 打印 `NLL`、RMSE 与未来预测标准差，便于快速 sanity check。

## R10

`demo.py` 模块结构：

- `simulate_local_level`: 生成合成数据
- `kalman_filter_local_level`: 滤波 + 对数似然
- `rts_smoother_local_level`: 反向平滑
- `fit_local_level_mle`: 估计 `q, r`
- `forecast_local_level`: 多步预测
- `main`: 串联实验流程并输出结果

## R11

运行方式（非交互）：

```bash
uv run python Algorithms/数学-时间序列-0329-时间序列分析_-_状态空间模型/demo.py
```

脚本会自动：
- 生成数据
- 注入少量缺失观测
- 拟合参数
- 输出滤波/平滑误差与未来5步预测

## R12

一次典型输出会包含：

- 真实参数 `q_true, r_true`
- MLE 参数 `q_hat, r_hat`
- 状态RMSE（滤波 vs 平滑）
- 一步预测RMSE
- 未来5步预测的 `(mean, std)`

一般可观察到：
- `q_hat, r_hat` 接近真实值（有限样本下允许偏差）
- 平滑RMSE通常不高于滤波RMSE

## R13

关键可调参数：

- `n`: 序列长度（默认220）
- `q_true`, `r_true`: 数据生成噪声
- `seed`: 随机种子
- 缺失模式：当前示例为每37个点缺失一个
- `horizon`: 未来预测步数（默认5）

## R14

适用场景：

- 传感器噪声去噪
- 经济/金融时间序列的平滑与短期预测
- 存在缺失观测的在线估计任务

局限：

- 当前模型只有“水平项”，没有趋势/季节项
- 假设高斯噪声，重尾噪声下可能偏弱

## R15

可扩展方向：

- 本地线性趋势模型（增加斜率状态）
- 季节状态空间模型
- 外生变量（动态回归 / DLM）
- EM 算法估计参数
- 非线性扩展（EKF/UKF）或粒子滤波

## R16

最小验证清单：

- `README.md` 与 `demo.py` 无占位符残留
- `demo.py` 可直接运行，无交互输入
- 输出包含参数估计、误差指标、未来预测
- 缺失值不会导致程序崩溃

## R17

参考资料（建议进一步阅读）：

1. Durbin, J., & Koopman, S. J. *Time Series Analysis by State Space Methods*.
2. Harvey, A. C. *Forecasting, Structural Time Series Models and the Kalman Filter*.
3. Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems.
4. Rauch, H. E., Tung, F., & Striebel, C. T. (1965). Maximum Likelihood Estimates of Linear Dynamic Systems.

## R18

demo.py 的“源码级算法流程”可拆成 8 步：

1. 用 `simulate_local_level` 生成 `x_t`（隐状态）和 `y_t`（观测），形成可控实验数据。
2. 在 `y_t` 中人为置入 `NaN`，模拟真实场景中的缺测。
3. 调用 `fit_local_level_mle`，把 `(log q, log r)` 作为优化变量。
4. 在目标函数里反复调用 `kalman_filter_local_level`，每次根据当前 `q, r` 计算整段序列 `loglik`。
5. `L-BFGS-B` 最小化 `-loglik`，得到 `q_hat, r_hat`。
6. 用估计参数再次滤波，得到 `predicted_mean/var` 与 `filtered_mean/var`。
7. 调用 `rts_smoother_local_level` 做反向递推，得到 `smoothed_mean/var`。
8. 计算 RMSE 与未来多步预测，打印可核验结果，完成“估计-平滑-预测”闭环。

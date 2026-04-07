# Kaplan-Meier估计

- UID: `MATH-0289`
- 学科: `数学`
- 分类: `生存分析`
- 源序号: `289`
- 目标目录: `Algorithms/数学-生存分析-0289-Kaplan-Meier估计`

## R01

Kaplan-Meier 估计（又称 product-limit estimator）是生存分析中最常用的非参数方法之一，用来估计生存函数：

\\[
S(t)=P(T>t)
\\]

它能在存在右删失（right censoring）的数据下，给出随时间变化的阶梯状生存曲线。

## R02

给定每个样本的观察时长 \\(t_i\\) 与事件指示 \\(\delta_i\\in\{0,1\}\\)：

- \\(\delta_i=1\\)：在 \\(t_i\\) 发生了目标事件（如死亡、故障）。
- \\(\delta_i=0\\)：在 \\(t_i\\) 时仍未发生事件，属于右删失。

目标是在不假设具体分布（如 Weibull、Log-normal）的前提下估计 \\(S(t)\\)，并支持在任意查询时刻做生存概率预测。

## R03

KM 估计的核心公式是：

\\[
\hat{S}(t)=\prod_{t_j \le t}\left(1-\frac{d_j}{n_j}\right)
\\]

其中：

- \\(t_j\\) 是按时间排序后的不同事件时点（实现中对所有唯一时间点更新，只有 `d_j>0` 时曲线下降）。
- \\(n_j\\) 是时点 \\(t_j\\) 之前仍在风险集（at risk）中的人数。
- \\(d_j\\) 是时点 \\(t_j\\) 发生事件的人数。

## R04

不确定性估计使用 Greenwood 公式：

\\[
\mathrm{Var}(\hat S(t))=\hat S(t)^2\sum_{t_j\le t}\frac{d_j}{n_j(n_j-d_j)}
\\]

并在实现中给出 log-log 变换的近似置信区间：

\\[
\hat S(t)^{\exp(\pm z_{\alpha/2}\cdot SE_{\log\log})}
\\]

这样比直接正态近似更容易保证区间落在 `[0,1]`。

## R05

优势：

- 非参数：不强依赖分布假设。
- 原生支持右删失数据。
- 结果可解释性强：生存曲线是事件累计作用的直接表达。

局限：

- 只能处理单变量时间到事件，不直接建模协变量影响（协变量通常交给 Cox 模型）。
- 在样本极少或尾部风险集很小的区域，估计方差会增大。
- 默认假设删失是独立删失（non-informative censoring）。

## R06

本目录 MVP 的建模边界：

- 数据类型：仅支持右删失，`events` 为二值 `{0,1}`。
- 输入形态：`durations/events` 均为 1D 数组。
- 置信区间：默认 `alpha=0.05`，即 95% CI。
- 输出能力：支持 `fit/predict/predict_interval/median_survival_time/event_table`。

## R07

算法流程（概念级）：

1. 按观察时间对样本排序。
2. 枚举唯一时间点，统计每个时间点的 `n_at_risk / n_events / n_censored`。
3. 若该时间点有事件（`d_j>0`），用乘积极限公式更新生存率。
4. 累加 Greenwood 项，得到方差轨迹。
5. 由方差构造 log-log 置信区间。
6. 用阶梯函数规则回答任意查询时间（`searchsorted`）。

## R08

设样本数为 \\(n\\)，唯一时间点数为 \\(u\\)，查询时间数为 \\(m\\)：

- 拟合排序：\\(O(n\log n)\\)。
- 拟合主循环：逐个唯一时间统计，当前实现总体约 \\(O(n+u)\\) 到 \\(O(nu)\\) 之间（取决于掩码统计方式）；在常见规模下可接受。
- 查询预测：`searchsorted` 为 \\(O(m\log u)\\)。
- 空间复杂度：主要存储事件表列，约 \\(O(u)\\)。

## R09

数值与工程注意事项：

- 当某时点 `n_j == d_j` 时，Greenwood 分母出现零；实现中对该项不再累加，且此后生存率通常已到 0。
- 当 \\(\hat S(t)=1\\) 或 \\(\hat S(t)=0\\) 时，log-log 变换不可直接使用；实现里做了分支保护。
- 由于浮点误差，单调性检查使用 `<= 1e-12` 容差。
- 输入若包含负时间、空数组、非二值事件，会显式报错。

## R10

典型应用场景：

- 医学：患者生存时间、复发时间分析。
- 工程可靠性：设备失效时间分析。
- 互联网产品：留存/流失时间分析（事件可定义为流失）。
- 金融风控：贷款违约前存续时间分析。

## R11

与常见方法对比：

- 对比 Nelson-Aalen：Nelson-Aalen 先估累计风险函数再映射到生存函数；KM 直接估生存函数。
- 对比 Cox 比例风险模型：Cox 能估协变量效应和风险比；KM 更适合单组或分组描述性生存曲线。
- 对比参数模型（如 Weibull）：参数模型在分布假设正确时效率高；KM 更稳健但外推能力弱。

## R12

`demo.py` 结构：

- `KaplanMeierEstimator`：核心估计器，包含 `fit/predict/predict_interval`。
- `simulate_right_censored_data`：生成两组风险水平不同的模拟删失数据。
- `summarize_model`：按查询时间打印生存率与置信区间。
- `main`：执行整体组与分组估计，并输出事件表头部。

## R13

MVP 数据与验证设置：

- 样本量：默认 320。
- 机制：高风险组使用更高事件 hazard，删失时间来自独立指数分布。
- 观测值：`duration = min(event_time, censor_time)`。
- 事件标签：`event_time <= censor_time` 记为 1，否则为 0。
- 自检项：输出总体删失率、曲线单调性检查结果、分组中位生存时间。

## R14

运行方式：

```bash
cd Algorithms/数学-生存分析-0289-Kaplan-Meier估计
uv run python demo.py
```

依赖（本实现实际使用）：

- `numpy`
- `pandas`
- `scipy`（仅在 `alpha != 0.05` 时用于标准正态分位数）

## R15

输出解读：

- `Overall censoring rate`：删失样本占比。
- `Monotone survival check`：KM 曲线是否保持非增（应为 `True`）。
- 每组 `time | survival | 95% CI`：查询时刻的生存率与区间估计。
- `median_survival_time`：首个使生存率不高于 0.5 的时间点（若未降到 0.5，则输出 `inf`）。
- `Event table head`：展示前若干时点的风险集、事件数、删失数和生存率。

## R16

边界条件与异常处理：

- `alpha` 必须在 `(0,1)`。
- `durations/events` 必须同长度、1D、非空。
- `durations` 不能为负。
- `events` 必须是 `{0,1}`。
- 预测前必须先 `fit`，否则抛出 `RuntimeError`。
- 查询时间也必须是一维数组。

## R17

可扩展方向：

- 增加 log-rank 检验，做分组生存曲线显著性比较。
- 支持左截断（left truncation）和区间删失（interval censoring）。
- 增加 bootstrap 置信区间或 Hall-Wellner 带。
- 输出绘图（阶梯曲线 + CI 阴影）并保存为文件。
- 扩展为与 Cox 模型联动的“描述 + 建模”完整流程。

## R18

`demo.py` 的源码级算法流可拆为 8 步：

1. `simulate_right_censored_data` 生成 `event_time` 与 `censor_time`，通过 `min` 与比较运算得到 `(duration, event)`。
2. `fit` 接收数组并完成输入校验，然后用 `np.argsort` 将样本按时间升序重排。
3. 用 `np.unique` 得到所有唯一时间点；循环每个时点，用布尔掩码统计 `d_j`（事件）和 `c_j`（删失）。
4. 记录该时点风险集人数 `n_j`，若 `d_j>0`，按 \\(1-d_j/n_j\\) 更新累计生存率。
5. 同步累加 Greenwood 项 \\(d_j/[n_j(n_j-d_j)]\\)，计算方差轨迹 \\(\hat S(t)^2 \cdot \text{sum}\\)。
6. 对每个时点用 log-log 变换构造置信区间；对 `S=0/1` 的边界情形做分支裁剪。
7. `predict` / `predict_interval` 用 `np.searchsorted(..., side="right")-1` 实现阶梯函数查询。
8. `main` 分别拟合总体、低风险组、高风险组，输出中位生存时间、查询点生存率和事件表，形成可复现实验闭环。

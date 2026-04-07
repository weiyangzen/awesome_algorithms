# 伞形采样 (Umbrella Sampling)

- UID: `PHYS-0317`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `320`
- 目标目录: `Algorithms/物理-计算物理-0320-伞形采样_(Umbrella_Sampling)`

## R01

伞形采样（Umbrella Sampling）用于解决“自由能面高势垒导致常规采样跨不过去”的问题。  
核心思想是：在反应坐标上放置多个带偏置势的窗口，让每个窗口只负责局部区域高质量采样，再把这些局部样本重加权拼接回无偏分布。

本条目给出 1D 最小可运行 MVP：`Metropolis-Hastings` 采样 + `WHAM`（Weighted Histogram Analysis Method）重构 PMF（potential of mean force）。

## R02

MVP 问题定义：
- 目标：估计一维坐标 `x` 的无偏自由能 `F(x) = -kBT ln P(x) + C`。
- 体系：一维双阱势能 `U(x)`（用于模拟存在势垒的体系）。
- 偏置：第 `i` 个窗口使用谐和伞形势 `w_i(x)=0.5*kappa*(x-c_i)^2`。
- 采样：每个窗口在 `U(x)+w_i(x)` 下进行 MH 采样。
- 重构：把所有窗口直方图输入 WHAM，得到无偏概率 `P(x)` 与 PMF。

## R03

数学模型（无量纲，脚本中设 `beta = 1/(kBT) = 1`）：

1. 无偏分布：
`P(x) ∝ exp[-beta * U(x)]`

2. 第 `i` 个窗口的偏置分布：
`P_i^b(x) ∝ exp[-beta * (U(x) + w_i(x))]`

3. 离散到直方图 bin `l` 后，WHAM 固定点方程为：
`p_l = (sum_i n_il) / (sum_i N_i * exp(beta * (f_i - w_i(x_l))))`

`exp(-beta * f_i) = sum_l p_l * exp(-beta * w_i(x_l))`

其中：
- `n_il`：窗口 `i` 在 bin `l` 的计数；
- `N_i`：窗口 `i` 的总样本数；
- `f_i`：窗口归一化自由能偏移；
- `p_l`：无偏离散概率（归一化后 `sum_l p_l = 1`）。

## R04

算法总流程：
1. 定义双阱势 `U(x)`，设定窗口中心 `c_i`、偏置强度 `kappa`、直方图网格。  
2. 对每个窗口做 Metropolis-Hastings，采样 `x` 并记录接受率。  
3. 统计每个窗口直方图 `n_il` 与样本总数 `N_i`。  
4. 预计算偏置矩阵 `w_i(x_l)`。  
5. 用 WHAM 迭代更新 `{f_i}` 与 `{p_l}` 直到收敛。  
6. 计算 PMF：`F_l = -ln(p_l)`，并减去最小值做零点平移。  
7. 额外跑一条“无偏长链”作为参考 PMF，对重构质量做数值核验。  
8. 输出窗口诊断、WHAM 收敛信息、PMF 对照表和 PASS/FAIL。

## R05

`demo.py` 的核心数据结构：
- `WindowSample`：单窗口中心、偏置参数、样本数组、接受率。  
- `WhamResult`：bin 中心、概率、PMF、`f_i`、收敛标志、迭代轮数、终止增量。  
- `ValidationReport`：验收指标与失败原因列表。  
- `numpy.ndarray`：存储窗口样本、直方图、偏置矩阵、PMF。  
- `pandas.DataFrame`：打印窗口统计表与 PMF 采样点表。

## R06

正确性依据：
- MH 采样保证每个窗口收敛到其偏置目标分布。  
- WHAM 方程本质是对多窗口偏置采样的一致性重加权估计，固定点解对应全局无偏概率估计。  
- PMF 与概率满足单调映射 `F=-ln P + C`，故只确定到常数平移。  
- 通过“WHAM PMF vs 无偏长链 PMF”做独立交叉验证，避免只做自洽检查。

## R07

复杂度分析（`W` 窗口数，`S` 每窗口样本数，`B` bin 数，`T` 为 WHAM 迭代轮数）：
- 多窗口采样：`O(W*S)`。  
- 直方图统计：`O(W*S)`。  
- WHAM 每轮更新：`O(W*B)`。  
- WHAM 总计：`O(T*W*B)`。  
- 空间复杂度：`O(W*B + W*S_kept)`（MVP 保留样本用于可审计输出）。

## R08

数值稳定与鲁棒性策略：
- WHAM 分母和归一化项使用 `scipy.special.logsumexp`，避免指数下溢/上溢。  
- 概率 `p_l` 在对数前做下界截断（`1e-300`）以防 `log(0)`。  
- PMF 用 `inf` 表示零概率 bin，并在有限区域内平移到最小值为 0。  
- 验证中同时检查收敛、窗口重叠、接受率、覆盖率和参考 RMSE，避免单指标误判。

## R09

MVP 取舍说明：
- 只做 1D 反应坐标，不扩展到高维 CV。  
- 只实现谐和偏置和 WHAM，不引入 MBAR 等更复杂估计器。  
- 参考解用无偏 MH 长链近似，不追求解析真值。  
- 目标是“最小、诚实、可复现、可核验”，而不是完整分子模拟平台。

## R10

技术栈（最小可运行）：
- `numpy`：向量化数值计算、随机采样、直方图统计。  
- `scipy`：`logsumexp` 进行稳定的对数域求和。  
- `pandas`：终端表格化输出诊断结果。  
- Python 标准库 `dataclasses`：结构化结果对象。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0320-伞形采样_(Umbrella_Sampling)
uv run python demo.py
```

脚本无交互输入，运行后自动完成采样、WHAM 重构和验收。

## R12

主要输出字段解释：
- `window / center / kappa`：窗口编号、中心、伞形刚度。  
- `samples`：该窗口保留样本数。  
- `acceptance`：MH 接受率。  
- `sample_mean / sample_std`：窗口样本均值和标准差。  
- `converged / iterations / max_delta`：WHAM 收敛状态、迭代轮数、最后增量。  
- `min adjacent overlap`：相邻窗口 Bhattacharyya 最小重叠系数。  
- `finite PMF coverage`：有限 PMF bin 占比。  
- `PMF RMSE vs reference`：与无偏参考 PMF 的均方根误差（单位 `kBT`）。

## R13

内置验收条件（全部满足才 `PASS`）：
1. WHAM 在 `max_iter` 内收敛。  
2. 平均接受率在 `[0.15, 0.85]`。  
3. 最小相邻窗口重叠 `>= 0.03`。  
4. 有限 PMF 覆盖率 `>= 0.75`。  
5. 参考对比 bin 数 `>= 30`。  
6. `PMF RMSE <= 1.0 kBT`。

## R14

关键可调参数：
- `centers`：窗口中心分布范围与密度。  
- `kappa`：偏置强度（过小窗口泄露，过大重叠不足）。  
- `proposal_sigma`：MH 提议步长（影响接受率和自相关）。  
- `n_steps / burn_in / thin`：采样质量与耗时的平衡。  
- `BIN_EDGES`：PMF 空间分辨率。  
- `tol / max_iter`：WHAM 收敛阈值与上限。

## R15

方法对比：
- 对比直接无偏采样：伞形采样在高势垒体系中更容易覆盖全坐标空间。  
- 对比单窗口偏置：多窗口 + WHAM 可恢复全局无偏 PMF，而不是局部形状。  
- 对比黑箱调用：本实现把“采样、建直方图、重加权、验收”全部显式写出，便于调试与教学。

## R16

典型应用：
- 分子构象转变自由能剖面（沿反应坐标）。  
- 化学反应路径中的势垒估计。  
- 蛋白配体解离坐标上的 PMF 近似。  
- 任何“稀有事件导致直接采样困难”的统计物理问题。

## R17

当前局限与扩展方向：
- 局限：1D CV、单温度、无动力学时间尺度分析。  
- 可扩展：
  - 多维 CV 与分块直方图；
  - 自适应窗口布置；
  - MBAR 替代 WHAM；
  - 不确定性评估（bootstrap / block averaging）；
  - 与真实 MD 轨迹文件对接。

## R18

`demo.py` 源码级流程拆解（9 步）：
1. `base_potential` 与 `harmonic_bias` 定义无偏势和伞形偏置，形成每个窗口的总能量。  
2. `metropolis_samples` 在指定窗口上执行 MH：提议、接受拒绝、烧入、抽稀，输出样本和接受率。  
3. `run_umbrella_windows` 遍历所有窗口中心，分别采样，形成 `WindowSample` 列表。  
4. `build_window_histograms` 把每个窗口样本映射到统一 `BIN_EDGES`，得到 `n_il` 与 `N_i`。  
5. `run_wham` 预置 `f_i=0`，然后迭代：
   - 用 `logsumexp` 计算分母 `sum_i N_i exp(beta(f_i-w_i(x_l)))`；
   - 更新 `p_l` 并归一化；
   - 再由 `p_l` 更新 `f_i`，直到 `max_delta < tol`。  
6. `run_wham` 将 `p_l` 转成 PMF：`F_l=-ln(p_l)`，并做常数平移归零。  
7. `unbiased_reference_pmf` 运行一条无偏长链，构建参考 PMF（同 bin 网格）。  
8. `validate_result` 汇总收敛、接受率、窗口重叠、覆盖率、RMSE 等指标并给出失败原因。  
9. `main` 打印窗口诊断表、WHAM 诊断和 PMF 采样点，最终输出 `Validation: PASS/FAIL`。

即使调用了 `numpy/scipy` 基础数值函数，算法主干（偏置采样、WHAM 固定点、重构与验收）都在源码中显式展开，没有把伞形采样留成黑箱。

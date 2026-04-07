# 混沌理论 (Chaos Theory)

- UID: `PHYS-0117`
- 学科: `物理`
- 分类: `非线性动力学`
- 源序号: `117`
- 目标目录: `Algorithms/物理-非线性动力学-0117-混沌理论_(Chaos_Theory)`

## R01

混沌理论研究“确定性系统中的不可长期精确预测”现象：系统演化规则是明确的，但对初始条件极端敏感，导致长期行为看似随机。

本条目使用 Logistic 映射作为最小可审计模型，演示三件核心事实：
- 非线性迭代可以从周期行为过渡到混沌；
- 正 Lyapunov 指数对应指数级初值敏感性；
- 分岔结构体现参数微调引起的全局动力学重排。

## R02

本目录 MVP 要解决的问题：

1. 手写实现离散非线性系统 `x_{n+1} = r x_n (1 - x_n)` 的迭代过程。
2. 计算 Lyapunov 指数并区分典型参数下的周期/混沌状态。
3. 用两条极近初值轨迹拟合对数距离增长率，量化“蝴蝶效应”。
4. 使用自相关峰检测给出周期提示（period hint）。
5. 生成分岔采样表（`r`-`x` 点云数据）并做数值合法性检查。
6. 用 NumPy 与 PyTorch 的同构实现做一致性校验，避免黑盒依赖。

## R03

离散动力系统定义：

- 状态变量：`x_n in (0,1)`。
- 控制参数：`r in [0,4]`。
- 迭代映射：
  - `f_r(x) = r x (1-x)`
  - `x_{n+1} = f_r(x_n)`

在 `r` 增大时，系统会出现固定点、倍周期分岔以及混沌区间。本 MVP 选取 `r=3.50, 3.57, 3.90` 作为代表点。

## R04

Lyapunov 指数（最大指数的一维版本）计算公式：

- 映射导数：`f_r'(x) = r(1-2x)`
- 有限样本估计：
  - `lambda ≈ (1/N) * sum_{n=1..N} log(|f_r'(x_n)|)`

解释：
- `lambda < 0`：邻近轨道收敛，常见于稳定周期轨道。
- `lambda > 0`：邻近轨道指数发散，混沌特征明显。

`demo.py` 对导数绝对值做了下界截断，防止 `log(0)` 数值崩溃。

## R05

除 Lyapunov 指数外，脚本还输出两个辅助诊断：

1. 自相关周期提示：
   - 对后段轨迹去均值后做自相关；
   - 用 `scipy.signal.find_peaks` 检测显著峰，给出最显著 lag 作为 `period_hint`；
   - 若无显著峰，则返回 `-1`（无明确周期提示）。
2. 分布熵：
   - 对轨迹做直方图，调用 `scipy.stats.entropy` 计算 Shannon 熵；
   - 用于粗略表征轨迹分布的“铺展程度”。

## R06

算法主流程（高层）：

1. 固定初值 `x0` 与代表参数集合 `r`；
2. 对每个 `r` 生成长轨迹（先 burn-in 再采样）；
3. 计算 Lyapunov 指数、周期提示与分布熵；
4. 在混沌参数下计算两条近邻轨迹距离并线性拟合 `log(distance)`；
5. 构造 `r` 网格，分别用 NumPy 与 Torch 批量迭代，计算 RMSE 与最大误差；
6. 生成分岔采样表并汇总统计；
7. 输出检查项布尔值与总验收结论。

## R07

核心数据结构：

- `series: np.ndarray(shape=(N,))`
  - 单参数下的 Logistic 轨迹。
- `case_df: pandas.DataFrame`
  - 每个代表参数的诊断结果，字段：
  - `r, lyapunov, period_hint, hist_entropy, regime`
- `np_batch / torch_batch: np.ndarray(shape=(T, M))`
  - 多参数批量轨迹，用于跨框架一致性校验。
- `bif_df: pandas.DataFrame`
  - 分岔采样点，字段：`r, x`。
- `checks: Dict[str, bool]`
  - 汇总验收结果。

## R08

正确性与一致性校验：

- 动力学性质校验：
  - `r=3.50` 的 Lyapunov 指数应为负；
  - `r=3.90` 的 Lyapunov 指数应为正。
- 初值敏感性校验：
  - `log(distance)` 拟合斜率应为正。
- 数值实现校验：
  - NumPy 与 Torch 批量迭代误差足够小。
- 物理/数学合法域校验：
  - 分岔样本中的 `x` 维持在 `[0,1]`；
  - 全部输出为有限数值（无 `nan/inf`）。

## R09

复杂度分析（设单轨迹长度为 `N`，参数个数为 `M`）：

- 单条轨迹迭代：时间 `O(N)`，空间 `O(N)`。
- Lyapunov 与熵计算：时间 `O(N)`。
- 批量一致性对照（`M` 条轨迹）：时间 `O(NM)`，空间 `O(NM)`。
- 分岔采样（每个 `r` 保留 `K` 个尾部点）：
  - 时间约 `O(MN)`；
  - 输出空间 `O(MK)`。

MVP 的主要开销来自分岔采样与批量轨迹构造。

## R10

边界与异常处理：

- 参数 `r` 必须在 `[0,4]`；
- 初值 `x0` 必须在 `(0,1)`；
- `steps / fit_points / tail_keep / max_lag` 必须为正整数；
- `burn_in` 不可为负；
- `delta0` 必须为正且 `x0 + delta0 < 1`；
- 输入数组维度和有限性均有显式检查。

这保证了脚本在失败时抛出可解释异常，而不是静默输出错误结果。

## R11

MVP 取舍说明：

- 选择 Logistic 映射而非高维连续系统（如 Lorenz）以保持最小闭环；
- 重点做“可审计实现 + 可复现实验”，不追求图形化展示；
- SciPy/sklearn/PyTorch 用于验证与统计，而非替代核心动力学公式；
- 周期检测只提供提示值，不作为严格拓扑分类器。

## R12

`demo.py` 主要函数职责：

- `logistic_step`：单步映射。
- `simulate_logistic_numpy`：NumPy 单参数轨迹生成。
- `simulate_logistic_torch`：Torch 多参数批量轨迹生成。
- `lyapunov_exponent_logistic`：Lyapunov 指数估计。
- `estimate_period_autocorr`：自相关峰周期提示。
- `finite_time_divergence_rate`：近邻轨迹对数距离增长率拟合。
- `bifurcation_dataframe`：分岔采样点构建。
- `classify_regime`：基于指标给出粗粒度状态标签。
- `main`：组织实验、打印表格与验收项。

## R13

运行方式：

```bash
cd Algorithms/物理-非线性动力学-0117-混沌理论_(Chaos_Theory)
uv run python demo.py
```

脚本不需要任何命令行参数，也不需要交互输入。

## R14

关键输出字段说明：

- `lyapunov`：Lyapunov 指数估计值。
- `period_hint`：自相关峰给出的主周期提示（无明显峰时为 `-1`）。
- `hist_entropy`：轨迹直方图 Shannon 熵。
- `log_distance_slope`：初值敏感性拟合斜率。
- `linear_fit_rmse`：对数距离线性拟合误差。
- `rmse_np_vs_torch` / `max_abs_np_vs_torch`：跨框架一致性误差。
- `bif_df` 与 `bif_summary`：分岔采样明细与聚合统计。
- `all_core_checks_pass`：核心检查总开关。

## R15

最小验收标准（建议）：

- `all_core_checks_pass=True`；
- `r=3.50` 的 Lyapunov 指数为负；
- `r=3.90` 的 Lyapunov 指数为正；
- 初值敏感性拟合斜率为正；
- NumPy 与 Torch 轨迹误差在浮点精度范围内；
- 分岔样本不越界且无非有限值。

## R16

与其他混沌模型关系（简述）：

- Logistic 映射：
  - 一维离散，公式简单，适合教学与指标验证。
- Lorenz / Rössler：
  - 连续时间常微分方程，状态维度更高，几何结构更丰富。
- Hénon 映射：
  - 二维离散映射，更接近“奇异吸引子”平面结构研究。

本条目定位为“混沌基础指标的最小实现”，可作为更复杂模型的前置模块。

## R17

可扩展方向：

1. 引入 Feigenbaum 常数估计，量化倍周期分岔尺度比；
2. 增加 permutation entropy / sample entropy 等复杂度指标；
3. 对 Lyapunov 指数做参数扫描并输出谱线表；
4. 扩展到 Lorenz 系统并比较离散/连续混沌特征；
5. 增加噪声扰动，分析随机扰动下的稳健性与可预报窗口。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `main` 设定 `x0` 与代表参数 `r in {3.50, 3.57, 3.90}`。
2. 对每个 `r` 调用 `simulate_logistic_numpy`，在循环中逐步执行 `x <- r*x*(1-x)`，生成 burn-in 后轨迹。
3. 调用 `lyapunov_exponent_logistic`，按 `mean(log(|r*(1-2*x_n)|))` 计算 Lyapunov 指数。
4. 调用 `estimate_period_autocorr`，对后段轨迹做自相关并用 `find_peaks` 提取最显著滞后作为 `period_hint`。
5. 对同一轨迹做直方图并调用 `entropy`，得到 `hist_entropy`。
6. 在 `r=3.9` 下调用 `finite_time_divergence_rate`：并行迭代两条近邻轨迹，记录距离，在线性回归上拟合 `log(distance)` 斜率。
7. 构造 `r` 网格，分别用 NumPy（逐列）与 Torch（批量）生成轨迹，再用 `mean_squared_error` 计算 RMSE 并取最大绝对误差。
8. 调用 `bifurcation_dataframe`，遍历参数网格、提取尾部点形成 `(r,x)` 样本，并做 groupby 聚合统计。
9. 汇总布尔检查项（Lyapunov 符号、敏感性斜率、跨框架一致性、值域合法性、有限性），打印 `all_core_checks_pass` 作为最终验收信号。

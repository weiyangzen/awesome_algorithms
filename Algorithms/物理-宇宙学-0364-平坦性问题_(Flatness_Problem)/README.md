# 平坦性问题 (Flatness Problem)

- UID: `PHYS-0346`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `364`
- 目标目录: `Algorithms/物理-宇宙学-0364-平坦性问题_(Flatness_Problem)`

## R01

平坦性问题是标准热大爆炸宇宙学中的经典初始条件精细调谐问题：
- 定义曲率偏离量 `delta = |Omega - 1| = |k|/(a^2 H^2)`。
- 在非暴胀演化下，`delta` 会随宇宙膨胀被放大（辐射时代约 `~a^2`，物质时代约 `~a`）。
- 因此若今天观测到 `|Omega_k|` 很小，则早期必须被极端精细地设定到接近 0，形成“为什么初始这么平坦”的问题。

## R02

本目录 MVP 目标：用最小可审计模型把“平坦性问题 + 暴胀解法”数值化。

实现闭环：
1. 设定再加热尺度 `T_reh` 与平衡红移 `z_eq`。
2. 计算无暴胀时从 `a_reh` 到今天的偏离增长因子 `G`。
3. 给定暴胀前 `delta_pre ~ O(1)`，计算今天偏离 `delta_today`。
4. 引入暴胀抑制 `delta -> delta * exp(-2N)`，反求满足今天约束所需最小 e-folds `N_min`。

## R03

符号与参数（见 `demo.py` 的 `FlatnessParams`）：
- `T_reh`：再加热温度，默认 `1e15 GeV`。
- `T0`：今天 CMB 温度对应能标，默认 `2.348e-13 GeV`。
- `g_s0, g_s_reh`：熵自由度，用于 `aT g_s^{1/3} = const`。
- `z_eq`：物质-辐射平衡红移，默认 `3400`。
- `omega_k_bound_today`：今天 `|Omega_k|` 目标上界，默认 `1e-2`。
- `delta_pre_inflation`：暴胀前曲率偏离（教学模型默认 `1`）。

## R04

MVP 采用的核心关系式：

1. 曲率偏离定义
`delta = |Omega - 1| = |k|/(a^2 H^2)`

2. 标准膨胀阶段标度律
- 辐射主导：`delta ~ a^2`
- 物质主导：`delta ~ a`

3. 再加热尺度因子（熵守恒近似）
`a_reh = (T0/T_reh) * (g_s0/g_s_reh)^(1/3)`

4. 平衡尺度因子
`a_eq = 1/(1+z_eq)`

5. 无暴胀总增长因子
`G = (a_eq/a_reh)^2 * (1/a_eq)`

6. 含暴胀的今天偏离
`delta_today(N) = delta_pre * exp(-2N) * G`

## R05

本任务“算法输出”包括：
- `a_reh, a_eq`：关键阶段尺度因子；
- `G_rad, G_mat, G_total`：辐射、物质、总增长因子；
- `delta_today_no_inflation`：无暴胀预测的今天偏离；
- `N_min`：满足 `delta_today <= omega_k_bound_today` 的最小 e-folds；
- 多个 `N` 采样点下的 `delta_today(N)` 与是否达标判定。

## R06

`demo.py` 的计算流程：
1. 用 `scale_factor_reheating` 与 `scale_factor_equality` 得到 `a_reh, a_eq`。
2. 用 `flatness_growth_factor_no_inflation` 计算 `G_rad, G_mat, G_total`。
3. 用 `omega_deviation_today(..., N=0)` 得到无暴胀今天偏离。
4. 构造残差函数 `f(N)=delta_pre*exp(-2N)*G_total-bound`。
5. 用 `required_efolds_numeric`（`scipy.optimize.brentq`）求数值根。
6. 用 `required_efolds_analytic` 给闭式根，做一致性对照。
7. 生成 `N` 扫描表并附带灵敏度分析（`T_reh` 与 `delta_pre`）。

## R07

正确性依据：
- 物理层：平坦性问题本质是 `delta` 在标准演化中增长，暴胀通过 `exp(-2N)` 指数压低 `delta`。
- 数学层：`delta_today(N)` 对 `N` 严格单调递减，因此阈值方程在 `delta_today(0)>bound` 时存在唯一根。
- 程序层：数值根与解析根同时输出，差值应接近数值精度极限。

## R08

复杂度（`m` 为 `N` 扫描点数）：
- 主计算（尺度因子、增长因子）均为 `O(1)`。
- `brentq` 求根迭代记为 `k`，复杂度 `O(k)`（通常几十步内）。
- 扫描表构建 `O(m)`。
- 空间开销主要来自结果表，`O(m)`。

## R09

数值稳定性与工程处理：
- 使用双精度 `numpy` 即可覆盖本任务参数范围。
- 指数项只在 `N` 数十量级评估，`exp(-2N)` 不会导致灾难性溢出。
- `required_efolds_numeric` 先判定 `delta_no_inflation <= bound`，可直接返回 `0`，避免不必要求根。
- `brentq` 使用有界区间与严格容差（`xtol=1e-12`, `rtol=1e-10`）。

## R10

代码模块划分：
- `FlatnessParams`：集中管理物理与观测参数；
- `scale_factor_reheating` / `scale_factor_equality`：关键时刻尺度因子；
- `flatness_growth_factor_no_inflation`：阶段性增长因子计算；
- `omega_deviation_today`：给定 `N` 的今天偏离；
- `required_efolds_numeric` / `required_efolds_analytic`：数值与解析双解；
- `build_scan_table`：生成 `pandas` 结果表；
- `run_demo`：组织实验、打印主结果和灵敏度扫描。

## R11

最小依赖栈：
- `numpy`：数组、指数与对数运算；
- `scipy.optimize`：`brentq` 一维求根；
- `pandas`：结果表展示。

未引入大型宇宙学框架，保持“可读 + 可验证”的最小 MVP。

## R12

运行方式（仓库根目录）：

```bash
uv run python "Algorithms/物理-宇宙学-0364-平坦性问题_(Flatness_Problem)/demo.py"
```

或切换到目标目录后：

```bash
uv run python demo.py
```

程序无需交互输入，会直接打印结果。

## R13

输出字段解释：
- `a_reh`, `a_eq`：模型中辐射末期与平衡时刻的尺度因子；
- `Growth in radiation/matter era`：两个宇宙学阶段对 `delta` 的放大倍数；
- `Total no-inflation growth G`：无暴胀总放大因子；
- `Predicted today without inflation`：无暴胀下今天偏离；
- `Required e-folds`：达到 `omega_k` 约束所需最小暴胀 e-folds；
- 扫描表中的 `passes_bound`：该 `N` 是否满足今天曲率上界。

## R14

自检建议：
1. 把 `omega_k_bound_today` 从 `1e-2` 放宽到 `1e-1`，应观察到 `N_min` 降低。
2. 把 `t_reh_gev` 提高（如 `1e12 -> 1e15`），通常会导致 `a_reh` 更小、增长更大、`N_min` 增加。
3. 把 `delta_pre_inflation` 从 `1` 改到 `10`，`N_min` 应增加约 `0.5*ln(10)`。
4. 检查数值根与解析根误差是否接近机器精度。

## R15

模型边界与局限：
- 这是教学型缩并模型，忽略了再加热动力学细节与暗能量阶段微调。
- 使用分段标度律（辐射 `a^2`、物质 `a`）而非全 Friedmann 数值积分。
- `delta_pre_inflation ~ O(1)` 是动机化假设，不代表唯一初态模型。
- 约束值使用示意级今天曲率上界，可按观测更新替换。

## R16

可扩展方向：
- 用完整 `H(a)`（含辐射、物质、暗能量）做数值积分替代分段近似。
- 引入不同再加热历史（非瞬时 reheating）并传播到 `a_reh`。
- 把 `delta_pre_inflation`、`T_reh` 设为分布，做蒙特卡洛不确定性传播。
- 接入观测后验（如 CMB/BAO）将 `omega_k_bound_today` 改为统计区间。

## R17

与暴胀宇宙学的关系：
- 平坦性问题、视界问题、磁单极子问题是暴胀的三大经典动机。
- 在本实现里，平坦性问题被压缩成一个单参数核心机制：
  `delta_today ~ exp(-2N)`。
- 因此只要 `N` 足够大，初始 `O(1)` 的曲率偏离也能被指数压低到今天观测允许范围。

## R18

本实现的源码级算法流（8 步）：
1. `run_demo` 构造 `FlatnessParams` 与 `N` 采样网格。
2. `scale_factor_reheating` 用 `aT g_s^{1/3}=const` 计算 `a_reh`，`scale_factor_equality` 由 `z_eq` 得到 `a_eq`。
3. `flatness_growth_factor_no_inflation` 计算辐射放大 `(a_eq/a_reh)^2`、物质放大 `(1/a_eq)` 与总因子 `G`。
4. `omega_deviation_today` 在 `N=0` 下给出无暴胀今天偏离，直接展示“平坦性问题”的量级冲突。
5. `required_efolds_numeric` 构造残差 `f(N)=delta_pre*exp(-2N)*G-bound`，并设置有界区间 `[0, n_hi]`。
6. `scipy.optimize.brentq` 在区间端点异号前提下迭代求根（混合二分/插值），得到数值 `N_min`。
7. `required_efolds_analytic` 计算闭式 `0.5*ln(delta_pre*G/bound)`，与数值根做一致性校验。
8. `build_scan_table` 生成 `N`-`delta_today`-`passes_bound` 表格，`run_demo` 继续执行 `T_reh` 与 `delta_pre` 灵敏度扫描。

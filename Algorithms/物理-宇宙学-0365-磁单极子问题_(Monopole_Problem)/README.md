# 磁单极子问题 (Monopole Problem)

- UID: `PHYS-0347`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `365`
- 目标目录: `Algorithms/物理-宇宙学-0365-磁单极子问题_(Monopole_Problem)`

## R01

磁单极子问题是标准大统一宇宙学中的经典“丰度灾难”：
- 在 GUT 相变（典型 `T_c ~ 10^16 GeV`）后，Kibble 机制通常预测每个因果视界量级会形成 `O(1)` 个拓扑缺陷（磁单极子）。
- 这种初始产额在后续绝热演化下难以被自然稀释到观测允许范围，导致今天应看到远高于上限的单极子通量。
- 该矛盾即“磁单极子问题”，是暴胀宇宙学被提出并广泛接受的重要动机之一。

## R02

本目录 MVP 的目标不是构建全细节早期宇宙代码，而是实现一个可审计的最小计算闭环：
1. 用 Kibble 一视界一单极近似估算初始丰度 `Y_M = n_M/s`。
2. 将其映射为今天的各向同性通量 `F_0`。
3. 引入暴胀稀释 `n_M -> n_M e^{-3N}`，计算 `F(N)`。
4. 以 Parker 磁场生存界（`F_bound ~ 10^-16 cm^-2 s^-1 sr^-1`）反求最小 e-folds `N_min`。

这给出“为什么暴胀能解单极子问题”的数值化直观结论。

## R03

符号与参数（见 `demo.py` 的 `MonopoleParams`）：
- `T_c`：GUT 相变温度（默认 `1e16 GeV`）。
- `g_*`：相对论自由度（默认 `100`）。
- `kappa`：每视界单极子数的无量纲系数（默认 `1`，用于 `O(1)` 不确定性）。
- `M_pl`：普朗克质量（`1.22e19 GeV`）。
- `s_0`：今天熵密度（`2891.2 cm^-3`）。
- `v/c`：今天单极子速度尺度（默认 `1e-3`）。
- `F_bound`：通量上界（Parker 界，默认 `1e-16 cm^-2 s^-1 sr^-1`）。

## R04

MVP 使用的核心方程：

1. 辐射主导哈勃率
`H(T) = 1.66 * sqrt(g_*) * T^2 / M_pl`

2. 熵密度
`s(T) = (2*pi^2/45) * g_* * T^3`

3. Kibble 初始数密度（最简近似）
`n_M(T_c) = kappa * H(T_c)^3`

4. 初始丰度
`Y_M = n_M / s`

5. 今天通量（各向同性近似）
`F = n_0 * v / (4*pi)`, 其中 `n_0 = Y_M * s_0`

6. 暴胀稀释
`F(N) = F_0 * exp(-3N)`

## R05

本任务的“算法输出”是以下可验证量：
- `Y_M(T_c)`：无暴胀时初始丰度估计；
- `F_0`：无暴胀时今天预测通量；
- `N_min`：满足 `F(N)<=F_bound` 的最小 e-folds；
- 若干 `N` 采样点下的 `F(N)` 与“是否过界”判定表。

其中 `N_min` 同时由：
- 解析式 `N = ln(F_0/F_bound)/3`
- 数值根求解 `scipy.optimize.brentq`
双重给出并交叉验证。

## R06

`demo.py` 的计算流程：
1. 固定一组物理参数。
2. 计算 `H(T_c)`、`s(T_c)` 与 `Y_M`。
3. 计算无暴胀通量 `F_0`。
4. 建立函数 `F(N)-F_bound`。
5. 用 `brentq` 在区间 `[0, N_hi]` 求根得到 `N_min`。
6. 扫描 `N` 网格（如 `0,5,10,...,60`）输出通量表。
7. 做 `kappa=0.1,1,10` 灵敏度扫描，评估模型不确定性。

## R07

正确性依据：
- 物理上：单极子问题本质是“初始拓扑缺陷过多 + 常规演化稀释不足”，指数因子 `e^{-3N}` 是暴胀解决机制的核心。
- 数学上：`F(N)` 对 `N` 严格单调递减，因此 `F(N)=F_bound` 在 `F_0>F_bound` 时有唯一根，`brentq` 适用。
- 程序上：数值根与解析根同步输出，二者差值应接近机器精度量级。

## R08

复杂度（`m` 为扫描的 `N` 点数）：
- 基础物理量计算均为 `O(1)`。
- 根求解 `brentq` 迭代次数记为 `k`，成本 `O(k)`（通常几十步内）。
- 扫描表生成 `O(m)`。
- 空间开销主要为结果表，`O(m)`。

## R09

数值稳定性与工程处理：
- `brentq` 采用保守容差（`xtol=1e-12`, `rtol=1e-10`），并确保初末端异号。
- 当 `F_0 <= F_bound` 时直接返回 `N_min=0`，避免无意义求根。
- 对 `N` 较大时 `exp(-3N)` 极小，使用 `numpy` 双精度即可稳定覆盖本任务区间（`N<=60`）。

## R10

代码模块划分（见 `demo.py`）：
- `MonopoleParams`：参数集中管理，保证可复现实验。
- `hubble_rate_radiation` / `entropy_density_radiation`：基础宇宙学量。
- `initial_monopole_yield`：Kibble 初始丰度。
- `present_flux_from_yield`：丰度到今天通量映射。
- `flux_after_inflation`：暴胀稀释模型。
- `required_efolds_numeric` / `required_efolds_analytic`：数值与解析双解。
- `build_scan_table` / `run_demo`：组织实验与输出。

## R11

最小依赖栈：
- `numpy`：科学计算与指数/对数运算。
- `scipy.optimize`：`brentq` 一维有界根求解。
- `pandas`：输出扫描表（便于验证与后续导出）。

未引入大型框架，保持 MVP 简洁可审计。

## R12

运行方式（仓库根目录）：

```bash
uv run python "Algorithms/物理-宇宙学-0365-磁单极子问题_(Monopole_Problem)/demo.py"
```

或切换到该目录后运行：

```bash
uv run python demo.py
```

程序不需要交互输入，直接打印结果。

## R13

输出字段解释：
- `Initial yield Y_M(T_c)`：初始单极子丰度；
- `No-inflation flux F0`：若无暴胀，今天预测通量；
- `Required e-folds (numeric/analytic)`：最小暴胀 e-folds；
- `Consistency |numeric-analytic|`：根求解一致性校验；
- `Flux scan over selected e-fold values`：不同 `N` 下是否满足 Parker 上界；
- `Sensitivity to kappa`：Kibble 系数不确定性传播到 `N_min` 的影响。

## R14

自检建议：
1. 将 `kappa` 改为 `10`，应看到 `F_0` 增大且 `N_min` 增大。
2. 将 `parker_flux_bound` 放宽为 `1e-14`，应看到 `N_min` 降低。
3. 将 `v_over_c` 改小（如 `1e-4`），通量线性降低，`N_min` 相应下降。
4. 观察数值根与解析根差值，确认在高精度容差下非常小。

## R15

模型边界与局限：
- 仅为“教学型最小模型”，未显式模拟再加热细节、非平衡产生、后期捕获等过程。
- `n_M ~ H^3` 是数量级估算，不代表完整场论缺陷网络演化。
- Parker 界用于演示约束量级；不同观测约束可替换 `F_bound` 重新评估。

## R16

可扩展方向：
- 加入再加热温度和熵注入，改写 `Y_M` 的演化链路。
- 加入随速度分布积分的通量模型，而非单一 `v/c` 常数。
- 将观测约束扩展为多约束并行（磁场、天体物理、实验搜索上限）。
- 引入蒙特卡洛抽样传播 `T_c, g_*, kappa` 的不确定性。

## R17

与宇宙学框架的关系：
- 在非暴胀标准热大爆炸中，单极子过剩难以避免。
- 暴胀把任何预先存在的重粒子/拓扑缺陷数密度按 `e^{-3N}` 指数稀释，天然给出解法。
- 因此“磁单极子问题”与“平坦性问题、视界问题”一起，构成暴胀的三大经典动机之一。

## R18

本实现的源码级算法流（8 步）：
1. `run_demo` 构造 `MonopoleParams`，并设置 `N` 扫描网格。
2. `initial_monopole_yield` 调用 `hubble_rate_radiation` 与 `entropy_density_radiation`，按 `Y_M = kappa*H^3/s` 得到初值。
3. `present_flux_from_yield` 将 `Y_M` 乘以 `s0` 得到 `n0`，再按 `F=n0*v/(4*pi)` 得到 `F_0`。
4. `required_efolds_numeric` 构造残差函数 `r(N)=F_0*exp(-3N)-F_bound`，并设置求根区间 `[0, N_hi]`。
5. `scipy.optimize.brentq` 在区间端点异号前提下迭代：混合二分、割线与反二次插值，逐步收缩包围区间直到满足容差。
6. `required_efolds_analytic` 给出闭式 `ln(F_0/F_bound)/3`，用于对 `brentq` 结果做一致性校验。
7. `build_scan_table` 对每个采样 `N` 计算 `F(N)` 并打标签 `passes_parker_bound`，形成 `pandas.DataFrame`。
8. `run_demo` 打印关键物理量、根求解一致性、通量扫描和 `kappa` 灵敏度，构成可复现的最小验证闭环。

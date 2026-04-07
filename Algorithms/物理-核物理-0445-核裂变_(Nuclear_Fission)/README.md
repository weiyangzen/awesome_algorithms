# 核裂变 (Nuclear Fission)

- UID: `PHYS-0425`
- 学科: `物理`
- 分类: `核物理`
- 源序号: `445`
- 目标目录: `Algorithms/物理-核物理-0445-核裂变_(Nuclear_Fission)`

## R01

本条目实现“核裂变链式反应”的最小可运行算法闭环，目标是把核裂变里最关键的中子世代放大机制落成可执行代码：
1. 用显式概率模型模拟每一代中子触发裂变与泄漏。
2. 估计有效增殖因子 `k`，判断亚临界/临界/超临界状态。
3. 统计累计裂变数并换算能量释放量。

该 MVP 重点在算法透明性与可验证性，而不是高保真反应堆工程细节。

## R02

核裂变可抽象为“中子分支过程”：一代中子进入燃料后，部分诱发裂变，每次裂变释放多个新中子，新中子再进入下一代。
- 若 `k < 1`，中子数平均衰减（亚临界）。
- 若 `k ≈ 1`，中子数大致稳定（临界）。
- 若 `k > 1`，中子数平均增长（超临界）。

`demo.py` 用代际模型复现这个核心机制，并用多种估计方式交叉验证 `k`。

## R03

MVP 使用的核心关系：

1. 理论增殖因子

a = `k_theory = p_fission * nu_prompt * (1 - p_leak)`

其中 `p_fission` 是中子诱发裂变概率，`nu_prompt` 是每次裂变平均瞬发中子数，`p_leak` 是中子泄漏概率。

2. 代际统计模型

b = `N_{g+1} ~ Poisson(k * N_g)`

这里 `N_g` 为第 `g` 代中子数，适合做 Poisson 似然估计。

3. 能量估算

c = `E = N_fission_total * E_fission(MeV) * 1e6 * eV_to_J`

默认 `E_fission = 200 MeV`，用于把累计裂变事件换算为焦耳级能量。

## R04

`demo.py` 的输入输出设计（无交互）：
- 输入：脚本内部固定参数（随机种子、初始中子数、代数、`p_fission`、`p_leak`、`nu_prompt`）。
- 输出：
1. 每代 `n_in`、裂变次数、产生中子数、下一代中子数、代际比值。
2. `k` 的多种估计（理论值、矩估计、对数回归、PyTorch-Poisson 拟合）。
3. Pearson `chi2/ndf` 与 `p-value`、累计裂变数与总释能。

## R05

变量与单位约定：
- `n_in`, `n_out`：某代输入/输出中子数（计数，无量纲）
- `n_fissions`：该代发生裂变事件数（计数）
- `nu_prompt_mean`：单次裂变平均瞬发中子数（无量纲）
- `p_fission`, `p_leak`：概率量（`[0,1]`）
- `k_*`：有效增殖因子（无量纲）
- `energy_mev_per_fission`：单次裂变释能，单位 `MeV`
- `released_energy`：总释放能量，单位 `J`

## R06

数据策略：
- 不依赖外部核数据库，直接使用可复现随机模拟生成世代数据。
- 通过固定随机种子构建稳定“观测样本”，便于验证算法流程。
- 这样能把重点放在链式反应算法本身，而非数据抓取与清洗。

## R07

世代模拟算法（源码实现）分解：
1. 对每代给定 `n_in`，先采样裂变事件数 `n_fissions ~ Binomial(n_in, p_fission)`。
2. 每次裂变中子产额近似 Poisson，利用可加性合并为 `Poisson(n_fissions * nu_prompt)`。
3. 对产生的中子做泄漏筛选 `Binomial(n_prompt_produced, 1-p_leak)` 得到 `n_out`。
4. 更新到下一代并累计裂变事件数，循环 `n_generations` 次。

该过程即 Galton-Watson 分支过程在核裂变语境下的离散实现。

## R08

一致性检验采用 4 条路径：
1. 理论值：直接由参数计算 `k_theory`。
2. 矩估计：`k_moment = sum(N_{g+1}) / sum(N_g)`。
3. `scikit-learn` 对数线性回归：拟合 `log(N_g) = a + g*log(k)` 得 `k_log`。
4. `PyTorch` Poisson 似然拟合：最小化 `mean(lambda - y*log(lambda))` 得 `k_torch`。

若这 4 条路径结果接近，说明模拟与估计流程内部一致。

## R09

复杂度分析（代数为 `G`）：
- 当前实现每代仅做常数次随机采样与标量运算，时间复杂度 `O(G)`。
- 结果表存储 `G` 行，空间复杂度 `O(G)`。
- 若改为逐中子显式追踪，复杂度会接近 `O(sum_g N_g)`，代价显著更高。

## R10

数值稳定性处理：
- Poisson 均值 `lambda` 和对数项都做下界截断（`1e-12`），避免 `log(0)`。
- `k` 在 PyTorch 中通过 `softplus(theta)` 参数化，确保始终为正。
- 当世代中子数为 0 时模拟函数直接返回 0，避免无意义随机采样。
- 统计检验自由度使用 `max(1, n-1)`，防止极端小样本除零问题。

## R11

正确性检查（脚本内断言 + 诊断输出）：
1. 所有世代 `n_in`、`n_out` 非负。
2. 累计裂变数应大于 0（链反应确实发生）。
3. `k_torch` 与 `k_theory` 偏差应在合理阈值内（默认 `< 0.20`）。
4. 总释能应为正值。

脚本末尾打印 `All checks passed.` 表示最小验证通过。

## R12

依赖与职责分工：
- `numpy`：随机采样、向量运算
- `pandas`：代际结果表整理与打印
- `scipy.stats`：`chi2` 分布计算 Pearson 检验 `p-value`
- `scikit-learn`：对数线性回归估计 `k`
- `torch`：Poisson 似然的梯度优化拟合

## R13

运行方式：

```bash
uv run python Algorithms/物理-核物理-0445-核裂变_(Nuclear_Fission)/demo.py
```

或在当前目录直接执行：

```bash
uv run python demo.py
```

## R14

物理近似与边界：
- 单群（one-group）中子近似：不区分能群谱细节。
- 只建模“诱发裂变 + 泄漏”，未显式建模慢化、反射层、延迟中子动力学。
- 参数视为代际常数，不随温度、燃耗和几何反馈动态变化。

因此本实现用于算法演示与参数敏感性实验，不替代反应堆工程设计代码。

## R15

可扩展方向：
1. 引入延迟中子群和点堆动力学方程（更贴近控制问题）。
2. 将 `p_fission` 与 `p_leak` 设为状态相关参数，耦合温度反馈。
3. 多区域耦合（燃料区/慢化区/反射区）做空间离散近似。
4. 用贝叶斯方法联合反演 `k` 与不确定度区间。

## R16

工程复现性：
- 固定随机种子 `20260407`。
- 所有参数在源码显式给定，无外部文件依赖。
- 输出为稳定文本表格，便于自动化快照对比与 CI 验证。

## R17

为何这不是“黑盒调用”：
- 链式反应的分支机制（Binomial + Poisson + 泄漏筛选）在源码中逐步展开。
- `k` 的三类估计（矩估计、回归、Poisson 似然）均可追踪到明确公式。
- SciPy、sklearn、PyTorch 只承担数值/优化工具角色，不替代物理建模本身。

## R18

`demo.py` 的源码级流程可拆为 8 步：

1. `main()` 固定随机种子并声明 `ReactorParams`、初始中子数与模拟代数。  
2. 调用 `simulate_chain()` 进入世代循环，逐代执行 `simulate_one_generation()`。  
3. 在单代函数中先采样裂变次数，再采样裂变产额，最后做泄漏筛选得到下一代中子数。  
4. `simulate_chain()` 汇总每代 `n_in/n_out/ratio` 与累计裂变数，构成 `DataFrame`。  
5. `estimate_k_moment()` 用总量比值给出 `k` 的矩估计。  
6. `estimate_k_log_linear()` 用 `LinearRegression` 拟合 `log(N_g)` 对 `g` 的斜率并转成 `k`。  
7. `fit_k_torch_poisson()` 对 Poisson 负对数似然做梯度下降，得到 `k_torch`，再用 `poisson_pearson_test()` 计算 `chi2/ndf` 与 `p-value`。  
8. `main()` 计算总释能、打印代际表与全局指标，执行断言，最终输出 `All checks passed.`。  

# 巨正则系综 (μVT Ensemble)

- UID: `PHYS-0334`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `341`
- 目标目录: `Algorithms/物理-统计力学-0341-巨正则系综_(μVT_Ensemble)`

## R01

巨正则系综（`muVT`）描述与“热库 + 粒子库”交换能量与粒子的平衡系统：

- 温度 `T` 固定
- 体积 `V` 固定
- 化学势 `mu` 固定
- 粒子数 `N` 可以涨落

其核心对象是巨配分函数 `Xi`：

`Xi(mu, V, T) = sum_N exp(beta * mu * N) * Z_N(V, T)`，其中 `beta = 1 / (k_B T)`。

## R02

本条目做一个最小但可验证的 `muVT` MVP：

- 体系选择为经典理想气体（避免复杂势函数，保留系综本质）；
- 算法采用 Grand Canonical Monte Carlo（GCMC）的插入/删除 moves；
- 用解析解（`N` 的泊松分布）对采样结果做自动校验。

这样可以把“巨正则系综的数学定义 -> 可运行采样器 -> 统计验证”完整打通。

## R03

理想气体在 `muVT` 下有封闭形式结果。定义活度样量：

`z = exp(beta * mu) / lambda_th^d`

其中 `lambda_th` 是热德布罗意波长（本 MVP 取约化单位 `lambda_th = 1`）。
则粒子数分布为：

`P(N) = exp(-Vz) * (Vz)^N / N!`

即 `N ~ Poisson(Vz)`，所以：

- `<N> = Vz`
- `Var(N) = Vz`
- `Fano = Var(N)/<N> = 1`

## R04

GCMC 的两类提案（理想气体，`Delta U = 0`）：

1. 插入：`N -> N+1`
   - `A_ins = min(1, V*z/(N+1))`
2. 删除：`N -> N-1`
   - `A_del = min(1, N/(V*z))`

`demo.py` 采用对数形式实现：

- `logA_ins = log(Vz) - log(N+1)`
- `logA_del = log(N) - log(Vz)`

并用 `log(u) < min(0, logA)` 判定接受，减少指数下溢风险。

## R05

`demo.py` 的目标不是做大规模分子模拟，而是验证算法正确性：

- 对多个 `mu` 点独立采样；
- 对比经验统计与理论统计；
- 自动断言：
  - `<N>` 随 `mu` 单调上升；
  - `mean/variance` 相对误差足够小；
  - 经验分布与泊松分布的 `L1` 距离不过大。

## R06

输出字段说明：

- `mu`：化学势；
- `mean_n_emp`, `mean_n_theory`：经验/理论粒子数均值；
- `var_n_emp`, `var_n_theory`：经验/理论方差；
- `fano_emp`, `fano_theory`：经验/理论 Fano 因子；
- `mean_rel_error`, `var_rel_error`：关键矩相对误差；
- `distribution_l1`：经验 `P(N)` 与理论泊松分布的 `L1` 距离；
- `insert_acceptance`, `delete_acceptance`：两类 move 接受率。

## R07

复杂度分析（单个 `mu` 点）：

- 总步数 `S = burn_in + sample_steps * thin`；
- 每步只更新标量 `N`，时间复杂度 `O(S)`；
- 存储 `sample_steps` 个样本，空间复杂度 `O(sample_steps)`。

多个 `mu` 点线性扩展为 `O(K*S)`。

## R08

环境依赖：

- Python `>=3.10`
- `numpy`
- `pandas`

未使用黑箱物理引擎。系综权重、接受率、分布比较都在源码中显式实现。

## R09

默认参数（见 `MuVTConfig`）：

- `temperature = 1.0`
- `volume = 40.0`
- `chemical_potentials = (-2.0, -1.0, -0.2)`
- `burn_in = 5000`
- `sample_steps = 14000`
- `thin = 4`
- `seed = 20260407`

该参数组合保证运行快速，同时统计误差足够低，适合自动化校验。

## R10

正确性直觉：

1. `mu` 越大，`exp(beta*mu)` 越大，插入 move 更易接受；
2. 因而平衡时 `N` 的均值应随 `mu` 增大；
3. 理想气体下 `N` 的完整分布已知（泊松），可用来做“分布级”检验；
4. 若均值、方差、分布形状都贴近理论，可高置信说明采样器实现正确。

## R11

数值稳定性设计：

- 使用对数接受判据，避免直接 `exp` 带来的溢出/下溢；
- 固定随机种子，确保结果复现；
- 加入 `burn_in + thin`，降低初值偏差和自相关影响；
- 若采样点数不匹配会抛出异常，避免静默失败。

## R12

这个 MVP 适合：

- 教学展示 `muVT` 如何在代码层实现；
- 快速验证 GCMC 插入/删除公式是否写对；
- 为后续“有相互作用体系（LJ、格点气体）”做基线。

不适合：

- 直接用于真实流体相图计算；
- 高精度有限尺寸标度分析；
- 需要复杂势函数和并行优化的生产模拟。

## R13

保证类型：

- 近似比保证：N/A（不是组合优化问题）；
- 概率成功下界：无统一闭式保证（属于 MCMC）。

本实现给出的“可执行保障”是：

- 多个 `mu` 点上都通过统计断言；
- 关键矩误差和分布距离都在阈值内；
- 否则程序会 `assert` 失败，便于自动化发现问题。

## R14

常见失效模式：

1. 插入/删除接受率公式写反；
2. 忽略 `N=0` 的删除边界情况；
3. 采样步数过少导致方差估计不稳；
4. 只验证 `<N>`，不验证 `Var(N)` 和分布形状，容易误判；
5. 将理想气体结果误外推到强相互作用系统。

## R15

可扩展方向：

1. 从理想气体扩展到含势能的连续体系（如 Lennard-Jones）；
2. 增加位移 move（不仅做插入/删除）以提高构型混合；
3. 增加误差条估计（block averaging / bootstrap）；
4. 输出 `N` 时间序列与直方图 CSV 用于可视化；
5. 比较不同 `V` 下的有限尺寸效应。

## R16

相关知识点：

- 化学势与粒子交换平衡；
- 巨配分函数与巨势 `Omega = -k_B T ln Xi`；
- 粒子数涨落与可压缩性关系；
- `NVE/NVT/NPT/muVT` 不同系综的控制变量差异。

## R17

运行方式（无交互输入）：

```bash
cd Algorithms/物理-统计力学-0341-巨正则系综_(μVT_Ensemble)
uv run python demo.py
```

完成条件核对：

- `README.md` 的 `R01-R18` 已全部填充；
- `demo.py` 为可执行 MVP；
- `meta.json` 与该任务元数据一致；
- 目录可独立验证。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `MuVTConfig` 定义 `T, V, mu` 网格、采样步数与随机种子。  
2. `activity` 计算 `z = exp(beta*mu)/lambda_th^d`，`theoretical_stats` 给出 `Poisson(Vz)` 的解析矩。  
3. `run_gcmc_ideal` 在每个 `mu` 下初始化 `N`（泊松初值）并进入主循环。  
4. 每步以 `1/2` 概率选择插入或删除提案，分别构造 `logA_ins` / `logA_del`。  
5. 用 `log(u) < min(0, logA)` 执行 Metropolis 接受-拒绝并更新 `N`。  
6. 经过 `burn_in` 后按 `thin` 间隔记录样本，得到 `N` 的时间序列。  
7. 由样本计算经验 `<N>`、`Var(N)`、`Fano`，并与理论值比较；同时计算经验分布与泊松分布的 `L1` 距离。  
8. `main` 汇总成 `pandas.DataFrame` 打印，并执行断言（单调性、矩误差、分布误差、接受率健康度）。

第三方库说明：`numpy` 仅用于随机采样和统计，`pandas` 仅用于展示表格；`muVT` 算法核心（权重、接受率、采样与验证）均由源码逐步展开，没有调用黑箱求解器。

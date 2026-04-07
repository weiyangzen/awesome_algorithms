# 详细平衡 (Detailed Balance)

- UID: `PHYS-0312`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `315`
- 目标目录: `Algorithms/物理-统计力学-0315-详细平衡_(Detailed_Balance)`

## R01

详细平衡（Detailed Balance）是马尔可夫过程在平衡态下的“成对通量守恒”条件。对于离散状态空间，若平稳分布为 `pi`，转移矩阵为 `P`，则对任意状态对 `(i, j)` 有：

`pi_i * P_ij = pi_j * P_ji`

它比全局平稳条件 `pi P = pi` 更强：不仅总流入等于总流出，而且每对互逆跃迁都逐对抵消，因此不会存在净概率环流。

## R02

本条目的 MVP 目标是把详细平衡从公式落到可运行数值实验：

1. 构造一个有限状态统计力学系统（4 自旋一维 Ising）。
2. 用“单自旋翻转 + Metropolis 接受率”显式构建转移核。
3. 解析验证详细平衡与平稳性残差。
4. 通过 Monte Carlo 采样得到经验分布和经验转移核，再验证经验详细平衡近似成立。

## R03

模型设定：

- 状态：`s in {-1, +1}^n`，本实现 `n=4`，总状态数 `2^n = 16`。
- 能量函数（周期边界）：

`E(s) = -J * sum_i s_i s_{i+1} - h * sum_i s_i`

- 目标分布（玻尔兹曼分布）：

`pi(s) = exp(-beta * E(s)) / Z`

`Z = sum_{s'} exp(-beta * E(s'))`

其中 `beta` 为逆温，`J` 为耦合强度，`h` 为外场。

## R04

转移机制采用标准 Metropolis 单自旋翻转：

1. 从状态 `i` 均匀随机选一个自旋位置 `k`（提议概率 `1/n`）。
2. 翻转该自旋得到候选状态 `j`（只相差一位比特）。
3. 接受率：

`a(i->j) = min(1, exp(-beta * (E_j - E_i)))`

4. 非对角元：`P_ij = (1/n) * a(i->j)`；
5. 对角元：`P_ii = 1 - sum_{j!=i} P_ij`。

由于单自旋翻转提议是对称的（`q(i->j)=q(j->i)`），该构造满足详细平衡。

## R05

本实现的验证链路包含“理论 + 经验”两层：

- 理论层：
  - 构造 `pi` 与 `P`；
  - 计算 `|pi_i P_ij - pi_j P_ji|` 的最大残差；
  - 检查 `|piP - pi|` 的平稳性残差。
- 经验层：
  - 运行长链采样，统计转移计数矩阵 `C` 与访问频率；
  - 行归一化 `C` 得 `P_hat`，访问频率归一化得 `pi_hat`；
  - 再计算 `|pi_hat_i P_hat_ij - pi_hat_j P_hat_ji|` 与 `||pi_hat-pi||_1`。

## R06

为什么这个 MVP 合理：

1. 状态空间小（16 个状态）可以精确构造全矩阵，不必依赖黑盒求解器。
2. 单自旋翻转是统计力学中最常见局域动力学之一，物理语义清晰。
3. 同时输出解析残差和采样残差，便于区分“公式错误”和“统计噪声”。
4. 代码短小且完全自包含，适合作为详细平衡的最小可复现实验。

## R07

复杂度分析（`n` 为自旋数，`S=2^n` 为状态数，`T` 为采样步数）：

- 状态枚举与能量计算：`O(S * n)`。
- 转移矩阵构造：每个状态尝试 `n` 个翻转，`O(S * n)`。
- 理论残差矩阵计算：`O(S^2)`。
- 采样：每步常数操作，`O(T)`。
- 经验核构造与残差：`O(S^2)`。

本例 `n=4`，`S=16`，主耗时基本来自 `T=250000` 的采样循环。

## R08

数值稳定性与鲁棒性措施：

- 计算玻尔兹曼权重前对能量做平移 `E - min(E)`，减少指数下溢风险。
- 对 `n_spins`、`beta`、`n_steps` 与 `burn_in` 做参数合法性检查。
- 通过 `row_normalize_counts` 在零行情况下安全归一化。
- 结果校验使用多个指标（理论/经验 DB 残差、平稳残差、`L1` 偏差、接受率范围）。

## R09

`demo.py` 关键函数职责：

- `enumerate_ising_states`：枚举全部自旋态。
- `ising_energies`：计算每个状态能量。
- `boltzmann_distribution`：计算理论平衡分布 `pi`。
- `build_metropolis_kernel`：显式构造 `P`、候选状态映射和接受率表。
- `simulate_chain`：执行 Metropolis 采样并记录转移计数。
- `detailed_balance_residual`：计算详细平衡残差矩阵。
- `build_edge_flux_table`：输出单翻转边上的通量与比率检查。
- `run_detailed_balance_demo` / `main`：组织实验、打印表格、执行断言。

## R10

运行方式（无交互输入）：

```bash
cd Algorithms/物理-统计力学-0315-详细平衡_(Detailed_Balance)
uv run python demo.py
```

脚本会自动输出：

- Summary 指标表；
- 理论/经验状态概率对照表；
- 单自旋翻转边上的成对通量检查表；
- 最终断言结果 `All checks passed.`。

## R11

输出指标解释：

- `kernel_row_sum_max_error`：`P` 每行和偏离 1 的最大误差。
- `max_db_residual_theory`：理论 `max |pi_iP_ij - pi_jP_ji|`。
- `max_db_residual_empirical`：经验版本残差最大值。
- `stationarity_max_abs_theory`：`max |(piP - pi)_i|`。
- `L1(empirical_pi, theory_pi)`：经验分布与理论分布的 `L1` 距离。
- `chi2_statistic / chi2_pvalue`：经验访问频数相对理论分布的卡方检验统计量。
- `max_rel_err(P_ratio, exp(-beta*deltaE))`：转移率比值与局域详细平衡公式的最大相对误差。

## R12

核心参数与调参建议：

- `beta`：越大分布越集中在低能区，接受率通常下降。
- `coupling_j`：控制相邻自旋一致性偏好。
- `field_h`：打破自旋反演对称，偏向某一磁化方向。
- `n_steps`：越大经验误差越小。
- `burn_in`：越大越能削弱初始态影响。

建议先固定 `n_spins=4`，逐步提高 `n_steps` 观察经验残差收敛。

## R13

正确性保证说明：

- 近似比保证：N/A（该任务不是优化近似算法）。
- 闭式概率保证：N/A（本任务是数值验证）。

本实现提供的工程保证：

1. 理论层面严格检查 `P` 行和、详细平衡、平稳性；
2. 经验层面检查详细平衡残差与分布偏差；
3. 若关键指标超阈值，程序会触发断言失败。

## R14

常见错误与排查：

1. 将 `DeltaE` 符号写反，导致高能态被错误偏好。
2. 忘记把拒绝提议计入 `i->i`，使 `P` 非随机矩阵。
3. 提议分布不对称但仍套用对称 Metropolis 接受率。
4. 采样步数太短，经验残差过大但理论残差正常。
5. 只看 `piP=pi`，不看成对通量，可能遗漏非平衡环流。

## R15

适用与不适用场景：

- 适用：统计力学教学、MCMC 入门验证、局域动力学正确性自检。
- 不适用：高维连续变量后验采样、复杂自适应 MCMC 性能评测、工业级大规模模拟。

## R16

可扩展方向：

1. 扩展到 2D Ising 并比较不同温度区间行为；
2. 对比 Metropolis 与 Glauber/heat-bath 更新规则；
3. 加入非平衡驱动，展示“平稳但不满足详细平衡”的环流；
4. 结合自相关时间估计，评估采样效率而不仅是正确性；
5. 增加 bootstrap 误差条，量化经验残差不确定性。

## R17

交付核对：

1. `README.md` 保留并完整填写 `## R01` 到 `## R18`。
2. `demo.py` 为可运行 MVP，且无交互输入。
3. `meta.json` 保持 UID/学科/分类/源序号与任务一致。
4. 所有修改仅在 `Algorithms/物理-统计力学-0315-详细平衡_(Detailed_Balance)` 目录内。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 创建 `DetailedBalanceConfig`，调用 `run_detailed_balance_demo`。  
2. `run_detailed_balance_demo` 调用 `enumerate_ising_states` 与 `ising_energies` 枚举全部 16 个状态并计算能量。  
3. 调用 `boltzmann_distribution` 计算理论平衡分布 `pi`。  
4. 调用 `build_metropolis_kernel`：对每个状态和每个翻转位点显式计算候选态、`DeltaE`、接受率，构造 `P`。  
5. 调用 `detailed_balance_residual(pi, P)` 与 `pi@P-pi` 评估理论详细平衡/平稳性残差。  
6. 调用 `simulate_chain` 进行长链采样，统计 `transition_counts` 与 `visit_counts`；再通过 `row_normalize_counts` 得到经验核 `P_hat` 和 `pi_hat`。  
7. 再次调用 `detailed_balance_residual(pi_hat, P_hat)`，并在 `build_edge_flux_table` 中逐边比较 `P_ij/P_ji` 与 `exp(-beta*DeltaE)`。  
8. 汇总 `summary_df`、`state_df`、`edge_df` 打印并执行断言，全部通过后输出 `All checks passed.`。

第三方库边界说明：`numpy`/`pandas`/`scipy` 只提供数值、表格与统计函数；详细平衡相关核心逻辑（状态枚举、转移核构造、采样、残差计算）均在源码中手工实现，不依赖黑盒 MCMC 框架。

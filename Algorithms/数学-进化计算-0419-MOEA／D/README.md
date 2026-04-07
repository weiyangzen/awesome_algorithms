# MOEA/D

- UID: `MATH-0419`
- 学科: `数学`
- 分类: `进化计算`
- 源序号: `419`
- 目标目录: `Algorithms/数学-进化计算-0419-MOEA／D`

## R01

`MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)` 的核心思想是：
把一个多目标优化问题分解为一组标量子问题（每个子问题对应一个权重向量），再让这些子问题通过邻域协同进化。

本目录 MVP 采用二目标 `ZDT1` 基准函数，使用：
- Tchebycheff 分解函数；
- 权重向量邻域更新；
- SBX 交叉 + 多项式变异。

目标是提供一个可运行、可审计、非黑盒的最小实现。

## R02

本实现要解决的问题：
- 在决策空间 `x in [0,1]^n` 上同时最小化 `f1(x), f2(x)`；
- 不把多个目标直接加权成单一固定目标，而是通过多组权重并行覆盖 Pareto 前沿。

输入（`demo.py` 内置）：
- 种群规模、代数、邻域大小、交叉/变异参数、随机种子。

输出：
- 进化后的种群目标值；
- 非支配解集合；
- 与 ZDT1 理论前沿 `f2 = 1 - sqrt(f1)` 的平均偏差（粗略质量指标）。

## R03

关键数学定义：

1. 多目标问题：
   - `min F(x) = (f1(x), f2(x), ..., fm(x))`。
2. 分解思想（Tchebycheff 标量化）：
   - `g_te(x | lambda, z*) = max_j lambda_j * |f_j(x) - z*_j|`。
   - 其中 `lambda` 是子问题权重向量，`z*` 是理想点（各目标当前最小值）。
3. 邻域协同更新：
   - 对子问题 `i` 产生候选解 `y` 后，遍历其邻域 `B(i)`；
   - 若 `g_te(y | lambda_j, z*) <= g_te(x_j | lambda_j, z*)`，则用 `y` 替换邻域内第 `j` 个解。

该机制让每个子问题只与局部邻居交换信息，兼顾分布性和效率。

## R04

算法流程（高层）：
1. 生成均匀权重向量集合 `lambda_1...lambda_N`。  
2. 基于权重距离构建每个子问题的邻域 `B(i)`。  
3. 随机初始化种群并计算目标值，初始化理想点 `z*`。  
4. 对每一代、每个子问题 `i`：
   - 在邻域（或全局）选父代；
   - 通过交叉变异生成子代；
   - 评估子代并更新 `z*`；
   - 用 Tchebycheff 标量值比较，更新邻域解。  
5. 迭代结束后提取非支配解，作为近似 Pareto 集。

## R05

核心数据结构：
- `MOEADConfig`：算法配置（种群规模、代数、邻域、变异参数等）。
- `weights: np.ndarray (N, M)`：每个子问题对应的权重向量。
- `neighborhoods: np.ndarray (N, T)`：每个子问题的邻域索引。
- `pop: np.ndarray (N, D)`：当前决策变量种群。
- `objs: np.ndarray (N, M)`：当前每个个体的多目标值。
- `ideal: np.ndarray (M,)`：当前理想点。
- `history: list[dict]`：每代简要日志（代数、理想点、替换次数）。

## R06

正确性要点：
- 权重分解正确性：每个子问题都对应一条固定偏好的优化方向。  
- 理想点单调更新：`ideal = min(ideal, child_obj)` 保持各目标的历史最优下界。  
- 邻域替换规则：用同一标量化准则比较新旧解，保证子问题目标一致。  
- 非支配集合检查：最终用支配关系筛选输出，避免只看单一子问题最优。

## R07

复杂度分析（`N` 种群规模，`T` 邻域规模，`G` 代数，`D` 决策维度，`M` 目标维度）：
- 预处理：
  - 邻域构建需权重两两距离，约 `O(N^2 * M)`。
- 每代：
  - 变异/交叉与目标评估约 `O(N * D)`；
  - 邻域替换比较约 `O(N * T * M)`。
- 总体：
  - `O(N^2*M + G*(N*D + N*T*M))`。

在常见参数下，主要耗时集中在每代的邻域替换与目标评估。

## R08

边界与异常处理：
- `n_obj != 2`：本 MVP 直接报错（只做双目标版本）。
- `pop_size < 2`：无法构建有效权重网格，报错。
- `neighborhood_size <= 0`：报错。
- 邻域大小超过种群时自动截断到 `N`。
- Tchebycheff 权重中零元素做 `1e-6` 保护，避免数值退化。

## R09

MVP 取舍：
- 只做 `ZDT1` + 双目标，避免扩展过多造成代码噪声。  
- 仅使用 `numpy`，不依赖现成多目标优化库。  
- 不实现外部档案（external archive）与高级约束处理。  
- 用文本输出替代绘图，保证在纯终端环境即可验证。

## R10

`demo.py` 主要函数职责：
- `zdt1_objectives`：计算单个解的二目标值。
- `generate_weight_vectors`：生成双目标权重向量网格。
- `build_neighborhood`：按权重欧氏距离构建邻域索引。
- `tchebycheff_value`：计算标量化值。
- `choose_parents`：按 `delta` 在邻域/全局进行父代选择。
- `sbx_crossover`、`polynomial_mutation`：产生新候选解。
- `run_moead`：主迭代流程（评估、更新理想点、邻域替换）。
- `non_dominated_indices`：提取非支配解索引。
- `summarize_results`：打印结果摘要。

## R11

运行方式：

```bash
cd Algorithms/数学-进化计算-0419-MOEA／D
uv run python demo.py
```

脚本无命令行参数、无交互输入。

## R12

输出字段说明：
- `Population size`：最终种群大小。  
- `Non-dominated set size`：最终非支配解数量。  
- `Mean |f2 - (1-sqrt(f1))| on ND set`：非支配解相对 ZDT1 理论前沿的平均偏差（越小通常越好）。  
- `First 8 non-dominated points`：非支配解样例。  
- `Last generation diagnostics`：最后一代的
  - `generation`：代数；
  - `ideal`：理想点估计；
  - `replacements`：该代邻域替换次数。

## R13

建议最小测试集：
- 默认配置（已内置）：验证流程可运行与结果可解释。  
- 小代数测试（如 `generations=5`）：做烟雾测试。  
- 小邻域测试（如 `neighborhood_size=3`）：观察局部协作更强时的收敛变化。  
- 低 `delta` 测试（如 `delta=0.2`）：观察更多全局交配带来的探索变化。

## R14

可调参数与建议：
- `pop_size`：增大可提升前沿覆盖，但更慢。  
- `generations`：越大越接近稳定前沿。  
- `neighborhood_size`：小值更局部，大值更全局。  
- `delta`：邻域交配概率，高值强调局部协作。  
- `eta_c/eta_m`：SBX 与变异分布指数；值越大扰动越温和。  
- `mutation_prob`：默认 `1/n_var`，是常见经验值。

## R15

与相近方法对比：
- 对比 NSGA-II：
  - NSGA-II 依赖非支配排序和拥挤距离；
  - MOEA/D 通过“分解 + 邻域替换”推进，通常在目标维度升高时更易结构化扩展。
- 对比加权和法：
  - 固定加权和通常需要多次独立求解；
  - MOEA/D 在一次运行中协同优化整组权重子问题。
- 对比单目标进化算法：
  - 单目标只输出一个折中点；
  - MOEA/D 能输出整条近似前沿。

## R16

典型应用场景：
- 需要同时优化成本/性能/风险等冲突指标的工程设计。  
- 模型压缩中的多指标权衡（精度、时延、参数量）。  
- 调度与资源分配中的多目标决策。  
- 任何可做黑盒评估、但目标不止一个的问题。

## R17

可扩展方向：
- 从双目标扩展到多目标（`M>2`）并引入系统化权重生成。  
- 支持外部非支配档案，增强解集多样性。  
- 增加约束处理（可行性优先或罚函数）。  
- 用真实仿真/训练任务替换 ZDT1，形成端到端实验。  
- 增加 HV/IGD 等质量指标做多种子统计评估。

## R18

`demo.py` 源码级算法流（8 步）：

1. `main()` 构造 `MOEADConfig`，固定 `pop_size/generations/neighborhood_size` 等参数。  
2. `run_moead()` 调用 `generate_weight_vectors()` 生成每个子问题的权重，再调用 `build_neighborhood()` 预计算邻域索引。  
3. 随机初始化种群 `pop`，用 `evaluate_population()` 计算目标矩阵 `objs`，并由 `np.min` 得到初始理想点 `ideal`。  
4. 每一代中，对随机顺序的子问题 `i`，`choose_parents()` 按概率 `delta` 在邻域或全局选择父代索引。  
5. 通过 `sbx_crossover()` 和 `polynomial_mutation()` 生成子代 `child`，再由 `zdt1_objectives()` 评估其目标值 `child_obj`。  
6. 用 `ideal = np.minimum(ideal, child_obj)` 更新理想点。  
7. 遍历 `i` 的邻域 `B(i)`，对每个邻居 `j` 计算 `tchebycheff_value`：若 `child` 在 `lambda_j` 对应子问题上的标量值不劣于当前解，则替换 `pop[j], objs[j]`。  
8. 迭代结束后，`non_dominated_indices()` 提取非支配解，`summarize_results()` 输出前沿样例和偏差指标。

该实现未调用任何多目标优化黑箱库，分解、邻域更新、交叉变异与非支配筛选都可在源码中逐行追踪。

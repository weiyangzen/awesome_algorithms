# 遗传算法

- UID: `MATH-0354`
- 学科: `数学`
- 分类: `进化计算`
- 源序号: `354`
- 目标目录: `Algorithms/数学-进化计算-0354-遗传算法`

## R01

遗传算法（Genetic Algorithm, GA）是一类模拟生物进化机制的全局优化方法。其核心思想是维护一个候选解种群，通过“选择-交叉-变异-保留”迭代搜索，在无梯度、非凸、多峰问题上常作为稳健基线。

本目录提供一个最小可运行、可审计的实数编码 GA MVP：
- 纯 `numpy` 手写实现，不调用黑盒优化器；
- 采用锦标赛选择、算术交叉、高斯变异、精英保留；
- 使用固定随机种子，结果可复现。

## R02

本条目问题定义如下：
- 输入：
  - 目标函数 `f(x)`（默认最小化）；
  - 连续变量边界 `bounds`（每维 `[low, high]`）；
  - GA 超参数（`pop_size`、`max_gens`、`crossover_rate`、`mutation_rate`、`elite_count` 等）。
- 输出：
  - 近似最优解 `x_best`；
  - 最优目标值 `f_best`；
  - 每代统计轨迹（`best_f/mean_f/diversity/mutated_genes`）；
  - 终止信息（代数、收敛标志、终止原因）。

`demo.py` 内置两个无交互案例：
1. `Sphere (8D)`：单峰函数，验证稳定收敛；
2. `Rastrigin (8D)`：多峰函数，验证全局搜索能力。

## R03

MVP 采用实数编码遗传算法，关键机制如下：

1. 选择（Tournament Selection）
- 随机抽取 `k` 个个体，选择其中目标值最小者作为父代；
- 重复两次得到一对父代。

2. 交叉（Arithmetic Crossover）
- 概率 `pc` 触发交叉；
- 对每维采样 `alpha_j in [0, 1]`：
  - `child1_j = alpha_j * p1_j + (1 - alpha_j) * p2_j`
  - `child2_j = alpha_j * p2_j + (1 - alpha_j) * p1_j`

3. 变异（Gaussian Mutation）
- 每个基因以概率 `pm` 发生变异：
  - `x_j <- x_j + N(0, (sigma_ratio * range_j)^2)`
- 变异后执行边界裁剪 `clip(low_j, high_j)`。

4. 精英保留（Elitism）
- 每代保留前 `elite_count` 个最优个体直接进入下一代。

## R04

算法主流程：
1. 校验参数和边界合法性，初始化随机种子。  
2. 在边界内均匀采样初始种群并计算适应度（目标值）。  
3. 每代先按目标值排序并提取精英个体。  
4. 反复执行“锦标赛选择 -> 算术交叉 -> 高斯变异”生成后代。  
5. 用“精英 + 后代”组成新种群并重新评估。  
6. 记录每代统计量（`best_f/mean_f/diversity/mutated_genes`）。  
7. 判断终止条件（最大代数或 `std(fitness) < tol`）。  
8. 输出最优解、轨迹和终止状态。

## R05

主要数据结构：
- `population: np.ndarray`，形状 `(pop_size, dim)`；
- `fitness: np.ndarray`，形状 `(pop_size,)`，保存每个个体目标值；
- `HistoryItem = (gen, best_f, mean_f, diversity, mutated_genes)`；
- `result: dict`：
  - `x_best/f_best/generations/converged/message`；
  - `history/final_fitness_std`。

该结构兼顾简洁与可追踪，便于后续可视化或批量实验。

## R06

正确性与稳定性要点：
- 最小化语义一致：锦标赛与精英排序均以“更小目标值更优”为准。  
- 交叉概率为 0 时退化为复制父代，概率为 1 时始终执行混合。  
- 变异尺度按维度区间缩放，避免不同量纲下步长失衡。  
- 每次变异后进行边界裁剪，防止无效解传播。  
- 每代强制重算 `fitness` 并检查有限值，防止 `nan/inf` 污染。  
- 精英保留确保最好个体不被随机扰动破坏。

## R07

复杂度分析（维度 `d`，种群规模 `N`，代数 `G`）：
- 每代主要成本：
  - 选择、交叉、变异约 `O(N * d)`；
  - 目标函数评估约 `O(N * C_f(d))`；
  - 排序精英约 `O(N log N)`（通常次要）。
- 总时间复杂度近似：`O(G * (N * d + N * C_f(d) + N log N))`。  
- 空间复杂度：`O(N * d)`。

对黑盒目标函数，实际瓶颈通常在 `C_f(d)`。

## R08

边界与异常处理规则：
- `bounds` 必须是 `(dim, 2)` 且每维满足 `high > low`；
- `pop_size >= 4`，`max_gens > 0`；
- `crossover_rate`、`mutation_rate` 必须在 `[0, 1]`；
- `mutation_scale > 0`；
- `elite_count` 满足 `0 <= elite_count < pop_size`；
- `tournament_size` 满足 `[2, pop_size]`；
- `tol > 0`；
- 初始或演化中出现非有限目标值时抛出异常。

## R09

MVP 取舍说明：
- 只依赖 `numpy`，符合“最小工具栈”原则；
- 不调用 `scipy`/`sklearn`/`torch` 的现成 GA 黑盒接口；
- 使用实数编码而非位串编码，直接面向连续优化；
- 未引入复杂自适应参数策略，优先保证逻辑透明、可读、可复现。

## R10

`demo.py` 函数职责：
- `ensure_bounds`：检查并规范边界输入；
- `sphere` / `rastrigin`：基准函数；
- `init_population`：边界内初始化种群；
- `tournament_select_index`：锦标赛选择；
- `arithmetic_crossover`：实数算术交叉；
- `gaussian_mutation`：高斯变异并边界裁剪；
- `genetic_algorithm`：GA 主循环；
- `print_history`：打印分代统计；
- `run_case`：执行单案例并计算误差；
- `main`：组织多案例并汇总结果。

## R11

运行方式：

```bash
cd Algorithms/数学-进化计算-0354-遗传算法
uv run python demo.py
```

脚本不接收交互输入，直接输出每个案例与汇总统计。

## R12

输出字段说明：
- `gen`：代编号；
- `best_f`：该代最优目标值；
- `mean_f`：该代平均目标值；
- `diversity`：个体到种群中心的平均二范数；
- `mutated_genes`：该代发生变异的基因总数；
- `f_best`：最终最优目标值；
- `abs_x_error / rel_x_error`：最优解与参考解的绝对/相对误差；
- `optimality_gap`：`f_best - known_optimum`；
- `Summary`：跨案例聚合指标。

## R13

最小测试集（内置）：
1. `Sphere (8D)`
- 边界：`[-5, 5]^8`；
- 已知最优：`x*=0, f*=0`；
- 用于验证稳定收敛和数值健壮性。

2. `Rastrigin (8D)`
- 边界：`[-5.12, 5.12]^8`；
- 已知最优：`x*=0, f*=0`；
- 用于验证多峰情况下的全局搜索能力。

## R14

关键参数与经验建议：
- `pop_size`：种群规模，增大可提升全局搜索但增加开销；
- `crossover_rate (pc)`：交叉概率，通常 `0.8~0.95`；
- `mutation_rate (pm)`：基因变异概率，过低易早熟，过高易随机游走；
- `mutation_scale`：变异步长比例，决定局部微调与跳跃强度；
- `elite_count`：精英保留数量，过大可能降低多样性；
- `tournament_size`：选择压力，越大越偏向当前优者。

调参顺序建议：先固定 `pc/pm`，再调整 `pop_size` 与 `mutation_scale`，最后微调 `elite_count` 和 `tournament_size`。

## R15

与相关方法对比：
- 对比随机搜索：GA 通过选择和重组利用历史信息，通常效率更高。  
- 对比爬山法：GA 维护群体并行搜索，更不容易陷入局部最优。  
- 对比差分进化：DE 依赖差分向量更新；GA 更强调“选择压力 + 重组”。  
- 对比梯度法：GA 不要求可导，但函数评估次数通常更多。

## R16

典型应用场景：
- 黑盒连续参数优化；
- 不可导或多峰目标的近似优化；
- 工程参数调优（控制、结构、工艺）；
- 作为混合优化前端，先粗搜再接局部方法精修。

## R17

可扩展方向：
- 自适应参数（动态 `pm` / `mutation_scale`）；
- 更丰富交叉算子（BLX-`alpha`、SBX）；
- 多目标扩展（NSGA-II 思路）；
- 并行评估与批量实验；
- 引入约束处理（罚函数、可行性优先）。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构造 `Sphere` 与 `Rastrigin` 两个固定案例，写死参数与随机种子。  
2. `run_case` 调用 `genetic_algorithm`，并统一计算误差与 gap 指标。  
3. `genetic_algorithm` 中先 `ensure_bounds` 校验边界，再 `init_population` 初始化种群并计算 `fitness`。  
4. 每代开始按 `fitness` 排序，复制前 `elite_count` 个体到 `elites`。  
5. 通过 `tournament_select_index` 选父代，`arithmetic_crossover` 生成两个子代（或直接复制）。  
6. 对子代调用 `gaussian_mutation`，按 `mutation_rate` 决定逐基因扰动并 `clip` 到边界。  
7. 组装 `next_population = elites + offspring`，重新评估 `fitness`，记录 `best_f/mean_f/diversity/mutated_genes` 到 `history`。  
8. 若 `std(fitness) < tol` 则提前收敛，否则继续；最终返回 `x_best/f_best/history`，并在 `main` 打印汇总统计。

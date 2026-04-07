# 差分进化

- UID: `MATH-0357`
- 学科: `数学`
- 分类: `进化计算`
- 源序号: `357`
- 目标目录: `Algorithms/数学-进化计算-0357-差分进化`

## R01

差分进化（Differential Evolution, DE）是一种面向连续变量全局优化的群体智能算法。它通过“向量差分”构造变异方向，再配合交叉与贪婪选择，在保持实现简洁的同时具备较强的全局搜索能力。

本目录给出一个可运行、可审计、非黑盒的最小 MVP：
- 手写 `DE/rand/1/bin` 变异-交叉-选择流程；
- 使用固定随机种子保证可复现；
- 通过两个标准基准函数展示算法行为。

## R02

本条目解决的问题定义：
- 输入：
  - 连续目标函数 `f(x)`；
  - 变量边界 `bounds`（每维 `[low, high]`）；
  - DE 超参数（`pop_size`、`F`、`CR`、`max_gens`、`tol` 等）。
- 输出：
  - 近似最优解 `x_best`；
  - 最优目标值 `f_best`；
  - 每代演化轨迹（最优值、群体均值、多样性、改进个体数）；
  - 终止状态（收敛与否、代数、终止原因）。

`demo.py` 内置两个固定案例，无需交互输入：
1. `Rastrigin (5D)`：多峰函数，考察全局搜索能力；
2. `Rosenbrock (5D)`：狭长谷底，考察精细收敛能力。

## R03

`DE/rand/1/bin` 的核心数学关系：

1. 变异（Mutation）
- 对目标个体 `x_i^g`，随机选取互不相同的 `r1,r2,r3`：
  - `v_i^g = x_{r1}^g + F * (x_{r2}^g - x_{r3}^g)`

2. 交叉（Binomial Crossover）
- 对每个维度 `j`：
  - 若 `rand_j < CR` 或 `j == j_rand`，取 `u_{i,j}^g = v_{i,j}^g`；
  - 否则取 `u_{i,j}^g = x_{i,j}^g`。
- `j_rand` 保证每个 trial 至少继承一个变异分量。

3. 选择（Greedy Selection）
- 若 `f(u_i^g) <= f(x_i^g)`，则 `x_i^{g+1} = u_i^g`；
- 否则 `x_i^{g+1} = x_i^g`。

4. 边界处理
- 本 MVP 使用逐维裁剪：`x <- clip(x, low, high)`。

## R04

算法主流程：
1. 校验 `bounds/参数` 合法性，初始化随机种子。  
2. 在边界内均匀采样初始种群并计算初始适应度。  
3. 进入代循环：对每个个体执行变异、交叉，生成 trial。  
4. 对 trial 计算目标值并做贪婪选择。  
5. 每代统计 `best/mean/diversity/improved_count`。  
6. 判断停止条件（达到最大代数或群体方差小于 `tol`）。  
7. 返回最优解与历史轨迹。

## R05

MVP 的主要数据结构：
- `population: np.ndarray`，形状 `(pop_size, dim)`，保存当前种群；
- `fitness: np.ndarray`，形状 `(pop_size,)`，保存当前目标值；
- `HistoryItem = (gen, best_f, mean_f, diversity, improved)`；
- `result: dict`：
  - `x_best/f_best/generations/converged/message`；
  - `history/final_fitness_std`。

这种结构让每一步都可追踪，便于后续做参数对比和可视化。

## R06

正确性与稳定性要点：
- 变异索引必须满足“与目标个体不同且互不重复”，否则差分信息退化。  
- `j_rand` 机制是 binomial 交叉的关键，避免 trial 与 target 完全相同。  
- 贪婪选择保证种群最优值不升高（最小化问题下单调不增）。  
- 边界裁剪避免非法解进入目标函数，提升数值稳定性。  
- 非有限目标值（`nan/inf`）按“拒绝该 trial”处理，防止污染种群。

## R07

复杂度分析（设维度 `d`、种群规模 `NP`、代数 `G`）：
- 每个个体一次变异+交叉约 `O(d)`；
- 每个 trial 评估代价记为 `C_f(d)`；
- 总时间复杂度约：`O(G * NP * (d + C_f(d)))`；
- 空间复杂度约：`O(NP * d)`（种群和少量统计量）。

DE 的优势在于实现与并行化都比较直接，适合做连续优化基线算法。

## R08

边界与异常处理：
- `bounds` 必须是二维数组，形状 `(dim, 2)`，且逐维满足 `high > low`；
- `pop_size >= 4`（`rand/1` 至少需要 3 个互异候选，加上目标个体）；
- `F` 要求在 `(0, 2]`，`CR` 要求在 `[0, 1]`；
- `max_gens > 0`、`tol > 0`；
- 若初始种群评估出现非有限值，立即报错；
- trial 非有限值按“拒绝 trial、保留原个体”处理。

## R09

MVP 取舍说明：
- 只依赖 `numpy`，保持依赖最小化；
- 不直接调用 `scipy.optimize.differential_evolution` 黑盒接口；
- 固定使用 `DE/rand/1/bin`，不引入多策略自适应，优先保证可读性；
- 采用边界裁剪而非反射/重采样，逻辑更直观。

## R10

`demo.py` 主要函数职责：
- `ensure_bounds`：检查并规范边界输入；
- `rastrigin` / `rosenbrock_nd`：基准测试函数；
- `init_population`：在边界内生成初始种群；
- `differential_evolution`：实现 DE 核心循环；
- `population_diversity`：统计种群分散度；
- `print_history`：输出分代日志；
- `run_case`：执行单个案例并计算误差指标；
- `main`：组织案例、汇总统计。

## R11

运行方式：

```bash
cd Algorithms/数学-进化计算-0357-差分进化
uv run python demo.py
```

脚本不需要命令行参数，也不会请求交互输入。

## R12

输出字段说明：
- `gen`：代编号；
- `best_f`：当前代最优目标值；
- `mean_f`：当前代种群平均目标值；
- `diversity`：当前代种群离散度（到群体均值的平均二范数）；
- `improved`：该代通过贪婪选择被替换的个体数；
- `f_best`：最终最优目标值；
- `abs_x_error / rel_x_error`：与已知参考最优点的误差；
- `optimality_gap`：`f_best - known_optimum`；
- `Summary`：跨案例聚合统计。

## R13

最小测试集（内置）：
1. `Rastrigin (5D)`
- `bounds=[-5.12, 5.12]^5`，已知最优点 `x*=0`，`f*=0`。

2. `Rosenbrock (5D)`
- `bounds=[-3, 3]^5`，已知最优点 `x*=1`，`f*=0`。

建议后续补充：
- Sphere（验证收敛速度）；
- Ackley/Griewank（更复杂多峰）；
- 非法参数单元测试（确保异常路径正确）。

## R14

关键参数与调参建议：
- `pop_size`：种群规模；维度升高时应适当增大（常见起点为 `8d~15d`）。
- `F`：差分缩放因子；常用 `0.4~0.9`，过小探索弱，过大震荡强。
- `CR`：交叉概率；高 `CR` 通常利于快速传播新信息。
- `max_gens`：代数预算；预算不足会导致早停。
- `tol`：收敛阈值；越小越严格但耗时更长。

经验策略：先固定 `F=0.7, CR=0.9`，优先调 `pop_size` 与 `max_gens`。

## R15

方法对比：
- 对比随机搜索：DE 利用种群差分方向，效率通常明显更高。  
- 对比模拟退火：SA 是单轨迹概率跳跃，DE 是群体并行进化。  
- 对比粒子群（PSO）：PSO 依赖速度更新与历史最优引导；DE 依赖差分变异，机制更直接。  
- 对比梯度法：DE 不需要梯度，适合不可导/多峰目标，但函数评估开销通常更大。

## R16

典型应用场景：
- 黑盒连续优化（目标函数不可导或噪声较大）；
- 工程参数标定（控制、结构、材料参数寻优）；
- 超参数优化基线（中低维连续空间）；
- 作为混合优化前端提供高质量初值。

## R17

可扩展方向：
- 自适应参数版本（如 jDE / SHADE）；
- 多策略并行（`rand/1`、`best/1`、`current-to-best/1`）；
- 更精细边界处理（反射、重采样）；
- 混合局部优化（DE + L-BFGS）实现“全局搜索 + 局部精修”；
- 增加收敛曲线可视化与批量实验脚本。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构造两个固定测试案例（Rastrigin、Rosenbrock），并配置统一或分案例 DE 参数。  
2. `run_case` 调用 `differential_evolution`，统一获取最优解与历史轨迹。  
3. `differential_evolution` 先通过 `ensure_bounds` 校验边界，再用 `init_population` 在边界内生成初始种群并计算 `fitness`。  
4. 每代对每个目标个体 `i` 随机采样 `r1/r2/r3`，按 `v = x_r1 + F*(x_r2-x_r3)` 生成 mutant，并执行边界裁剪。  
5. 通过 binomial 交叉（含 `j_rand` 强制位）生成 trial 向量。  
6. 评估 `trial_f`，若 `trial_f <= fitness[i]` 则用 trial 替换目标个体（贪婪选择）。  
7. 每代结束统计 `best_f/mean_f/diversity/improved` 写入 `history`，并检查 `std(fitness) < tol` 的停止条件。  
8. 循环结束后返回 `x_best/f_best/generations/converged/message`，`run_case` 计算误差与 gap，`main` 汇总跨案例结果。

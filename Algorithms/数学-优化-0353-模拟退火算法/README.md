# 模拟退火算法

- UID: `MATH-0353`
- 学科: `数学`
- 分类: `优化`
- 源序号: `353`
- 目标目录: `Algorithms/数学-优化-0353-模拟退火算法`

## R01

模拟退火算法（Simulated Annealing, SA）是一种基于随机搜索与概率接受机制的全局优化方法。它借鉴物理退火过程：
- 高温阶段允许以一定概率接受“更差解”，帮助跳出局部最优；
- 低温阶段逐渐收敛，更多执行局部精修。

本目录实现一个可运行、可审计、非黑盒的 Python MVP：
- 手写 Metropolis 接受准则；
- 手写温度衰减与邻域采样；
- 提供固定样例与分阶段日志输出。

## R02

本实现对应的问题定义：
- 输入：
  - 目标函数 `f(x)`（连续向量输入）；
  - 初始解 `x0`；
  - 变量边界 `bounds`（每维 `low/high`）；
  - 退火参数（初温、终温、降温率、每温度采样步数等）。
- 输出：
  - 搜索过程中发现的最优解 `best_x` 与 `best_energy`；
  - 最终状态 `final_x/final_energy`；
  - 迭代历史（温度、当前目标值、最优值、接受率）。

`demo.py` 内置 2 个二维固定案例，无需交互输入：
1. Rastrigin（多峰）
2. Himmelblau（多局部极小）

## R03

核心数学机制：

1. 目标最小化：
   `min f(x), x in Omega`。

2. 邻域提案（本实现）：
   `x' = clip(x + eps, bounds)`，其中 `eps ~ N(0, sigma(T)^2)`。

3. 能量差：
   `Delta E = f(x') - f(x)`。

4. Metropolis 接受准则：
   - 若 `Delta E <= 0`，必接受；
   - 否则以概率
     `p = exp(-Delta E / T)`
     接受更差解。

5. 温度更新：
   `T_{k+1} = cooling * T_k`，其中 `0 < cooling < 1`。

该机制保证高温探索、低温收敛的平衡。

## R04

算法流程（高层）：
1. 校验 `x0` 与 `bounds` 形状、有限性与范围合法性。  
2. 初始化 `current <- clip(x0)`，并设 `best <- current`。  
3. 在每个温度阶段执行固定次数邻域采样。  
4. 生成候选解并计算 `Delta E`。  
5. 按 Metropolis 规则决定是否接受候选。  
6. 若候选更优则更新全局最优 `best`。  
7. 记录该温度阶段统计（接受率、当前值、最优值等）。  
8. 降温并进入下一温度阶段，直到终温或最大迭代。

## R05

核心数据结构：
- `HistoryItem = (iters, temperature, current_energy, best_energy, acceptance_rate, ||best-current||)`
  - `iters`：累计迭代步数；
  - `temperature`：该阶段温度；
  - `current_energy`：阶段结束时当前点目标值；
  - `best_energy`：历史最优目标值；
  - `acceptance_rate`：该温度阶段接受比例；
  - `||best-current||`：当前点与最优点距离。
- `result: dict`
  - `best_x/best_energy/final_x/final_energy`；
  - `iterations/accepted_total/acceptance_rate_total`；
  - `history/final_temperature`。

## R06

正确性与有效性要点：
- “下降必收、上升按概率收”由 Metropolis 准则保证，是 SA 的核心。  
- 当温度较高时，`exp(-Delta E / T)` 较大，允许越过局部势垒。  
- 温度降低后，上升步接受概率下降，搜索逐步稳定在低能区域。  
- 本实现全程保存 `best`，即便当前状态波动，也不丢失历史最优解。  
- 通过固定随机种子实现可复现实验结果。

## R07

复杂度分析（设变量维度 `d`、总采样步数 `N`）：
- 单步成本：
  - 邻域采样 `O(d)`；
  - 目标函数评估 `C_f(d)`（与具体函数有关）；
  - 其他标量计算 `O(1)`。
- 总时间复杂度：
  - `O(N * (d + C_f(d)))`。
- 空间复杂度：
  - 不保存历史时为 `O(d)`；
  - 保存分温度日志时增加 `O(K)`（`K` 为温度阶段数）。

## R08

边界与异常处理：
- `x0` 非一维或含 `nan/inf`：抛 `ValueError`。  
- `bounds` 形状非 `(dim, 2)` 或上下界不合法：抛 `ValueError`。  
- `temp_init <= 0`、`temp_min <= 0`、`temp_min >= temp_init`：抛 `ValueError`。  
- `cooling` 不在 `(0, 1)`：抛 `ValueError`。  
- `iters_per_temp <= 0` 或 `max_iters <= 0`：抛 `ValueError`。  
- 若目标函数返回非有限值：抛 `RuntimeError`。

## R09

MVP 取舍说明：
- 仅依赖 `numpy`，避免额外框架，提高可运行性。  
- 不调用 `scipy.optimize.dual_annealing` 等黑盒接口，保持算法透明。  
- 聚焦“连续变量 + 盒约束”版本，便于讲清 SA 主干机制。  
- 使用固定温度日程与固定迭代预算，优先保证复现性与可审计性。

## R10

`demo.py` 主要函数职责：
- `ensure_vector`：检查并规范化输入向量。  
- `ensure_bounds`：检查边界矩阵合法性。  
- `rastrigin` / `himmelblau`：两组基准目标函数。  
- `propose_neighbor`：按温度自适应方差生成候选解。  
- `simulated_annealing`：执行 SA 主循环并返回完整结果。  
- `print_history`：打印阶段性迭代日志。  
- `run_case`：运行单案例并输出最优性差距。  
- `main`：组织固定案例、汇总结果并给出通过判据。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0353-模拟退火算法
python3 demo.py
```

脚本不读取命令行参数，不请求用户输入。

## R12

输出字段说明：
- `temp_step`：累计迭代步数（到该温度阶段末）。  
- `temperature`：当前温度。  
- `current_energy`：当前状态目标值。  
- `best_energy`：历史最优目标值。  
- `accept_rate`：该温度阶段接受率。  
- `||best-current||`：当前状态与历史最优状态距离。  
- `best_x`：全局最优点估计。  
- `optimality_gap`：`best_energy - known_optimum`。  
- `overall_acceptance_rate`：全流程总体接受率。  
- `Summary`：跨案例最大/平均最优性差距及通过标志。

## R13

最小测试集（已内置）：
1. `2D Rastrigin`
- 多峰目标，验证“跳出局部最优”的能力。

2. `2D Himmelblau`
- 多个全局极小，验证退火过程可找到低能区域。

建议补充异常测试：
- 传入非法 `bounds`（上界不大于下界）；
- 传入 `cooling=1.0` 或负温度参数；
- 构造返回 `nan` 的目标函数检查异常路径。

## R14

可调参数与建议：
- `temp_init`：初温，越高探索越强。  
- `temp_min`：终温，越低收敛越细但耗时增加。  
- `cooling`：降温率，越接近 1 降温越慢。  
- `iters_per_temp`：每温度阶段采样次数。  
- `max_iters`：总预算上限。  
- `seed`：随机种子（可复现）。

调参经验：
- 先调 `temp_init/cooling` 解决“卡局部最优”；
- 再增大 `iters_per_temp/max_iters` 提升末期精度；
- 观察接受率：过低说明探索不足，过高说明温度偏高或步长偏大。

## R15

方法对比：
- 对比梯度下降：
  - SA 不依赖梯度，可处理不可导/多峰目标；
  - 梯度法在光滑凸问题通常更快。
- 对比遗传算法：
  - SA 单轨迹、实现更轻；
  - GA 群体搜索更强但参数更多、成本更高。
- 对比粒子群：
  - PSO 更偏群体协同；
  - SA 更偏单解概率跳跃，机制更简洁。

## R16

典型应用场景：
- 组合/连续混合优化的基线求解。  
- 非凸参数寻优（如模型超参数、工程参数整定）。  
- 作为更复杂元启发式算法中的局部搜索或后处理模块。  
- 教学场景下展示“探索-收敛”平衡机制。

## R17

可扩展方向：
- 自适应降温（根据接受率在线调节温度）。  
- 多起点并行退火，提升全局搜索稳定性。  
- 更丰富邻域算子（柯西扰动、坐标分块扰动）。  
- 混合局部优化器（SA + 梯度法/BFGS）形成两阶段求解。  
- 增加日志落盘与收敛曲线可视化。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 定义统一退火参数（`temp_init/temp_min/cooling/...`）与两个固定测试函数。  
2. `run_case` 调用 `simulated_annealing`，并在案例结束后计算最优性差距。  
3. `simulated_annealing` 先通过 `ensure_vector/ensure_bounds` 完成输入合法性检查并初始化 `current/best`。  
4. 在每个温度阶段，`propose_neighbor` 基于当前温度生成高斯扰动候选并做边界裁剪。  
5. 计算 `Delta E = f(candidate) - f(current)`，按 Metropolis 规则决定接受与否。  
6. 若接受则更新 `current`，若更优则同步更新 `best` 与 `best_energy`。  
7. 每个温度阶段结束后记录 `(iters, temperature, current_energy, best_energy, acceptance_rate, ||best-current||)` 到 `history`，再执行 `temperature *= cooling`。  
8. 达到终温或迭代预算后返回结果；`run_case` 与 `main` 分别输出单案例指标和跨案例 Summary。

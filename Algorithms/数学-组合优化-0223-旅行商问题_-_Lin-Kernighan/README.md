# 旅行商问题 - Lin-Kernighan

- UID: `MATH-0223`
- 学科: `数学`
- 分类: `组合优化`
- 源序号: `223`
- 目标目录: `Algorithms/数学-组合优化-0223-旅行商问题_-_Lin-Kernighan`

## R01

Lin-Kernighan（LK）是 TSP 里最经典的局部搜索框架之一。核心思想不是固定只做 2-opt 或 3-opt，而是根据当前增益动态决定交换深度（variable-depth），在可行时间内逼近高质量路线。

本目录给出一个可审计的 LK 风格 MVP：
- 先用最近邻构造初始回路；
- 用候选邻居集合限制搜索分支；
- 使用“正增益链式 2-opt”做可变深度改进。

## R02

目标问题：给定 `n` 个城市（本 demo 使用二维欧氏坐标），求一个 Hamilton 回路，使总路径长度最小。

形式化：
- 输入：坐标矩阵 `coords in R^(n*2)`，距离矩阵 `D`；
- 输出：排列 `tour`（城市访问顺序）；
- 目标：最小化
  `L(tour) = sum_i D[tour[i], tour[i+1]]`（末尾回到起点）。

## R03

为什么选择 LK 风格而不是只写纯 2-opt：
- 2-opt 作为基线容易陷入局部最优；
- LK 的“可变深度 + 增益判据”是更实用的工程策略；
- 直接调用黑盒库（例如一键求解器）虽然方便，但不利于理解路径改进机制。

因此本实现选择“轻量 LK 思路 + 明确代码流程”，在可读性和效果之间取平衡。

## R04

MVP 中采用的关键机制：
- 候选集（candidate list）：每个城市只考虑最近的 `k` 个邻居，减少组合爆炸；
- 2-opt 增益公式：
  对边 `(a,b)` 与 `(c,d)`，交换后增益
  `gain = D[a,b] + D[c,d] - D[a,c] - D[b,d]`；
- 链式改进：一次改善后继续在“受影响节点”附近寻找下一步改善，直到深度上限或无正增益；
- 多锚点扫描：一轮中对多个起始边尝试链式搜索，取收益最大的方案。

## R05

算法高层流程：

1. 生成（或读取）城市坐标并构造欧氏距离矩阵；
2. 最近邻法得到初始回路；
3. 构建候选邻居列表；
4. 进入主循环：
   - 执行一轮 LK 风格链式搜索；
   - 若有收益则更新当前回路并记录历史；
   - 连续多轮无收益则提前停止；
5. 输出最终路径长度、改进比例与迭代表。

## R06

正确性与可行性要点：
- `tour` 始终维护为城市排列，2-opt 通过区间反转保持 Hamilton 回路结构；
- 每次接受新解都要求“真实长度下降”(`old_length - new_length > 0`)；
- 距离矩阵在入口做对称性、非负、对角为 0 的一致性检查；
- 输入路径会验证是 `0..n-1` 的全排列，避免非法状态传播。

## R07

复杂度（近似）：
- 记城市数 `n`、候选邻居数 `k`、链深 `d`、外层迭代轮数 `T`。
- 一次候选 2-opt 搜索约为 `O(n * k)`；
- 一次链式 pass 最多深度 `d`，并扫描多个锚点，近似 `O(n * d * n * k)` 的启发式上界；
- 实际运行中因候选集与提前停止显著快于全量边对枚举 `O(n^2)`。
- 空间复杂度：
  - 距离矩阵 `O(n^2)`；
  - 候选集 `O(nk)`；
  - 路径与索引结构 `O(n)`。

## R08

边界与异常处理：
- 坐标必须是 `(n,2)`；
- 距离矩阵必须方阵、有限、对称、非负、对角为 0；
- `candidate_k/max_depth/max_iters` 必须为正；
- `n_cities` 过小（<8）时拒绝运行，以避免示例失真；
- 对 2-opt 索引做合法性检查，排除相邻边和退化交换。

## R09

MVP 取舍说明：
- 仅依赖 `numpy` 和标准库，避免重型框架；
- 未实现完整论文版 LK 的全部规则（例如更复杂的 backtracking 与可行闭环判据组合）；
- 采用“LK 风格链式 2-opt”作为教学和审计友好的最小实现；
- 优先代码透明、可运行、可解释，而不是追求竞赛级最优解。

## R10

`demo.py` 主要函数职责：
- `euclidean_distance_matrix`：由坐标构造距离矩阵。
- `validate_distance_matrix / validate_tour`：输入合法性校验。
- `nearest_neighbor_tour`：生成初始可行解。
- `build_candidate_lists`：为每个城市准备近邻候选。
- `two_opt_gain / apply_two_opt_inplace`：2-opt 增益计算与原地应用。
- `find_best_candidate_two_opt`：在候选集内找最优单步交换。
- `lk_style_chain_pass`：执行可变深度链式改进。
- `lin_kernighan_style_tsp`：整体求解循环与历史记录。
- `main`：构造实例、运行求解并打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-组合优化-0223-旅行商问题_-_Lin-Kernighan
python3 demo.py
```

脚本无交互输入，直接输出实验结果。

## R12

输出字段说明：
- `cities`：城市数量。
- `instance seed`：实例随机种子（坐标生成）。
- `solver seed`：搜索随机种子（锚点顺序）。
- `initial length`：初始路径长度（最近邻解）。
- `final length`：优化后路径长度。
- `improvement`：绝对改进值与百分比。
- `iterations`：外层迭代次数。
- `iter | old_length | new_length | gain`：每轮改进日志。

## R13

最小验证集合（脚本已覆盖）：
- 固定随机种子、`n=120` 城市的二维欧氏 TSP；
- 使用最近邻初始解；
- 运行 LK 风格链式搜索并打印历史；
- 通过最终长度与改进比例确认算法确实优化了路径。

可选扩展测试：
- 改变 `candidate_k`（如 12/24/40）观察速度与质量折中；
- 改变 `max_depth`（如 3/6/8）观察局部搜索强度变化。

## R14

关键参数建议：
- `candidate_k`：候选邻居数。越大搜索更充分，但更慢；
- `max_depth`：链式交换最大深度。越大可能更强，但单轮开销增加；
- `max_iters`：外层最大轮数；
- `stale_rounds`（代码内）：连续无改进轮数阈值，触发提前停止。

经验起点（本 demo 默认）：
- `candidate_k=24`
- `max_depth=6`
- `max_iters=80`

## R15

与相关方法对比：
- 最近邻：极快但质量一般，只作为初始化。
- 纯 2-opt：实现简单，但局部最优明显。
- LK 风格：通过可变深度链式交换，通常优于纯 2-opt。
- 完整 LKH/Lin-Kernighan-Helsgaun：性能更强，但实现复杂度也显著更高。

本目录定位是“可运行且可读”的中间层实现。

## R16

典型应用场景：
- 物流配送路径初始优化；
- 巡检/采样/拜访顺序规划；
- 作为更复杂元启发式（GA/SA/ACO）的局部改进算子；
- 组合优化课程中的 TSP 启发式教学样例。

## R17

可扩展方向：
- 加入双桥扰动（double-bridge）做逃逸重启；
- 将候选集从“距离最近”改为 alpha-nearness 等更强策略；
- 增加 3-opt/5-opt 子动作与回溯规则；
- 支持 TSPLIB 文件读入与结果对标；
- 将距离矩阵替换为非欧氏或带约束代价模型。

## R18

`demo.py` 源码级流程（8 步）：

1. `main` 固定 `n_cities/seed`，调用 `make_euclidean_instance` 生成二维坐标。
2. `euclidean_distance_matrix` 构建 `n x n` 距离矩阵，随后进入求解流程。
3. `nearest_neighbor_tour` 生成初始 Hamilton 回路，并用 `tour_length` 计算初值。
4. `lin_kernighan_style_tsp` 内部先做合法性校验，再通过 `build_candidate_lists` 构造候选邻居。
5. 每轮调用 `lk_style_chain_pass`：按随机锚点顺序尝试“正增益链式 2-opt”。
6. 链内每一步由 `find_best_candidate_two_opt` 在候选集搜索最优交换，使用 `two_opt_gain` 计算增益，并由 `apply_two_opt_inplace` 执行区间反转。
7. 若本轮真实长度下降，则接受新回路并写入 `history`；连续多轮无改进则提前停止。
8. `main` 打印初始/最终长度、提升比例、迭代表和路径前缀，形成可直接复现的 MVP 结果。

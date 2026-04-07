# facility location算法

- UID: `MATH-0228`
- 学科: `数学`
- 分类: `组合优化`
- 源序号: `228`
- 目标目录: `Algorithms/数学-组合优化-0228-facility_location算法`

## R01

`facility location`（设施选址）关注“开哪些设施、客户分配给谁”以最小化总成本。  
本目录实现的是**无容量设施选址（Uncapacitated Facility Location, UFL）**：
- 每个候选设施 `i` 有固定开设成本 `f_i`；
- 每个客户 `j` 必须分配给一个已开设施；
- 分配运输成本 `c_{ij}` 按“欧氏距离 × 客户需求”计算。

MVP 采用“可追踪而非黑箱”的方案：
- 贪心 add 初始化；
- add/drop/swap 局部搜索改进；
- 小规模用穷举精确解做对照。

## R02

问题定义（UFL）：
- 决策变量：
  - `y_i in {0,1}`：设施 `i` 是否开设；
  - `x_ij in {0,1}`：客户 `j` 是否由设施 `i` 服务。
- 目标函数：
  - `min sum_i f_i y_i + sum_j sum_i c_ij x_ij`
- 约束：
  - `sum_i x_ij = 1`（每个客户必须且只被一个设施服务）；
  - `x_ij <= y_i`（未开设施不能服务客户）。

`demo.py` 生成确定性随机实例，不需要任何交互输入。

## R03

本实现的成本建模：
- 坐标：设施点 `facility_xy`、客户点 `customer_xy`；
- 需求：`demand[j] > 0`；
- 运输成本：`c_ij = demand[j] * ||facility_i - customer_j||_2`；
- 总成本：`opening_cost + shipping_cost`。

说明：这是常见连续平面上的离散选址近似模型，利于演示算法结构。

## R04

算法总流程：
1. 生成实例并构造运输成本矩阵 `C`。  
2. 在所有“单设施开设”方案中选最优，作为初始解。  
3. 执行贪心 add：反复尝试新增一个设施，直到无改进。  
4. 以该解为起点做局部搜索，邻域包含 `add/drop/swap`。  
5. 每轮选择“最优改进”动作并更新解，直到局部最优。  
6. 对小规模设施数执行穷举，得到精确最优作质量对照。  
7. 输出成本、开设数量、动作轨迹和相对 gap。

## R05

核心数据结构：
- `FacilityLocationInstance`：
  - `facility_xy (m,2)`、`customer_xy (n,2)`；
  - `demand (n,)`、`open_cost (m,)`；
  - `service_cost (m,n)`。
- `FacilityLocationSolution`：
  - `open_mask (m,)` 布尔向量；
  - `assignment (n,)` 客户到设施的索引；
  - `opening_cost / shipping_cost / total_cost`。
- `MoveRecord`：
  - `iteration, move_type, detail, delta, total_cost`。

## R06

正确性要点：
- `evaluate_solution` 对给定 `open_mask` 精确执行“每个客户选最便宜已开设施”，因此该子问题是最优分配。  
- 局部搜索每一步只接受 `delta < 0` 的动作，总成本严格下降，有限状态下必终止。  
- 穷举器遍历所有非空设施子集，对每个子集调用同一评估函数，得到全局最优参考值（小规模）。

## R07

复杂度（`m` 设施，`n` 客户）：
- 单次方案评估：`O(m_open * n)`，最坏 `O(mn)`；
- 贪心 add：最多 `m` 轮，每轮评估最多 `m` 个候选，约 `O(m^3 n)` 上界；
- 局部搜索单轮邻域规模：
  - add `O(m)`、drop `O(m)`、swap `O(m^2)`，
  - 总体评估约 `O(m^2)` 次，每次 `O(mn)`，单轮 `O(m^3 n)`；
- 穷举精确解：`O(2^m * m * n)`，仅用于小 `m` 校验。

## R08

健壮性与异常处理：
- 维度检查：二维坐标矩阵、需求向量长度、`open_mask` 形状。  
- 数值检查：输入必须有限；需求必须为正。  
- 可行性检查：不允许“零设施开设”方案。  
- 穷举保护：`m > max_facilities` 时直接抛错，避免不可控运行时间。

## R09

MVP 取舍：
- 重点是**算法透明度**，不引入 MIP 求解器黑箱；
- 不实现容量约束、服务半径约束、多级设施等扩展版本；
- 用局部搜索得到高质量解，再用小规模精确解评估质量；
- 依赖极简：仅 `numpy`。

## R10

`demo.py` 函数职责：
- `pairwise_euclidean`：计算设施-客户距离矩阵；
- `build_service_cost`：生成运输成本矩阵；
- `evaluate_solution`：给定开设集合后做最优分配并算总成本；
- `greedy_add_initialization`：贪心初始解；
- `local_search`：add/drop/swap 最优改进；
- `brute_force_optimal`：小规模精确最优；
- `generate_instance`：可复现实例生成；
- `run_case/main`：执行案例并打印摘要。

## R11

运行方式：

```bash
cd Algorithms/数学-组合优化-0228-facility_location算法
python3 demo.py
```

脚本固定两组案例（不同 seed 和规模），无命令行参数、无交互输入。

## R12

输出解读：
- 每个案例会打印：
  - `Initial (greedy-add)`：初始化解成本；
  - `Final (local-search)`：局部搜索后成本；
  - `Exact (brute-force)`：精确最优成本；
  - `Relative gap to exact`：启发式相对最优差距。
- 还会打印两段动作轨迹：
  - `Greedy-add move trace`；
  - `Local-search move trace`。
- 最后汇总平均成本与平均 gap。

## R13

最小测试建议：
- 默认内置两组案例可作为冒烟测试；
- 建议补充：
  - 提高开设成本，观察“少开站点”行为；
  - 降低开设成本，观察“多开站点”行为；
  - 不同客户簇分布，验证 swap 动作是否被触发。

## R14

关键可调参数：
- `n_facilities / n_customers`：问题规模；
- `max_iter`：局部搜索迭代上限；
- `max_facilities`：穷举允许的最大设施数；
- `open_cost` 生成公式系数：控制开设成本与运输成本的相对权重。

经验：当开设成本更高时，最优解倾向少开设施；反之倾向多开设施以降低运输成本。

## R15

与其他解法对比：
- 相对纯贪心：
  - 本实现增加 drop/swap，可跳出“只增不减”的结构性偏差。  
- 相对整数规划（MIP）：
  - MIP 可给全局最优但依赖外部求解器；本实现更轻量、可读性更高。  
- 相对仅随机搜索：
  - 局部搜索利用结构化邻域，通常更稳定、收敛更快。

## R16

典型应用场景：
- 仓储中心/前置仓布局；
- 物流站点和配送半径规划的原型分析；
- 边缘节点或服务机房选址的成本评估；
- 需要快速给出“可解释方案”的教学与研究原型。

## R17

可扩展方向：
- 容量约束设施选址（Capacitated FLP）；
- 多层级选址（工厂-仓库-门店）；
- 加入固定服务半径或时效 SLA 约束；
- 把邻域搜索替换为 tabu search / simulated annealing / LNS；
- 与 MIP 或拉格朗日松弛结合做大规模求解。

## R18

`demo.py` 源码级流程（8 步）：
1. `main` 设定两个固定案例（seed、设施数、客户数），逐个调用 `run_case`。  
2. `run_case` 调用 `generate_instance`：生成设施/客户坐标、需求、开设成本，并构造 `service_cost`。  
3. 调用 `greedy_add_initialization`：
   - 枚举所有单设施方案选最佳起点；
   - 反复尝试“新增一个设施”的最佳改进，得到初始高质量解。  
4. 调用 `local_search`，在当前解上评估三类邻域：`add`、`drop`、`swap`。  
5. 每轮从全部候选动作中选 `delta` 最小（最负）者执行，保证目标值下降；无改进则停机。  
6. 调用 `brute_force_optimal` 穷举全部非空设施子集，用同一 `evaluate_solution` 计算精确最优。  
7. 计算启发式解与精确解的相对 gap，并输出开设数量、成本、动作轨迹。  
8. 所有案例结束后输出整体平均成本与平均 gap。

整个流程不依赖第三方优化黑箱，选址决策、客户分配和邻域改进都可直接在源码中逐行追踪。

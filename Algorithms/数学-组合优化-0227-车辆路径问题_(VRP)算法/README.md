# 车辆路径问题 (VRP)算法

- UID: `MATH-0227`
- 学科: `数学`
- 分类: `组合优化`
- 源序号: `227`
- 目标目录: `Algorithms/数学-组合优化-0227-车辆路径问题_(VRP)算法`

## R01

车辆路径问题（Vehicle Routing Problem, VRP）研究如何让多辆车从同一配送中心出发，为一组客户提供服务并返回中心，使总代价最小。  
本目录实现的是 `CVRP`（容量约束 VRP）的最小可运行 MVP：
- 每个客户需求已知；
- 每辆车容量固定；
- 每个客户必须且仅能被访问一次；
- 目标是最小化总行驶距离。

实现采用两阶段启发式：
1. `Clarke-Wright Savings` 构造初始可行路径；
2. 对每条路径执行 `2-opt` 局部改进。

## R02

本实现的问题定义：
- 输入：
  - 节点坐标 `coords`（`0` 为仓库，`1..n` 为客户）；
  - 客户需求 `demands`（`demands[0] = 0`）；
  - 车辆容量 `capacity`；
  - 可选车辆上限 `max_vehicles`。
- 约束：
  - 每条路线形式为 `[0, ..., 0]`；
  - 每个客户必须被且仅被服务一次；
  - 单条路线客户需求和不超过 `capacity`；
  - 若给出 `max_vehicles`，路线条数不能超过该值。
- 输出：
  - 路线集合 `routes`；
  - 总路程 `total_distance`。

`demo.py` 内置随机种子和数据生成，不需要交互输入。

## R03

数学建模（CVRP 常见形式）：
- 给定图 `G=(V,E)`，其中 `V={0,1,...,n}`，`0` 为仓库；
- 客户需求 `q_i`，车辆容量 `Q`；
- 边代价（本实现为欧氏距离）`c_ij`。

典型目标：
- 最小化 `sum(c_ij * x_ij)`，其中 `x_ij in {0,1}` 表示是否走边 `(i,j)`。

主要约束：
1. 每个客户恰好一次入流、一次出流；
2. 路线均从仓库出发并回到仓库；
3. 每条路线载重不超过 `Q`；
4. 消除子回路（在精确法中通常通过额外约束处理）。

本 MVP 不做 MILP 精确求解，而使用可解释启发式快速生成可行近似解。

## R04

算法总流程：
1. 计算完整距离矩阵 `dist`；
2. 初始解设为每个客户独立一条路线 `[0, i, 0]`；
3. 计算任意客户对 `(i,j)` 的节约值：
   `s_ij = d(0,i) + d(0,j) - d(i,j)`；
4. 按 `s_ij` 从大到小尝试合并路线；
5. 合并时检查两点是否位于各自路线端点、合并后是否超容量；
6. 得到 `Clarke-Wright` 初始可行解；
7. 对每条路线做 `2-opt` 反转优化；
8. 做可行性校验并输出摘要。

## R05

核心数据结构：
- `CVRPInstance`：问题实例。
  - `coords: np.ndarray`
  - `demands: np.ndarray`
  - `capacity: int`
  - `max_vehicles: Optional[int]`
- `CVRPSolution`：候选解。
  - `routes: List[List[int]]`
  - `total_distance: float`
- 中间映射（节约算法内部）：
  - `routes: Dict[route_id, List[customer]]`（不含仓库端点）
  - `loads: Dict[route_id, int]`
  - `customer_to_route: Dict[customer, route_id]`

## R06

正确性要点（实现级）：
- 访问唯一性：每次合并只重连两条互不相同的路线，客户归属映射同步更新；
- 容量合法性：合并前检查 `loads[ri] + loads[rj] <= capacity`；
- 路径合法性：仅当连接点位于各自路径端点时才允许拼接，避免中间断裂；
- 完整校验：`validate_solution` 会检查
  - 路径起终点是否是仓库；
  - 是否覆盖且只覆盖 `1..n`；
  - 每条路径是否超载；
  - 路径长度和距离是否有限；
  - 车辆数是否超过上限。

## R07

复杂度分析（`n` 为客户数）：
- 距离矩阵计算：`O(n^2)` 时间，`O(n^2)` 空间；
- 节约值枚举与排序：
  - 枚举 `O(n^2)`；
  - 排序 `O(n^2 log n)`；
- 合并扫描：最坏 `O(n^2)` 次候选检查；
- 2-opt：对单条长度 `m` 的路径为 `O(m^2)`，全局取决于路径划分，最坏可近似看作 `O(n^2)` 级别（启发式迭代通常远低于最坏上界）。

整体由节约排序与局部搜索主导，适合中小规模快速求可行优质解。

## R08

边界与异常处理：
- `coords` 维度非 `(k,2)` 或含 `nan/inf`：抛 `ValueError`；
- 客户数 `<= 0`：抛 `ValueError`；
- 路线起终点非 `0`：校验失败；
- 存在未访问客户、重复访问客户、节点越界：校验失败；
- 路线载重超 `capacity`：校验失败；
- 若给定 `max_vehicles` 且解超限：抛异常。

## R09

MVP 取舍：
- 采用 `numpy` + 手写启发式，不依赖黑盒 OR 求解器；
- 不引入时间窗（VRPTW）、多仓库、异构车队等扩展约束；
- 不做全局最优保证，目标是“可运行、可审计、可扩展”的基础版本；
- `2-opt` 只在单条路径内优化，不做跨路径 customer relocate/exchange。

## R10

`demo.py` 函数职责：
- `euclidean_distance_matrix`：构造距离矩阵；
- `route_load` / `route_distance` / `total_distance`：计算载重与距离；
- `validate_solution`：统一可行性校验；
- `try_merge`：判断并执行两条客户序列的端点合并；
- `clarke_wright_savings`：节约算法主过程；
- `two_opt_route`：单路线 2-opt；
- `improve_with_two_opt`：对全部路线做局部改进；
- `trivial_single_customer_routes`：构造基线解；
- `generate_demo_instance`：生成可复现实例；
- `print_solution`：格式化打印；
- `main`：组织流程、输出结果与对比指标。

## R11

运行方式：

```bash
cd Algorithms/数学-组合优化-0227-车辆路径问题_(VRP)算法
python3 demo.py
```

脚本不读取命令行参数，也不会请求用户输入。

## R12

输出解读：
- `customers / total demand / capacity / vehicle limit`：实例规模与约束；
- `Baseline`：每个客户单独一条路线的对照解；
- `After Clarke-Wright savings`：节约算法构造结果；
- `After per-route 2-opt`：局部改进后结果；
- 每条路线明细：
  - `load`：载重；
  - `dist`：该路线路程；
  - `path`：访问序列（含仓库）；
- `distance reduction vs baseline`：相对基线降幅；
- `distance reduction from 2-opt stage`：2-opt 额外降幅；
- `feasibility checks: PASS`：约束检查通过。

## R13

建议测试项：
1. 正常随机实例：验证完整流程可运行并输出可行解；
2. 小规模手工实例（如 5-8 客户）：便于人工核对路由与载重；
3. 压力测试（增大 `n_customers`）：观察运行时间与解质量变化；
4. 异常测试：
   - 注入非法坐标（`nan`）；
   - 将 `capacity` 设得过小；
   - 人为篡改路线触发 `validate_solution` 报错。

## R14

关键参数：
- `n_customers`：客户数；
- `area_size`：坐标采样范围；
- `demand_low/high`：需求区间；
- `target_vehicles`：用于估计容量，使路线数接近期望；
- `max_vehicles`：可行路线数上限。

调参建议：
- 想减少路线数：增大 `capacity` 或放宽 `max_vehicles`；
- 想制造更难实例：增大客户数、扩大区域、提高需求波动；
- 想看 2-opt 价值：固定实例，比较优化前后总路程。

## R15

与常见方法对比：
- 对比精确法（MILP/Branch-and-Cut）：
  - 精确法可给全局最优或最优界，但规模上去后代价高；
  - 本方法速度快、实现轻量，适合工程预解和教学演示。
- 对比纯构造法（仅 Savings）：
  - 仅构造通常可行但局部次优；
  - 加 2-opt 后常能稳定再降一段路程。
- 对比元启发式（GA/SA/TS/ACO）：
  - 元启发式解质量上限更高，但实现和参数调试更重；
  - 本 MVP 结构清晰，便于后续叠加高级策略。

## R16

典型应用：
- 城配/城际配送路径规划；
- 末端快递分拨后的车辆派送；
- 售后巡检与服务工程师派工；
- 冷链或药品配送的基础路径基线；
- 作为更复杂 VRP（时间窗、取送货、多仓）的起点版本。

## R17

可扩展方向：
- 增加跨路线操作：relocate、swap、2-opt*；
- 引入时间窗形成 VRPTW；
- 使用并行多初始解 + 迭代改进；
- 增加固定车辆成本，联合优化“车数 + 路程”；
- 对接真实路网距离/时长矩阵（替代欧氏距离）。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `main` 调用 `generate_demo_instance` 生成固定随机种子的 CVRP 数据。  
2. `euclidean_distance_matrix` 构造全节点两两距离矩阵。  
3. 计算基线解：`trivial_single_customer_routes`（每客户单独往返仓库）。  
4. `clarke_wright_savings` 初始化 `n` 条单客户路径，并建立 `customer_to_route` 映射。  
5. 在 `clarke_wright_savings` 内枚举全部客户对，按 `s_ij = d(0,i)+d(0,j)-d(i,j)` 降序排序。  
6. 依次尝试候选对：调用 `try_merge` 检查两端点可拼接性，并同时检查容量约束，成功则合并并更新映射。  
7. 构造得到的路径集合后，`improve_with_two_opt` 对每条路线调用 `two_opt_route`，通过子段反转迭代降低路径长度。  
8. `validate_solution` 对节约解与 2-opt 解分别做完整可行性审计（覆盖、容量、起终点、车辆数）。  
9. `print_solution` 与 `Summary` 输出三种解（基线/节约/节约+2-opt）及距离降幅指标。

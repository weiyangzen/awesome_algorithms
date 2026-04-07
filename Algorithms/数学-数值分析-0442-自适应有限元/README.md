# 自适应有限元

- UID: `MATH-0442`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `442`
- 目标目录: `Algorithms/数学-数值分析-0442-自适应有限元`

## R01

自适应有限元（Adaptive FEM）是在有限元求解中动态调整网格分辨率的策略，核心闭环是：

- `SOLVE`：在当前网格上求离散解
- `ESTIMATE`：计算后验误差指示子
- `MARK`：选出需要细化的单元
- `REFINE`：细化网格并进入下一轮

相比一次性均匀加密，自适应方法能把自由度集中在“误差高、解变化快”的区域，以更少计算量获得更高精度。

## R02

本条目用 1D Poisson 方程做最小可运行演示：

- 强形式：`-u''(x)=f(x), x in (0,1)`
- 边界条件：`u(0)=u(1)=0`
- 右端项：`f(x)=pi^2 sin(pi x)`
- 精确解：`u(x)=sin(pi x)`

选该问题是因为精确解已知，可直接计算 `L2` 与 `H1` 误差，便于核对自适应循环是否有效。

## R03

离散设置：

- 单元类型：1D 线性单元（`P1`）
- 自由度：网格节点值（两端点因 Dirichlet 固定）
- 初始网格：`[0,1]` 上均匀划分为 `initial_elements=4`
- 细化策略：仅做 `h`-refinement（对被标记单元二分）

这是最小实现，不涉及高阶基函数或 hp 混合策略。

## R04

弱形式：在 `V=H_0^1(0,1)` 上求 `u` 使得

`a(u,v)=l(v), forall v in V`

其中：

- `a(u,v)=int_0^1 u' v' dx`
- `l(v)=int_0^1 f v dx`

离散后得到线性系统 `K u_h = F`。在 `demo.py` 中，局部刚度矩阵直接使用线性单元解析表达：

`K_e = [[1,-1],[-1,1]] / h`。

## R05

局部载荷积分与误差积分使用 Gauss-Legendre 数值积分：

- 组装载荷：4 点积分
- 残差估计中的 `||f||_{L2(K)}`：5 点积分
- 精确误差评估：6 点积分

这样做的目的是把积分误差压低，避免把收敛行为误判为“算法问题”。

## R06

后验误差估计器采用 1D 残差型指标：

`eta_K^2 = h_K^2 ||f||^2_{L2(K)} + 0.5 h_K (J_left^2 + J_right^2)`

其中 `J` 是节点处导数跳跃（相邻单元斜率差）。该形式结合了：

- 单元体残差项（右端激励强度）
- 面跳跃项（离散梯度不连续程度）

总估计量：`eta = sqrt(sum_K eta_K^2)`。

## R07

标记策略使用 Doerfler bulk marking（体积分数标记）：

- 将 `eta_K^2` 按降序排序
- 取最小集合 `M`，使 `sum_{K in M} eta_K^2 >= theta * sum_all`
- 本实现设 `theta=0.5`

这比固定比例标记更稳健：误差高度集中时会自动少标记，误差分散时会自动多标记。

## R08

细化策略：

- 对所有被标记单元做中点二分
- 非标记单元保持不变
- 重新生成严格递增节点数组

因为是 1D，二分细化实现简单且无悬挂节点处理成本；在 2D/3D 中通常需要额外网格一致性修复。

## R09

`demo.py` 的函数划分：

- `solve_poisson_p1`: 组装并求解线性系统
- `residual_estimator`: 计算每单元 `eta_K^2` 与总估计量
- `mark_doerfler`: 执行 bulk marking
- `bisect_marked_elements`: 对标记单元做二分细化
- `exact_errors`: 计算 `L2` / `H1` 精确误差
- `adaptive_fem_1d`: 驱动自适应主循环
- `main`: 固定参数运行并打印结果表

## R10

线性求解实现：

- 优先使用 `scipy.sparse` + `spsolve`（稀疏求解）
- 若 SciPy 不可用，自动回退到 `numpy` 密集矩阵求解

回退逻辑保证 MVP 在不同环境下都能执行，不依赖交互输入或手工参数。

## R11

复杂度（本 MVP）概览：

- 单轮组装：`O(nelem)`（1D P1，每单元常数规模局部计算）
- 标记排序：`O(nelem log nelem)`
- 线性求解：稀疏直接法通常介于线性到超线性（实现相关）
- 迭代轮次：由 `max_iterations` 与 `estimator_tol` 控制

尽管是教学级实现，流程完整覆盖自适应 FEM 关键环节。

## R12

程序输出每轮的关键指标：

- `elem/nodes/dof`
- `L2 error`
- `H1-semi err`
- `estimator`
- `marked`

可直接观察“网格变细 -> 误差下降 -> 估计量下降”的趋势。

## R13

运行方式（无交互）：

```bash
python3 Algorithms/数学-数值分析-0442-自适应有限元/demo.py
```

或先进入目录再运行：

```bash
cd Algorithms/数学-数值分析-0442-自适应有限元
python3 demo.py
```

## R14

默认参数：

- `initial_elements=4`
- `max_iterations=10`
- `theta=0.5`
- `estimator_tol=1e-3`
- `max_elements=4096`

在该设置下，通常会看到误差随迭代显著下降；若估计量已低于阈值会提前停止。

## R15

当前 MVP 的边界：

- 仅 1D、仅标量椭圆问题
- 仅 `P1` 线性单元
- 仅 `h` 自适应，不含 `p` 或 `hp`
- 细化不含“合并粗化（coarsening）”

因此它是算法闭环演示，不是工程级通用有限元框架。

## R16

可扩展方向：

- 扩展到 2D 三角形/四边形网格
- 引入层次基与高阶单元（`p` 或 `hp`）
- 使用更强后验估计器（如 equilibrated flux）
- 增加 coarsening 与误差-复杂度自适应控制
- 结合 AMG/多重网格提升大规模求解效率

## R17

最小验收清单：

- `README.md` 不含模板占位符
- `demo.py` 不含模板占位符
- `python3 demo.py` 可直接运行并输出迭代表格
- 输出中 `estimator` 与误差总体呈下降趋势
- `meta.json` 的 UID/名称/目录与任务元数据一致

## R18

源码级算法流程（对应 `demo.py`，非黑盒）：

1. `adaptive_fem_1d` 初始化网格节点 `nodes=linspace(0,1,initial_elements+1)`，进入迭代循环。
2. 每轮先调用 `solve_poisson_p1`：逐单元构造 `K_e` 与 `F_e`，累加到全局稀疏矩阵/向量。
3. 在 `solve_poisson_p1` 内去除两端 Dirichlet 自由度，解 `K_ff u_f = F_f`，回填完整节点解 `u`。
4. 调用 `residual_estimator`：先算各单元斜率，再在内部节点计算导数跳跃 `J`。
5. `residual_estimator` 对每个单元积分 `||f||_{L2(K)}`，组合成 `eta_K^2`，最后得到 `eta=sqrt(sum eta_K^2)`。
6. 调用 `mark_doerfler`：按 `eta_K^2` 从大到小累积，直到达到 `theta * total`，得到布尔标记数组。
7. 调用 `bisect_marked_elements`：仅对被标记单元插入中点，生成新网格并进入下一轮。
8. 每轮同时调用 `exact_errors`（利用已知解析解）计算 `L2` 与 `H1` 误差，用于验证收敛趋势。
9. 满足 `eta <= estimator_tol` 或达到迭代/单元上限时停止，`main` 打印完整迭代统计与最终误差。

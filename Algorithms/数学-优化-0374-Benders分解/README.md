# Benders分解

- UID: `MATH-0374`
- 学科: `数学`
- 分类: `优化`
- 源序号: `374`
- 目标目录: `Algorithms/数学-优化-0374-Benders分解`

## R01

Benders 分解用于求解“少量耦合变量 + 大量条件变量”的大规模优化问题。核心思想是把原问题拆成：
- 主问题（Master）：只保留一阶段变量 `x` 与一个近似变量 `theta`。
- 子问题（Subproblem）：在给定 `x` 后，求二阶段最优值 `Q(x)`。

通过迭代地向主问题添加割（cuts），逐步逼近真实的 `Q(x)`，最终得到全局最优解。

## R02

本目录实现的是连续型两阶段线性规划（LP）的标准场景：

- 原问题：
  - `min c^T x + q^T y`
  - `s.t. A x <= b`
  - `W y >= h - T x, y >= 0`

当 `x` 维度较小、`y` 和约束规模较大时，Benders 分解相比直接求解整体模型更有结构优势。

## R03

在实现中，我们把二阶段价值函数写成：

- `Q(x) = min { q^T y | W y >= h - T x, y >= 0 }`

并将主问题写成：

- `min c^T x + theta`
- `s.t. A x <= b`
- `theta >=` 已生成的所有 Benders 割

其中 `theta` 是 `Q(x)` 的下界近似。

## R04

分解逻辑：

1. 在主问题中给出一个候选 `x^k`。
2. 固定 `x^k` 解子问题，得到真实二阶段值 `Q(x^k)`。
3. 用子问题对偶解生成一条新的线性下界（最优性割）。
4. 把该割加入主问题，重复迭代。

由于 `Q(x)` 是分段线性凸函数，这种逐步加割的方式会收敛到最优解。

## R05

本 MVP 的主问题具体为：

- 变量：`x1, x2, theta`
- 目标：`min 0.9*x1 + 1.1*x2 + theta`
- 约束：
  - `x1 + x2 <= 8`
  - `0 <= x1 <= 6`
  - `0 <= x2 <= 6`
  - `theta >= 0`（初始下界）
  - 以及迭代生成的 Benders 最优性割

## R06

子问题（固定 `x`）采用对偶形式求值：

- 原始子问题：`min q^T y, s.t. W y >= h - T x, y >= 0`
- 对偶子问题：
  - `max pi^T (h - T x)`
  - `s.t. W^T pi <= q, pi >= 0`

`demo.py` 直接求解对偶 LP，获得：
- 子问题值 `Q(x)`
- 对偶极点 `pi`（用于生成割）

## R07

由对偶极点 `pi` 可得一条最优性割：

- `theta >= pi^T (h - T x)`
- 等价于 `theta >= alpha - beta^T x`
  - `alpha = pi^T h`
  - `beta = T^T pi`

代码中把它转换为 `linprog` 的 `<=` 形式：
- `-beta^T x - theta <= -alpha`

## R08

终止判据使用双重检查：

- 割违背度：`Q(x^k) - theta^k <= tol`
- 上下界差：`UB - LB <= tol`

其中：
- `LB` 是当前主问题最优值。
- `UB` 是迄今最好可行解的真实目标值 `c^T x + Q(x)`。

## R09

复杂度（粗略）：

- 每次迭代求解 2 个 LP（主问题 + 子问题）。
- 若迭代次数为 `K`，总体复杂度约为 `K` 次 LP 复杂度之和。
- 主问题维度随割数线性增长；子问题维度固定。

在真实大规模场景中，Benders 的价值在于：避免直接构建包含全部二阶段变量的大型整体模型。

## R10

数值实现要点：

- 使用 `scipy.optimize.linprog(method="highs")`，数值稳定性较好。
- 对 `theta` 施加下界，避免初始主问题无界。
- 在收敛判据里同时用“割违背度”和“UB-LB gap”，减少误判。
- 由于本示例满足 complete recourse，省略可行性割；工程中若可能不可行需额外加入可行性割逻辑。

## R11

`demo.py` 的模块划分：

- `build_demo_instance()`：构建示例数据。
- `solve_master_problem()`：解带当前割的主问题。
- `solve_dual_subproblem()`：解对偶子问题，返回 `Q(x)` 和 `pi`。
- `benders_decomposition()`：主循环，维护 `LB/UB`、收敛判断、加割。
- `solve_extensive_form()`：直接解整体 LP 做结果校验。
- `main()`：打印迭代日志和最终对比。

## R12

示例数据特点：

- `x` 维度为 2，`y` 维度为 2，子问题约束为 3 条。
- 二阶段约束右端为 `h - T x`，体现一阶段决策对二阶段需求的影响。
- `W` 与 `q` 选择保证可解释的分段线性 recourse 函数。
- complete recourse：对任意给定 `x` 都可通过足够大的 `y` 满足 `W y >= h - T x`。

## R13

运行方式（无交互输入）：

```bash
uv run python Algorithms/数学-优化-0374-Benders分解/demo.py
```

脚本会输出：
- 每轮迭代的 `x, theta, sub, LB, UB, gap`
- Benders 与直接整体 LP 的最优值对比

## R14

输出解释：

- `theta`：主问题对 `Q(x)` 的当前近似。
- `sub`：固定当前 `x` 后子问题真实值 `Q(x)`。
- `cut_violation = sub - theta`：若显著大于 0，说明当前近似偏松，需要加割。
- `gap = UB - LB`：全局最优性缺口。

当二者都接近 0 时，算法收敛。

## R15

正确性校验策略：

- 同时实现“分解求解”和“整体 LP 直接求解”。
- 结束时比较两者目标值差 `|obj_benders - obj_direct|`。
- 若差值超过阈值（`1e-6`），脚本抛出异常。

这保证了 MVP 不只是“能跑”，而是结果可核验。

## R16

局限与边界：

- 当前版本是单割（single-cut）Benders，未做多割并行加速。
- 假设线性、连续变量且满足 complete recourse。
- 未实现可行性割（当子问题可能不可行时需补充 Farkas 射线相关逻辑）。
- 未加入整数主变量；若主问题含整数，会变成 MIP + Benders 框架。

## R17

可扩展方向：

- 加入多割（multi-cut）提升收敛速度。
- 为不可行子问题加入可行性割。
- 支持主问题整数变量（MILP Benders / Logic-based Benders）。
- 在随机规划中扩展为多场景 L-shaped 方法。
- 加入 cut 管理策略（老割删除、稳定化、信赖域等）。

## R18

`demo.py` 的源码级算法流（8 步）：

1. `build_demo_instance` 初始化 `c, A, b, h, T, W, q`，构造两阶段 LP 数据结构。
2. `benders_decomposition` 启动循环，初始割集为空，设置 `best_ub=+inf`。
3. `solve_master_problem` 用 `linprog` 解 `min c^T x + theta`，得到候选 `x^k, theta^k` 与 `LB`。
4. `solve_dual_subproblem` 在 `x^k` 下解对偶 LP `max pi^T(h-Tx^k)`，得到 `Q(x^k)` 与 `pi^k`。
5. 计算可行上界 `UB_candidate = c^T x^k + Q(x^k)`，更新全局 `best_ub`。
6. 计算 `cut_violation = Q(x^k)-theta^k` 与 `gap = best_ub - LB`，判断是否达到容差。
7. 若未收敛，用 `pi^k` 生成割 `theta >= (pi^k)^T h - (T^T pi^k)^T x`，加入主问题继续迭代。
8. 收敛后调用 `solve_extensive_form` 直接解整体 LP，并对比目标值以完成结果验证。

这 8 步完整映射了 Benders 分解在代码中的实际执行路径，而不是把 `linprog` 当作黑盒一行带过。

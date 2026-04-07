# 拉格朗日松弛

- UID: `MATH-0376`
- 学科: `数学`
- 分类: `组合优化`
- 源序号: `376`
- 目标目录: `Algorithms/数学-组合优化-0376-拉格朗日松弛`

## R01

拉格朗日松弛（Lagrangian Relaxation）用于把“难约束”吸收到目标函数中，通过乘子把原问题分解为更易求解的子问题，并通过对偶最小化获得上界（最大化原问题场景）。

它常用于整数规划、网络设计、排程、车辆路径、设施选址等组合优化问题，尤其适合“部分约束结构简单、部分约束耦合复杂”的模型。

## R02

本目录的 MVP 使用一个二元选择问题作为演示原型：

```text
maximize   sum_i v_i x_i
subject to sum_i w_i x_i <= W
           sum_i t_i x_i <= T
           x_i in {0, 1}
```

- `w` 约束保留为硬约束（子问题结构）
- `t` 约束作为被松弛的耦合约束

## R03

对第二条约束引入拉格朗日乘子 `lambda >= 0`，得到：

```text
L(x, lambda) = sum_i v_i x_i + lambda * (T - sum_i t_i x_i)
             = lambda*T + sum_i (v_i - lambda*t_i) x_i
```

对固定 `lambda`，只需解：

```text
max  sum_i (v_i - lambda*t_i) x_i
s.t. sum_i w_i x_i <= W, x_i in {0,1}
```

这是一个 0-1 背包子问题。

## R04

定义对偶函数：

```text
theta(lambda) = max_x L(x, lambda)
```

对偶问题：

```text
min_{lambda >= 0} theta(lambda)
```

在本最大化原问题里：
- `theta(lambda)` 是原问题最优值的上界（Upper Bound）
- 由可行解得到的目标值是下界（Lower Bound）

因此可用 `UB - LB` 衡量当前质量。

## R05

子问题求解采用动态规划（DP）实现的 0-1 背包：

- 状态：`dp[i][c]` 表示前 `i` 个物品、容量 `c` 下的最大改造后收益
- 转移：不选第 `i` 个 vs 选第 `i` 个
- 回溯：通过 `take` 表恢复 `x`

这部分完全开源可见，不依赖黑箱优化器。

## R06

`lambda` 更新使用子梯度法：

- 子梯度可取 `g = sum_i t_i x_i - T`（约束违反量）
- 迭代：`lambda <- max(0, lambda + step * g)`
- `step` 使用衰减步长，并在违反量符号频繁翻转时缩小尺度，减少震荡

这对应对偶问题 `min theta(lambda)` 的标准投影子梯度更新。

## R07

松弛解不一定满足被松弛约束，需要恢复可行解：

1. 若超时（`sum t_i x_i > T`），先按价值密度删除低质量物品。
2. 在两条约束都满足后，再进行贪心补充（尽量提高收益）。
3. 进行一次 1-1 交换局部搜索，尝试提升目标值。

该策略保证能持续产出原问题可行解，用于更新下界。

## R08

停止条件与本实现策略：

- 最多 `max_iter=120` 轮
- 若 `UB - LB` 非常小则提前停止
- 若步长已经很小且违反量接近 0，可提前停止

演示场景下默认跑满迭代，便于展示末尾迭代日志。

## R09

正确性要点：

- 对任意 `lambda>=0`，`theta(lambda)` 不低于原问题最优值（上界性质）
- 可行恢复解始终满足原约束，因此其目标值是下界
- 代码里包含简单边界校验：
  - 若 `exact_opt > best_dual_ub` 抛错
  - 若 `best_primal_lb > exact_opt` 抛错

## R10

复杂度（`n` 个物品，背包容量 `W`，迭代 `K`）：

- 子问题 DP：`O(nW)`
- 每轮迭代：`O(nW + n log n + n^2)`（DP + 修复 + 交换）
- 总体：`O(K * (nW + n^2))`

在本 MVP 的小规模数据（`n=14, W=28`）下可瞬时运行。

## R11

工程实现说明：

- 语言：Python 3
- 依赖：`numpy`
- 文件：
  - `demo.py`：完整可运行演示（含精确枚举对照）
  - `README.md`：方法说明与结果解释

`demo.py` 内还提供了 `brute_force_optimal`，用于小规模实例验证上下界逻辑。

## R12

运行方式：

```bash
cd Algorithms/数学-组合优化-0376-拉格朗日松弛
python3 demo.py
```

无需交互输入，直接打印数据规模、上下界、可行解和迭代日志。

## R13

输出字段解释：

- `Exact primal optimum (bruteforce)`：小规模精确最优值（仅用于验证）
- `Lagrangian best primal lower bound`：迭代过程中最优可行解值
- `Lagrangian best dual upper bound`：迭代过程中最优对偶上界
- `Duality gap`：`UB-LB`
- `Relative gap`：`(UB-LB)/max(|LB|,1)`

## R14

当前实例一次运行结果（固定数据、确定性输出）：

- 精确最优值：`102.0000`
- 最佳下界：`102.0000`
- 最佳上界：`103.0000`
- 绝对间隙：`1.0000`
- 相对间隙：`0.9804%`

说明该 MVP 在该样例上获得了最优可行值，并给出了接近的对偶上界。

## R15

结果解读：

- 最优可行值达到 `102`，与精确枚举一致，说明“松弛 + 修复”策略在该例上有效。
- 对偶上界停在 `103`，表明仍存在小的对偶间隙（离散问题常见）。
- 拉格朗日方法核心价值不在于直接给出最终最优，而在于快速提供高质量上界并引导可行搜索。

## R16

适用场景：

- 约束中存在“耦合约束”导致直接求解困难
- 去掉某些约束后可转成可高效求解的结构（背包、最短路、匹配、流等）
- 需要在有限时间内先获得稳定上下界，用于分支定界或启发式框架

## R17

局限与改进方向：

- 子梯度法可能震荡，收敛速度依赖步长策略
- 可行恢复策略会影响下界质量
- 可进一步引入：
  - 更稳健的步长规则（Bundle、Volume、Polyak 变体）
  - 多乘子/多约束联合松弛
  - 与分支定界（Branch-and-Bound）或列生成耦合

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `build_instance()` 构造固定二元组合优化实例（`v,w,t,W,T`）。
2. `lagrangian_subproblem()` 依据当前 `lambda` 计算改造收益 `v-lambda*t`。
3. `solve_knapsack_dp()` 用二维 DP 解松弛子问题，得到 `x_relaxed` 与 `theta(lambda)`。
4. 由 `sum(t*x_relaxed)-T` 计算子梯度（违反量）`violation`。
5. `repair_to_feasible()` 对 `x_relaxed` 先删后补，并做 1-1 交换，得到原问题可行解 `x_feasible`。
6. 用 `x_feasible` 更新下界 `best_primal_lb`，用 `theta(lambda)` 更新上界 `best_dual_ub`。
7. 按投影子梯度公式更新乘子：`lambda <- max(0, lambda + step*violation)`，并自适应收缩步长尺度。
8. 迭代结束后输出上下界、间隙、解向量与末尾迭代日志；并用 `brute_force_optimal()` 做小规模正确性核验。

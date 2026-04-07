# 次梯度方法

- UID: `MATH-0377`
- 学科: `数学`
- 分类: `凸优化`
- 源序号: `377`
- 目标目录: `Algorithms/数学-凸优化-0377-次梯度方法`

## R01

次梯度方法（Subgradient Method）用于求解**凸但可能不可导**的优化问题。  
当目标函数在某些点不可微（例如含 `L1` 范数、hinge loss、`max` 结构）时，经典梯度下降无法直接使用单一梯度，次梯度方法用“可行下降方向集合中的一个向量”继续迭代。

本条目给出一个最小可运行 MVP：
- 目标函数选为 `0.5*||x-c||^2 + lam*||x||_1`（凸、非光滑）；
- 次梯度更新完全手写实现，不调用黑盒优化器；
- 用闭式解（软阈值）做数值对照，验证实现正确性。

## R02

本目录中的问题定义：

- 输入：
  - 向量 `c in R^n`；
  - 正则系数 `lam >= 0`；
  - 初值 `x0 in R^n`；
  - 步长参数 `step_scale`、最大迭代 `max_iter`、停止阈值 `tol`。
- 目标函数：
  - `f(x) = 0.5 * ||x-c||_2^2 + lam * ||x||_1`。
- 输出：
  - 算法找到的最优候选 `x_best`；
  - 最后一次迭代点 `x_last`；
  - 历史记录 `[(iter, objective, subgrad_norm, step_norm), ...]`。

`demo.py` 内置三组固定测试样例，无需交互输入。

## R03

核心数学关系：

1. `L1` 正则项不可导，但存在次梯度：
   - `∂|x_i| = {sign(x_i)}` 当 `x_i != 0`；
   - `∂|x_i| = [-1, 1]` 当 `x_i = 0`。
2. 整体目标的一个次梯度可写为：
   - `g(x) = (x-c) + lam * s`，其中 `s_i in ∂|x_i|`。
3. 迭代更新：
   - `x_{t+1} = x_t - eta_t * g_t`；
   - 本实现步长 `eta_t = step_scale / sqrt(t)`。
4. 该目标有闭式最优解（软阈值）：
   - `x* = sign(c) * max(|c| - lam, 0)`。

## R04

算法高层流程：

1. 校验输入向量维度、有限性和超参数合法性。  
2. 初始化 `x <- x0`，并把当前点作为 `best`。  
3. 每轮选择一个 `L1` 次梯度（本实现对零点取 `0`）。  
4. 组合得到总次梯度 `g_t = (x-c) + lam*s`。  
5. 按 `eta_t = step_scale/sqrt(t)` 执行更新。  
6. 记录目标值、`||g_t||`、`||x_{t+1}-x_t||`。  
7. 若新目标更小则更新 `x_best`。  
8. 若步长变化极小则提前停止，否则继续到 `max_iter`。

## R05

核心数据结构：

- `HistoryItem = (iter, objective, subgrad_norm, step_norm)`：
  - `iter`：迭代编号；
  - `objective`：该轮点的目标函数值；
  - `subgrad_norm`：次梯度二范数；
  - `step_norm`：本轮更新向量二范数。
- `CaseConfig`（`dataclass`）：
  - 每个测试样例的 `name/c/lam/step_scale/max_iter`。
- `results: list[dict]`：
  - 汇总每个样例的目标 gap、参数误差、迭代次数。

## R06

正确性说明（本实现层面）：

- 次梯度表达式正确：
  - 平滑项导数为 `x-c`；
  - 非光滑项用 `lam*subgradient_l1(x)` 表达。  
- 更新规则正确：
  - 始终按 `x <- x - eta_t * g` 做一步法迭代。  
- 可验证性：
  - 对同一目标，使用软阈值闭式解 `x*` 作为真值对照；
  - 输出并检查 objective gap 与参数误差。  
- 鲁棒性：
  - 每轮检测非有限值并即时抛错，避免静默失败。

## R07

复杂度分析（向量维数 `n`，迭代轮数 `T`）：

- 单轮主要操作：
  - 计算次梯度 `O(n)`；
  - 计算目标值 `O(n)`；
  - 向量更新 `O(n)`。
- 单轮时间复杂度：`O(n)`。  
- 总时间复杂度：`O(Tn)`。  
- 空间复杂度：
  - 参数与中间向量 `O(n)`；
  - 全历史记录额外 `O(T)`。

## R08

边界与异常处理：

- `c/x0` 不是一维向量，或形状不一致：抛 `ValueError`。  
- `c/x0` 存在 `nan` 或 `inf`：抛 `ValueError`。  
- `lam < 0`、`step_scale <= 0`、`max_iter <= 0`、`tol <= 0`：抛 `ValueError`。  
- 迭代过程中若 `objective`、`subgrad_norm`、`step_norm` 出现非有限值：抛 `RuntimeError`。

## R09

MVP 取舍：

- 选择 `L1` 正则二次目标，而不是泛化到所有非光滑凸目标；
- 只实现最基础的确定性次梯度法，不引入投影、镜像下降或加速器；
- 不调用 `scipy.optimize.minimize` / `sklearn` 优化器求解主问题；
- 用闭式解做强对照，优先保证“可核验”和“源码透明”。

## R10

`demo.py` 主要函数职责：

- `check_vector`：检查一维性与有限值。  
- `objective_l1_quadratic`：计算目标函数值。  
- `soft_threshold`：计算闭式最优解。  
- `subgradient_l1`：给出 `||x||_1` 的一个合法次梯度。  
- `subgradient_method`：主迭代循环。  
- `print_history`：格式化打印前若干轮日志。  
- `build_cases`：生成可复现测试样例。  
- `run_case`：执行单例、对照闭式解、执行阈值检查。  
- `main`：批量运行所有样例并输出汇总。

## R11

运行方式：

```bash
cd Algorithms/数学-凸优化-0377-次梯度方法
uv run python demo.py
```

脚本无命令行参数、无交互输入。

## R12

输出字段说明：

- `iter`：当前迭代轮次。  
- `objective`：当前点目标值。  
- `||subgrad||`：当前次梯度范数。  
- `||step||`：本轮更新幅度。  
- `objective(best iterate)`：历史最优点对应目标值。  
- `objective(last iterate)`：最后迭代点目标值。  
- `objective(optimal closed-form)`：软阈值闭式最优值。  
- `best/last objective gap`：与闭式最优值差距。  
- `||x_best - x_star||_2`：参数向量误差。  
- `sparsity ratio`：近零元素占比。  
- `Summary`：全样例最大/平均相对目标 gap 与最大误差。

## R13

内置测试样例（`build_cases`）：

- `Low-dimensional mixed signs`：低维混合符号数据；
- `Sparse-inducing medium case`：含大量接近零分量，验证稀疏性；
- `Higher-dimensional random case`：更高维随机向量，验证可扩展性。

每个样例都会检查：
- `x_best` 全有限；
- 相对目标 gap 不超过阈值 `3e-3`。

## R14

关键超参数：

- `lam`：`L1` 强度；越大越稀疏。  
- `step_scale`：步长基准；越大下降更快但更易振荡。  
- `max_iter`：上限迭代次数；影响最终精度。  
- `tol`：步长停止阈值；越小通常需要更多迭代。

调参建议：
- 若目标震荡明显，减小 `step_scale`；
- 若 gap 收敛慢，增大 `max_iter`；
- 若想更稀疏，增大 `lam`。

## R15

与相关方法对比：

- 对比梯度下降：
  - 梯度下降要求可微；
  - 次梯度法可处理不可导点，但收敛通常更慢。  
- 对比近端梯度法（ISTA）：
  - ISTA 对 `L1` 问题往往更高效；
  - 次梯度法结构更朴素，教学和原型验证更直接。  
- 对比坐标下降：
  - 坐标下降在 Lasso 上通常更快；
  - 次梯度法不依赖坐标可分性，泛用性更强。

## R16

典型应用场景：

- 含 `L1` 正则的稀疏建模原型；
- 含 `max` / hinge 等非光滑损失的一阶求解 baseline；
- 凸优化课程中“不可导优化”入门演示；
- 更复杂算法（投影次梯度、镜像下降、近端法）上线前的对照实现。

## R17

可扩展方向：

- 增加 Polyak 步长或自适应步长策略；
- 增加迭代平均（ergodic averaging）改善稳定性；
- 增加约束集合投影，扩展为投影次梯度法；
- 替换目标为 hinge loss / TV 正则等其他非光滑凸问题；
- 增加 CSV 日志和收敛曲线可视化。

## R18

`demo.py` 源码级流程拆解（8 步）：

1. `main` 调用 `build_cases` 构造三组固定 `c/lam` 配置，保证可复现。  
2. `run_case` 为每个样例设 `x0=0`，调用 `subgradient_method` 进入主优化循环。  
3. `subgradient_method` 先做输入合法性检查，再初始化 `x`、`x_best` 与 `best_obj`。  
4. 每轮通过 `subgradient_l1(x)` 生成 `L1` 次梯度，并与平滑项梯度合成 `g_t=(x-c)+lam*s_t`。  
5. 按 `eta_t = step_scale/sqrt(t)` 做更新 `x_{t+1}=x_t-eta_t*g_t`，并记录目标值、`||subgrad||`、`||step||`。  
6. 若本轮目标更小则刷新 `x_best`；若出现非有限数值则立即报错退出。  
7. 迭代结束后，`run_case` 用 `soft_threshold` 计算闭式最优解 `x_star`，对比 objective gap 与参数误差。  
8. `main` 汇总全样例最大/平均相对 gap 与最大参数误差，并在全部检查通过后输出 `All checks passed.`。

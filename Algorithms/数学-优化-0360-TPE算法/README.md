# TPE算法

- UID: `MATH-0360`
- 学科: `数学`
- 分类: `优化`
- 源序号: `360`
- 目标目录: `Algorithms/数学-优化-0360-TPE算法`

## R01

TPE（Tree-structured Parzen Estimator）是贝叶斯优化家族中的一种“密度比”思路：
- 不直接拟合 `p(y|x)`；
- 而是把历史样本按目标值分成“好样本/坏样本”，分别拟合 `p(x|good)` 与 `p(x|bad)`；
- 通过最大化两者比值来挑选下一次评估点。

本目录的 MVP 聚焦 1 维连续变量最小化，目标是给出一个可运行、可追踪、无黑盒优化器依赖的 TPE 最小实现。

## R02

本实现求解的问题：
- 目标：
  - `min_{x in [l, u]} f(x)`，其中 `f` 是黑盒函数，仅可点查询。
- 输入：
  - 边界 `bounds=(l,u)`；
  - 总评估次数 `n_trials`；
  - 随机预热轮数 `n_startup`；
  - 分位参数 `gamma`；
  - 候选数 `n_candidates`。
- 输出：
  - 最优 `best_x, best_y`；
  - 全部观测 `x_obs, y_obs`；
  - 每轮日志（采样来源、阈值、密度比得分等）。

`demo.py` 默认优化一个 1D 多峰目标函数，无需交互输入。

## R03

核心数学关系（最小化场景）：

1. 阈值分割：
   - `y* = Quantile_gamma({y_i})`
   - `good = {x_i | y_i <= y*}`
   - `bad  = {x_i | y_i >  y*}`
2. 两个条件密度：
   - `l(x) = p(x | y <= y*)`
   - `g(x) = p(x | y >  y*)`
3. TPE 的候选打分：
   - 近似最大化 `l(x) / g(x)`。
4. 本实现的 Parzen 窗：
   - `p(x) = (1/n) * sum_i N(x | x_i, h^2)`；
   - `h` 采用 Silverman 风格带宽并做上下限裁剪。

直观上，`l(x)/g(x)` 高的区域表示“更像历史好解、又不像坏解”，因此更值得评估。

## R04

算法流程（MVP）：
1. 在边界内随机采样若干预热点，得到初始观测。  
2. 每轮按 `gamma` 分位将观测拆为 good/bad 两组。  
3. 分别对 good 与 bad 构建高斯 Parzen 密度估计。  
4. 从 `l(x)` 对应的混合高斯里采样候选点。  
5. 计算每个候选的 `l(x)/g(x)`。  
6. 选择得分最高候选作为下一评估点。  
7. 评估黑盒目标并更新历史最优。  
8. 达到预算后返回最优值与完整日志。

## R05

核心数据结构：
- `TrialRecord(dataclass)`：
  - `iter_id, source, x, y, best_x, best_y, y_star, score`；
- `TPEResult(dataclass)`：
  - `best_x, best_y, x_obs, y_obs, records`；
- `x_obs/y_obs`：
  - 历史样本值，分别用于下一轮的 good/bad 分组与密度估计。

## R06

正确性要点：
- 预热阶段保证可行：样本直接 `Uniform([l,u])` 生成。  
- 阈值分割正确：通过对 `y_obs` 排序后取前 `ceil(gamma*n)` 个作为 good。  
- 密度计算稳定：`g(x)` 分母有 `1e-12` 下限，避免除零。  
- 候选有效性：所有候选点都会 `clip` 回边界区间。  
- 最优记录单调：每轮执行 `best_y = min(best_y, y_next)`，历史最优不会变差。

## R07

复杂度分析（第 `t` 轮已有 `n_t` 个样本，候选数 `m`）：
- 分组排序：`O(n_t log n_t)`；
- KDE 打分：
  - `l(x)` 与 `g(x)` 评估都为向量化高斯和，约 `O(m * n_t)`；
- 每轮总时间：`O(n_t log n_t + m * n_t)`；
- 空间开销：
  - 观测历史 `O(n_t)`；
  - 向量化中间矩阵约 `O(m * n_t)`。

## R08

边界与异常处理：
- `bounds` 非法（长度不是 2、非有限值、`lower>=upper`）会抛 `ValueError`；
- `n_trials/n_startup/n_candidates/gamma` 非法会抛 `ValueError`；
- 空样本调用 Parzen 密度会抛 `ValueError`；
- 目标函数返回非有限值会抛 `RuntimeError`；
- 当 good/bad 样本不足时自动回退随机采样，避免不稳定 KDE。

## R09

MVP 取舍：
- 仅实现 1 维连续变量版本，优先清晰可读。  
- 仅使用高斯 Parzen + `l/g` 打分，不扩展树结构条件变量。  
- 不依赖 Optuna/Hyperopt 等框架，核心步骤全部源码可追踪。  
- 不做并行评估、早停、复杂约束处理。

## R10

`demo.py` 主要函数职责：
- `rugged_objective`：构造演示用多峰黑盒函数；
- `validate_bounds`：检查边界合法性；
- `silverman_bandwidth`：计算鲁棒带宽；
- `parzen_density`：计算高斯 Parzen 密度；
- `sample_from_parzen`：从 good 组混合高斯采样候选；
- `split_good_bad`：按 `gamma` 分位分组；
- `tpe_suggest`：根据 `l/g` 最大化给出下一点；
- `run_tpe`：主循环（提议、评估、更新）；
- `approximate_minimum`：网格近似参考最优；
- `print_iteration_table`：打印迭代摘要；
- `main`：组织实验与结果汇总。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0360-TPE算法
python3 demo.py
```

脚本不需要命令行参数，也不需要任何交互输入。

## R12

输出字段说明：
- 迭代表：
  - `iter`：迭代编号；
  - `source`：提议来源（`random/tpe/random-fallback`）；
  - `x, y`：本轮采样点与函数值；
  - `best_y`：截至当前的最优目标值；
  - `y_star`：good/bad 分割阈值；
  - `l/g`：候选点的密度比得分。
- 最终摘要：
  - `TPE best x/y`：算法找到的最优观测；
  - `Reference grid best x/y`：高密网格近似最优；
  - `Absolute error`：两者绝对误差；
  - `Pass loose check`：宽松正确性检查结果。

## R13

最小测试建议：
1. 默认多峰函数（已内置）：验证 TPE 在探索-利用平衡下收敛到较优区域；
2. 单峰凸函数（如 `(x-1.3)^2`）：验证快速定位能力；
3. 含噪目标（`f(x)+N(0,sigma^2)`）：验证带宽和 `gamma` 对噪声鲁棒性；
4. 极小预算（如 `n_trials=20`）与较大预算（如 `n_trials=200`）对比，观察样本效率变化。

## R14

关键参数与调优建议：
- `gamma`：good 集比例；
  - 小 `gamma` 更“贪心”，大 `gamma` 更“平滑”；
- `n_startup`：随机预热轮数；
  - 太小会导致早期 KDE 不稳，太大则降低模型驱动轮数；
- `n_candidates`：每轮候选规模；
  - 值越大，`l/g` 近似最大化越充分，但计算开销更高；
- `bandwidth`（自动估计）：
  - 过小会尖峰化，过大易过度平滑。

## R15

方法对比：
- 对比随机搜索：
  - 随机搜索不利用历史分布；
  - TPE 用 good/bad 密度比引导采样，样本效率通常更高。
- 对比 GP 贝叶斯优化：
  - GP 更偏全局后验建模，低维表现强；
  - TPE 在复杂/离散/条件空间常更灵活。
- 对比网格搜索：
  - 网格搜索在高分辨率时成本高；
  - TPE 在固定预算下能把更多评估投向潜在优区。

## R16

典型应用场景：
- 机器学习超参数优化（学习率、正则强度、结构超参）；
- 黑盒仿真参数寻优；
- 自动化实验中的低预算调参；
- 不可导、不可解析、仅能点查询的目标优化。

## R17

可扩展方向：
- 扩展到多维和树结构条件空间（接近原始 TPE 的完整形态）；
- 支持分类/离散变量的专用 Parzen 估计；
- 加入批量并行建议（一次输出多个候选点）；
- 引入约束处理（可行性密度或惩罚项）；
- 增加可视化（密度曲线、收敛轨迹）与实验报告落盘。

## R18

`demo.py` 的源码级算法流（8 步）：
1. `main` 固定边界、预算与随机种子，调用 `run_tpe`。  
2. `run_tpe` 先进行随机预热，积累初始 `(x_obs, y_obs)`。  
3. 每轮调用 `split_good_bad`：按 `gamma` 分位把历史样本拆为 good 与 bad，并得到阈值 `y_star`。  
4. `tpe_suggest` 对 good/bad 分别用 `silverman_bandwidth` 计算带宽，然后用 `parzen_density` 构建 `l(x)` 与 `g(x)`。  
5. 同函数里通过 `sample_from_parzen` 从 good 组混合高斯采样 `n_candidates` 个候选。  
6. 对候选逐点评估 `score=l(x)/max(g(x),1e-12)`，取最高分对应的 `x_next`；若分组不足则回退随机提议。  
7. 回到 `run_tpe`，评估黑盒 `y_next`，更新 `x_obs/y_obs` 与 `best_x/best_y`，写入 `TrialRecord`。  
8. 迭代结束后 `main` 再调用 `approximate_minimum` 做网格参考最优，对比误差并输出通过标志。

该流程没有调用现成 TPE 优化器，分组、密度估计、候选生成、密度比选择都在源码中逐步展开。

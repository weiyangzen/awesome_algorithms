# 粒子群优化 (PSO)

- UID: `MATH-0355`
- 学科: `数学`
- 分类: `群体智能`
- 源序号: `355`
- 目标目录: `Algorithms/数学-群体智能-0355-粒子群优化_(PSO)`

## R01

粒子群优化（Particle Swarm Optimization, PSO）是典型群体智能算法。  
它用一组“粒子”在搜索空间内协同移动，通过个体经验（`pbest`）和群体经验（`gbest`）共同引导，逐步逼近最优解。

本目录给出一个最小可运行 MVP：
- 目标函数选用连续优化常见基准 `Rastrigin`；
- 使用标准全局最优 PSO（global-best PSO）；
- 运行后打印迭代过程和最终收敛摘要。

## R02

问题定义（本实现）：
- 决策变量：`x in R^d`，并满足边界约束 `x_i in [L, U]`。
- 优化目标：最小化 `f(x)`，这里选择
  `f(x) = 10d + sum(x_i^2 - 10*cos(2*pi*x_i))`。
- 该函数全局最优点是 `x*=0`，最优值 `f(x*)=0`。

`demo.py` 默认 `d=5`，边界为 `[-5.12, 5.12]`，无需外部输入。

## R03

PSO 核心思想：
- 每个粒子维护当前位置 `x` 和速度 `v`；
- 每个粒子记住历史最佳位置 `pbest`；
- 全体粒子共享当前全局最佳位置 `gbest`；
- 每一轮由惯性项、个体认知项、社会学习项共同更新速度，然后更新位置。

这种机制本质上在“探索（exploration）”与“开发（exploitation）”之间做动态平衡。

## R04

标准更新公式（最小化）：
- 速度更新  
  `v_{i,t+1} = w_t * v_{i,t} + c1*r1*(pbest_i - x_{i,t}) + c2*r2*(gbest - x_{i,t})`
- 位置更新  
  `x_{i,t+1} = x_{i,t} + v_{i,t+1}`

其中：
- `w_t` 为惯性权重（本实现随迭代线性递减）；
- `c1/c2` 为认知/社会系数；
- `r1/r2 ~ U(0,1)` 为逐维随机数。

实现中还使用：
- 速度裁剪：`v in [-vmax, vmax]`；
- 位置裁剪：`x in [L, U]`。

## R05

高层伪代码：

```text
初始化粒子位置 x 和速度 v
计算每个粒子适应度，初始化 pbest 与 gbest
for t in 1..T:
  计算当前惯性权重 w_t
  for 每个粒子 i:
    采样随机向量 r1, r2
    更新速度 v_i
    更新位置 x_i，并做边界裁剪
  重新计算适应度
  若粒子当前解优于其 pbest，则更新 pbest
  在所有 pbest 中更新 gbest
  若达到目标精度则提前停止
输出 gbest 及收敛历史
```

## R06

时间复杂度：
- 设粒子数 `N`、维度 `D`、迭代轮数 `T`。
- 每轮主要开销：
  - 速度和位置更新：`O(ND)`；
  - 目标函数评估（向量化 Rastrigin）：`O(ND)`。
- 总时间复杂度：`O(TND)`。

这是 PSO 易于落地的重要原因之一：单轮运算规则简单、并行友好。

## R07

空间复杂度（MVP 实现）：
- `positions`、`velocities`、`pbest_positions`：各 `O(ND)`；
- `pbest_fitness`、临时适应度向量：`O(N)`；
- 全局历史 `history`：`O(T)`。

因此总空间约为 `O(ND + T)`，主项通常是 `O(ND)`。

## R08

默认参数含义（`PSOConfig`）：
- `dimensions=5`：问题维度。
- `swarm_size=40`：粒子数。
- `max_iters=200`：最大迭代轮数。
- `lower_bound/upper_bound`：搜索区间。
- `inertia_start=0.90`, `inertia_end=0.40`：惯性权重退火区间。
- `cognitive_coeff=1.80`：个体学习强度 `c1`。
- `social_coeff=1.80`：群体学习强度 `c2`。
- `velocity_clip_ratio=0.20`：速度上限比例，`vmax = ratio*(U-L)`。
- `target_fitness=1e-8`：达到该目标值则提前停止。
- `seed=42`：随机种子，保证可复现。

## R09

实现假设与边界：
- 仅覆盖连续变量、盒约束（box constraints）场景。
- 目标函数要求可数值评估，且返回有限实数。
- 未实现复杂约束处理（可行性规则、罚函数、多目标等）。
- 这是“教学可审计 MVP”，不追求工业级并行优化框架。

## R10

运行方式（非交互）：

```bash
uv run python demo.py
```

程序会：
- 输出固定间隔迭代日志（当前 `global_best`）；
- 打印最终最优值、到零向量距离、收敛历史统计。

## R11

输出解读建议：
- `global_best` 应整体下降，且通常后期趋于稳定。
- `best fitness` 越接近 `0` 越好（Rastrigin 的理论最优是 0）。
- `distance to zero vector` 越小表示解越接近已知最优点。
- `history monotonic non-increasing` 预期为 `True`，因为记录的是“历史最好值”。

## R12

正确性快速检查：
- 检查 `best fitness >= 0`（Rastrigin 理论下界为 0）。
- 检查 `history` 单调不增。
- 检查最终位置各维都在给定边界内。
- 更换多个 `seed` 时应观察到“多数运行接近 0，但速度有随机波动”。

## R13

常见失败模式与症状：
- `swarm_size` 太小：容易早熟收敛到局部最优。
- `inertia` 过大且不衰减：粒子震荡明显，难以收敛。
- `c1/c2` 过大：速度频繁触发裁剪，搜索不稳定。
- `velocity_clip_ratio` 过小：移动步幅不足，收敛很慢。
- 边界过窄：真实最优可能不在可行域内，结果被边界限制。

## R14

与常见优化方法对比：
- 对比梯度法：PSO 不依赖梯度，适合不可导或噪声目标，但样本效率通常较低。
- 对比遗传算法（GA）：两者都属于群体搜索；PSO 参数更少、状态更新更连续。
- 对比随机搜索：PSO 能利用历史最优信息形成引导，通常更快逼近高质量解。

## R15

工程实践建议：
- 先固定随机种子做可复现调参，再做多种子统计。
- 优先调整 `swarm_size` 与 `max_iters`，再微调 `c1/c2/w`。
- 对不同尺度变量做归一化，避免某些维度主导更新。
- 记录每轮最优值与耗时，便于判断是“卡局部最优”还是“算力不足”。

## R16

可扩展方向：
- 多目标 PSO（MOPSO）。
- 约束 PSO（罚函数、可行解优先规则）。
- 自适应参数 PSO（动态 `c1/c2/w`）。
- 邻域拓扑 PSO（环形/局部通信）替代全局 `gbest`，提升多峰问题鲁棒性。
- 与局部优化器混合（PSO 全局探索 + 梯度法局部精修）。

## R17

MVP 依赖与自包含性：
- 依赖：`numpy`
- 输入：无外部数据文件、无交互输入
- 输出：标准输出日志与摘要

目录内 `README.md + demo.py + meta.json` 即可独立复现实验流程。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main()` 构建 `PSOConfig` 并打印实验配置（维度、粒子数、迭代轮数、边界、随机种子）。  
2. `run_pso()` 先调用 `validate_config()` 做参数合法性检查。  
3. 在 `run_pso()` 中随机初始化 `positions` 与 `velocities`，并调用 `rastrigin_population()` 评估初始适应度。  
4. 根据初始适应度建立 `pbest_positions / pbest_fitness`，再选出 `gbest_position / gbest_fitness`。  
5. 每轮迭代先通过 `inertia_weight()` 计算当前惯性权重 `w_t`，再生成随机矩阵 `r1/r2`。  
6. 用速度公式组合惯性项、认知项、社会项，更新 `velocities`；随后更新 `positions`，并进行速度/位置裁剪。  
7. 重新评估适应度，按“更优则替换”规则更新各粒子 `pbest`，并在全体 `pbest` 中刷新 `gbest`；将 `gbest_fitness` 追加到 `history`。  
8. 达到 `target_fitness` 则提前停止；返回结果后由 `main()` 输出最终最优值、到理论最优点距离和收敛历史检查。  

以上流程是手写 PSO 主循环，不依赖第三方黑箱优化器调用。

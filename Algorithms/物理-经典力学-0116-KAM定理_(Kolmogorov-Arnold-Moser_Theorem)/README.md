# KAM定理 (Kolmogorov-Arnold-Moser Theorem)

- UID: `PHYS-0116`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `116`
- 目标目录: `Algorithms/物理-经典力学-0116-KAM定理_(Kolmogorov-Arnold-Moser_Theorem)`

## R01

KAM 定理讨论的是：在“近可积哈密顿系统”中，加入足够小的扰动后，并非所有不变环面都会被破坏，满足非共振（通常用 Diophantine 条件）的一大部分环面会保留下来。

本条目采用“数值代理实验”而非形式化证明：
- 用标准映射（Chirikov standard map）作为经典近可积模型；
- 通过旋转数曲线平滑度与有限时间李雅普诺夫指数，观察从准周期到混沌的过渡；
- 展示随扰动强度 `K` 增大，规则轨道比例下降、混沌指示量上升的趋势。

## R02

KAM 定理的典型背景形式：

`H(I, theta) = H0(I) + epsilon * H1(I, theta)`。

当 `epsilon` 足够小、`H0` 满足非退化扭转条件、频率满足适当非共振条件时，很多不变环面在扰动后仍存在，仅发生形变。

物理含义：
- 保留下来的环面阻止全局相空间完全混合；
- 这解释了“同一个系统里规则运动和混沌运动可长期共存”的现象。

## R03

MVP 选用标准映射：

- `p_{n+1} = p_n + K sin(x_n)`
- `x_{n+1} = (x_n + p_{n+1}) mod 2pi`

说明：
- `K` 是扰动强度（可理解为踢转子受迫强度）；
- `K -> 0` 时更接近可积，通常保留更多 KAM 型不变结构；
- `K` 增大后，共振重叠增强，混沌区域扩展。

## R04

本实现的理论-数值桥接：
1. 用离散辛映射替代连续哈密顿流，保留“面积守恒/辛结构”核心特征。  
2. 用初值族 `(x0, p0)` 扫描相空间。  
3. 对每条轨道估计旋转数（长时平均角增量）。  
4. 对每条轨道估计有限时间李雅普诺夫指数（FTLE）。  
5. 把统计量汇总为 `regular_fraction`、`rotation_roughness` 等可比较指标。

这属于 KAM 现象的“计算证据”，不是数学证明。

## R05

`demo.py` 默认参数：
- 扰动强度：`K = (0.20, 0.80, 1.40)`
- 初值数：`n_initial_conditions = 72`
- 演化步数：`n_steps = 4000`
- 舍弃前期：`burn_in = 800`
- 初始动量区间：`p0 in [-pi, pi]`
- 初始角度：按黄金比例共轭数生成确定性低相关序列
- 规则轨道阈值：`lyapunov < 0.02`

输出含两部分：
- `summary_by_K`：每个 `K` 的聚合指标；
- `per_ic_sample`：每个 `K` 的部分单轨道样本。

## R06

代码结构（`demo.py`）：
- `KAMConfig`：集中管理实验参数
- `standard_map_step`：标准映射一步更新
- `generate_initial_conditions`：构造初值族
- `estimate_rotation_numbers`：矢量化估计旋转数
- `finite_time_lyapunov`：基于雅可比连乘求 FTLE
- `rotation_curve_roughness` / `plateau_fraction`：旋转数曲线诊断
- `analyze_single_k`：单个 `K` 的轨道与统计汇总
- `run_kam_proxy`：遍历全部 `K`
- `main`：打印结果并做趋势断言

## R07

伪代码：

```text
for K in K_values:
  (x0_i, p0_i)_{i=1..N} = deterministic ensemble

  # rotation number
  for step in 1..n_steps:
    p <- p + K*sin(x)
    x <- (x + p) mod 2pi
    if step > burn_in:
      lift_sum += p
  rot_i = lift_sum_i / ((n_steps-burn_in)*2pi)

  # FTLE
  for each orbit i:
    iterate tangent vector v_{n+1} = J(x_n) v_n
    renormalize and accumulate log growth
    lyap_i = mean(log ||J v||)

  aggregate:
    regular_fraction = mean(lyap_i < threshold)
    rotation_roughness = mean(abs(second_diff(sort(rot_i by p0))))
```

## R08

复杂度分析（`M = len(K_values)`, `N = n_initial_conditions`, `T = n_steps`）：
- 时间复杂度：`O(M * N * T)`
  - 旋转数估计：`O(M * N * T)`（矢量化常数较小）
  - FTLE 估计：`O(M * N * T)`
- 空间复杂度：`O(N)` 到 `O(M*N)`（取决于是否保留全量轨道表）

对当前默认规模（`M=3, N=72, T=4000`）可在普通 CPU 上快速运行。

## R09

数值稳定性和可复现性策略：
- 初值使用确定性序列，避免随机波动导致结论不稳定；
- FTLE 每步对切向量归一化，防止溢出/下溢；
- 旋转数在 `burn_in` 之后统计，减弱瞬态影响；
- 使用多个 `K` 做趋势比较，而非依赖单点结论。

## R10

与相关建模方式对比：
- KAM 严格证明：要求函数空间正则性、非退化与小除数估计，难以直接在 MVP 完整实现。
- 本条目方法：保留辛映射核心结构，用可计算指标观测“环面保留/破碎”趋势。
- 全连续系统高精度辛积分：物理解释更直接，但工程实现更长；标准映射更适合作为最小教学入口。

## R11

参数调优建议：
- 想看更明显的 KAM 保留：减小 `K`（如 `0.05~0.3`）。
- 想看混沌增强：增大 `K`（如 `>1`）。
- 想减少统计噪声：增大 `n_steps` 与 `n_initial_conditions`。
- 若运行时间受限：先减小 `N`，再减小 `T`，并保留 `burn_in/T` 的相对比例。

## R12

实现细节说明：
- 旋转数使用“lifted 增量平均”，而不是直接对 `x mod 2pi` 平均，避免角变量环绕带来的偏差；
- FTLE 的雅可比矩阵来自映射显式导数，不依赖外部黑盒求导；
- `rotation_roughness` 用二阶差分平均绝对值近似曲线光滑度；
- `rotation_plateau_fraction` 用局部斜率阈值统计平台段比例，辅助观察共振台阶结构。

## R13

运行方式：

```bash
cd "Algorithms/物理-经典力学-0116-KAM定理_(Kolmogorov-Arnold-Moser_Theorem)"
uv run python demo.py
```

或在项目根目录执行：

```bash
uv run python Algorithms/物理-经典力学-0116-KAM定理_(Kolmogorov-Arnold-Moser_Theorem)/demo.py
```

脚本无交互输入。

## R14

输出解读：
- `median_lyapunov` / `p90_lyapunov`：越大表示指数发散越明显，混沌倾向越强；
- `regular_fraction`：低于阈值的轨道占比，越大表示规则区占比越高；
- `rotation_roughness`：旋转数曲线粗糙度，越大通常表示环面结构越破碎；
- `rotation_plateau_fraction`：近平台局部斜率比例，反映共振台阶特征。

典型预期：随 `K` 增大，`median_lyapunov` 上升、`regular_fraction` 下降、`rotation_roughness` 上升。

## R15

常见问题排查：
- 若断言失败（趋势不明显）：
  - 增大 `n_steps`（如 6000+）；
  - 增加 `n_initial_conditions`；
  - 拉开 `K` 取值间距（如 `0.15, 0.8, 1.6`）。
- 若运行太慢：
  - 先把 `n_steps` 调到 `2000` 做冒烟测试；
  - 再逐步恢复高精度配置。
- 若结果波动：
  - 保持初值生成方式不变，避免引入随机种子差异。

## R16

可扩展方向：
- 计算 Greene residue 或频率图谱，做更细粒度环面破裂检测；
- 增加相空间网格可视化（Poincare 截面散点图）；
- 把标准映射扩展到更一般扭转映射族，比较 KAM 阈值变化；
- 引入连续系统辛积分（如双摆近可积区）进行交叉验证。

## R17

适用边界与限制：
- 本实现是 KAM 现象的数值代理，不是定理证明；
- FTLE 为有限时间估计，不等于无限时 Lyapunov 指数；
- 旋转数曲线指标依赖初值采样与阈值选取；
- 结论用于教学与算法验证，不直接替代严格数学判定。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main()` 构造 `KAMConfig`，固定 `K` 集合、轨道数、步数和阈值。  
2. `run_kam_proxy()` 调用 `generate_initial_conditions()` 生成 `(x0, p0)` 初值族。  
3. 对每个 `K` 调 `analyze_single_k()`，先用 `estimate_rotation_numbers()` 做矢量化轨道推进并计算旋转数。  
4. `estimate_rotation_numbers()` 在每一步执行标准映射更新；仅在 `burn_in` 之后累计 lifted 角增量并做时间平均。  
5. `analyze_single_k()` 对每条轨道调用 `finite_time_lyapunov()`；该函数按显式雅可比矩阵更新切向量、归一化并累计 `log` 增长率。  
6. 对按 `p0` 排序后的旋转数，计算 `rotation_curve_roughness()`（二阶差分）与 `plateau_fraction()`（小斜率比例）。  
7. 汇总 `median_lyapunov`、`regular_fraction`、`rotation_span` 等统计量，构成 `summary_by_K`。  
8. `main()` 打印聚合表和样本轨道表，并断言“大 `K` 相比小 `K`：Lyapunov 更高、规则占比更低、粗糙度更高”。

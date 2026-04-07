# 玻尔兹曼分布 (Boltzmann Distribution)

- UID: `PHYS-0031`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `31`
- 目标目录: `Algorithms/物理-统计力学-0031-玻尔兹曼分布_(Boltzmann_Distribution)`

## R01

玻尔兹曼分布描述热平衡下“系统处于某微观能量态”的概率规律。  
对离散能级 `E_i`，其平衡概率为：

`p_i = exp(-beta E_i) / Z`, 其中 `beta = 1/(k_B T)`，`Z = sum_j exp(-beta E_j)`。

直观上：
- 温度低（`beta` 大）时，系统更偏向低能态；
- 温度高（`beta` 小）时，不同能级的占据更均匀；
- 同温度下，能量越高，概率指数级下降。

## R02

本条目把玻尔兹曼分布落地为“可执行统计任务”：

- 输入：温度 `T`、玻尔兹曼常数 `k_B`、离散能级表 `E_0...E_{n-1}`、采样数 `n_samples`、随机种子 `seed`；
- 输出：
  - 理论概率 `p_i`；
  - 按 `p_i` 抽样得到的经验频率 `hat(p_i)`；
  - 配分函数 `Z`、平均能量 `⟨E⟩`；
  - 从样本反推 `beta_hat`/`T_hat` 的估计值；
  - 误差指标（TV、RMSE、KL）和自动通过/失败结论。

`demo.py` 无交互，直接运行并输出完整报告。

## R03

核心公式（离散情形）：

1. 概率分布：
   `p_i = exp(-beta E_i) / Z`
2. 配分函数：
   `Z = sum_i exp(-beta E_i)`
3. 对数配分函数：
   `log Z = log(sum_i exp(-beta E_i))`
4. 平均能量：
   `⟨E⟩ = sum_i p_i E_i`
5. 自由能：
   `F = -k_B T log Z`

本实现用 `scipy.special.logsumexp` 计算 `log Z`，避免指数上溢/下溢。

## R04

温度与分布的关系：

- `T -> 0+`（`beta -> +inf`）时，概率集中到最低能级（基态）；
- `T -> +inf`（`beta -> 0+`）时，`p_i -> 1/n`（近似均匀）；
- `p_i/p_j = exp[-beta(E_i-E_j)]`，只与能量差有关。

这也解释了一个工程结论：若能级整体平移 `E_i + C`，概率不变（归一化会抵消常数因子）。

## R05

MVP 算法流程（离散能级版）：

1. 给定能级 `E` 与温度，计算 `beta`；
2. 用公式算理论概率向量 `p`；
3. 按 `p` 进行多次随机抽样，得到状态样本；
4. 统计样本频率 `hat(p)` 与样本平均能量 `hat(E)`；
5. 通过 `hat(E)` 求解 MLE 方程，反推出 `beta_hat`（再得 `T_hat`）；
6. 输出误差指标并执行断言验证。

优势：流程短、可复现、每一步都可解释，不依赖黑箱物理库。

## R06

微型算例（`k_B=1, T=1, E=[0,1,2]`）：

- `beta=1`
- `Z = 1 + e^{-1} + e^{-2} ≈ 1.5032`
- 概率约为：
  - `p0 ≈ 0.6652`
  - `p1 ≈ 0.2447`
  - `p2 ≈ 0.0900`

可见低能态占据显著更高，且满足指数衰减规律。

## R07

玻尔兹曼分布的重要性：

- 是正则系综的基础概率模型；
- 连接微观能级与宏观热力学量（`Z, F, U, S`）；
- 在化学反应、材料相变、退火优化、马尔可夫链采样中广泛出现；
- 是费米-狄拉克/玻色-爱因斯坦统计的经典极限参考。

## R08

理论推导的工程化摘要：

1. 系统与热库接触，温度固定为 `T`；
2. 态权重由能量惩罚决定：`w_i = exp(-beta E_i)`；
3. 归一化得到概率 `p_i = w_i/sum_j w_j`；
4. 任意可观测量期望都可写成 `sum_i p_i A_i`；
5. 从 `Z` 的导数可恢复热力学量并进行一致性校验。

`demo.py` 对应实现了上述链路中的概率构造、采样、统计与反演。

## R09

适用条件与边界：

- 适用于：经典统计、热平衡、固定温度、可枚举离散能级场景；
- 不适用于：
  - 强非平衡动力学过程；
  - 严格量子占据限制主导（需 Fermi/Bose 分布）；
  - 能级模型定义不合理（漏态/重度并计错误）。

如果系统偏离这些前提，经验占据会显著偏离玻尔兹曼形式。

## R10

本实现的正确性验证设计：

1. 数学层：`p_i` 总和应为 1；高能态概率不应高于低能态（在能级升序下）。
2. 统计层：样本频率 `hat(p_i)` 应逼近理论 `p_i`。
3. 物理层：样本均能量应接近理论均能量。
4. 反演层：由样本均能量求得的 `T_hat` 应接近设定 `T`。
5. 指标层：TV、RMSE、KL 同时受控，避免单指标偶然“过关”。

## R11

误差来源与数值稳定性：

- 有限样本噪声：`hat(p_i)-p_i` 量级约 `O(1/sqrt(n_samples))`；
- 稀有高能态：样本计数少，局部相对误差可能大；
- 指数计算风险：`exp(-beta E)` 可能下溢；
- 反演求根误差：若括区不充分，`beta_hat` 可能失败。

对应措施：
- 提高样本量并固定随机种子；
- 用 `logsumexp` 稳定归一化；
- 用 `brentq` 且动态扩展括区保证求根稳定。

## R12

复杂度分析（`L=能级数`, `N=n_samples`）：

- 计算理论概率：`O(L)`；
- 抽样 `N` 次：`O(N)`；
- 统计频率：`O(N)`；
- 反演 `beta_hat`：每次函数评估 `O(L)`，总计约 `O(L * iters)`；
- 总体：`O(N + L * iters)`，通常由 `N` 主导。

空间复杂度：
- 状态样本 `O(N)`；
- 概率与统计表 `O(L)`。

## R13

本目录验收保证：

- 近似比保证：N/A（不是组合优化问题）。
- 可执行保证：
  - `demo.py` 无交互、可重复；
  - 输出理论/经验逐级对照表；
  - 输出 `beta_hat`、`T_hat` 与误差指标；
  - 若误差超阈值，抛出 `AssertionError`；
  - 通过时打印 `All checks passed.`。

## R14

常见失效模式与排查建议：

- `n_samples` 太小：增大样本量到 `>= 5e4`；
- 能级顺序错误：先按升序整理 `energy_levels`；
- 温度或 `k_B` 非正：立即拒绝输入；
- 把“能级平移不变性”误写成“能级缩放不变性”（后者错误）；
- 只看一种误差指标：建议同时看 TV、RMSE、KL 与温度反演误差。

## R15

`demo.py` 结构说明：

- `BoltzmannConfig`：参数定义与合法性检查；
- `boltzmann_probabilities`：按公式计算 `p_i`、`Z`、`logZ`；
- `sample_states`：按离散分布抽样状态；
- `empirical_probabilities`：统计样本计数与频率；
- `estimate_beta_mle`：通过 `⟨E⟩_model(beta)=⟨E⟩_sample` 求根得到 `beta_hat`；
- `build_report`：汇总 DataFrame 和指标；
- `run_checks`：断言阈值；
- `main`：主流程串联与打印。

## R16

可扩展方向：

- 连续能谱：把求和换成积分（如密度状态 `g(E)`）；
- 多参数模型：加入简并度 `g_i`，概率改为 `p_i ∝ g_i exp(-beta E_i)`；
- 多温度扫描：生成 `T`-`U`、`T`-`F` 曲线；
- 与 Metropolis 采样联动：从“直接抽样已知分布”扩展到“未知归一化常数采样”；
- 与 Ising/Potts 模型耦合，研究相变附近的占据变化。

## R17

运行方式（无交互）：

```bash
cd Algorithms/物理-统计力学-0031-玻尔兹曼分布_(Boltzmann_Distribution)
uv run python demo.py
```

预期输出：
- 每个能级的理论概率与经验频率对照；
- `beta_true/beta_hat`、`temperature_true/temperature_hat`；
- `TV`、`RMSE`、`KL(emp||theory)`；
- 最后一行 `All checks passed.`（若未通过则抛异常）。

## R18

源码级算法流程拆解（`demo.py` 主链路，8 步）：

1. `BoltzmannConfig.validate()` 检查 `T>0`、`k_B>0`、样本量与能级数组合法且升序。  
2. `beta_from_temperature` 计算 `beta = 1/(k_B T)`。  
3. `boltzmann_probabilities` 用 `-beta*E` 构造对数权重，借助 `logsumexp` 得到 `logZ`，再恢复 `p_i`。  
4. `sample_states` 使用 `numpy.random.Generator.choice` 按 `p_i` 抽取 `N` 个状态索引。  
5. `empirical_probabilities` 通过 `bincount` 得到计数和经验概率 `hat(p_i)`，并计算经验均能量。  
6. `estimate_beta_mle` 设定方程 `⟨E⟩_model(beta)-⟨E⟩_sample=0`，动态扩括区后调用 `scipy.optimize.brentq` 求 `beta_hat`。  
7. `build_report` 汇总分层对照表与指标：`TV`、`RMSE`、`KL`、`Z/logZ`、`T_hat`。  
8. `run_checks` 执行阈值断言（温度误差、均能量误差、分布误差），通过后在 `main` 打印 `All checks passed.`。  

第三方库使用说明：
- `numpy` 负责数组、抽样、统计；
- `pandas` 仅用于结果表格化输出；
- `scipy` 只用于 `logsumexp`（数值稳定）和 `brentq`（一维求根）。

玻尔兹曼分布的核心构造、估计和验证步骤都在源码中逐步展开，不是黑箱调用。

# 拟Monte Carlo方法

- UID: `MATH-0139`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `139`
- 目标目录: `Algorithms/数学-数值分析-0139-拟Monte_Carlo方法`

## R01

本条目实现“拟 Monte Carlo（Quasi-Monte Carlo, QMC）积分”的最小可运行版本，核心任务是：
- 在单位超立方体 `[0,1]^d` 上估计积分 `I=\int f(x)dx`；
- 对比普通 Monte Carlo（伪随机采样）与拟 Monte Carlo（低差异序列采样）的误差收敛表现；
- 给出可直接运行、可复现实验结果的 `demo.py`。

## R02

问题定义（数值计算版）：
- 输入：
  - 维度 `d`；
  - 样本数 `n`；
  - 重复次数 `R`（用于统计误差）；
  - 被积函数 `f: [0,1]^d -> R`。
- 输出：
  - MC 与 QMC 的积分估计值；
  - 每个 `n` 下的均方根误差（RMSE）与估计标准差；
  - 对数坐标拟合的经验收敛斜率（`log(error)` 对 `log(n)` 的线性拟合斜率）。

## R03

数学基础：

1. Monte Carlo 估计
- `I_n = (1/n) * sum_{i=1}^n f(U_i)`，其中 `U_i` 为独立均匀随机样本；
- 在常见条件下，RMSE 通常呈 `O(n^{-1/2})` 量级。

2. 拟 Monte Carlo 思路
- 用低差异点集 `{x_i}` 代替独立随机点，使样本在 `[0,1]^d` 上覆盖更均匀；
- 对有界变差函数，Koksma-Hlawka 不等式给出误差上界：
  `|I - (1/n)sum f(x_i)| <= V_HK(f) * D_n^*`，
  其中 `D_n^*` 为星差异度；低差异序列可使 `D_n^*` 比随机采样更小。

3. 随机化 QMC
- 为得到统计误差估计，常对低差异序列进行随机化（如 Sobol scramble 或随机平移）；
- 这样可在保留低差异结构的同时进行多次重复并计算 RMSE。

## R04

MVP 算法总览：
1. 固定实验维度 `d`、样本规模序列 `n=2^k`、重复次数 `R`。
2. 定义可解析积分的可分离函数 `f(x)=\prod_j exp(-x_j)*(1+0.1*cos(2\pi x_j))`。
3. 根据解析公式计算真值 `I_exact`。
4. 对每个 `n`：
   - 做 `R` 次 MC 估计；
   - 做 `R` 次 QMC 估计（优先 Sobol，缺失 SciPy 时回退 Halton+随机平移）。
5. 汇总每个 `n` 的 RMSE、均值、标准差。
6. 在 `log(error)-log(n)` 空间拟合经验收敛斜率并打印结论。

## R05

核心数据结构：
- `numpy.ndarray`：
  - `points: shape=(n,d)` 采样点；
  - `values: shape=(n,)` 函数值；
  - `estimates: shape=(R,)` 多次重复估计值。
- `LevelMetrics`（`dataclass`）：
  - `n`、`exact`、`mc_mean`、`qmc_mean`；
  - `mc_std`、`qmc_std`；
  - `mc_rmse`、`qmc_rmse`。

## R06

正确性要点：
- MC 部分是经典样本均值积分估计，代码中直接实现 `mean(f(points))`；
- QMC 部分使用低差异序列保证更均匀覆盖，减少“聚团”采样导致的方差；
- 采用随机化 QMC（scramble 或随机平移）后可重复试验，RMSE 统计可直接与 MC 对比；
- 被积函数真值由解析公式给出，误差评估不是“相对比较”而是“对真实值比较”。

## R07

复杂度分析：
- 单次估计（MC 或 QMC）均需 `n` 次函数评估，时间 `O(n*d)`；
- 每个样本规模做 `R` 次重复，总时间 `O(R*n*d)`；
- 若样本规模列表长度为 `L`，总体约 `O(R*d*sum(n_l))`；
- 空间主要为单批采样点 `O(n*d)`。

## R08

边界与异常处理：
- `n <= 0`、`d <= 0`、`R <= 0` 直接抛 `ValueError`；
- QMC-Sobol 路径要求 `n` 是 2 的幂（`random_base2` 约束）；
- 函数输入维度不匹配（不是二维数组）抛 `ValueError`；
- 结果出现非有限值（`nan/inf`）抛 `ValueError`。

## R09

MVP 取舍：
- 优先“小而诚实”的对比实验：一类函数、一组维度、一组样本规模；
- 优先使用 `numpy`，并在可用时使用 `scipy.stats.qmc.Sobol`；
- 不引入大型实验框架，不做图形绘制，直接输出表格即可复核；
- 保留 Halton 回退路径，避免 SciPy 缺失时脚本不可运行。

## R10

`demo.py` 函数职责：
- `integrand`：向量化被积函数；
- `exact_integral`：给出解析真值；
- `mc_estimate_once`：单次 MC 积分估计；
- `qmc_estimate_once`：单次 QMC 积分估计；
- `halton_sequence`/`radical_inverse`：无 SciPy 时生成低差异序列；
- `evaluate_level`：对固定 `n` 汇总 MC/QMC 统计量；
- `fit_loglog_rate`：拟合经验收敛斜率；
- `main`：组织实验并打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0139-拟Monte_Carlo方法
python3 demo.py
```

脚本无需交互输入，会直接输出每个样本规模下的 MC/QMC 误差统计及经验收敛斜率。

## R12

输出解读：
- `MC_RMSE` / `QMC_RMSE`：相对真值的均方根误差，越小越好；
- `MC_STD` / `QMC_STD`：重复实验估计值的标准差；
- `MC_MEAN` / `QMC_MEAN`：重复实验均值估计；
- `log-log slope`：经验收敛速率，数值越负表示误差下降越快。

典型现象：
- MC 斜率通常接近 `-0.5`；
- QMC 在光滑函数上常优于 MC，斜率通常更接近 `-1`（实际受维度、函数光滑性、随机化策略影响）。

## R13

建议最小测试集：
- 正常路径：`d=8, n=2^5..2^12, R=16`，观察 MC/QMC 差异；
- 参数错误：`d=0`、`n=0`、`R=0` 应抛异常；
- Sobol 约束：当启用 Sobol 且 `n` 非 2 的幂时应抛异常；
- 数值稳定性：检查所有输出均为有限数。

## R14

可调参数：
- `dimension`：维度；
- `n_values`：样本规模列表；
- `repeats`：重复次数；
- `base_seed`：随机种子基值。

调参建议：
- 先用较小 `d` 和 `n` 确认流程正确；
- 再增大 `n` 观察收敛斜率是否符合理论预期；
- 若维度很高，QMC 优势可能减弱，可增加随机化重复次数提升统计稳定性。

## R15

方法对比：
- 与普通 MC：
  - MC 简单、通用、理论成熟，但方差收敛慢（约 `n^{-1/2}`）；
  - QMC 通过“均匀覆盖”降误差，常在中低维光滑问题更高效。
- 与确定性求积（如高斯求积、稀疏网格）：
  - 确定性求积在低维可非常高效；
  - QMC 更适合中高维且结构较平滑的积分任务。

## R16

应用场景：
- 金融工程中多因子期权定价与风险度量；
- 贝叶斯推断中的高维期望估计；
- 不确定性量化（UQ）中的参数空间积分；
- 计算物理中高维统计量估计。

## R17

后续扩展方向：
- 引入 Brownian bridge / PCA 变量重排，提升金融路径问题中的 QMC 效果；
- 加入 Owen scrambling 等更系统的随机化策略；
- 增加控制变量、重要性采样，与 QMC 组合降方差；
- 增加可视化（误差曲线）与单元测试。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 固定 `d/n_values/R/seed`，并调用 `exact_integral(d)` 得到解析真值。  
2. `evaluate_level` 对每个样本规模 `n` 做 `R` 次重复，每次分别调用 MC 与 QMC 单次估计。  
3. `mc_estimate_once` 用 `numpy.random.default_rng(seed).random((n,d))` 生成伪随机点并计算 `mean(f(x))`。  
4. `qmc_estimate_once` 先判断可用后端：若有 SciPy，走 Sobol；否则走 Halton 回退。  
5. Sobol 路径中，`scipy.stats.qmc.Sobol(d, scramble=True, seed=...)` 构造序列生成器；`random_base2(m)` 生成 `2^m=n` 个点。  
6. Halton 回退路径中，`halton_sequence` 对每一维用不同质数基的 `radical_inverse` 生成低差异点，再做随机平移 `(x+shift) mod 1` 实现随机化。  
7. 每个 `n` 汇总 `MC/QMC` 的均值、标准差和相对真值的 RMSE，形成 `LevelMetrics`。  
8. `fit_loglog_rate` 对 `log(error)-log(n)` 做线性拟合，输出经验收敛斜率，完成可验证的算法对比闭环。  

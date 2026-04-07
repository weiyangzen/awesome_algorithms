# 吉布斯-杜安方程 (Gibbs-Duhem Equation)

- UID: `PHYS-0289`
- 学科: `物理`
- 分类: `热力学`
- 源序号: `292`
- 目标目录: `Algorithms/物理-热力学-0292-吉布斯-杜安方程_(Gibbs-Duhem_Equation)`

## R01

吉布斯-杜安方程是热力学中“偏摩尔量不独立”的约束关系。对恒温恒压多组分体系：

`sum_i n_i d(mu_i) = 0`。

写成摩尔分数形式可得：

`sum_i x_i d(mu_i) = 0`。

在二元溶液活度系数表述中，常用等价式：

`x1 d(ln gamma1) + x2 d(ln gamma2) = 0`，其中 `x2 = 1 - x1`。

## R02

本条目 MVP 的目标不是做大而全相平衡求解器，而是最小闭环验证：

1. 给定二元体系 `ln(gamma1)(x1)` 数据（含噪声）；
2. 用 Gibbs-Duhem 微分关系反推出 `ln(gamma2)(x1)`；
3. 检查重构误差和方程残差，证明“从一组活度系数可推另一组”的可计算性。

## R03

`demo.py` 输入输出（无交互）：

- 输入（脚本内置）：
1. 一参数 Margules 真值模型参数 `A`；
2. 组成网格 `x1 in (0,1)`；
3. 观测噪声标准差 `noise_std` 与随机种子。
- 输出：
1. `ln(gamma2)` 重构 RMSE；
2. `ln(gamma1)` 平滑拟合 MAE；
3. Gibbs-Duhem 数值残差（公式残差与数值导数残差）；
4. 头尾样本表格与断言结果 `All checks passed.`。

## R04

MVP 使用的核心公式：

1. Gibbs-Duhem（二元、恒 `T/P`）：
`x1 dln(gamma1) + x2 dln(gamma2) = 0`。

2. 由上式得到重构微分方程：
`dln(gamma2)/dx1 = -(x1/x2) * dln(gamma1)/dx1`。

3. 数值积分恢复 `ln(gamma2)`：
`ln(gamma2)(x) = ln(gamma2)(x_min) + integral_(x_min->x) dln(gamma2)/dxi dxi`。

4. 本脚本真值模型（用于可验证基准）：
`ln(gamma1)=A*x2^2`，`ln(gamma2)=A*x1^2`（一参数 Margules）。

## R05

高层算法流程：

1. 构造组成网格并生成 Margules 真值 `ln(gamma1), ln(gamma2)`；
2. 给 `ln(gamma1)` 加噪声，模拟实验测量；
3. 用平滑样条拟合 `ln(gamma1)` 并求导；
4. 用 Gibbs-Duhem 公式计算 `dln(gamma2)/dx1`；
5. 对导数做累计积分得到 `ln(gamma2)` 重构曲线；
6. 与真值对比，输出误差与残差，执行断言验收。

## R06

正确性与实现对应关系：

- 物理关系显式落地在 `reconstruct_ln_gamma2_from_gibbs_duhem`；
- 不是黑盒“直接拟合 `gamma2`”，而是先求 `dln(gamma1)/dx1` 再通过约束方程积分；
- `gd_residual_formula` 验证离散点上微分关系是否严格成立；
- `gd_residual_numeric` 用 `np.gradient` 对重构曲线做独立导数检查，避免只看同一公式自洽。

## R07

复杂度分析（`N` 为组成网格点数）：

- 样条拟合：约 `O(N)` 到 `O(N*k)`（`k=3` 固定时可视作线性量级）；
- 导数评估：`O(N)`；
- 累计积分（梯形）：`O(N)`；
- 残差与误差统计：`O(N)`；
- 总体时间复杂度近似 `O(N)`，空间复杂度 `O(N)`。

## R08

边界与异常处理：

- `validate_composition_grid` 强制 `x1` 一维、严格递增、且位于 `(0,1)`，避免 `x1/x2` 奇异；
- 网格点数过少或存在非有限值直接报错；
- `dln_gamma1_dx1` 形状不匹配直接报错；
- 组成区间避开 0 和 1，避免端点处分母 `x2=0`。

## R09

MVP 取舍：

- 采用二元、恒温恒压、一参数 Margules 教学模型，突出 Gibbs-Duhem 主体逻辑；
- 不覆盖多元体系、VLE 联立、EOS/状态方程耦合；
- 不进行参数反演与置信区间估计；
- 重点是“约束方程 -> 可执行重构”的可审计实现。

## R10

`demo.py` 主要函数职责：

- `validate_composition_grid`：组成网格合法性检查；
- `margules_ln_gamma`：生成真值 `ln(gamma1), ln(gamma2)`；
- `reconstruct_ln_gamma2_from_gibbs_duhem`：核心微分约束与数值积分；
- `rms`：误差度量；
- `main`：构造数据、平滑求导、重构、评估与断言。

## R11

运行方式：

```bash
cd Algorithms/物理-热力学-0292-吉布斯-杜安方程_(Gibbs-Duhem_Equation)
uv run python demo.py
```

脚本会自动打印指标与样本表，无需输入参数。

## R12

输出字段说明：

- `rmse_ln_gamma2_reconstruction`：重构 `ln(gamma2)` 与真值的 RMSE；
- `mae_ln_gamma1_smoothing`：平滑后的 `ln(gamma1)` 与真值 MAE；
- `max_abs_gd_residual_formula`：用公式导数直接计算的最大残差；
- `max_abs_gd_residual_numeric`：对重构曲线数值求导后的最大残差；
- `profile` 表中同时给出 `x1/x2`、真值、观测值、平滑值和重构值，便于审计。

## R13

最小验收项（脚本内断言）：

1. `rmse_ln_gamma2 < 0.035`；
2. `mae_ln_gamma1_fit < 0.020`；
3. `max_abs_gd_residual_formula < 1e-10`；
4. `max_abs_gd_residual_numeric < 0.06`。

满足后打印 `All checks passed.`。

## R14

关键参数与调参建议：

- `noise_std`：噪声越大，导数估计越不稳定；
- `smooth_s = lambda*N*sigma^2`：样条平滑强度，脚本默认 `lambda=2`；过小会过拟合噪声，过大则过度平滑；
- `n_points`：网格越密，积分更平滑，但计算量增加；
- `x_min/x_max`：越靠近 0 或 1，`x1/x2` 放大效应越明显，数值更敏感。

## R15

与其他做法对比：

- 直接同时拟合 `gamma1/gamma2`：可能违反热力学一致性；
- 本方法先拟合一侧，再用 Gibbs-Duhem 约束推出另一侧，天然满足一致性；
- 代价是对导数质量敏感，因此需平滑和边界控制。

## R16

适用场景：

- 二元溶液活度系数教学与一致性检查；
- 实验数据预处理后从单侧数据推断另一侧趋势；
- 相平衡建模前的快速物理约束校验。

不适用场景：

- 强电解质、化学反应耦合体系；
- 多元复杂体系的工业级高精度设计；
- 需要完整状态方程和实验回归链路的生产环境。

## R17

可扩展方向：

- 扩展到 `N` 元体系：`sum_i x_i dln(gamma_i)=0` 的约束求解；
- 用正则化微分或贝叶斯平滑替代基础样条，提升抗噪性能；
- 引入参数反演（如拟合 Margules/Wilson/NRTL 参数）；
- 将 Gibbs-Duhem 作为损失项嵌入机器学习模型，强制物理一致性。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 创建 `BinarySystemConfig`，生成 `x1` 网格并通过 `validate_composition_grid` 保证定义域合法。  
2. 调用 `margules_ln_gamma` 生成真值 `ln_gamma1_true` 与 `ln_gamma2_true`。  
3. 用固定随机种子给 `ln_gamma1_true` 叠加高斯噪声，得到模拟观测 `ln_gamma1_obs`。  
4. 调用 `UnivariateSpline(x1, ln_gamma1_obs, s, k=3)` 做平滑，再用 `spline.derivative(1)` 求 `dln_gamma1/dx1`。  
5. 进入 `reconstruct_ln_gamma2_from_gibbs_duhem`：按 `dln_gamma2/dx1 = -(x1/x2)*dln_gamma1/dx1` 逐点构造导数。  
6. 在同一函数内使用 `scipy.integrate.cumulative_trapezoid` 对 `dln_gamma2/dx1` 做累计积分，得到 `ln_gamma2_reconstructed`。  
7. `main` 计算 RMSE/MAE 与两类 Gibbs-Duhem 残差（公式残差、数值导数残差），并输出 `summary` 与 `profile` 表。  
8. 通过四个断言阈值完成最小验收，全部通过后输出 `All checks passed.`。  

补充：`UnivariateSpline` 和 `cumulative_trapezoid` 仅承担“平滑求导”和“数值积分”两个基础数值步骤，热力学约束关系本身在源码中显式展开，不是黑箱调用。

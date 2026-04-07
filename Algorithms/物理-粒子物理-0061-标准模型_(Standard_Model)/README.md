# 标准模型 (Standard Model)

- UID: `PHYS-0061`
- 学科: `物理`
- 分类: `粒子物理`
- 源序号: `61`
- 目标目录: `Algorithms/物理-粒子物理-0061-标准模型_(Standard_Model)`

## R01

标准模型（SM）本质是规范群 `SU(3)_c x SU(2)_L x U(1)_Y` 下的量子场论框架。对“算法化 MVP”而言，最可执行的最小闭环不是做大型蒙特卡洛，而是验证三条核心一致性链路：

1. 电弱电荷关系：`Q = T3 + Y/2`
2. 规范反常与引力-超荷反常消失
3. 规范耦合在一环近似下的跑动行为

本条目即围绕这三点实现一个可直接运行、可断言、可复核的脚本。

## R02

问题定义：给定标准模型最小粒子量子数与 `MZ` 处实验输入常数，构建一个程序同时输出并验证：

- 多个粒子组分的 `Q_pred` 与 `Q_expected` 差值
- 3 代费米子的四类反常系数
- `alpha1, alpha2, alpha3` 的一环跑动快照与“最接近交汇”尺度

目标是让脚本以“失败即抛错”的方式自动验收一致性，而不依赖人工判断。

## R03

物理背景（与本 MVP 直接相关）：

- 电荷关系 `Q = T3 + Y/2` 把可观测电荷与电弱量子数联系起来。
- 反常消失（如 `[SU(2)]^2U(1)_Y`、`[U(1)_Y]^3`）是量子一致性的必要条件。
- 一环 RG 方程
  `alpha_i^{-1}(mu) = alpha_i^{-1}(MZ) - (b_i/2pi) ln(mu/MZ)`
  给出最小 SM 下耦合常数随能标变化的趋势。

这三项共同构成“标准模型是否自洽”的基础数值切面。

## R04

`demo.py` 输入/输出约定：

- 输入：脚本内置常量（无交互输入）
  - 电弱输入：`alpha_em^{-1}(MZ), sin^2(theta_W), alpha_s(MZ)`
  - 粒子量子数：`T3, Y` 与手工期望电荷
  - 手征场集合：每代 `Q_L, L_L, u_R^c, d_R^c, e_R^c`
- 输出：控制台三段表格 + 诊断指标
  1. 电荷关系表（含逐项误差）
  2. 反常系数表（分数与浮点双格式）
  3. 一环跑动快照表与最近交汇诊断

## R05

核心数学关系：

1. 电荷关系
   `Q = T3 + Y/2`

2. 反常系数（以左手 Weyl 场求和）
   - `[SU(3)]^2U(1)_Y: sum(Y * T3(R) * d2(R))`
   - `[SU(2)]^2U(1)_Y: sum(Y * T2(R) * d3(R))`
   - `[U(1)_Y]^3: sum(Y^3 * d3(R) * d2(R))`
   - `[Gravity]^2U(1)_Y: sum(Y * d3(R) * d2(R))`

3. 一环 beta 系数（最小 SM）
   `b1 = 41/10, b2 = -19/6, b3 = -7`

4. GUT 归一化耦合
   - `alpha1 = (5/3) * alpha_em / cos^2(theta_W)`
   - `alpha2 = alpha_em / sin^2(theta_W)`
   - `alpha3 = alpha_s`

## R06

MVP 主流程：

1. 构造电弱组分表（夸克、轻子、Higgs 组分）并计算 `Q_pred`。
2. 与 `Q_expected` 对比得到误差表与最大误差。
3. 构造每代手征场集合并使用有理数 `Fraction` 精确求和反常系数。
4. 把四类反常系数转为表格，检查是否全为 0。
5. 从 `MZ` 输入计算 `alpha1^{-1}, alpha2^{-1}, alpha3^{-1}`。
6. 在 `mu in [MZ, 1e17 GeV]` 上做一环跑动并计算三者离散度 `spread`。
7. 给出最小 `spread` 对应尺度、`alpha1-alpha2` 交叉尺度、以及该处与 `alpha3` 的偏差。
8. 所有检查通过后打印总结；若任一检查失败则抛 `RuntimeError`。

## R07

复杂度分析：

- 电荷关系与反常求和均是常量规模列表，时间复杂度 `O(1)`。
- 跑动扫描主开销来自 `N` 个能标点、3 个耦合：`O(N)`。
- 空间复杂度为存储 `N x 3` 数组，即 `O(N)`。

在默认 `N=2400` 下运行成本极低，适合快速验证。

## R08

正确性依据（程序内可验证）：

- 若 `Q=T3+Y/2` 实现错误，`max_charge_error` 会显著偏离 0 并触发断言。
- 反常系数用 `Fraction` 精确算术，避免浮点抵消误差；理论上应严格为 0。
- 一环跑动应体现“SM 下三耦合不精确统一”：
  - 程序计算最近接近尺度 `mu*`
  - 同时计算 `alpha1=alpha2` 时与 `alpha3` 的残差
- 三类检查共同形成可复核的一致性链。

## R09

伪代码：

```text
components <- standard_model_components()
charge_table, err <- evaluate_Q_relation(components)
assert err < 1e-12

fields <- standard_model_chiral_fields()
anomaly <- sum_anomalies_with_fraction(fields, generations=3)
assert all(anomaly[k] == 0)

inv_alpha_MZ <- build_initial_inverse_alphas()
mu_grid <- logspace(log10(MZ), 17)
inv_alpha_grid <- one_loop_run(mu_grid, inv_alpha_MZ, b)
spread <- row_max(inv_alpha_grid) - row_min(inv_alpha_grid)
mu_star <- mu_grid[argmin(spread)]

mu12 <- analytical_crossing(alpha1, alpha2)
mismatch <- |inv_alpha3(mu12) - inv_alpha1(mu12)|
assert min(spread) > 1.0

print(all tables and diagnostics)
```

## R10

边界与异常处理：

- 跑动函数要求 `mu > 0`，否则抛 `ValueError`。
- 电荷关系、反常消失、跑动结论都设置了硬断言，失败即抛错。
- 反常计算使用 `Fraction`，避免“理论为零但浮点不为零”的假失败。
- 脚本不依赖外部文件，避免 I/O 失败路径。

## R11

默认参数（`demo.py`）：

- `MZ = 91.1876 GeV`
- `alpha_em^{-1}(MZ) = 127.955`
- `sin^2(theta_W)(MZ) = 0.23122`
- `alpha_s(MZ) = 0.1179`
- 扫描区间：`[MZ, 1e17 GeV]`
- 扫描点数：`2400`

这些参数足以稳定复现“电荷正确、反常消失、非精确统一”的结果。

## R12

实现边界（MVP 范围）：

- 覆盖内容：量子数一致性、反常一致性、一环耦合跑动。
- 未覆盖内容：
  - 二环 RG 与阈值修正
  - Yukawa 跑动、真空稳定性
  - 事件级散射截面与探测器仿真

即：它是“理论一致性校验器”，不是完整 phenomenology 框架。

## R13

运行方式：

```bash
uv run python Algorithms/物理-粒子物理-0061-标准模型_(Standard_Model)/demo.py
```

无需任何交互输入。

## R14

输出结构示例（数值会随输入常量版本微调）：

```text
[1] Electroweak charge relation Q = T3 + Y/2
component ... Q_pred Q_expected error
...
max |Q_pred - Q_expected| = 0.000e+00

[2] Gauge/gravitational anomaly coefficients (3 generations)
anomaly             value_fraction  value_float
[SU(3)]^2 U(1)_Y    0               0.000000e+00
...

[3] One-loop running snapshots ...
mu_GeV log10_mu alpha1_inv alpha2_inv alpha3_inv
...

Diagnostics
closest approach scale mu* [GeV] = ...
minimal spread ...               = ...
```

重点是两类“零检验”（电荷误差、反常系数）和一类“非零检验”（统一残差）。

## R15

最小验收清单：

- `README.md` 已完成 R01-R18，无占位符。
- `demo.py` 无占位符，且可一键运行。
- 运行后电荷最大误差为 0（或机器零）。
- 四类反常系数均为 0。
- 一环跑动显示最小 SM 非精确统一（残差明显非零）。

## R16

当前局限：

- 使用一环近似，忽略二环与高阶阈值效应。
- 输入常数固定在 `MZ`，未做误差传播与参数拟合。
- 仅验证最小 SM 结构，不包含 `nu_R` 扩展或 GUT 新粒子。
- 无实验数据接口，仅做理论一致性的数值化演示。

## R17

可扩展方向：

1. 引入二环 beta 函数并比较一环/二环差异。
2. 加入阈值匹配（`m_t, m_H, m_{SUSY}` 等）研究统一敏感性。
3. 扩展到 SMEFT 参数扰动并追踪反常/跑动变化。
4. 接入 `scipy` 做参数拟合，把输入常数误差传到统一指标。

## R18

源码级算法流（对应 `demo.py`，8 步）：

1. `standard_model_components` 构造电弱组分列表（含 `T3, Y, Q_expected`）。
2. `charge_consistency_table` 逐项执行 `electric_charge`，形成 `pandas.DataFrame` 并计算最大误差。
3. `standard_model_chiral_fields` 生成每代左手 Weyl 场清单与群表示信息。
4. `anomaly_coefficients` 用 `Fraction` 对四类反常项逐项求和，输出精确有理数结果（避免浮点黑箱抵消）。
5. `initial_inverse_alphas` 把 `MZ` 处实验输入转换为 `alpha1^{-1}, alpha2^{-1}, alpha3^{-1}` 初值。
6. `one_loop_inverse_alpha` 用向量化公式
   `inv_alpha(mu) = inv_alpha(MZ) - (b/2pi) ln(mu/MZ)`
   在整段 `mu` 网格上批量生成跑动结果。
7. `running_summary` 计算最近接近尺度 `argmin(max-min)`，并用解析式 `alpha1=alpha2` 求 `mu12`，再评估与 `alpha3` 的偏差。
8. `main` 汇总三段结果并执行硬断言：电荷误差阈值、反常系数全零、最小统一残差下界；通过后输出“MVP checks passed”。

# AdS/CFT对应 (AdS/CFT Correspondence)

- UID: `PHYS-0381`
- 学科: `物理`
- 分类: `量子引力`
- 源序号: `400`
- 目标目录: `Algorithms/物理-量子引力-0400-AdS／CFT对应_(AdS／CFT_Correspondence)`

## R01

AdS/CFT 对应（规范/引力对偶）是一个“强耦合边界量子场论 = 弱耦合体引力/弦论”的字典。

本条目不尝试复现全息重整化、弦谱或黑洞热力学全套框架，而是做一个可运行最小原型（MVP）：

- 只取标量场扇区；
- 显式实现 `m^2 <-> Δ` 字典；
- 从边界两点函数幂律数据反演 `Δ` 与 `m^2`；
- 用近边界传播子斜率验证 `K_Δ(z,r) ~ z^Δ`。

## R02

MVP 的目标是打通“体到边界、边界回体”的完整闭环：

1. 给定 AdS 体标量质量 `m^2`（满足 BF bound）；
2. 计算共形维度分支 `Δ_-, Δ_+`，选标准量子化 `Δ=Δ_+`；
3. 生成边界两点函数 `G(r) = A / r^(2Δ)` 的带噪样本；
4. 用三种数值估计器回归 `Δ`（sklearn / scipy / torch）；
5. 由估计的 `Δ` 反推 `m^2`，并做一致性检查。

## R03

采用的理论关系（Euclidean Poincare patch，标量扇区）：

- 维度关系：`m^2 L^2 = Δ(Δ-d)`；
- 分支：`Δ_± = d/2 ± sqrt(d^2/4 + m^2 L^2)`；
- BF bound：`m^2 L^2 >= -d^2/4`；
- 边界两点函数尺度律：`G(r) = A / r^(2Δ)`；
- 体到边界传播子：
  `K_Δ(z,r) = C_Δ * ( z / (z^2 + r^2) )^Δ`，
  其中 `C_Δ = Gamma(Δ) / (pi^(d/2) Gamma(Δ-d/2))`。

## R04

建模与数值假设：

- `d=4`，`L=1`；
- 默认真值 `m^2=-3`（远高于 BF 下界 `-4`）；
- 仅考虑标量两点函数，不涉及自旋场与三点函数结构常数；
- 噪声模型为对数域高斯（对应乘性噪声）；
- 所有随机性由固定种子控制，结果可复现。

## R05

算法主流程（高层）：

1. 配置参数，检查 BF bound；
2. 用 `m^2 -> Δ_±` 计算字典主值 `Δ_true`；
3. 采样半径 `r` 并构造 `G_obs(r)`；
4. 三路拟合得到 `Δ_est` 与振幅 `A_est`；
5. 每路 `Δ_est -> m^2_est`，输出误差统计；
6. 取 `Δ` 共识值，检查传播子近边界斜率；
7. 汇总表格并执行 PASS/FAIL 断言。

## R06

`demo.py` 输入输出约定：

- 无命令行输入、无交互输入；
- 直接运行 `uv run python demo.py` 即可；
- 输出包括：
1. 理论参数（`Δ_true`, BF bound）；
2. 三种估计器结果表（`delta_est`, `m2_est`, `r2`, `mae`）；
3. 共识反演结果；
4. 传播子近边界斜率检查；
5. 验证清单与 `Validation: PASS/FAIL`。

## R07

核心函数与职责：

- `bf_bound(d)`：返回 BF 下界；
- `delta_branches(d, m2_L2)`：体质量映射到两条共形维度分支；
- `mass_from_delta(d, delta, L)`：边界维度反推体质量；
- `cft_two_point(r, A, Δ)`：两点函数幂律模型；
- `bulk_to_boundary_kernel(z, r, Δ, d)`：体到边界传播子；
- `fit_log_linear / fit_curve_power_law / fit_torch_log_model`：三路估计器；
- `estimate_near_boundary_slope`：近边界尺度律斜率检查。

## R08

复杂度分析（设样本数为 `N`，Torch 迭代步数为 `T`）：

- 生成数据：`O(N)`；
- sklearn 对数线性拟合：`O(N)`；
- scipy 非线性拟合：每次函数评估 `O(N)`，总成本近似 `O(kN)`；
- torch Adam 拟合：`O(TN)`；
- 统计汇总与验证：`O(N)`。

总体时间复杂度可记为 `O((k+T)N)`，空间复杂度为 `O(N)`。

## R09

数值稳健措施：

- 在 `delta_branches` 前检查判别式，避免复数 `Δ`；
- `curve_fit` 使用参数上下界，避免不物理解；
- Torch 拟合中对 `Δ` 进行 `clamp(0.1, 20.0)`；
- 采用对数域拟合降低幂律量级跨度带来的病态；
- 噪声水平默认较小（`sigma=0.03`）用于稳定演示。

## R10

最小工具栈与用途：

- `numpy`：幂律模型、向量化计算；
- `scipy`：`curve_fit` 与 `linregress`，以及 `Gamma` 特殊函数；
- `pandas`：结果汇总表与验证清单输出；
- `scikit-learn`：对数线性估计与 `R^2/MAE` 指标；
- `torch`：梯度下降拟合（Adam）做独立交叉验证。

## R11

运行方式：

```bash
cd Algorithms/物理-量子引力-0400-AdS／CFT对应_(AdS／CFT_Correspondence)
uv run python demo.py
```

正常情况下脚本末尾输出 `Validation: PASS`；
若任一验证失败，进程以非零状态退出。

## R12

关键输出字段解释：

- `delta_est`：估计得到的共形维度；
- `delta_abs_err`：与 `Δ_true` 的绝对误差；
- `m2_est`：由 `Δ_est` 反演得到的体质量；
- `bf_margin_m2L2`：相对 BF 下界的裕量（应非负）；
- `r2`：观测 `G_obs` 与拟合 `G_pred` 的决定系数；
- `mae`：两点函数拟合平均绝对误差；
- `slope_r2`：近边界 `log K - log z` 线性关系质量。

## R13

内置验收条件：

1. 三路估计器 `delta_abs_err < 0.08`；
2. 三路估计器 `r2 > 0.99`；
3. 三路估计器 `mae < 0.40`；
4. 所有 `m2_est` 满足 BF bound；
5. 共识质量误差 `|m2_consensus - m2_true| < 0.30`；
6. 传播子斜率误差 `|slope - Δ_consensus| < 0.04`。

全部成立才输出 `Validation: PASS`。

## R14

当前 MVP 的边界与限制：

- 仅覆盖 AdS/CFT 字典的标量两点函数子问题；
- 不含有限温度黑洞背景、熵与输运系数；
- 不含全息重整化的完整反项与关联函数正规化流程；
- 不处理 `Δ_-` 替代量子化窗口内的边界条件细节；
- 不是严格 EFT/弦论推导代码，而是教学级数值演示。

## R15

可扩展方向：

- 增加动量空间两点函数与 Bessel 型径向方程数值求解；
- 引入多自旋场（矢量/张量）并映射到守恒流/应力张量；
- 在 AdS-Schwarzschild 背景下提取准正则模频谱；
- 增加三点函数拟合，估计 OPE 系数；
- 接入符号推导与自动微分框架提升可探索性。

## R16

适用场景：

- AdS/CFT 入门教学中的“字典可计算化”演示；
- 强耦合 QFT 课程作业的最小可复现脚本；
- 研究代码前的数值校准 smoke test；
- 用于检验参数反演链路与 BF bound 合规性。

## R17

MVP 覆盖清单：

- [x] 体质量与共形维度的双向映射；
- [x] BF bound 显式检查；
- [x] 边界两点函数幂律数据生成；
- [x] 三种独立拟合器交叉验证；
- [x] 近边界传播子尺度律数值检查；
- [x] 表格化输出与自动 PASS/FAIL 验证。

## R18

`demo.py` 源码级算法流（8 步）：

1. `AdsCftConfig` 固定 `d, L, m2_true` 与噪声、采样、优化超参数，保证实验可复现。  
2. `main` 调用 `bf_bound` 与 `delta_branches`，先做 BF 合规，再由 `m^2 L^2` 得到 `Δ_-, Δ_+`，选 `Δ_true=Δ_+`。  
3. `generate_noisy_boundary_data` 用 `cft_two_point` 生成 `G_clean=A/r^(2Δ_true)`，再施加对数高斯噪声得到 `G_obs`。  
4. `fit_log_linear` 在 `log r - log G` 空间做线性回归，显式把斜率映射为 `Δ=-slope/2`。  
5. `fit_curve_power_law` 直接对 `G(r)=A/r^(2Δ)` 非线性拟合，给出独立的 `Δ` 与 `A`。  
6. `fit_torch_log_model` 用 Torch 参数化 `logA, Δ`，通过 Adam 最小化对数域 MSE，形成第三路估计。  
7. `summarize_fits` 将每路 `Δ_est` 通过 `mass_from_delta` 回推 `m2_est`，并汇总 `delta_abs_err/r2/mae/BF裕量`。  
8. `estimate_near_boundary_slope` 对 `K_Δ(z,r0)` 做 `log-log` 斜率回归验证 `~ z^Δ`；随后 `main` 执行 6 条验收断言并输出 `Validation`。  

第三方库未作为黑盒“直接给 AdS/CFT 结论”：

- `scipy/sklearn/torch` 仅用于通用回归与优化；
- 体-边字典公式、BF 约束、传播子尺度律与验收逻辑全部在源码中逐步展开。

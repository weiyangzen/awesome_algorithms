# DGLAP方程 (DGLAP Equations)

- UID: `PHYS-0400`
- 学科: `物理`
- 分类: `QCD`
- 源序号: `419`
- 目标目录: `Algorithms/物理-QCD-0419-DGLAP方程_(DGLAP_Equations)`

## R01

DGLAP 方程描述了 parton distribution functions（PDF）随硬标度 `Q^2` 的演化：

`d f_i(x,Q^2) / d ln Q^2 = sum_j (alpha_s(Q^2)/(2pi)) * [P_ij ⊗ f_j](x,Q^2)`

其中 `x` 是动量分数，`P_ij(z)` 是 splitting kernels，`⊗` 表示卷积。它解释了深度非弹散射中的尺度破缺（scaling violation），是 QCD 现象学的核心工具之一。

## R02

本条目目标是给出一个最小可运行的 DGLAP 数值 MVP，而不是完整全球拟合框架。

MVP 聚焦：

1. 仅做 `LO`、`non-singlet` 演化（避免完整耦合矩阵）；
2. 在 `x` 空间直接实现卷积与 `plus-prescription` 的有限化表达；
3. 使用一圈 `alpha_s(Q^2)`，从 `Q0` 演化到更高 `Q`；
4. 输出演化前后 PDF 对比与守恒量漂移检查。

## R03

`demo.py` 的输入输出约定（无交互）：

- 输入（脚本内固定参数）：
1. `N_f=4`, `Lambda_QCD=0.2 GeV`, `Q0=2 GeV`, `Q1=20 GeV`；
2. `x` 网格：`[1e-3, 0.90]` 的几何网格；
3. 初始分布：`q(x,Q0^2)=x^a (1-x)^b`。
- 输出：
1. 运行参数与 `alpha_s(Q0^2), alpha_s(Q1^2)`；
2. 初末态非奇异 PDF 在若干 `x` 点的对照表；
3. `∫ q(x) dx` 的相对漂移（数值守恒检查）；
4. 所有断言通过后打印 `All checks passed.`。

## R04

采用的数学模型（LO, non-singlet）：

1. 主方程：
`d q_NS(x,Q^2)/d lnQ^2 = (alpha_s(Q^2)/(2pi)) * (P_qq ⊗ q_NS)(x,Q^2)`。

2. 一圈跑动耦合：
`alpha_s(Q^2)=4pi / [beta0 ln(Q^2/Lambda^2)]`，
`beta0 = 11 - 2N_f/3`。

3. LO kernel：
`P_qq(z)=C_F[(1+z^2)/(1-z)]_+ + (3/2)C_F delta(1-z)`，`C_F=4/3`。

4. 数值实现中把 `plus` 分布写成可积形式：
- 先把 kernel 等价拆分为：`2/(1-z)_+ - (1+z) + (3/2)delta(1-z)`；
- 用 `phi(z)=q(x/z)/z` 处理 `plus` 项：  
  `∫_x^1 [1/(1-z)]_+ phi(z)dz = ∫_x^1 (phi(z)-phi(1))/(1-z) dz - phi(1)∫_0^x dz/(1-z)`；
- 其中 `∫_0^x dz/(1-z) = -ln(1-x)`，可解析计算。

## R05

设 `N_x` 为 `x` 网格点数，`N_z` 为每个卷积积分的局部采样点数，`N_t` 为演化步数：

- 单次 RHS 计算：`O(N_x * N_z)`；
- 全部时间推进：`O(N_t * N_x * N_z)`；
- 空间复杂度：`O(N_x)`（不存完整时间历史时）。

MVP 主要成本来自每个 `x_i` 上的卷积积分与插值。

## R06

算法闭环：

1. 构建 `x` 网格与初始分布 `q(x,Q0^2)`；
2. 在每个 `ln Q^2` 步计算当前 `alpha_s(Q^2)`；
3. 对每个 `x_i` 计算 `P_qq ⊗ q`（含 plus-prescription 修正）；
4. 得到 `dq/dlnQ^2`；
5. 使用显式 Euler 在 `lnQ^2` 上推进；
6. 得到 `q(x,Q1^2)` 并输出关键点对照；
7. 断言检查正性、耦合跑动方向、守恒漂移阈值。

## R07

优点：

- 方程、kernel、卷积都在源码中显式给出，可审计；
- 代码体量小，便于教学与二次扩展；
- 不依赖外部 PDF 拟合库，避免“黑盒调用即完成”。

局限：

- 仅 `LO + non-singlet`，未覆盖 singlet/gluon 耦合系统；
- 仅一圈 `alpha_s`，无阈值匹配和高圈修正；
- 数值积分是教学级实现，不追求全球拟合精度。

## R08

前置知识与环境：

- QCD 中 PDF 与 splitting function 的基本概念；
- `plus-distribution` 与卷积积分；
- Python `>=3.10`；
- 依赖：`numpy`, `pandas`（用于表格输出）。

## R09

适用场景：

- QCD 课程中演示 DGLAP 机制；
- 需要可读可改的小型演化原型；
- 在研究代码前做方法级 sanity check。

不适用场景：

- 需要 NNLO/N3LO 精度或全局 PDF 拟合；
- 需要阈值匹配、重味夸克细节、实验系统误差传播；
- 需要与 LHAPDF 等生产级工具直接对接的工程环境。

## R10

正确性直觉：

1. DGLAP 本质是“辐射导致动量分布重排”的流方程；
2. 大 `x` 端通常因辐射而被削弱，小 `x` 端可相对抬升；
3. `alpha_s` 随 `Q` 增大而减小（渐近自由），所以高能处演化速率变慢；
4. non-singlet 总量 `∫ q dx` 理论上守恒，数值结果应仅有小漂移。

## R11

数值稳定策略：

- 在 `t=lnQ^2` 上推进，避免跨尺度直接步进；
- `x_max` 选在 `0.90`，降低 `z→1` 端点奇异放大；
- 卷积上限使用 `1-eps`，并用 plus-subtraction 取消可积奇异；
- 每一步后把 `q` 裁到小正值，避免负值传播；
- 断言 `alpha_s(Q1^2) < alpha_s(Q0^2)` 与结果有限性。

## R12

关键参数及影响：

- `N_f`：决定 `beta0`，影响耦合跑动速度；
- `Lambda_QCD`：控制 `alpha_s` 绝对大小；
- `Q0,Q1`：决定总演化长度；
- `N_x,N_z,N_t`：决定精度与耗时；
- 初始指数 `a,b`：决定初始 PDF 形状。

调参建议：

- 先固定物理参数，再增大 `N_z`/`N_t` 做收敛检查；
- 若出现高 `x` 噪声，优先提高 `N_z` 并适度减小步长。

## R13

- 近似比保证：N/A（非优化近似算法）。
- 随机成功率保证：N/A（算法完全确定性）。

可验证保证（脚本断言）：

1. `alpha_s(Q^2)` 在给定区间单调下降；
2. 演化后 `q(x)` 全部有限且正；
3. `∫ q dx` 的相对漂移在设定阈值内（教学级数值容差）；
4. 高 `x` 端出现预期抑制（代表辐射重排方向正确）。

## R14

常见失效模式：

1. `Q^2 <= Lambda^2` 导致 `alpha_s` 发散；
2. 把 plus-prescription 写成裸 `1/(1-z)` 积分，造成数值爆炸；
3. `x` 网格过粗导致卷积振荡或伪负值；
4. 时间步过大导致显式积分不稳定；
5. 忽略端点裁剪，插值落到非物理区域。

## R15

工程扩展方向：

- 加入 singlet + gluon 的二维耦合演化；
- 升级到 NLO/NNLO splitting kernels；
- 引入阈值匹配与分段 `N_f(Q)`；
- 对接实验数据或 LHAPDF 初始条件；
- 使用更高阶时间积分器（RK/自适应步长）与并行卷积。

## R16

相关条目：

- Altarelli-Parisi splitting functions；
- QCD 渐近自由与 running coupling；
- DIS 中的 scaling violation；
- Mellin 空间 DGLAP 解法。

## R17

`demo.py` 交付能力清单：

- 显式实现 LO non-singlet DGLAP 方程；
- 显式实现 plus-prescription 的有限化卷积；
- 输出可读表格（`pandas`）展示演化前后变化；
- 内置断言进行数值与物理方向检查；
- 无交互，单命令可运行。

运行方式：

```bash
cd Algorithms/物理-QCD-0419-DGLAP方程_(DGLAP_Equations)
uv run python demo.py
```

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `EvolutionConfig` 固定 `N_f, Lambda_QCD, Q0, Q1, N_x, N_t, N_z`，给出完整可复现实验条件。  
2. `alpha_s_lo` 按一圈公式计算每个演化步的 `alpha_s(Q^2)`，并检查 `Q^2 > Lambda^2`。  
3. `initial_non_singlet_pdf` 生成初始 `q(x,Q0^2)=x^a(1-x)^b` 作为边界条件。  
4. `plus_subtraction_integral_0_to_x` 给出 `∫_0^x dz/(1-z)` 的解析表达（即 `-ln(1-x)`），为 plus-prescription 提供有限修正项。  
5. `non_singlet_convolution_lo` 在每个 `x_i` 上：采样 `z`、插值 `q(x_i/z)`、计算有限 integrand、再加上 `delta(1-z)` 项，得到 `(P_qq ⊗ q)(x_i)`。  
6. `dglap_rhs_lo` 把卷积乘上 `alpha_s/(2pi)`，形成 `dq/dlnQ^2`。  
7. `evolve_pdf_euler` 在 `lnQ^2` 上做显式 Euler 推进，逐步得到 `q(x,Q1^2)`。  
8. `main` 汇总初末态表格、积分漂移与断言检查并打印结果；第三方库只用于数组计算和表格打印，核心物理算法在源码中逐步展开。  

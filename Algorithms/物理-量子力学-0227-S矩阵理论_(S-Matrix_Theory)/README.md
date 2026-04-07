# S矩阵理论 (S-Matrix Theory)

- UID: `PHYS-0226`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `227`
- 目标目录: `Algorithms/物理-量子力学-0227-S矩阵理论_(S-Matrix_Theory)`

## R01

S 矩阵理论（S-Matrix Theory）用“入射态 -> 出射态”的线性映射描述散射过程。与直接求解全时域波函数不同，它把关注点放在可观测量（散射振幅、截面、共振参数）上，核心对象是：

`|out> = S |in>`

对中心势问题，S 矩阵可以按分波 `l=0,1,2,...` 对角化，每个通道由复数 `S_l` 表示。该表示直接连接相移、幺正性和光学定理，是量子散射中的标准框架。

## R02

本条目 MVP 的目标场景：
- 单粒子、三维、中心势散射。
- 通过方势阱模型先得到相移 `delta_l`，再构造 `S_l`，最后输出截面与一致性检查。
- 演示两种工况：
  - 纯弹性 `eta_l=1`（S 矩阵幺正）；
  - 人工引入低阶通道非弹性 `eta_l<1`（出现反应截面）。

这样既覆盖 S 矩阵的基础物理结构，也展示“幺正 -> 非幺正”的可计算差异。

## R03

核心公式（分波表象）：
- `S_l = eta_l * exp(2 i delta_l)`，其中 `0 <= eta_l <= 1`。
- 弹性极限：`eta_l = 1`，故 `|S_l|=1`。
- 散射振幅：
  `f(theta) = (1/(2ik)) * sum_l (2l+1)(S_l-1) P_l(cos theta)`
- 微分截面：`dσ/dΩ = |f(theta)|^2`
- 弹性截面：
  `σ_el = (π/k^2) * sum_l (2l+1)|S_l-1|^2`
- 反应截面：
  `σ_re = (π/k^2) * sum_l (2l+1)(1-|S_l|^2)`
- 总截面（两种等价写法）：
  `σ_tot = σ_el + σ_re = (2π/k^2) * sum_l (2l+1)(1-Re S_l)`
- 光学定理：`σ_tot = (4π/k) Im f(0)`。

## R04

`demo.py` 的输入输出约定：
- 输入：无交互输入。脚本内部固定参数 `E=4, V0=10, a=1, l_max=8, m=1, hbar=1`。
- 计算流程：
  1. 方势阱边界匹配求 `delta_l`；
  2. 构造 `S_l` 与对角 S 矩阵；
  3. 计算角分布和各类截面；
  4. 进行幺正性、求和公式、光学定理一致性检查。
- 输出：
  - 每个 `l` 的 `delta_l`、`eta_l`、`|S_l|`、`arg(S_l)`；
  - `σ_el, σ_re, σ_tot(parts), σ_tot(ReS), σ_tot(optical)`；
  - 多个角度的 `dσ/dΩ`。

## R05

算法伪代码：

```text
给定 E, V0, a, l_max:
  计算波数 k, q

for l = 0..l_max:
  用球贝塞尔函数匹配边界，求相移 delta_l

给定 eta_l:
  S_l = eta_l * exp(2 i delta_l)

由 S_l 计算:
  f(theta) = (1/(2ik)) sum_l (2l+1)(S_l-1) P_l(cos theta)
  dσ/dΩ = |f(theta)|^2
  σ_el, σ_re, σ_tot

验证:
  1) 弹性时 |S_l| 是否接近 1
  2) σ_tot(parts) 与 σ_tot(ReS) 是否一致
  3) σ_tot(parts) 与光学定理结果是否一致
```

## R06

正确性依据：
- 方势阱相移来自径向波函数在 `r=a` 处的连续匹配，公式可直接追溯到分波散射标准推导。
- 用 `atan2(num, den)` 取相移，避免单纯 `atan(num/den)` 的象限歧义。
- `S_l` 构造后，弹性情形应满足 `|S_l|=1` 与矩阵幺正性；非弹性情形应出现 `σ_re>0`。
- `σ_tot` 的三种表达（`σ_el+σ_re`、`Re S_l` 求和、光学定理）应相互一致；差值可作为数值自检。

## R07

复杂度分析（`L=l_max+1`, `N_theta` 为角度数）：
- 相移计算：`O(L)`（每个 `l` 常数次特殊函数计算）。
- 角分布：`O(L * N_theta)`（每个角度累加全部分波）。
- 截面与检查项：`O(L)`。
- 空间复杂度：`O(L + N_theta)`。

## R08

与相邻方法对比：
- 相比 Born 近似：S 矩阵框架可自然表达非微扰结构（通过 `delta_l, eta_l`），并显式展示幺正约束。
- 相比只做相移法：S 矩阵形式把“可观测量”和“约束关系”（光学定理、反应截面）统一在同一记号下。
- 相比黑箱散射求解器：本实现把从 `delta_l` 到 `S_l` 再到 `σ` 的链路完整展开，便于审计与教学。

## R09

代码结构（`demo.py`）：
- `wave_numbers`：计算内外区波数 `k, q`。
- `phase_shift_square_well`：单个分波的边界匹配求 `delta_l`。
- `compute_phase_shifts`：批量得到 `l=0..l_max` 相移。
- `build_partial_s_matrix` / `to_matrix`：生成 `S_l` 与对角 S 矩阵。
- `scattering_amplitude` / `differential_cross_section`：振幅和角分布。
- `sigma_elastic` / `sigma_reaction` / `sigma_total_*`：截面计算。
- `matrix_unitarity_residual`：矩阵幺正性残差。
- `run_case` / `main`：组织两种工况并打印结果。

## R10

边界与异常处理：
- `energy<=0`、`mass<=0`、`hbar<=0`、`a<=0`、`l_max<0` 时抛 `ValueError`。
- `energy+well_depth<=0` 时拒绝计算（避免虚波数进入当前 MVP）。
- 当 `j_l(q a)` 过小（接近节点）时抛 `RuntimeError`，避免对数导数数值爆炸。
- `eta_l` 强制限制在 `[0,1]`，并检查数组形状与 `delta_l` 一致。

## R11

MVP 技术栈：
- `numpy`：数组、复数运算、向量化统计。
- `scipy.special`：`spherical_jn/spherical_yn/eval_legendre`。
- `pandas`：结果表格输出。

依赖保持最小，不依赖任何“散射黑箱包”。

## R12

运行方式（在仓库根目录或本目录均可）：

```bash
uv run python Algorithms/物理-量子力学-0227-S矩阵理论_(S-Matrix_Theory)/demo.py
```

或先切到该目录再执行：

```bash
uv run python demo.py
```

## R13

预期输出特征：
- 弹性工况中：
  - `|S_l|` 全部接近 1；
  - `σ_re` 接近 0；
  - `matrix_unitarity_res`、`optical_res`、`sumrule_res` 很小。
- 非弹性示例工况中：
  - 低阶通道 `|S_l|<1`；
  - `σ_re` 明显大于 0；
  - `σ_tot(parts)` 仍应与 `σ_tot(optical)` 近似一致。

## R14

常见实现错误：
- 把 `f(theta)` 公式写成 `1/k` 而漏掉 `1/(2i)` 因子。
- 只计算 `σ_el` 就当成总截面，忽略 `σ_re`。
- 使用 `atan(num/den)` 导致相移跨象限错误。
- 忽略 `eta_l` 的物理约束（把它设成大于 1）。
- 只检查 `|S_l|` 而不交叉验证光学定理。

## R15

最小测试清单：
- 可运行性：`uv run python demo.py` 无异常退出。
- 结构性：输出包含两组工况（elastic 与 inelastic toy）。
- 弹性一致性：`σ_re≈0` 且 `matrix_unitarity_res` 足够小。
- 总截面一致性：`|σ_tot(parts)-σ_tot(optical)|` 足够小。
- 数值合法性：无 `nan/inf`，`dσ/dΩ >= 0`。

## R16

扩展方向：
- 把方势阱替换为一般短程势并用数值积分提取 `delta_l(E)`。
- 做能量扫描，分析共振（`delta_l` 快速变化）和阈值行为。
- 扩展到耦合道 S 矩阵（非对角块），演示道间跃迁。
- 引入实验拟合：由角分布反推低阶 `delta_l, eta_l`。

## R17

当前 MVP 的取舍与局限：
- 仅覆盖单通道、中心势、无自旋耦合模型。
- 非弹性工况是教学化“手动 eta”示例，不对应具体微观吸收势。
- 高能情况下 `l_max` 截断误差会增大，需要更高分波截断与收敛分析。
- 以可解释性优先，未做大规模参数扫描与性能优化。

## R18

源码级算法流程拆解（8 步，非黑箱）：
1. `main` 读取固定配置并调用 `wave_numbers`，计算外区/内区波数 `k, q`。  
2. `compute_phase_shifts` 遍历 `l=0..l_max`，逐通道进入 `phase_shift_square_well`。  
3. 在 `phase_shift_square_well` 中调用 `scipy.special.spherical_jn/spherical_yn` 及导数，构造匹配式并用 `atan2` 得到 `delta_l`。  
4. `build_partial_s_matrix` 按 `S_l = eta_l exp(2i delta_l)` 生成分波 S 元素；`to_matrix` 将其嵌入对角矩阵表示。  
5. `scattering_amplitude` 按 `(S_l-1)P_l(cosθ)` 的分波和计算复振幅；`differential_cross_section` 对每个角度取模平方。  
6. `sigma_elastic`、`sigma_reaction`、`sigma_total_from_reS`、`sigma_total_optical` 分别计算不同定义下的截面。  
7. `matrix_unitarity_residual` 计算 `S S^† - I` 的最大模残差；同时比较三种总截面，得到 `optical_res/sumrule_res`。  
8. `run_case` 先后输出弹性与非弹性示例的分波表和角分布表，构成完整、可审计的 S 矩阵最小验证链路。

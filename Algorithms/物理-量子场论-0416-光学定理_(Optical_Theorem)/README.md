# 光学定理 (Optical Theorem)

- UID: `PHYS-0397`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `416`
- 目标目录: `Algorithms/物理-量子场论-0416-光学定理_(Optical_Theorem)`

## R01

光学定理把“前向散射振幅的虚部”与“总截面”直接联系起来。常见形式为：
\[
\sigma_{\text{tot}} = \frac{4\pi}{k}\,\operatorname{Im} f(0)
\]
其中 `k` 是入射波数，`f(0)` 是散射角 `\theta=0` 的振幅。

在相对论量子场论中也常写作：
\[
2\,\operatorname{Im}\,\mathcal M_{ii} = \sum_f \int d\Phi_f\,|\mathcal M_{fi}|^2
\]
本质是 `S^\dagger S=I` 的直接后果。

## R02

它的核心意义是“守恒与可观测量闭环”：
- 左边是前向弹性振幅的虚部（振幅信息）；
- 右边是对所有末态求和后的总概率流（截面信息）。

因此光学定理常被用于：
- 检查散射模型是否满足幺正性；
- 用前向振幅预测总截面；
- 在有效场论和现象学拟合中做一致性约束。

## R03

本目录 MVP 的输入输出定义如下。

输入（脚本内置，无交互）：
- 波数 `k > 0`；
- 最大部分波阶数 `l_max`；
- 每个 `l` 的相移 `\delta_l` 与非弹性参数 `\eta_l \in [0,1]`。

输出：
- 前向振幅 `f(0)`；
- 光学定理两端：`(4\pi/k)Im f(0)` 与部分波求和得到的 `\sigma_tot`；
- `\sigma_el`、`\sigma_reac` 及其和；
- 角分布积分得到的弹性截面数值校验。

## R04

部分波记号下：
\[
S_l = \eta_l e^{2i\delta_l},\quad
f(\theta)=\frac{1}{2ik}\sum_{l=0}^{l_{\max}}(2l+1)(S_l-1)P_l(\cos\theta)
\]

则可得：
\[
\sigma_{\text{tot}}=\frac{2\pi}{k^2}\sum_l (2l+1)\bigl(1-\Re S_l\bigr)
\]
\[
\sigma_{\text{el}}=\frac{\pi}{k^2}\sum_l (2l+1)|1-S_l|^2
\]
\[
\sigma_{\text{reac}}=\frac{\pi}{k^2}\sum_l (2l+1)(1-|S_l|^2)
\]
且 `\sigma_tot = \sigma_el + \sigma_reac`。

## R05

从 `f(0)` 推导光学定理：
\[
\operatorname{Im} f(0)
=\frac{1}{2k}\sum_l (2l+1)(1-\Re S_l)
\]
两边乘以 `4\pi/k` 得：
\[
\frac{4\pi}{k}\operatorname{Im} f(0)
=\frac{2\pi}{k^2}\sum_l(2l+1)(1-\Re S_l)
=\sigma_{\text{tot}}
\]

这说明只要给定满足物理约束的 `S_l`，光学定理是代数恒等式，不是经验拟合。

## R06

MVP 伪代码：

```text
输入 k, l_max, {delta_l}, {eta_l}
构造 S_l = eta_l * exp(2 i delta_l)
计算 forward amplitude:
    f0 = (1/(2ik)) * sum_l (2l+1) * (S_l - 1)
计算 sigma_tot_optical = (4pi/k) * Im(f0)
计算 sigma_tot_pw      = (2pi/k^2) * sum_l (2l+1) * (1 - Re(S_l))
计算 sigma_el_pw, sigma_reac_pw 并验证 sigma_tot_pw = sigma_el_pw + sigma_reac_pw
在 theta 网格上计算 f(theta), dσ/dΩ=|f|^2，并积分得到 sigma_el_num
输出以上量与误差
```

## R07

正确性依据分三层：
1. 部分波展开是中心势散射的标准完备基展开；
2. `S_l` 参数化后，`f(0)` 与 `\sigma_tot` 的关系可直接由代数恒等式推出；
3. `\sigma_el + \sigma_reac = \sigma_tot` 是 `S` 矩阵概率流守恒的分解形式。

因此该算法不是“数值猜测”，而是对幺正性公式的离散实现与校验。

## R08

复杂度（`N_\theta` 为角度网格点数，`L=l_max+1`）：
- 计算前向量与部分波求和：`O(L)`；
- 递推 Legendre 多项式表并算 `f(\theta)`：`O(N_\theta L)`；
- 总体时间复杂度：`O(N_\theta L)`；
- 空间复杂度：`O(N_\theta L)`（保存 `P_l(cos\theta)` 表）。

对教学和小型验证任务足够轻量。

## R09

数值注意事项：
- `\eta_l` 必须截断在 `[0,1]`，否则会出现非物理增益；
- `\theta=0` 时 Legendre 递推应保持 `P_l(1)=1` 的稳定性；
- 用有限 `l_max` 截断时，角分布细节可能受影响，但前向光学定理仍应在截断模型内自洽；
- 弹性截面积分依赖角网格分辨率，建议 `N_\theta` 足够大。

## R10

与量子场论中的关系：
- 在场论层面，光学定理等价于振幅虚部由“切割后中间态”给出；
- 与 Cutkosky cutting rules 一致，都是幺正性在图级别的体现；
- 部分波版本可看作对角动量通道的离散化表达。

MVP 选择部分波框架，是因为它最直接、可审计、且易数值验证。

## R11

常见错误：
- 把 `\sigma_tot` 误写为仅弹性截面，忽略吸收/非弹性通道；
- 前向振幅取实部而非虚部；
- `S_l` 指数写成 `e^{i\delta_l}` 而非 `e^{2i\delta_l}`；
- 角分布积分遗漏球坐标 Jacobian `sin\theta`；
- 将有限截断误差误认为光学定理失效。

## R12

本目录实现策略：
- 不依赖大型框架，只用 `numpy`；
- 用两组案例演示：
  - `elastic_only`: `\eta_l=1`，验证 `\sigma_reac\approx0`；
  - `inelastic`: `\eta_l<1`，验证反应截面为正；
- 同时输出：
  - 光学定理两端差值；
  - `\sigma_tot-(\sigma_el+\sigma_reac)` 守恒误差；
  - 角分布积分与部分波解析的弹性截面差值。

## R13

运行命令：

```bash
uv run python demo.py
```

若本地未使用 `uv`，也可直接：

```bash
python3 demo.py
```

## R14

预期输出应包含每个案例的：
- `k`, `l_max`, `f(0)`；
- `sigma_tot(optical)` 与 `sigma_tot(partial-wave)`；
- `sigma_el(partial-wave)`, `sigma_reac(partial-wave)`；
- `sigma_el(numerical integral)`；
- 三个误差量（应接近机器精度或离散积分误差级别）。

其中 `inelastic` 案例应明确显示 `sigma_reac > 0`。

## R15

典型应用场景：
- 强子散射总截面拟合的幺正约束；
- 核反应或原子碰撞中的吸收通道建模；
- 偏波分析与部分波分析（PWA）中的一致性检查；
- 数值振幅模型训练后的物理可行性验证。

## R16

建议最小测试集：
- `\eta_l=1`、随机平滑 `\delta_l`，检查纯弹性关系；
- 含 `\eta_l<1` 的吸收模型，检查 `\sigma_reac>0`；
- 改变 `l_max`，观察收敛行为；
- 改变 `N_\theta`，观察积分型 `\sigma_el` 的网格收敛。

## R17

可扩展方向：
- 从合成 `\delta_l, \eta_l` 换成具体势散射相移求解；
- 接入实验数据做参数拟合，并将光学定理作为硬约束；
- 推广到多道耦合 `S` 矩阵；
- 在相对论归一化下对接 `2Im\,\mathcal M` 形式与相空间积分。

## R18

`demo.py` 的源码级算法流程（8 步）如下：
1. `build_s_matrix`：从 `\delta_l`、`\eta_l` 计算 `S_l=\eta_l e^{2i\delta_l}`。  
2. `forward_amplitude`：按 `f(0)=\frac{1}{2ik}\sum_l(2l+1)(S_l-1)` 计算前向振幅。  
3. `cross_sections_from_partial_waves`：并行计算 `\sigma_tot`、`\sigma_el`、`\sigma_reac` 的闭式求和。  
4. `legendre_table`：用三项递推构造 `P_l(\cos\theta)` 数值表，避免黑箱调用。  
5. `scattering_amplitude`：合成任意角度振幅 `f(\theta)`。  
6. `elastic_cross_section_from_angles`：对 `|f(\theta)|^2` 乘 `\sin\theta` 后做数值积分，得到 `\sigma_el`。  
7. `run_case`：在一个案例内汇总全部观测量，并计算三种误差（光学定理误差、守恒误差、积分误差）。  
8. `main`：构造弹性/非弹性两案例并打印结果，形成可直接复验的最小闭环。

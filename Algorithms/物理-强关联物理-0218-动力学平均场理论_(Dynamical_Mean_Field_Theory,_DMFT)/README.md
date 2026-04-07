# 动力学平均场理论 (Dynamical Mean Field Theory, DMFT)

- UID: `PHYS-0217`
- 学科: `物理`
- 分类: `强关联物理`
- 源序号: `218`
- 目标目录: `Algorithms/物理-强关联物理-0218-动力学平均场理论_(Dynamical_Mean_Field_Theory,_DMFT)`

## R01

动力学平均场理论（DMFT）把晶格强关联系统映射为“单杂质 + 自洽电子浴”问题，用局域但频率依赖的自能 `Sigma(iwn)` 捕捉关联效应。它在无穷配位数极限下对局域关联是严格的，能描述 Mott 物理、准粒子重整化等关键现象。

## R02

本条目聚焦最经典的单带 Hubbard 模型（半填充、顺磁）：

`H = -t * sum_{<ij>,sigma}(c^dag_{i,sigma} c_{j,sigma} + h.c.) + U * sum_i n_{i,up} n_{i,down}`

目标不是搭建完整生产级求解器，而是做一个可运行、可解释的最小 DMFT 闭环：
- 给定 `U, beta`；
- 自洽求 `Delta(iwn), G_imp(iwn), Sigma(iwn)`；
- 输出收敛指标与低频物理量摘要。

## R03

MVP 采用 Bethe 格点（半带宽 `D=1`）的标准自洽关系。设 `t = D/2`，Matsubara 频率 `iwn = i(2n+1)pi/beta`：

1. 杂质 Dyson 方程：`G_imp^{-1}(iwn) = iwn - Delta(iwn) - Sigma(iwn)`。
2. Bethe 格点自洽：`Delta(iwn) = t^2 * G_loc(iwn)`。
3. 晶格局域格林函数满足二次方程：`G_loc = 1 / (iwn - Sigma - t^2 G_loc)`。

因此每次迭代核心是：`Delta -> impurity solver -> Sigma -> G_loc -> Delta_new`。

## R04

杂质求解器采用“单浴位离散 + 精确对角化（ED）”最小构型：
- 轨道：`d_up, d_dn, b_up, b_dn`（4 个自旋轨道，Hilbert 维度 `2^4=16`）；
- 杂质哈密顿量：

`H_imp = U(n_d_up-1/2)(n_d_dn-1/2) + eps_b(n_b_up+n_b_dn) + V * sum_sigma(d^dag_sigma b_sigma + h.c.)`

其中 `(eps_b, V)` 由当前 `Delta(iwn)` 拟合得到。

## R05

`demo.py` 的最小可运行策略：

- 用 `scipy.optimize.minimize` 拟合低频区 `Delta(iwn) ≈ V^2/(iwn-eps_b)`（单极点近似）；
- 用 Fock 基上的显式费米算符矩阵构造 `16x16` 哈密顿量；
- 用 `numpy.linalg.eigh` 精确对角化；
- 用 Lehmann 表达式计算 `G_imp(iwn)`；
- 按 DMFT 关系更新并混合 `Delta` 直到收敛。

这不是“黑盒调用库就结束”：拟合目标函数、Lehmann 求和、分支选择与自洽更新都在源码中显式实现。

## R06

输入（脚本内配置 `DMFTConfig`）：
- `U`：局域库仑相互作用；
- `beta`：逆温度；
- `n_iw`：Matsubara 点数；
- `mix, tol, max_iter`：自洽控制参数。

输出（终端打印）：
- 是否收敛、迭代步数、`max|Delta_new-Delta_old|`；
- 拟合浴参数 `eps_b, V` 与拟合损失；
- 双占据数 `<n_up n_dn>` 与低频 `Z` 估计；
- 低频 `Im G_imp(iwn)`、`Im Sigma(iwn)` 表格。

## R07

高层伪代码：

1. 初始化 `Delta(iwn)=t^2/iwn`。
2. 循环直到收敛或达到 `max_iter`：
3. 拟合 `(eps_b, V)` 使单极点浴逼近当前 `Delta`。
4. 构造两位点 Anderson 杂质模型哈密顿量并 ED 求解。
5. 由 Lehmann 展开计算 `G_imp(iwn)`。
6. 由 `Sigma = iwn - Delta - 1/G_imp` 得到自能。
7. 由 Bethe 二次方程求 `G_loc`，再算 `Delta_new=t^2*G_loc`。
8. 线性混合更新 `Delta` 并检查误差阈值。

## R08

设 Matsubara 点数为 `Nw`，每次自洽迭代复杂度近似为：
- ED 对角化：`O(16^3)`，常数规模；
- Lehmann 求和：`O(Nw * 16^2)`；
- 拟合（L-BFGS-B）由若干次目标函数评估组成，每次 `O(Nw_fit)`。

总体可视为 `O(n_iter * Nw)` 主导，内存占用也约 `O(Nw)`。

## R09

数值稳定策略：
- 采用 `Delta` 混合参数 `mix` 抑制震荡；
- 拟合时对低频点加权，优先保证低能行为；
- Bethe 二次方程在两根之间按高频渐近 `G~1/z` 选分支，避免错误支路；
- 玻尔兹曼因子按 `E-E_min` 平移，减少上溢风险。

## R10

正确性检查（MVP 层面）：
- 收敛误差 `max|Delta_new-Delta_old|` 随迭代下降；
- `G_imp` 与 `Sigma` 不出现 NaN/Inf；
- 低频 `Im G_imp(iwn)` 为负（费米格林函数常见符号约束）；
- 输出的双占据数在物理范围 `[0,1]`。

## R11

默认参数（`demo.py`）：
- `U=2.2, beta=30, n_iw=96, D=1`；
- `mix=0.55, tol=1e-4, max_iter=60, n_fit=40`。

调参建议：
- 若不收敛：降低 `mix`（如 `0.3~0.5`）或提高 `max_iter`；
- 若低频噪声偏大：增加 `beta` 与 `n_iw`；
- 若拟合不稳：减少 `n_fit` 只拟合更低频段。

## R12

运行方式：

```bash
cd Algorithms/物理-强关联物理-0218-动力学平均场理论_(Dynamical_Mean_Field_Theory,_DMFT)
uv run python demo.py
```

脚本无交互输入，直接输出收敛与物理量摘要。

## R13

输出解读要点：
- `converged=True` 表示在给定阈值内完成自洽；
- `fit_success_ratio` 反映每轮浴拟合成功比例；
- `<n_up n_dn>` 越小通常意味着关联抑制双占据越强；
- `Z_est` 是低频导数近似，仅作教学指标，不等同于高精度连续极限结果。

## R14

局限性：
- 单浴位离散太粗糙，不能精确还原连续杂质浴；
- 未实现 CTQMC/NRG 等高精度杂质求解器；
- 仅处理半填充顺磁情形，未覆盖掺杂、磁有序、多轨道；
- 没有做实频解析延拓，因此谱函数信息有限。

## R15

可扩展方向：
- 增加浴位数并做多极点拟合；
- 替换为 IPT / CTQMC 等更强杂质求解器；
- 扩展到多轨道 DMFT 与晶体场分裂；
- 增加解析延拓（如 MaxEnt）输出 `A(omega)`。

## R16

适用场景：
- 强关联课程中的 DMFT 最小闭环教学；
- 作为大规模 DMFT 代码开发前的原型验证；
- 快速测试“参数变动 -> 自洽结果变化”的定性趋势。

不适用场景：
- 需要定量对比实验数据的高精度研究；
- 需要动量分辨信息（超出单点 DMFT）的问题。

## R17

参考方向（概念层）：
- A. Georges, G. Kotliar, W. Krauth, M. J. Rozenberg, *Rev. Mod. Phys.* 68, 13 (1996).
- E. Pavarini et al. (eds.), *The LDA+DMFT approach to strongly correlated materials*.
- DMFT 两位点/离散浴教学模型相关文献与讲义。

## R18

`demo.py` 源码级算法流（9 步，含第三方函数拆解）：

1. `matsubara_frequencies` 生成 `iwn` 网格，并在 `run_dmft` 中初始化 `Delta=t^2/iwn`。  
2. `fit_single_pole_hybridization` 定义目标函数 `mean(w_n*|Delta - V^2/(iwn-eps_b)|^2)`。  
3. 调用 `scipy.optimize.minimize(L-BFGS-B)` 时，优化器会反复回调该目标函数；每次评估都由源码里显式的复数残差计算完成，不是黑盒物理解算。  
4. `build_fermion_operators` 在 Fock 基上构造费米产生/湮灭算符（含反对易符号），再得到数算符。  
5. `build_hamiltonian` 用这些算符拼出 `16x16` Anderson 杂质哈密顿量。  
6. `solve_impurity_ed` 用 `numpy.linalg.eigh` 对角化，并在 `impurity_green_function` 中按 Lehmann 公式逐频求 `G_imp(iwn)`。  
7. `run_dmft` 根据 Dyson 关系计算 `Sigma=iwn-Delta-1/G_imp`。  
8. `bethe_lattice_green` 解二次方程得到 `G_loc`，通过高频渐近条件选择正确根，再更新 `Delta_new=t^2*G_loc` 并做线性混合。  
9. `main` 汇总收敛误差、拟合质量、双占据、`Z` 估计与低频表格，并做有限性断言，形成一次性非交互验证。  

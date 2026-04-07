# Møller-Plesset微扰理论 (MP Perturbation Theory)

- UID: `PHYS-0211`
- 学科: `物理`
- 分类: `量子化学`
- 源序号: `212`
- 目标目录: `Algorithms/物理-量子化学-0212-Møller-Plesset微扰理论_(MP_Perturbation_Theory)`

## R01

Møller-Plesset（MP）微扰理论是以 Hartree-Fock（HF）参考态为零阶近似，对电子相关效应做逐阶修正的方法。最常用的是二阶 MP2：在保持实现简洁的前提下显著改进 HF 能量。

本目录提供一个“可运行、可审计”的最小原型（MVP）：
- 先做闭壳层 RHF 自洽；
- 再用规范轨道显式计算 MP2 二阶相关能；
- 给出主要激发贡献项和自动校验结果。

## R02

MP 理论分割思路：

- 原哈密顿量：`H`
- 选取零阶哈密顿量：`H0 = sum_i f(i)`（Fock 算符和）
- 扰动项：`V = H - H0`

能量按微扰展开：

`E = E(0) + E(1) + E(2) + ...`

在 HF 参考下，`E(0)+E(1)` 对应 HF 能量，`E(2)` 即 MP2 相关能修正。

## R03

本 MVP 的目标不是复现实验级量化结果，而是明确展示 MP2 的算法链路：

1. 构造一套小型“分子型”一电子与二电子积分；
2. 用这些积分完成 RHF SCF；
3. 在 RHF 规范轨道中计算 MP2 能量；
4. 输出可验证指标（收敛、能隙、分母符号、能量下降）。

## R04

模型设定（教学用）：

- 采用正交 AO 基（`S = I`），避免额外广义本征问题复杂度；
- 一电子哈密顿量 `h_core`：对称矩阵，带有能级梯度与小耦合；
- 二电子积分 `(pq|rs)`：由三维因子张量经
  `eri[pqrs] = sum_L L[pq,L] L[rs,L]`
  生成，再显式对称化，保证积分张量满足常见交换对称性；
- 设常数核排斥项 `E_nuc`，形成完整总能表达。

该设定可以完整演示算法，不依赖外部量化化学软件。

## R05

RHF 部分采用闭壳层公式：

- 密度矩阵：`P = 2 * C_occ * C_occ^T`
- Fock 矩阵：`F = h + J - 0.5K`
- 其中
  `J_pq = sum_rs P_rs (pq|rs)`，
  `K_pq = sum_rs P_rs (pr|qs)`
- 电子能：`E_elec = 0.5 * sum_pq P_pq (h_pq + F_pq)`
- 总 HF 能量：`E_HF = E_elec + E_nuc`

SCF 通过阻尼混合密度（damping）稳定收敛。

## R06

MP2 相关能使用闭壳层规范轨道表达式：

`E_MP2 = sum_{ijab} (2(ij|ab) - (ij|ba)) (ij|ab) / (eps_i + eps_j - eps_a - eps_b)`

其中：
- `i,j` 为占据轨道；
- `a,b` 为虚轨道；
- `eps` 为 RHF 轨道能；
- `(ij|ab)` 来自 AO->MO 四指标变换。

最终总能：`E_total(MP2) = E_HF + E_MP2`。

## R07

主流程（高层）如下：

1. 读取 `MP2Config`（轨道数、电子数、随机种子、SCF 阈值等）。
2. 构造 `h_core`、`eri`、`E_nuc`。
3. 从零密度出发执行 RHF SCF，得到收敛密度与规范轨道。
4. AO 二电子积分做四指标变换到 MO 表示。
5. 生成 `ijab` 张量和能量分母，逐项累加得到 `E_MP2`。
6. 统计最大贡献激发项。
7. 打印 SCF 尾部日志、能量摘要和检查结果。
8. 验证失败时返回非零退出码。

## R08

设 AO/MO 轨道数为 `N`，占据轨道数 `Nocc`，虚轨道数 `Nvir`：

- SCF 每轮 Fock 构造主成本约 `O(N^4)`；
- AO->MO 四指标变换主成本约 `O(N^5)`（本 MVP 用直接 einsum）；
- MP2 求和成本约 `O(Nocc^2 * Nvir^2)`；
- 总体由四指标变换主导，空间中 `eri` 为 `O(N^4)`。

## R09

数值稳定与健壮性措施：

- 输入合法性检查：闭壳层（偶电子数）与电子数范围；
- SCF 使用阻尼混合，抑制振荡；
- 同时监控 `dE` 与 `dP`；
- 输出 HOMO/LUMO 与 gap 检查轨道排序合理性；
- 校验 MP2 分母全负（在正常闭壳层下应满足）。

## R10

最小工具栈：

- `numpy`：线性代数、einsum 张量收缩、SCF 与 MP2 核心计算；
- `pandas`：迭代日志与摘要表展示。

未使用 `pyscf` 等黑盒电子结构库，关键公式都在 `demo.py` 源码中显式实现。

## R11

运行方式：

```bash
cd Algorithms/物理-量子化学-0212-Møller-Plesset微扰理论_(MP_Perturbation_Theory)
uv run python demo.py
```

脚本无交互输入。成功时会打印 `Validation: PASS`；失败会以非零退出。

## R12

主要输出字段：

- `E_HF`：RHF 总能；
- `E_MP2_corr`：MP2 二阶相关能修正；
- `E_MP2_total`：MP2 总能；
- `HOMO/LUMO/gap`：前线轨道能与能隙；
- `trace_P`：密度矩阵迹（应接近电子数）；
- `denom_min/denom_max`：MP2 分母范围；
- `Largest |MP2 term contributions|`：贡献最大的 `i,j,a,b` 激发项。

## R13

脚本内置最小验收：

1. SCF 收敛；
2. `trace(P)` 与电子数一致（误差阈值 `1e-4`）；
3. HOMO-LUMO gap 为正；
4. MP2 全部分母为负；
5. `E_MP2_corr` 有限；
6. `E_MP2_corr < 0`（相关能使总能低于 HF）。

全部通过时输出 `Validation: PASS`。

## R14

当前 MVP 局限：

- 积分来自教学型合成哈密顿量，不对应具体分子几何；
- 仅实现 RHF + MP2（闭壳层），未涵盖 UHF/ROHF；
- 未实现更高阶 MP3/MP4；
- 四指标变换是直接法，未做 RI-MP2、局域相关等加速。

## R15

可扩展方向：

- 引入真实分子积分（例如从外部量化计算前处理得到）；
- 加入 DIIS 加速 SCF；
- 扩展到自旋分辨的 UMP2；
- 加入分子轨道冻结（frozen core）策略；
- 用密度拟合（RI）降低 MP2 成本到更可扩展形式。

## R16

适用场景：

- 量子化学教学中解释“HF 为什么缺相关能”；
- 代码审计场景下验证 MP2 公式是否正确落地；
- 作为更大电子结构项目中的最小可运行参考实现；
- 用于单元测试（能量、分母符号、轨道排序）基线。

## R17

与近邻方法简对比：

- HF：速度快、实现简单，但缺电子相关；
- MP2（本条目）：成本较 HF 高，但能显式补偿二阶相关，通常显著降能；
- CCSD(T)：精度更高，但实现和计算成本明显更重；
- DFT：成本与精度依泛函而变，物理可解释性与系统误差模式不同于 MP2。

本 MVP 的定位是：以最小代码展示“从 RHF 到 MP2”的可追踪路径。

## R18

`demo.py` 源码级算法流（8 步）：

1. `MP2Config` 固定系统规模、电子数与 SCF 超参数。
2. `build_toy_hamiltonian` 生成 `h_core`、`eri`、`E_nuc`，并做积分对称化。
3. `rhf_scf` 循环调用 `build_fock`，对角化得到轨道，更新密度并记录 `E_total/dE/dP/gap`。
4. SCF 结束后重新对角化最终 Fock，输出规范轨道系数与轨道能。
5. `transform_eri_to_mo` 用四指标 `einsum` 完成 AO->MO 积分变换。
6. `mp2_correlation_energy` 构造 `g_ijab`、`g_ijba` 与分母 `eps_i+eps_j-eps_a-eps_b`，逐项求和得到 `E_MP2`。
7. 提取绝对值最大的若干 `ijab` 贡献项，形成可读表格。
8. `validate_results` 执行 6 项检查，`main` 输出 `Validation: PASS/FAIL` 并据此返回退出码。

# 反kT算法 (Anti-kT Algorithm)

- UID: `PHYS-0404`
- 学科: `物理`
- 分类: `粒子物理`
- 源序号: `424`
- 目标目录: `Algorithms/物理-粒子物理-0424-反kT算法_(Anti-kT_Algorithm)`

## R01

反kT算法（anti-kT）是高能物理中最常用的喷注重建算法之一，属于顺序复合（sequential recombination）家族。它通过定义粒子-粒子与粒子-束流距离，迭代执行“合并或出队（成为最终喷注）”，并天然形成边界较圆滑、对软辐射稳定的喷注。

## R02

任务定义：给定末态粒子集合 `P = {p_i}`，每个粒子带有 `(pt, eta, phi, m)`，构造喷注集合 `J = {j_k}`。

目标：
- 将空间上相近、运动学相关的粒子聚合为同一喷注。
- 使高 `pt` 粒子主导喷注核心。
- 具备红外/共线稳定性，便于理论计算与实验分析一致。

## R03

物理背景与价值：
- 硬散射产生的夸克/胶子在探测器中不可直接观测，只能观测其强子化后的末态粒子。
- 喷注重建是从“可探测末态”映射回“parton 级动力学信息”的关键步骤。
- anti-kT 在 LHC 分析中广泛使用，因为它的结果在实验上可解释性强、系统学行为成熟。

## R04

本目录 `demo.py` 的输入输出约定：
- 输入：`list[Particle]`，字段为 `pid, pt, eta, phi, mass`。
- 输出：`list[Jet]`，字段为 `jet_id, pt, eta, phi, mass, n_constituents, constituents`。
- 运行行为：脚本自行生成一个可复现的 toy event，无需交互输入。

## R05

anti-kT 核心距离定义（半径参数 `R`）：

- 粒子-粒子距离
`d_ij = min(1/pt_i^2, 1/pt_j^2) * (DeltaR_ij^2 / R^2)`

- 粒子-束流距离
`d_iB = 1/pt_i^2`

- 角距离
`DeltaR_ij^2 = (eta_i - eta_j)^2 + DeltaPhi_ij^2`

其中 `DeltaPhi` 要在 `[-pi, pi)` 内取最短角差。

四动量重组采用 E-scheme：
`p_mu(new) = p_mu(i) + p_mu(j)`。

## R06

MVP 算法流程：
1. 将每个输入粒子转换成 pseudojet（四动量 + constituent id 列表）。
2. 在每轮迭代中先计算所有 `d_iB`。
3. 再计算所有成对 `d_ij`。
4. 选取最小距离。
5. 若最小为 `d_ij`，合并这两个 pseudojet。
6. 若最小为 `d_iB`，该 pseudojet 标记为最终喷注并移出活动集合。
7. 活动集合为空后结束，按 `pt_min` 过滤并按 `pt` 降序输出。

## R07

复杂度（朴素实现）：
- 每轮成对距离扫描为 `O(n^2)`。
- 迭代轮数上界约 `O(n)`。
- 总体时间复杂度约 `O(n^3)`，空间复杂度约 `O(n)`（不含输入输出）。

该复杂度适合教学和小样本验证，不适合大规模实验生产。

## R08

为什么 anti-kT 会形成“硬核 + 圆滑边界”喷注：
- 高 `pt` 粒子对应更小的 `d_iB`，优先成为稳定核心。
- 软粒子若靠近硬核，会因 `d_ij` 较小而先并入硬核。
- 远离硬核的软粒子倾向作为独立软喷注或被阈值过滤。

这一机制使喷注形状在 `(eta, phi)` 平面上更接近圆锥。

## R09

伪代码：

```text
active <- particle_to_pseudojet for all particles
final <- []
while active not empty:
    best <- min over all d_iB and d_ij
    if best is pair(i, j):
        active.remove(i, j)
        active.append(merge(i, j))
    else:
        active.remove(i)
        final.append(i)
jets <- map(final pseudojet -> jet kinematics)
jets <- filter(jets, pt >= pt_min)
return sort(jets by pt desc)
```

## R10

数值与边界处理：
- `pt` 下限保护：`safe_pt = max(pt, 1e-12)`，避免除零。
- `phi` 周期性：通过 `wrap_phi` 将角度差规约到 `[-pi, pi)`。
- 质量计算：`m2 = E^2 - |p|^2`，对浮点误差做 `max(m2, 0.0)`。
- 空事件输入：返回空喷注列表并正常结束。

## R11

关键参数：
- `R`：喷注半径，示例默认 `0.6`。
- `pt_min`：输出喷注阈值，示例默认 `8.0`。
- `seed`：toy event 随机种子，示例默认 `424`，保证复现。

## R12

`demo.py` 模块职责：
- 运动学工具：`wrap_phi`、`delta_phi`、坐标与四动量互转。
- 距离函数：`beam_distance`、`anti_kt_pair_distance`。
- 聚类主过程：`anti_kt_cluster`。
- 数据生成：`generate_toy_event`（两个硬核心 + 软背景）。
- 展示：`format_jets_table` 和 `main`。

## R13

运行方式：

```bash
uv run python Algorithms/物理-粒子物理-0424-反kT算法_(Anti-kT_Algorithm)/demo.py
```

或进入目录后运行：

```bash
cd Algorithms/物理-粒子物理-0424-反kT算法_(Anti-kT_Algorithm)
uv run python demo.py
```

## R14

示例输出结构（具体数值会随随机采样略有变化）：

```text
Generated particles: 90
Clustered jets (pt >= 8.0): 3
jet_id        pt      eta      phi      mass  n_const
------------------------------------------------------
J1       301.245    0.531    0.768    51.102       28
J0       279.733   -0.825   -2.270    48.665       26
J3        22.904    1.742   -0.418    10.334       11
```

## R15

最小验收标准：
- `README.md` 与 `demo.py` 中无占位符残留。
- `uv run python demo.py` 能直接完成运行，无交互。
- 输出包含喷注数量与喷注表格。
- 每个喷注至少展示 `pt/eta/phi/mass/n_constituents`。

## R16

当前 MVP 局限：
- 使用 toy 数据，不包含真实探测器分辨率、噪声和触发效应。
- 未包含 pileup 抑制、JES/JER 校准、ghost area 修正。
- 算法为朴素 `O(n^3)`，未采用 FastJet 的几何加速结构。

## R17

后续扩展建议：
- 引入真实 HepMC/ROOT 事件输入并做单位一致性验证。
- 增加 `kt` 与 `Cambridge/Aachen` 对照，实现同事件算法比较。
- 输出喷注子结构观测量（如 `groomed mass`, `N-subjettiness`）。
- 将距离扫描向量化或引入加速数据结构，降低大样本计算成本。

## R18

源码级流程追踪（`demo.py`，非黑盒，8 步）：
1. `generate_toy_event` 构造粒子列表，赋予每个粒子 `(pt, eta, phi, m)` 与唯一 `pid`。
2. `anti_kt_cluster` 调用 `particle_to_pseudojet`，把每个粒子转换为可合并对象（`p4 + constituents`）。
3. 在每轮循环中先遍历活动对象，调用 `beam_distance` 计算所有 `d_iB = 1/pt^2`。
4. 再做双重循环，调用 `anti_kt_pair_distance` 计算所有 `d_ij`，其中内部会用 `kinematics_from_p4` 和 `delta_phi` 求 `DeltaR`。
5. 选取全局最小距离；若是 `pair(i, j)`，则调用 `merge_pseudojets` 做 E-scheme 四动量相加并合并 constituent 列表。
6. 若最小是 `beam(i)`，该对象从活动集合移入 `final_pseudojets`，不再参与后续聚类。
7. 循环结束后，把 `final_pseudojets` 映射为 `Jet`，重算 `pt/eta/phi/mass` 并按 `pt_min` 过滤。
8. `main` 对结果按 `pt` 排序打印，输出喷注计数与表格，形成一键可验证的最小实验闭环。

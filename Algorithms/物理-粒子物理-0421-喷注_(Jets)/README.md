# 喷注 (Jets)

- UID: `PHYS-0402`
- 学科: `物理`
- 分类: `粒子物理`
- 源序号: `421`
- 目标目录: `Algorithms/物理-粒子物理-0421-喷注_(Jets)`

## R01

喷注（Jet）是高能碰撞中由夸克/胶子强子化后形成的准共线粒子簇。算法目标是把探测器中离散的末态粒子（或能量簇）重建为少量“物理可解释”的喷注对象。这里实现一个最小可运行的 `anti-kt` 顺序复合算法 MVP。

## R02

问题定义：给定一个事件中的粒子集合 `P = {p_i}`，每个粒子具有 `(pT, eta, phi, m)`，构造喷注集合 `J = {j_k}`，使得：
- 喷注在 `(eta, phi)` 空间近似局域。
- 高 `pT` 粒子优先成为喷注核心。
- 聚类规则红外/共线稳定（`anti-kt` 具备该性质）。

## R03

粒子物理背景：
- 在 LHC 等实验中，硬散射产生的 parton（夸克/胶子）不会直接被探测。
- parton 经过 parton shower + hadronization，最终留下大量稳定强子。
- 喷注算法是从“可观测末态”反推“原始硬过程方向/能量尺度”的核心步骤。

## R04

输入输出约定（对应 `demo.py`）：
- 输入：`list[Particle]`，每个 `Particle` 包含 `pid, pt, eta, phi, mass`。
- 输出：`list[Jet]`，每个 `Jet` 包含 `jet_id, pt, eta, phi, mass, n_constituents, constituents`。
- 额外输出：控制台打印事件粒子数、聚类后喷注数、按 `pT` 排序的喷注摘要。

## R05

`anti-kt` 距离度量：

- 粒子-粒子距离
`d_ij = min(1/pT_i^2, 1/pT_j^2) * (DeltaR_ij^2 / R^2)`

- 粒子-束流距离
`d_iB = 1/pT_i^2`

- 角距离
`DeltaR_ij^2 = (eta_i - eta_j)^2 + DeltaPhi_ij^2`

其中 `DeltaPhi` 需要做 `[-pi, pi)` 周期折返。

合并方案采用 E-scheme：直接四动量相加
`p_mu(new) = p_mu(i) + p_mu(j)`。

## R06

MVP 算法步骤：
1. 把输入粒子转成活动对象（保存四动量和 constituent 列表）。
2. 在每轮迭代中，计算全部 `d_iB`。
3. 计算全部成对 `d_ij`。
4. 取最小距离：
5. 若最小值是 `d_ij`，合并两对象并回到步骤 2。
6. 若最小值是 `d_iB`，该对象出队并标记为最终喷注。
7. 活动对象清空后，按 `pt_min` 做最终筛选并按 `pT` 排序。

## R07

复杂度分析：
- 朴素实现每轮需要 `O(n^2)` 计算成对距离，最多约 `O(n)` 轮。
- 总复杂度约 `O(n^3)`，空间复杂度 `O(n)`（不含输入输出存储）。
- 该复杂度足够支撑教学/验证级小事件，但不适合实验级大规模数据。

## R08

正确性直觉：
- 高 `pT` 粒子的 `d_iB` 小，倾向先“冻结”为喷注核心。
- 与高 `pT` 核心足够接近的软粒子，其 `d_ij` 可小于自身 `d_iB`，会先并入核心。
- 这使 `anti-kt` 形成近似圆锥且边界平滑的硬核喷注，符合实验常用行为。

## R09

伪代码：

```text
active <- particles_as_pseudojets
final_jets <- []
while active not empty:
    best <- +inf
    action <- none
    for i in active:
        if d_iB(i) < best:
            best <- d_iB(i)
            action <- beam(i)
    for (i, j) in all pairs active:
        if d_ij(i, j) < best:
            best <- d_ij(i, j)
            action <- merge(i, j)
    if action is merge(i, j):
        active <- active \ {i, j} U {combine(i, j)}
    else:
        active <- active \ {i}
        final_jets.append(i)
return sort(filter(final_jets, pt >= pt_min), key=pt desc)
```

## R10

边界情况处理：
- `pT` 极小：在实现中对 `pT` 取 `max(pt, 1e-12)`，避免除零。
- `phi` 跨越 `-pi/pi`：通过 `wrap_phi` 规约，保证角差最短路径。
- 合并后质量平方数值负零：使用 `max(m2, 0)` 抑制浮点误差。
- 空输入：返回空喷注列表并正常结束。

## R11

关键超参数：
- `R`：喷注半径，默认 `0.6`。
- `pt_min`：最终喷注 `pT` 阈值，默认 `10 GeV`（示例单位）。
- `seed`：随机事件生成器种子，默认 `7`，保证演示可复现。

## R12

`demo.py` 的实现范围：
- 包含一个合成事件发生器（两束硬核 + 软背景）。
- 包含完整 `anti-kt` 朴素聚类逻辑（无外部 Jet 库）。
- 输出喷注列表和每个喷注的 constituent 数量，便于人工检查聚类行为。

## R13

运行方式：

```bash
uv run python Algorithms/物理-粒子物理-0421-喷注_(Jets)/demo.py
```

脚本无交互输入，直接打印结果。

## R14

示例输出结构（数值会因随机样本略有变化）：

```text
Generated particles: 96
Clustered jets (pt >= 10.0): 2
jet_id        pt      eta      phi      mass  n_const
J0       391.503   -0.681   -2.102    66.354       29
J1       353.965    0.767    0.525    56.182       30
...
```

前两大喷注通常对应合成事件中的两束硬过程方向。

## R15

最小验收清单：
- `README.md` 与 `demo.py` 均已完成填充，无占位符残留。
- `uv run python demo.py` 可直接运行结束。
- 输出中至少出现 1 个 `pt >= pt_min` 的喷注（在默认参数下通常多个）。
- 每个喷注提供 `pt/eta/phi/mass/n_constituents`。

## R16

当前 MVP 局限：
- 事件数据为玩具模拟，不代表真实探测器响应。
- 未做 pileup mitigation、Jet Area 校正、能量刻度校准。
- 朴素复杂度较高，不含 FastJet 式几何加速结构。

## R17

可扩展方向：
- 接入真实 HepMC/ROOT 事件并增加单位与坐标一致性检查。
- 增加 `kt`、`Cambridge/Aachen` 对照，实现算法行为比较。
- 增加 jet substructure 指标（如 groomed mass、N-subjettiness）。
- 用 `numba` 或向量化重构距离扫描，降低运行时延。

## R18

源码级算法流（对应 `demo.py`，非黑盒）：
1. `generate_synthetic_event` 生成粒子四元组 `(pt, eta, phi, m)` 并封装为 `Particle`。
2. `anti_kt_cluster` 调用 `particle_to_pseudojet` 把每个粒子转为四动量表示。
3. 在主循环内，先遍历活动对象计算所有 `d_iB = 1/pT^2`。
4. 再双重循环计算所有 `d_ij`：
   `anti_kt_pair_distance` 内部依次调用 `kinematics_from_p4`、`delta_phi`，并按 `deta^2 + dphi^2` 构造 `delta_r2`。
5. 若最小项为 `pair`，通过 `merge_pseudojets` 执行 E-scheme 四动量相加并合并 constituent 列表。
6. 若最小项为 `beam`，把该对象移入 `final_jets`，不再参与后续距离比较。
7. 循环结束后，把 `final_jets` 逐个映射为 `Jet`（重算 `pt/eta/phi/mass`），并执行 `pt_min` 过滤与排序。
8. `main` 打印事件和喷注摘要，确保一键运行即可验证聚类链路。

# 拓扑绝缘体 (Topological Insulator)

- UID: `PHYS-0080`
- 学科: `物理`
- 分类: `凝聚态物理`
- 源序号: `80`
- 目标目录: `Algorithms/物理-凝聚态物理-0080-拓扑绝缘体_(Topological_Insulator)`

## R01

拓扑绝缘体（Topological Insulator, TI）的核心特征是“体绝缘、边导电”，其相稳定性由拓扑不变量而不是局域序参量决定。

本条目实现一个最小可运行 MVP：在 2D 晶格 BHZ（Bernevig-Hughes-Zhang）模型上，数值计算两自旋块的 Chern 数，并由 spin Chern 给出 `Z2` 指标，区分平庸相与拓扑相。

## R02

采用的模型是 decoupled BHZ 两块哈密顿量（每块 `2x2`）：

- 自旋上块：
  `H_up(k) = sin(kx) * sigma_x + sin(ky) * sigma_y + (m + cos(kx) + cos(ky)) * sigma_z`
- 自旋下块（时间反演伙伴）：
  `H_down(k) = sin(kx) * sigma_x - sin(ky) * sigma_y + (m + cos(kx) + cos(ky)) * sigma_z`

其中 `k=(kx,ky)` 取第一布里渊区离散网格，`m` 为质量参数。该模型具有明确的拓扑相变点（能隙闭合点）。

## R03

拓扑指标定义：

- 每个块的 Chern 数：`C_up`, `C_down`
- spin Chern：`C_s = (C_up - C_down) / 2`
- `Z2` 指标：`nu = |round(C_s)| mod 2`

在该简化模型中，时间反演对称保证 `C_down ~= -C_up`，因此 `C_s` 接近整数，`nu` 可稳定判别拓扑相（`1`）与平庸相（`0`）。

## R04

数值上用 Fukui-Hatsugai-Suzuki（FHS）离散规范场方法计算 Chern 数：

1. 在每个 `k` 点求占据带本征矢 `|u(k)>`（最低本征值对应态）。
2. 构造规范链变量
   `Ux(k)=<u(k)|u(k+dx)>/|<u(k)|u(k+dx)>|`，
   `Uy(k)=<u(k)|u(k+dy)>/|<u(k)|u(k+dy)>|`。
3. 每个 plaquette 的离散 Berry 通量：
   `F(k)=arg[Ux(k) * Uy(k+dx) * conj(Ux(k+dy)) * conj(Uy(k))]`。
4. Chern 数：
   `C = (1 / 2pi) * sum_k F(k)`。

该公式是离散规范不变量，对局域相位选择鲁棒。

## R05

`demo.py` 的主任务是扫描一组 `m`：

- `m = -3, -1, 1, 3`

并输出每个参数点的：

- `C_up`, `C_down`
- `spin_chern`
- `z2`
- `min_gap`（用于确认样本点未落在相变闭隙点）

其中 `m=±1` 应对应非平庸相（`z2=1`），`m=±3` 对应平庸相（`z2=0`）。

## R06

输入输出约定：

- 输入：脚本内固定配置（`nk` 网格、`masses` 参数列表、容错阈值）
- 无命令行交互，无外部数据文件
- 输出：
1. 相图结果表（pandas DataFrame）
2. 验证项逐条 PASS/FAIL
3. 末行 `Validation: PASS` 或 `Validation: FAIL`

满足 `uv run python demo.py` 一次执行即可复现实验。

## R07

复杂度分析（设动量网格边长为 `N`）：

- 单个 `m`、单个自旋块：
1. 本征分解成本：`O(N^2)`（每点仅 `2x2` 常数规模矩阵）
2. 链变量与通量累计：`O(N^2)`
- 两块合计仍为 `O(N^2)` 量级常数倍。

若扫描 `M` 个质量参数，总时间复杂度 `O(M * N^2)`，空间复杂度 `O(N^2)`（存占据态网格）。

## R08

数值稳定策略：

1. 对重叠归一化分母设下界 `eps`，避免 `0/0`。
2. 周期边界通过取模索引实现，避免边界缺口。
3. Chern 数最终与最近整数比较，使用容差判断，不直接硬等于整数。
4. 额外计算 `min_gap`，确保当前点不是闭隙临界点（临界点附近数值会不稳定）。

## R09

正确性检查由三层构成：

1. 拓扑结构检查：`C_down ~= -C_up`。
2. 整数量子化检查：`C_up`、`C_down` 必须接近整数。
3. 相判别检查：
   `m in {-1, 1}` 时 `z2=1`，
   `m in {-3, 3}` 时 `z2=0`。

若任一检查失败，脚本以非零状态退出。

## R10

最小工具栈：

- `numpy`：矩阵、复数线性代数、网格与向量化
- `pandas`：结果表格化与打印

不依赖高层拓扑材料软件包，核心算法可在源码中直接审计。

## R11

运行方式：

```bash
cd Algorithms/物理-凝聚态物理-0080-拓扑绝缘体_(Topological_Insulator)
uv run python demo.py
```

脚本无交互输入，默认会完成全部样例并输出验证结果。

## R12

输出字段说明：

- `mass`：质量参数 `m`
- `chern_up`：自旋上块 Chern 数（浮点近整数）
- `chern_down`：自旋下块 Chern 数（应与上块符号相反）
- `spin_chern`：`(chern_up - chern_down)/2`
- `z2`：`|round(spin_chern)| mod 2`
- `min_gap`：该 `m` 下网格最小直接能隙（用于检查是否远离临界点）

## R13

预期数值现象：

1. `m=±3` 时 `chern_up/chern_down` 接近 `0`，`z2=0`。
2. `m=-1` 时 `chern_up≈-1`、`chern_down≈+1`，`z2=1`。
3. `m=+1` 时 `chern_up≈+1`、`chern_down≈-1`，`z2=1`。
4. 所有测试点 `min_gap` 为正，说明样本点位于绝缘相而非临界相。

## R14

当前 MVP 的局限：

1. 是 2D 格点 toy model，不是特定真实材料第一性原理模型。
2. `Z2` 通过 block 结构下的 spin Chern 推断，未实现通用多带 Wilson loop 流程。
3. 未加入无序、相互作用、自能修正等复杂效应。
4. 只给体拓扑指标，不直接求边缘态谱。

## R15

常见失败模式与应对：

1. `nk` 过小：Chern 数偏离整数。对策：增大 `nk`。
2. 选到临界质量（如 `m≈-2,0,2`）：闭隙导致指标不稳定。对策：避开相变点或做细化分析。
3. 重叠过小导致相位噪声。对策：提升网格密度并保持 `eps` 防护。
4. 仅看单块 Chern 数会与 TRS 整体 Chern 为零“矛盾”。对策：使用 spin Chern / Z2 判据。

## R16

可扩展方向：

1. 增加 Wilson-loop (Wannier center flow) 求通用 `Z2`。
2. 构造条带几何并求边缘态色散，验证“体-边对应”。
3. 加入随机势并研究拓扑 Anderson 绝缘体。
4. 迁移到 3D 模型并计算强/弱拓扑指标。

## R17

适用场景：

1. 凝聚态课程中拓扑不变量数值演示。
2. 作为更复杂 TI 代码的最小单元测试基线。
3. 快速验证格点哈密顿量参数区间的拓扑相图。
4. 给后续“边界态 + 传输”模块提供体相先验标签。

## R18

`demo.py` 源码级算法流（8 步）：

1. `bhz_block_hamiltonian` 用 `sin/cos` 项和泡利矩阵在每个 `k` 组装 `2x2` 块哈密顿量（上/下自旋分别构造）。
2. `occupied_state` 调用 `numpy.linalg.eigh` 求本征分解并提取最低本征值对应本征矢，形成占据带态。
3. `occupied_manifold` 在 `nk x nk` 周期网格上遍历，存储每个格点的占据态。
4. `normalized_overlap` 计算相邻格点态重叠并除以模长，得到单位复相位链变量 `Ux/Uy`（含 `eps` 保护）。
5. `chern_number_fhs` 对每个 plaquette 计算离散通量相位并全局累加，得到 `C = sum(angle)/2pi`。
6. `compute_phase_point` 分别计算 `C_up` 与 `C_down`，再得到 `spin_chern` 与离散 `z2`。
7. `minimum_direct_gap` 直接从 `|d(k)|` 计算最小能隙，过滤临界附近不稳定样本。
8. `validate_results` 检查量子化、TR 关系和预期相标签，`main` 输出表格与 `Validation: PASS/FAIL`。

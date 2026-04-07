# 外尔半金属 (Weyl Semimetal)

- UID: `PHYS-0269`
- 学科: `物理`
- 分类: `凝聚态物理`
- 源序号: `272`
- 目标目录: `Algorithms/物理-凝聚态物理-0272-外尔半金属_(Weyl_Semimetal)`

## R01

外尔半金属（Weyl Semimetal, WSM）的核心特征是三维动量空间中的线性色散简并点（Weyl 节点），其节点手性（chirality）为拓扑荷，且必须成对出现（Nielsen-Ninomiya 配对约束）。

本条目实现一个最小可运行 MVP：

1. 用 3D 两带格点模型构造一个含单对外尔节点的 WSM 相。
2. 在固定 `kz` 的 2D 切片上，用 FHS 离散方法计算切片 Chern 数 `C(kz)`。
3. 通过 `C(kz)` 的跳变定位节点，并与解析节点位置对照验证。

## R02

采用哈密顿量：

`H(k) = d_x(k) * sigma_x + d_y(k) * sigma_y + d_z(k) * sigma_z`

其中

- `d_x = sin(kx)`
- `d_y = sin(ky)`
- `d_z = m + 2 - cos(kx) - cos(ky) - cos(kz)`

当 `|m| < 1` 时，存在一对外尔节点：

- `k_W± = (0, 0, ±k0)`
- `k0 = arccos(m)`

默认参数 `m=0`，得到节点 `kz=±pi/2`。

## R03

MVP 观测量与物理判据：

1. 切片 Chern 数 `C(kz)`：每个固定 `kz` 的 2D 切片可看作一个有效 Chern 绝缘体。
2. 节点判据：`C(kz)` 跨越节点时发生整数跳变（本模型中为 `0 <-> 1`）。
3. 能隙判据：节点处切片最小能隙趋近 0，远离节点时切片保持有限能隙。

## R04

切片 Chern 数使用 Fukui-Hatsugai-Suzuki（FHS）离散规范方法：

1. 在 `nk_xy x nk_xy` 的 `(kx, ky)` 网格上求占据带本征态 `|u(k)>`。
2. 构造规范链变量：
   `Ux(k)=<u(k)|u(k+dx)>/|<u(k)|u(k+dx)>|`，
   `Uy(k)=<u(k)|u(k+dy)>/|<u(k)|u(k+dy)>|`。
3. 对每个 plaquette 计算 Berry 通量：
   `F(k)=arg[Ux(k)Uy(k+dx)conj(Ux(k+dy))conj(Uy(k))]`。
4. 求和得到：
   `C(kz) = (1/2pi) * sum_k F(k)`。

该离散公式对局域相位选择不敏感，适合最小实现。

## R05

默认实验配置：

- `mass = 0.0`
- `nk_xy = 41`
- `nk_z = 61`（在 `[-pi, pi]` 均匀采样 `kz`）

预期行为：

1. `kz` 在 `(-pi/2, +pi/2)` 内部时 `C(kz)` 为非平庸（约 `+1`，符号依约定）。
2. `kz` 在区间外时 `C(kz)` 为平庸（约 `0`）。
3. 在 `kz≈±pi/2` 附近出现切片拓扑跳变，对应外尔节点。

## R06

输入输出约定：

- 输入：脚本内固定配置，不需要命令行参数，不读取外部文件。
- 输出：
1. 模型参数与理论节点位置
2. 由 Chern 跳变得到的数值节点位置
3. `kz` 采样预览表（`kz`, `chern`, `chern_round`, `min_gap_slice`）
4. 多条验证检查与最终 `Validation: PASS/FAIL`

满足 `uv run python demo.py` 一次执行即可复现。

## R07

复杂度分析（`Nxy = nk_xy`, `Nz = nk_z`）：

1. 单个 `kz` 切片：
   - 本征态求解 `O(Nxy^2)`（每点 `2x2` 常数维矩阵）
   - 链变量与 plaquette 通量 `O(Nxy^2)`
2. 扫描所有 `kz`：总时间 `O(Nz * Nxy^2)`。
3. 空间复杂度：每切片主要存储占据态与链变量，为 `O(Nxy^2)`。

## R08

数值稳定处理：

1. 归一化重叠使用 `overlap_eps`，避免极小重叠导致数值溢出。
2. `(kx, ky)` 周期边界通过取模索引实现，保证离散环路闭合。
3. Chern 量子化检查仅在“充分开隙切片”上执行，避开节点附近的非绝热不稳定区。
4. 节点定位使用相邻点线性插值，仅作为网格分辨率下的一阶估计。

## R09

`demo.py` 的自动校验包含五项：

1. `|m|<1`，模型处于可含外尔节点的参数区。
2. 在开隙切片上，`C(kz)` 必须接近整数。
3. `kz≈0` 的内部切片为非平庸，`kz≈pi` 的外部切片为平庸。
4. 从 Chern 跳变检测出的节点位置与解析节点 `±arccos(m)` 一致（容差检查）。
5. 在理论节点坐标 `(0,0,±k0)` 直接能隙趋近 0，且远离节点处切片保持有限能隙。

任一检查失败即返回非零退出码。

## R10

最小工具栈：

- `numpy`：复数线性代数、网格遍历、离散 Berry 通量计算
- `pandas`：结果组织和表格化输出

不依赖专用拓扑材料黑盒库，核心公式在源码中可直接审计。

## R11

运行方式：

```bash
cd Algorithms/物理-凝聚态物理-0272-外尔半金属_(Weyl_Semimetal)
uv run python demo.py
```

脚本无交互输入，默认直接完成计算与验证。

## R12

输出字段含义：

- `kz`：切片动量
- `chern`：FHS 计算得到的切片 Chern 数（浮点近整数）
- `chern_round`：`chern` 四舍五入后的整数平台值
- `min_gap_slice`：该 `kz` 切片在 `(kx, ky)` 网格上的最小直接能隙

另外单独输出：

- `theoretical nodes (kz)`：解析节点
- `detected nodes (kz)`：基于 `chern_round` 跳变反演的数值节点

## R13

默认配置下的典型现象：

1. 在 `kz` 扫描中可见两次整数平台切换，表示两处外尔节点。
2. 节点之间的切片为非平庸拓扑相，节点之外为平庸相。
3. 节点附近切片能隙塌缩，体现 3D 体相的无隙点特征。
4. 检测到的节点位置应接近 `±pi/2`（当 `m=0`）。

## R14

当前 MVP 局限：

1. 是简化两带格点 toy model，不对应具体材料第一性原理参数。
2. 仅演示体拓扑与节点定位，未计算表面态（Fermi arc）。
3. 未引入无序、相互作用、倾斜锥（type-II Weyl）等高阶效应。
4. 节点定位精度受 `kz` 网格分辨率限制。

## R15

常见失败模式与处理：

1. `|m|>=1`：外尔节点消失。处理：把 `mass` 调回 `(-1, 1)`。
2. `nk_xy` 过小：Chern 偏离整数。处理：增大 `nk_xy`。
3. `nk_z` 过小：节点定位粗糙。处理：增大 `nk_z`。
4. 在节点附近强行做“整数量子化检查”会误判。处理：仅对开隙切片检查。

## R16

可扩展方向：

1. 增加表面格林函数或条带模型，显式计算 Fermi arc。
2. 扫描 `mass` 生成 `m-kz` 的切片拓扑相图。
3. 在哈密顿量中加入倾斜项，研究 type-I / type-II 转变。
4. 引入随机势，考察节点稳定性和异常霍尔响应的鲁棒性。

## R17

适用场景：

1. 凝聚态拓扑课程中的 WSM 数值演示与作业基线。
2. 作为更复杂 3D 拓扑代码的单元验证模块。
3. 快速测试新哈密顿量是否存在节点和切片拓扑跳变。
4. 为后续输运、表面态和响应函数计算提供相位先验。

## R18

`demo.py` 源码级算法流（9 步）：

1. `weyl_hamiltonian` 用 `sin/cos` 项和泡利矩阵构造 `2x2` 格点外尔模型。
2. `occupied_state` 在每个动量点调用 `numpy.linalg.eigh` 并提取占据带本征矢。
3. `occupied_manifold_slice` 在固定 `kz` 上构建 `(kx, ky)` 占据态网格。
4. `normalized_overlap` 计算相邻格点态重叠并做单位化，形成 `Ux/Uy` 链变量。
5. `chern_number_slice_fhs` 对每个 plaquette 计算离散 Berry 通量并求和得到 `C(kz)`。
6. `min_direct_gap_slice` 在同一切片网格上计算最小直接能隙，区分开隙区与节点邻域。
7. `scan_kz_slices` 扫描全部 `kz`，生成包含 `chern`、`chern_round`、`min_gap_slice` 的 DataFrame。
8. `detect_nodes_from_chern_jumps` 根据 `chern_round` 相邻跳变并线性插值反演数值节点位置。
9. `validate_results` 逐项检查外尔相条件、切片量子化、内外区拓扑标签、节点位置与能隙行为（含 `direct_gap_point` 节点坐标处校验），`main` 输出结果并给出 `PASS/FAIL`。

# Wannier函数 (Wannier Functions)

- UID: `PHYS-0437`
- 学科: `物理`
- 分类: `固体物理`
- 源序号: `457`
- 目标目录: `Algorithms/物理-固体物理-0457-Wannier函数_(Wannier_Functions)`

## R01

Wannier 函数是 Bloch 波函数在实空间晶格上的局域基函数。对单带情形，其定义可写为：

`|w_R> = (1/sqrt(N_k)) * sum_k exp(-i k R) |psi_k>`

其中 `|psi_k>` 是规范化 Bloch 态，`R` 是晶格平移。若在代码里直接对周期部分 `u_k` 做离散变换，常用实现会采用 `1/N_k` 归一化（本条目 `demo.py` 即此约定）。它把动量空间离域表述转换为实空间局域表述，是固体电子结构和紧束缚建模的核心桥梁。

## R02

典型用途：

- 从第一性原理能带提取局域轨道模型（tight-binding）
- 计算极化、轨道中心与 Berry 相几何量
- 构建电子相互作用模型（Hubbard/U 等）所需的局域基
- 分析绝缘体/拓扑体系中“局域化”与“相位规范（gauge）”关系

## R03

数学关键点：

- Bloch 态具有相位自由度：`|u_k> -> exp(i phi(k)) |u_k>`
- 不同 `phi(k)` 不改变能带本征值，但会显著改变 Wannier 的空间局域性
- Wannier 概率密度（单带）：
  `P(R)=<w_R|w_R>=sum_alpha |w(R,alpha)|^2`
- 二阶矩（spread）常用于量化局域性：
  `Omega = <R^2> - <R>^2`

## R04

本条目 MVP 的问题设定：

- 模型：1D SSH 两亚晶格哈密顿量
  `H(k) = [[0, t1+t2*e^{-ik}], [t1+t2*e^{ik}, 0]]`
- 对象：下能带（single isolated band）
- 目标：
  1) 离散 k 网格上计算 Bloch 本征矢
  2) 逆傅里叶构造 Wannier 振幅 `w(R,alpha)`
  3) 比较“随机 gauge”与“平滑 gauge”下的局域化指标

## R05

为什么要做 gauge 平滑：

- 数值 `eigh` 给出的本征矢在每个 `k` 点都有任意相位
- 若直接用“相位乱跳”的本征矢做逆变换，Wannier 会被非物理相位噪声拉宽
- 通过并行输运（parallel transport）让相邻 `k` 点重叠尽量为实正，可显著压缩 spread
- 这不是改物理系统，而是选取更合适的表示规范

## R06

复杂度（`N_k` 为 k 采样点数）：

- 每个 `k` 上 2x2 `eigh`：常数级，合计 `O(N_k)`
- gauge 平滑单次扫描：`O(N_k)`
- Wannier 逆 Bloch 求和（矩阵乘）：
  - 构造相位矩阵 `N_k x N_k`
  - 乘以 `u_k` 后整体 `O(N_k^2)`
- 在本 MVP（`N_k=121`）中，计算毫秒级，重点是算法流程可解释与可验证

## R07

`demo.py` 的核心函数：

- `ssh_hamiltonian(k, t1, t2)`：构造 Bloch 哈密顿量
- `sample_lower_band(config)`：离散 k 上提取下能带本征矢
- `apply_random_gauge(u_k, seed)`：施加随机 `U(1)` 相位
- `parallel_transport_gauge(u_k)`：并行输运平滑并做周期闭合
- `build_wannier(u_k, k_grid)`：逆 Bloch 求和得到 `w(R,alpha)`
- `localization_metrics(...)`：输出 `norm/center/spread/ipr`

## R08

MVP 使用的局域化度量：

- `norm = sum_R P(R)`，应接近 1
- `center = sum_R R*P(R)/norm`
- `spread = sum_R R^2*P(R)/norm - center^2`
- `IPR = sum_R P(R)^2`（越大越局域）

验证逻辑：平滑 gauge 后应满足 `spread` 下降、`IPR` 上升。

## R09

稳定性与可复现性：

- 固定随机种子：`seed=20260407`
- 固定模型参数：`t1=0.8, t2=1.2, nk=121`
- 无交互输入，脚本直接打印结果表与断言状态
- 输出中心附近 `R` 点的概率剖面，便于人工检查“是否更尖锐”

## R10

正确性检查（脚本内 assert）：

- 随机 gauge 的 Wannier 归一化 `|norm-1| < 1e-10`
- 平滑 gauge 的 Wannier 归一化 `|norm-1| < 1e-10`
- `spread_smooth < spread_random`
- `ipr_smooth > ipr_random`

这些检查保证：流程既保持物理归一化，又体现了规范选择对局域性的实际影响。

## R11

边界与异常处理：

- `build_wannier` 检查实空间网格与 `k` 网格点数一致
- 当相邻态重叠过小（数值零）时，平滑步骤跳过相位校正，避免除零
- 所有矩阵与向量均用复数 dtype，防止实数截断导致相位信息丢失

## R12

与相关概念的关系：

- Bloch 函数：动量空间表述，天然离域
- Wannier 函数：实空间局域表述，便于构建局域哈密顿量
- Berry 相/极化：本质依赖 `k` 空间规范连接；Wannier 中心可与其对应
- 最大局域 Wannier（MLWF）是更一般优化问题；本 MVP 只实现单带下最小可行 gauge 平滑

## R13

本实现的简化假设：

- 仅处理单个孤立能带，不含多带纠缠 disentanglement
- 模型是一维 SSH，不涉及三维晶体群对称约束
- 使用离散傅里叶求和，不引入 Wannier90 等外部工程框架

这些简化保证代码短小透明，适合教学和自动化验证。

## R14

工程实现注意点：

- `np.linalg.eigh` 只负责本征分解，无法自动处理跨 `k` 点相位连续性
- 周期闭合步骤会把末端残余相位均匀分摊到全 k 网格，避免边界跳相
- 由于是离散网格，`center` 可能不是整数格点，这是正常离散化现象
- `nk` 取奇数可让中心附近剖面展示更直观

## R15

运行方式：

```bash
cd Algorithms/物理-固体物理-0457-Wannier函数_(Wannier_Functions)
uv run python demo.py
```

或在仓库根目录：

```bash
uv run python Algorithms/物理-固体物理-0457-Wannier函数_(Wannier_Functions)/demo.py
```

## R16

可扩展方向：

- 多带情形：加入子空间旋转 `U_mn(k)`，做真正的 MLWF spread 最小化
- 更高维：`k=(kx,ky,kz)` 下处理三维晶格与各向异性 spread 张量
- 与第一性原理接口：从 ab-initio 导出的 Bloch 态构造有效模型
- 增加拓扑指标（Zak/Berry phase）并与 Wannier 中心演化联动展示

## R17

本条目交付物说明：

- `README.md`：R01-R18 已完整填写，覆盖定义、复杂度、实现、验证与扩展
- `demo.py`：可直接运行的 Python MVP，输出局域化度量与中心概率表
- `meta.json`：保持与任务元数据一致（UID/学科/分类/源序号/目录）

## R18

源码级算法流程拆解（对应 `demo.py`，8 步）：

1. **离散化 k 空间并构造 SSH 哈密顿量**  
   `sample_lower_band` 在均匀 `k` 网格上调用 `ssh_hamiltonian` 生成 `2x2` 复矩阵。

2. **逐点本征分解选取目标能带**  
   用 `np.linalg.eigh` 求每个 `k` 的本征对，选最小本征值对应本征矢作为下能带 `u_k`。

3. **施加任意 U(1) 相位扰动（基线）**  
   `apply_random_gauge` 为每个 `k` 乘上随机相位，构造“未平滑 gauge”作为对照。

4. **并行输运平滑相位**  
   `parallel_transport_gauge` 逐点最大化相邻重叠 `vdot(u_{k-1},u_k)`，消除相位乱跳。

5. **周期闭合修正**  
   计算末端与起点残余相位 `gamma`，并在全网格均匀分摊，得到周期一致的平滑规范。

6. **逆 Bloch 求和生成 Wannier 振幅**  
   `build_wannier` 计算 `w(R,alpha)=(1/N_k)*sum_k exp(-ikR)u_k(alpha)`，得到实空间振幅矩阵。

7. **计算局域化统计量**  
   `localization_metrics` 从 `P(R)` 提取 `norm/center/spread/ipr`，比较随机与平滑规范。

8. **输出剖面并执行断言验证**  
   打印中心区 `P(R)` 表格；断言归一化保持、`spread` 下降、`IPR` 上升，给出可自动校验结论。

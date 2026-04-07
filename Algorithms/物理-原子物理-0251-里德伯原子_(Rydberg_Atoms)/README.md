# 里德伯原子 (Rydberg Atoms)

- UID: `PHYS-0248`
- 学科: `物理`
- 分类: `原子物理`
- 源序号: `251`
- 目标目录: `Algorithms/物理-原子物理-0251-里德伯原子_(Rydberg_Atoms)`

## R01

本条目给出“里德伯原子”最小可运行实现（MVP），聚焦一个可验证的核心链路：
- 用量子亏损修正计算高主量子数束缚能级；
- 用幂律 `C6 ~ n^11` 估计范德瓦耳斯相互作用强度；
- 由 `Rb = (|C6|/(hbar*Omega))^(1/6)` 计算阻塞半径；
- 在二维原子云中构造阻塞图，并用随机贪心近似最大独立集，估计可同时激发的里德伯原子数。

## R02

问题定义（MVP范围）：
- 输入：
  - 量子数与谱线参数：`n`、`delta_l`、`n_transition_target`；
  - 相互作用参数：`n_ref`、`c6_ref_hz_um6`、`rabi_omega_rad_s`；
  - 原子云与算法参数：`n_atoms`、`cloud_side_um`、`restarts`、`seed`。
- 输出：
  - `n*`、`E_n`、相邻能级跃迁频率（GHz）；
  - `C6`（`J*m^6`）与阻塞半径 `Rb`（um）；
  - 阻塞图平均度、近似最大可同时激发原子数、激发分数；
  - 前若干个原子的坐标与是否被选入激发集合的预览表。
- 约束：
  - 本实现是教学/验证级近似模型，不直接拟合具体实验装置全部细节。

## R03

数学模型：
1. 量子亏损修正主量子数：`n* = n - delta_l`。
2. 里德伯束缚能近似：`E_n = -Ry / (n*)^2`。
3. 跃迁频率：`nu = |E_{n2} - E_{n1}| / h`。
4. 范德瓦耳斯系数缩放：`C6(n) = C6_ref * (n/n_ref)^11`。
5. 阻塞半径：`Rb = (|C6|/(hbar*Omega))^(1/6)`。
6. 阻塞图定义：若两原子距离 `r_ij < Rb`，则边 `A_ij = 1`，表示不能同时激发。
7. 同时可激发集合对应图的独立集；最大并发激发数对应最大独立集（MVP里用近似算法求解）。

## R04

算法流程（MVP）：
1. 读取默认参数并初始化随机数生成器。
2. 计算 `n*`、`E_n` 与 `n -> n+1` 的跃迁频率。
3. 把参考 `C6` 从 `Hz*um^6` 换算到 `J*m^6`，再用 `n^11` 缩放得到目标态 `C6`。
4. 根据驱动强度 `Omega` 计算阻塞半径 `Rb`。
5. 在二维正方形云中均匀采样原子坐标。
6. 构造两两距离矩阵并得到阻塞邻接矩阵。
7. 多次随机重启执行贪心独立集搜索，取最佳结果。
8. 计算统计量、做一致性检查并打印摘要与预览表。

## R05

核心数据结构：
- `numpy.ndarray`：
  - `positions_um`：形状 `(N, 2)` 的原子坐标；
  - `adjacency`：形状 `(N, N)` 的布尔邻接矩阵；
  - `selected_indices`：近似最大独立集索引。
- `RydbergSimulationResult`（`dataclass`）：
  - 统一保存物理量（能级、频率、`C6`、`Rb`）和图统计量（平均度、可并发激发数等）。

## R06

正确性要点：
- `effective_principal_quantum_number` 强制 `n* > 0`，避免无物理意义输入。
- `E_n < 0` 符合束缚态能量符号；`nu > 0` 符合跃迁频率定义。
- `Rb` 从 `C6`、`hbar`、`Omega` 的量纲关系直接计算，保证单位链路可追踪。
- 选中集合通过 `adjacency[np.ix_(s,s)]` 校验无边，确保满足“阻塞下可同时激发”的独立集定义。
- 随机多重启贪心并不保证全局最优，但能稳定给出可复现、可解释的近似上界估计。

## R07

复杂度分析：
- 设原子数为 `N`，随机重启次数为 `R`。
- 构图（全对距离）时间复杂度 `O(N^2)`，空间复杂度 `O(N^2)`。
- 每次贪心扫描最坏 `O(N^2)`（布尔阻塞向量更新），总计 `O(R*N^2)`。
- 整体复杂度约为 `O((R+1)*N^2)`，对 MVP 默认参数（`N=120`, `R=350`）可快速运行。

## R08

边界与异常处理：
- `n <= 0`、`n* <= 0`、`n_ref <= 0`、`C6_ref <= 0`、`Omega <= 0` 直接抛出 `ValueError`。
- `n_atoms <= 0` 或 `cloud_side_um <= 0` 直接抛出 `ValueError`。
- 坐标数组不是 `(N,2)` 时拒绝构图。
- `restarts <= 0` 时拒绝运行近似最大独立集搜索。

## R09

MVP取舍说明：
- 保留里德伯原子最关键的三个尺度关系：`E_n`、`C6 ~ n^11`、`Rb`。
- 用“阻塞图 + 独立集”替代完整多体主方程求解，换取更小实现体量和更高可解释性。
- 不调用外部黑盒求解器；每一步都在源码中显式展开。
- 结果定位为“结构正确、数量级合理”的原型，不替代高精度实验拟合工具。

## R10

`demo.py` 模块职责：
- `effective_principal_quantum_number`：计算并校验 `n*`。
- `rydberg_binding_energy_joule`：计算里德伯束缚能。
- `transition_frequency_ghz`：计算跃迁频率（GHz）。
- `c6_from_scaling_j_m6`：执行 `C6` 的 `n^11` 缩放。
- `blockade_radius_um`：计算阻塞半径。
- `sample_positions_square_um`：生成二维原子云。
- `blockade_adjacency`：构造阻塞邻接矩阵。
- `randomized_greedy_independent_set`：随机重启贪心近似 MIS。
- `run_rydberg_blockade_mvp`：串联完整流程。
- `run_checks` 与 `main`：做一致性检查并输出结果。

## R11

运行方式：

```bash
cd Algorithms/物理-原子物理-0251-里德伯原子_(Rydberg_Atoms)
uv run python demo.py
```

脚本无交互输入，运行后直接打印物理量、图统计量、样本预览，并在末尾输出 `All checks passed.`。

## R12

输出字段解读：
- `n*`：量子亏损修正后的有效主量子数。
- `Binding energy E_n`：该里德伯态的束缚能（J）。
- `Transition frequency`：`n -> n+1` 的能级差对应频率（GHz）。
- `Estimated C6`：按缩放律估计的范德瓦耳斯系数（`J*m^6`）。
- `Blockade radius Rb`：阻塞判据长度尺度（um）。
- `Mean blockade-graph degree`：平均每个原子的阻塞冲突邻居数。
- `Approx. max simultaneous excitations`：近似最大同时激发数。
- `Excitation fraction`：可同时激发占总原子数比例。

## R13

建议最小测试集：
- 正常场景：默认参数，应通过全部检查并输出合理数量级 `Rb`（微米级）。
- 低密度场景：增大 `cloud_side_um`，应提高可同时激发比例。
- 高密度场景：减小 `cloud_side_um`，应降低可同时激发比例。
- 异常输入：
  - `n <= delta_l`；
  - `Omega <= 0`；
  - `n_atoms <= 0`；
  - `restarts <= 0`。

## R14

关键可调参数：
- `n`：主量子数，增大后通常导致更强相互作用（经 `C6` 缩放）与更大阻塞半径。
- `delta_l`：量子亏损，影响 `n*` 与能级位置。
- `c6_ref_hz_um6`、`n_ref`：设置 `C6` 经验缩放锚点。
- `rabi_omega_rad_s`：驱动强度，增大时通常减小 `Rb`。
- `n_atoms`、`cloud_side_um`：控制空间密度与阻塞图稀疏度。
- `restarts`：近似 MIS 的搜索强度，越大通常越接近更优独立集。
- `seed`：保证可复现。

## R15

方法对比：
- 对比“直接给经验公式不做几何模拟”：
  - 经验公式只能给单点尺度；
  - 本实现额外给出有限尺寸原子云中可并发激发数估计。
- 对比“全量多体量子动力学求解”：
  - 全量求解更精确但实现重、参数多；
  - 本实现更适合教学、快速验证与参数敏感性初筛。
- 对比“第三方黑盒优化器直接求 MIS”：
  - 黑盒调用更短；
  - 本实现明确暴露随机重启贪心流程，便于审计与扩展。

## R16

应用场景：
- 里德伯阻塞效应教学演示与数量级估算；
- 中性原子量子比特阵列的初步几何布局评估；
- 参数扫描前的快速可行性检查（如 `Omega`、密度、`n` 的影响）；
- 图论视角解释多体阻塞约束（物理问题转图优化问题）。

## R17

可扩展方向：
- 从二维正方形扩展到三维云或规则晶格；
- 在 `C6` 中引入角向各向异性和态依赖修正；
- 用局部搜索/退火或分支界算法替代贪心，提升 MIS 质量；
- 进一步引入时间演化，连接到脉冲序列和门保真度评估；
- 结合实验标定数据，对 `C6_ref` 与量子亏损进行反推拟合。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 调用 `run_rydberg_blockade_mvp`，设定默认物理与仿真参数。  
2. `effective_principal_quantum_number` 计算 `n*`，随后 `rydberg_binding_energy_joule` 给出 `E_n`。  
3. `transition_frequency_ghz` 用 `|E_{n+1}-E_n|/h` 计算跃迁频率。  
4. `run_rydberg_blockade_mvp` 将参考 `C6_ref` 从 `Hz*um^6` 转到 `J*m^6`，并在 `c6_from_scaling_j_m6` 中按 `(n/n_ref)^11` 缩放到目标态。  
5. `blockade_radius_um` 按 `Rb=(|C6|/(hbar*Omega))^(1/6)` 计算阻塞半径。  
6. `sample_positions_square_um` 随机生成 `N` 个二维坐标，再由 `blockade_adjacency` 构造阻塞邻接矩阵。  
7. `randomized_greedy_independent_set` 进行多次随机重启贪心，返回最大已找到独立集作为“可同时激发原子集合”。  
8. `run_checks` 验证物理量与独立集约束，`main` 打印统计摘要和表格预览并输出 `All checks passed.`。  

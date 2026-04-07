# 马约拉纳费米子 (Majorana Fermion)

- UID: `PHYS-0271`
- 学科: `物理`
- 分类: `粒子物理`
- 源序号: `274`
- 目标目录: `Algorithms/物理-粒子物理-0274-马约拉纳费米子_(Majorana_Fermion)`

## R01

马约拉纳费米子是满足“粒子等于反粒子”的费米激发，其算符满足 `gamma = gamma^dagger`。在高能物理中常被讨论为中微子类型候选；在凝聚态中，更常见的是**马约拉纳零能模（Majorana zero mode）**，即拓扑超导边界处接近零能量的准粒子激发。

本条目以一维 Kitaev 链为算法载体：把哈密顿量写成 BdG（Bogoliubov-de Gennes）矩阵，数值对角化后通过“近零能谱 + 边缘局域性”识别 Majorana 模态。

## R02

典型应用场景：

- 拓扑超导相的数值判别（拓扑相 vs 平庸相）
- 量子器件建模中的零能模可见性评估
- 非阿贝尔任意子和拓扑量子计算的前置仿真
- 教学与研究中的最小模型验证（Kitaev chain 基线）

## R03

`demo.py` 使用的数学模型：开边界的一维 Kitaev 链

`H = -mu * sum_i (c_i^dagger c_i - 1/2) - t * sum_i (c_i^dagger c_{i+1} + h.c.) + Delta * sum_i (c_i c_{i+1} + h.c.)`

在 Nambu 基 `Psi = (c_1,...,c_N,c_1^dagger,...,c_N^dagger)^T` 下，写成 BdG 块矩阵：

`H_BdG = [[A, B], [-B, -A]]`

其中：

- `A`：常规跃迁与化学势（实对称）
- `B`：p-wave 配对（实反对称）

再对 `H_BdG` 做实对称特征分解得到能谱与本征态。

## R04

直观理解：

- 参数区间 `|mu| < 2|t|` 对应拓扑超导相（在无限长链近似下）
- 有限长度开链在拓扑相会出现一对接近零的能级 `±E`（`E` 很小）
- 这对态在空间上集中在链两端，体现“端点马约拉纳模”
- 平庸相通常无近零边界模，最低激发保持有限能隙

## R05

正确性要点（本 MVP 的可验证证据链）：

1. `H_BdG` 必须是厄米矩阵（本例为实对称），否则谱不具备物理意义。
2. 粒子-空穴对称要求 `tau_x H^* tau_x = -H`，能谱应成 `±E` 配对。
3. 在 `|mu| < 2|t|` 且开边界时，有限链可见接近 0 的一对本征值。
4. 对应本征态在边缘几格上的概率权重应显著高于体区。
5. 与 `|mu| > 2|t|` 的平庸参数组比较，可形成对照验证。

## R06

复杂度分析（链长 `N`）：

- 构造 BdG 矩阵维度为 `2N x 2N`
- 组装矩阵时间复杂度 `O(N)`，存储复杂度 `O(N^2)`（稠密表示）
- 稠密特征分解复杂度约 `O((2N)^3) = O(N^3)`
- 本 MVP 取 `N=60`，单机 CPU 上可快速运行

## R07

标准实现步骤：

1. 给定 `N, t, Delta, mu` 构造 `A` 与 `B` 两个块矩阵。
2. 拼装 `H_BdG = [[A, B], [-B, -A]]`。
3. 检查厄米性与粒子-空穴对称误差。
4. 用 `np.linalg.eigh` 对角化得到全部本征值/本征向量。
5. 按 `|E|` 排序，读取最小绝对值能级。
6. 用阈值 `zero_tol` 统计近零模数量。
7. 计算最低两模在边缘 `k` 个格点的局域权重。
8. 对拓扑参数组与平庸参数组做并排比较并输出结论。

## R08

`demo.py` 的 MVP 设计：

- 依赖：`numpy`
- 模型：开边界 Kitaev 链，实参数 `t, Delta, mu`
- 输出：
  - 最小 `|E|` 能级列表
  - 近零模计数
  - 边缘局域权重
  - 粒子-空穴对称残差
- 运行：`uv run python demo.py`（无需交互输入）

## R09

`demo.py` 主要接口：

- `KitaevParams`：参数数据类
- `build_kitaev_bdg(params) -> np.ndarray`
- `validate_bdg_hamiltonian(hamiltonian, atol=1e-10) -> None`
- `particle_hole_symmetry_error(hamiltonian) -> float`
- `diagonalize_bdg(hamiltonian) -> tuple[np.ndarray, np.ndarray]`
- `edge_localization_score(eigenvector, n_sites, edge_sites) -> float`
- `analyze_phase(label, params) -> dict[str, object]`
- `main() -> None`

## R10

测试策略：

- 结构测试：`H_BdG` 是否方阵、偶数维、厄米
- 对称性测试：`tau_x H^* tau_x + H` 的 Frobenius 残差应很小
- 物理对照测试：
  - 拓扑相（`mu=0`）应出现至少 2 个近零模
  - 平庸相（`mu=3`）应无近零模
- 局域性测试：拓扑相最低模边缘权重应显著高（本例阈值 > 0.5）

## R11

边界条件与异常处理：

- `n_sites < 3`：抛 `ValueError`
- `hopping == 0` 或 `pairing == 0`：抛 `ValueError`（本 MVP 聚焦拓扑超导区）
- BdG 维度非法或非厄米：抛 `ValueError`
- 若拓扑/平庸对照未满足预期：抛 `RuntimeError`

数值边界说明：有限链会导致零模能量劈裂，不一定精确等于 0，因此采用容差阈值 `zero_tol` 而非硬等式。

## R12

与相关模型关系：

- Kitaev 链是最小一维拓扑超导模型
- 连续模型可离散化到该格点形式
- 更高维或含自旋模型可扩展为更大 BdG 块结构
- 本条目关注“零模识别”而非编织（braiding）动力学模拟

## R13

示例参数（`demo.py`）：

- 链长：`N = 60`
- 跃迁：`t = 1.0`
- 配对：`Delta = 0.8`
- 近零阈值：`zero_tol = 1e-2`
- 边缘统计格点：`edge_sites = 3`

两组化学势：

- 拓扑相：`mu = 0.0`（满足 `|mu| < 2|t|`）
- 平庸相：`mu = 3.0`（满足 `|mu| > 2|t|`）

## R14

工程实现注意点：

- Nambu 基的上下半区索引必须一致，否则 `A/B` 块会错位
- `B` 的反对称号位（`B[i,i+1]=+Delta, B[i+1,i]=-Delta`）决定 p-wave 配对符号
- 近零模检测不应使用“等于 0”，必须用容差
- 边缘权重需把粒子分量和空穴分量同时计入

## R15

最小示例输出解读：

- 拓扑相会显示两条非常靠近 0 的能级（`±E`）
- 这两模边缘权重接近 1，表明端点局域
- 平庸相最低 `|E|` 远离 0，且边缘权重很小
- 粒子-空穴对称误差接近机器精度，说明矩阵构造一致

## R16

可扩展方向：

- 引入无序项 `mu_i` 研究拓扑稳定性
- 加入长程跃迁/长程配对做相图扫描
- 用稀疏矩阵与 Lanczos 只求低能谱（大系统）
- 扩展到时间依赖参数，研究淬火动力学与缺陷生成

## R17

本条目交付说明：

- `README.md`：完成 R01-R18 的模型、算法与工程说明
- `demo.py`：可运行 MVP，输出拓扑相/平庸相对照结果
- `meta.json`：保留与任务元数据一致

执行方式（目录内）：

```bash
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，9 步）：

1. **参数封装**
   `KitaevParams` 固化 `N, t, Delta, mu, zero_tol, edge_sites`，避免硬编码散落。

2. **构造正常块 `A`**
   `build_kitaev_bdg` 先填充 `A` 的对角项 `-mu`，再在相邻格点写入 `-t` 跃迁。

3. **构造配对块 `B`**
   在相邻格点写入反对称配对：`B[i,i+1]=+Delta, B[i+1,i]=-Delta`。

4. **拼装 BdG 总矩阵**
   以块矩阵形式生成 `H_BdG=[[A,B],[-B,-A]]`，显式保留粒子-空穴结构。

5. **物理一致性校验**
   `validate_bdg_hamiltonian` 检查方阵、偶数维、厄米；`particle_hole_symmetry_error` 计算 `tau_x H^* tau_x + H` 的相对 Frobenius 误差。

6. **特征分解内核调用**
   `diagonalize_bdg` 调用 `np.linalg.eigh`。其底层走 LAPACK 对称本征流程：
   - 先用 Householder 变换把实对称矩阵化为三对角矩阵；
   - 再用分治/QR 家族算法求三对角本征对；
   - 最后回代得到原矩阵本征向量。

7. **近零谱提取**
   对本征值按 `|E|` 排序，依据 `zero_tol` 统计近零模数并输出最小 `|E|` 列表。

8. **边缘局域度计算**
   `edge_localization_score` 把本征向量拆成粒子+空穴概率，累加两端 `edge_sites` 格点权重，量化“是否端点局域”。

9. **拓扑/平庸双相对照并断言**
   `main` 对 `mu=0` 与 `mu=3` 两组参数重复步骤 2-8，最后用断言保证：拓扑相有零模且边缘局域，平庸相无零模。

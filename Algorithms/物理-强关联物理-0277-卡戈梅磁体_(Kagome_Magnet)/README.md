# 卡戈梅磁体 (Kagome Magnet)

- UID: `PHYS-0274`
- 学科: `物理`
- 分类: `强关联物理`
- 源序号: `277`
- 目标目录: `Algorithms/物理-强关联物理-0277-卡戈梅磁体_(Kagome_Magnet)`

## R01

卡戈梅磁体描述的是自旋位于卡戈梅（Kagome）几何格点上的磁性体系。该格点由“角共享三角形”构成，典型特征是几何阻挫（geometric frustration）：最近邻反铁磁相互作用无法在每条键上同时最小化能量，从而产生高度简并的低能态族。

## R02

本条目采用经典最近邻 Heisenberg 反铁磁模型作为最小算法对象：

`H = J * sum_{<i,j>} (S_i · S_j),  J > 0,  |S_i| = 1`

其中 `S_i` 是三维单位向量。目标不是复现实验级材料细节，而是构建一个可运行、可检查的 MVP，展示 Kagome 体系的核心数值现象：
- 低净磁矩；
- 显著负的最近邻关联；
- 由阻挫导致的低能态复杂性（通过能量与手性统计体现）。

## R03

离散化与几何构造：
- Bravais 基矢：`a1=(1,0)`, `a2=(1/2, sqrt(3)/2)`；
- 三点基底：`b0=(0,0), b1=(1/2,0), b2=(1/4,sqrt(3)/4)`；
- 系统大小：`Lx x Ly` 个原胞，总站点数 `N=3*Lx*Ly`；
- 周期边界：通过超胞矢量 `A=Lx*a1, B=Ly*a2` 搜索最近镜像距离，自动识别最近邻键。

该几何法不硬编码邻接表，而是由坐标+最近距离判定生成图结构，便于调试与扩展。

## R04

求解策略使用模拟退火 + Metropolis 单自旋更新：

1. 随机初始化单位自旋场；
2. 设温度日程（`beta` 从小到大线性增长）；
3. 每个 sweep 对随机站点尝试局部旋转；
4. 用 `dE` 与 `exp(-beta*dE)` 判据接受/拒绝；
5. 记录能量、磁矩、最近邻关联与三角手性。

该算法直接对应统计物理的经典采样流程，代码体量小，适合 Stage0 的“可运行、可解释”目标。

## R05

`demo.py` 实现的 MVP 能力：
- 自动构建周期 Kagome 几何与最近邻键；
- 校验每个站点配位数是否为 4（Kagome 最近邻图的关键结构特征）；
- 执行退火采样并输出时间序列 DataFrame；
- 计算最终结构因子峰值（粗网格）作为磁结构诊断；
- 输出测量窗口均值，便于快速判断体系是否进入低能阻挫态。

## R06

输入（脚本内 `KagomeConfig`）：
- `Lx, Ly`：二维超胞尺寸；
- `J`：交换常数（默认 `1.0`，反铁磁）；
- `sweeps, burn_in`：总迭代与测量起点；
- `beta_start, beta_end`：退火温度区间；
- `proposal_sigma`：局部旋转幅度；
- `seed`：随机种子。

输出（终端）：
- 系统规模、键数、三角数、配位一致性；
- 最后若干 sweep 的表格（`energy_per_spin`, `magnetization`, `nn_dot`, `abs_chirality`）；
- 测量窗口平均值与结构因子峰值位置。

## R07

高层伪代码：

1. 生成 Kagome 坐标与周期超胞矢量。
2. 基于最近镜像距离识别最近邻键。
3. 由键构建邻接表并枚举三角形团簇。
4. 随机初始化单位向量自旋。
5. 按 `beta` 日程循环执行 Metropolis sweep。
6. 每个 sweep 记录能量、磁矩、最近邻点积、手性。
7. 在测量窗口统计均值并计算结构因子峰。
8. 打印诊断并执行有限性断言。

## R08

记 `N=3LxLy`，平均配位数固定为 4：
- 单次 sweep 尝试 `N` 次局部更新，每次 `O(1)` 访问邻居，因此 `O(N)`；
- 全部采样复杂度 `O(sweeps * N)`；
- 构图阶段用成对距离扫描，约 `O(N^2)`（一次性成本）；
- 内存主要为自旋数组与邻接关系，约 `O(N)`。

## R09

数值稳定与工程细节：
- 每次提案后强制归一化自旋，避免长度漂移；
- 周期最近镜像搜索限定在 `(-1,0,1)` 平移块，确保稳定识别邻接；
- 接受率过低时可增大温度或减小 `proposal_sigma`；
- 通过固定 `seed` 保证复现实验；
- 关键统计量上做 `np.isfinite` 断言，防止静默异常。

## R10

MVP 层面的正确性检查：
- 所有站点配位数应统一为 4；
- 最近邻键数应为 `2N`（由 `Nz/2` 给出，`z=4`）；
- `energy_per_spin` 在降温后应趋于负值并稳定；
- `magnetization` 在反铁磁阻挫体系中通常保持较小；
- 所有输出指标不得出现 NaN/Inf。

## R11

默认参数：
- `Lx=4, Ly=4`（`N=48`）；
- `sweeps=180, burn_in=70`；
- `beta_start=0.2, beta_end=6.5`；
- `proposal_sigma=0.35`。

调参建议：
- 更平滑退火：增加 `sweeps`；
- 更低温探索：提高 `beta_end`；
- 采样效率不足：调 `proposal_sigma` 使接受率约在 `0.2~0.6`；
- 统计波动大：增大 `Lx, Ly` 并延长测量窗口。

## R12

运行方式：

```bash
cd Algorithms/物理-强关联物理-0277-卡戈梅磁体_(Kagome_Magnet)
uv run python demo.py
```

脚本无交互输入，会直接打印诊断结果。

## R13

输出解读：
- `energy_per_spin`：越低表示越接近低能配置；
- `nn_dot`：最近邻平均点积，反铁磁下应偏负；
- `magnetization`：总磁矩模，若长期较小说明未出现简单铁磁序；
- `abs_chirality`：三角标量手性绝对值均值，反映非共面倾向；
- `structure_factor_peak` 与 `(h,k)`：粗粒度动量空间峰值指示可能的主导相关波矢。

## R14

局限性：
- 经典自旋模型，不含量子涨落与自旋子等量子效应；
- 仅最近邻各向同性交换，未含 DMI、各向异性、外场；
- 结构因子仅做粗网格扫描，非高分辨谱分析；
- 单链退火采样，未做严格误差棒与自相关时间评估。

## R15

可扩展方向：
- 增加次近邻交换 `J2`、Dzyaloshinskii-Moriya 相互作用与 Zeeman 项；
- 从经典 MC 升级到并行回火（parallel tempering）；
- 对接 PyTorch/JAX 实现可微分能量优化；
- 增加动力学结构因子与时间关联函数分析。

## R16

适用场景：
- 强关联/阻挫磁性课程的教学演示；
- 新模型项加入前的几何与采样正确性回归测试；
- 研究代码前期参数扫描的原型。

不适用场景：
- 需要与具体 Kagome 材料做定量拟合的高精度研究；
- 需要量子低温临界行为与精确误差控制的发表级计算。

## R17

参考方向（概念层）：
- C. Lacroix, P. Mendels, F. Mila (eds.), *Introduction to Frustrated Magnetism*.
- L. Balents, *Nature* 464, 199 (2010): Spin liquids in frustrated magnets.
- Kagome Heisenberg antiferromagnet 的经典与量子 Monte Carlo / 变分研究文献。

## R18

`demo.py` 源码级算法流（8 步，非黑盒拆解）：

1. `build_kagome_coordinates` 生成 Kagome 原胞坐标与超胞周期矢量 `A,B`。  
2. `build_kagome_bonds` 对每对站点执行最近镜像距离搜索，按 `d_nn=0.5` 判定最近邻键。  
3. `build_neighbor_list` 和 `find_triangles` 分别构造邻接表与三角团簇，并检查配位数约束。  
4. `random_unit_vectors` 生成初始自旋；`normalize_rows` 保证每个自旋长度为 1。  
5. `metropolis_sweep` 对随机站点提案 `S'_i`，显式计算 `dE = J*(S'_i-S_i)·sum_nn(S_j)` 并执行接受率判据。  
6. `total_energy_per_spin`、`mean_neighbor_dot`、`mean_abs_scalar_chirality` 在每个 sweep 后产出可观测量序列。  
7. `structure_factor_peak` 在离散 `(h,k)` 网格上逐点计算 `S(q)=|sum_i S_i exp(iq·r_i)|^2/N`，返回峰值与位置。  
8. `run_simulation` 汇总为 `pandas.DataFrame`，`main` 打印末段时间序列、窗口均值与有限性断言，形成一次性可复现实验。

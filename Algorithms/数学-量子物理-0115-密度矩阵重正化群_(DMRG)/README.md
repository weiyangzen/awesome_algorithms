# 密度矩阵重正化群 (DMRG)

- UID: `MATH-0115`
- 学科: `数学`
- 分类: `量子物理`
- 源序号: `115`
- 目标目录: `Algorithms/数学-量子物理-0115-密度矩阵重正化群_(DMRG)`

## R01

本条目实现一个可运行、可追踪的 DMRG 最小版本：针对自旋 `1/2` 反铁磁 Heisenberg 开链，使用**无限系统 DMRG（infinite-system DMRG）**估计基态能量，并用小规模精确对角化做交叉验证。

MVP 目标：

- 明确展示“构造超块 -> 求基态 -> 约化密度矩阵 -> 截断保留”主链路；
- 不依赖黑盒 DMRG 库；
- `python3 demo.py` 一键运行并输出可检查结果。

## R02

问题定义（本实现固定模型）：

- 哈密顿量：
  `H = sum_{i=1}^{N-1} (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})`
- 自旋算符：`Sx, Sy, Sz = sigma/2`
- 边界条件：开边界（OBC）
- 输入参数：目标链长 `target_length`（偶数，且 `>=4`）、截断维度 `m_keep`
- 输出：各增长阶段的总能量、每格点能量、截断误差，以及终点与精确对角化误差

## R03

DMRG 的核心思想是“变分 + 压缩”：

1. 在系统块与环境块组成的超块上求基态波函数；
2. 对系统块做约化密度矩阵 `rho_sys`；
3. 保留 `rho_sys` 最大本征值对应的前 `m` 个本征态，得到最优低维子空间（对当前目标态最优）；
4. 在该子空间中继续迭代。

这比直接在 `2^N` 维希尔伯特空间做全局求解更可扩展，特别适合一维低纠缠基态问题。

## R04

本实现使用的关键公式：

- 超块哈密顿量
  `H_super = H_sys ⊗ I_env + I_sys ⊗ H_env + H_boundary`
- 边界耦合
  `H_boundary = Sx_edge^sys ⊗ Sx_edge^env + Sy_edge^sys ⊗ Sy_edge^env + Sz_edge^sys ⊗ Sz_edge^env`
- 约化密度矩阵（由基态 `|psi0>`）
  `rho_sys = Tr_env(|psi0><psi0|)`
- 截断误差（丢弃权重）
  `eps_trunc = 1 - sum_{k in kept} lambda_k`

其中 `lambda_k` 为 `rho_sys` 本征值，按从大到小排序。

## R05

核心数据结构对应 `demo.py`：

- `Block(dataclass)`
  - `length`：块长度
  - `h`：块哈密顿量
  - `sx_edge/sy_edge/sz_edge`：块最右边界自旋算符
  - `trunc_error`：本轮截断误差
- `history`（列表字典）
  - `length`、`energy`、`energy_per_site`
  - `system_dim_before_trunc/system_dim_after_trunc`
  - `trunc_error`

该组织方式让“物理量”和“计算开销指标”都可直接打印审计。

## R06

正确性保障（脚本内已有）：

- 参数合法性检查：`target_length` 必须是偶数且 `>=4`，`m_keep>=2`；
- 基态求解使用 Hermitian 对角化 `np.linalg.eigh`；
- 约化密度矩阵显式厄米化，避免数值非对称污染；
- 结束时用同一模型的小规模精确对角化做对照，检查每格点误差阈值。

当前默认参数下（`N=8, m_keep=8`）可得到与精确值一致到数值精度范围内的结果。

## R07

复杂度（设截断维度为 `m`）：

- 系统块放大后维度约 `2m`；
- 超块维度约 `(2m)^2 = 4m^2`；
- 若用致密本征分解，单轮主成本近似 `O((4m^2)^3)=O(m^6)`；
- 存储主成本来自超块矩阵，约 `O((4m^2)^2)=O(m^4)`。

本 MVP 取小 `m`（默认 8），优先保证链路透明。

## R08

边界与异常处理：

- `target_length < 4` 或为奇数：抛 `ValueError`；
- `m_keep < 2`：抛 `ValueError`；
- 最终能量非有限：抛 `RuntimeError`；
- 与精确对照偏差超过阈值：抛 `RuntimeError`。

策略是“尽早失败”，避免无效结果继续传播。

## R09

MVP 取舍说明：

- 只实现 infinite-system DMRG，不做 finite-system sweep；
- 只覆盖 Heisenberg 开链，不做任意 MPO 输入；
- 用 `numpy` 致密线代替代大规模稀疏迭代，牺牲规模换取可读性；
- 额外保留精确对角化对照，仅用于小系统质量校验。

该版本定位是教学/验证，不是工业大规模求解器。

## R10

`demo.py` 函数职责：

- `spin_operators`：生成局域 `Sx/Sy/Sz`
- `init_single_site_block`：初始化长度 1 块
- `enlarge_block`：执行块扩展 `block + site`
- `build_superblock_hamiltonian`：组装超块哈密顿量
- `ground_state_dense`：求超块最低本征态
- `truncate_system_block`：约化密度矩阵截断
- `infinite_dmrg`：主迭代流程
- `exact_ground_energy_heisenberg`：小规模精确对角化基准
- `print_history/main`：结果输出与质量断言

## R11

运行方式：

```bash
cd Algorithms/数学-量子物理-0115-密度矩阵重正化群_(DMRG)
python3 demo.py
```

脚本无需交互输入，会自动完成 DMRG 与精确对照并打印检查结果。

## R12

输出字段说明：

- `length`：当前超块总链长
- `total_energy`：当前长度估计总基态能
- `energy/site`：每格点能量（常用于尺度比较）
- `dim_before`：截断前系统块维度
- `dim_after`：截断后保留维度
- `trunc_error`：截断丢弃权重
- `Absolute error / Per-site error`：与精确解的总误差/每格点误差

## R13

建议最小测试集：

1. 合法性测试：`target_length=8, m_keep=8`，应通过并输出 `All checks passed.`
2. 参数异常：`target_length=7`、`target_length=2`、`m_keep=1`，应抛 `ValueError`。
3. 精度敏感性：固定 `N=8`，比较 `m_keep=4/6/8` 的误差与截断误差变化。
4. 稳定性测试：重复运行多次，结果应一致（本脚本无随机项）。

## R14

关键可调参数：

- `target_length`：目标链长（默认 8）
- `m_keep`：保留本征态数（默认 8）

经验：

- `m_keep` 增大通常提升精度，但超块矩阵尺寸与开销上升；
- 在本致密实现中，先保证小规模正确，再考虑更大 `N`。

## R15

与相关方法对比：

- 全空间精确对角化：结果精确，但维度 `2^N` 爆炸；
- 无限系统 DMRG：通过密度矩阵截断显著降维，适合一维链基态；
- 有限系统 DMRG sweep：通常更精确，但实现复杂度更高。

本条目选择“无限系统 DMRG + 小系统精确对照”作为最小可审计组合。

## R16

典型应用场景：

- 一维强关联系统基态近似（Heisenberg、Hubbard 链等）
- 张量网络/矩阵乘积态（MPS）教学入门
- 研究截断维度与纠缠信息对精度影响的实验
- 在大规模算法实现前做快速原型验证

## R17

可扩展方向：

- 增加 finite-system sweep（左-右往返优化）
- 将哈密顿量推广到可配置参数（如 XXZ 各向异性）
- 用稀疏本征迭代替代致密对角化，提升可处理尺寸
- 显式输出纠缠熵、相关函数等物理观测量
- 过渡到 MPO/MPS 数据结构，贴近研究级实现

## R18

源码级算法流程拆解（`infinite_dmrg` 主链路，8 步）：

1. 初始化单站点 `Block`，其 `h=0`，边界算符为单站点 `Sx/Sy/Sz`。  
2. 调用 `enlarge_block` 把当前块扩成 `block + site`，得到系统块 `system`。  
3. 在 infinite-system 设定下令环境块 `env = system`，构造镜像超块。  
4. 调用 `build_superblock_hamiltonian` 显式组装 `H_super = H_sys⊗I + I⊗H_env + 边界耦合`。  
5. 调用 `ground_state_dense` 求 `H_super` 最低本征态 `|psi0>` 与能量 `E0`。  
6. 将 `|psi0>` reshape 成 `(d_sys, d_env)`，构造 `rho_sys = psi psi^†` 并做本征分解。  
7. 保留 `rho_sys` 最大的前 `m_keep` 个本征向量，投影 `h` 与边界算符到截断子空间，得到新 `Block`。  
8. 记录当前长度/能量/截断误差，循环迭代直到达到 `target_length`，最后与 `exact_ground_energy_heisenberg` 做误差校验。  

实现中没有把 DMRG 委托给第三方黑盒库，所有关键步骤都可在源码中逐行追踪。

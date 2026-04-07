# 声子色散关系 (Phonon Dispersion)

- UID: `PHYS-0446`
- 学科: `物理`
- 分类: `固体物理`
- 源序号: `469`
- 目标目录: `Algorithms/物理-固体物理-0469-声子色散关系_(Phonon_Dispersion)`

## R01

声子色散关系（Phonon Dispersion）描述晶格振动角频率 `omega` 与波矢 `q` 的函数关系 `omega(q)`。

本目录将该主题落成一个可运行最小模型：

- 一维单原子链（monatomic）声学支；
- 一维双原子链（diatomic）声学支 + 光学支；
- 通过断言验证关键物理性质（长波线性、Gamma 点与带隙结构）。

## R02

声子色散关系的核心价值在于：

- 决定晶格振动可激发频率范围与分支结构；
- 群速度 `v_g = d omega / d q` 直接影响热输运行为；
- 为后续 Debye 模型、电-声子耦合、非谐散射分析提供基础输入。

因此它是从“晶格参数（质量、弹簧常数）”走向“可观测热/输运性质”的第一层桥梁。

## R03

本 MVP 的目标与边界：

- 输入：`DispersionConfig` 中的 `K, a, m` 与 `q` 采样数；
- 输出：
  1. 单原子色散采样表（`q/pi/a`, `omega`, `v_g`）；
  2. 双原子色散采样表（声学支、光学支、分支间隔）；
  3. 5 个 sanity checks 指标。
- 约束：不调用黑盒声子软件包，核心公式与检查在 `demo.py` 显式实现。

## R04

`demo.py` 使用的物理公式（均为 1D 最近邻谐振近似）：

1. 单原子链：
`omega_mono(q) = 2*sqrt(K/M)*|sin(qa/2)|`

2. 单原子声速（长波极限）：
`v_s = a*sqrt(K/M)`，且 `omega(q) ~ v_s*q`（`q -> 0`）

3. 双原子链分支（`m1, m2`）：
`omega^2_{±}(q) = A ± sqrt(A^2 - B*sin^2(qa/2))`

其中：
`A = K*(1/m1 + 1/m2)`，`B = 4*K^2/(m1*m2)`。

`-` 对应声学支，`+` 对应光学支。

## R05

算法流程（实现级）如下：

1. 校验配置合法性（正质量、正晶格常数、采样点数下限等）；
2. 在第一布里渊区构造 `q in [0, pi/a]` 均匀网格；
3. 用解析式计算单原子支 `omega_mono(q)`；
4. 计算单原子群速度 `v_g(q)`；
5. 用双原子解析式计算 `omega_acoustic(q)` 与 `omega_optical(q)`；
6. 构造可读采样表（9 个代表性 `q` 点）；
7. 运行 5 条物理断言；
8. 打印参数摘要、两张采样表、校验指标。

## R06

复杂度（`Nq = n_q`）：

- 时间复杂度：`O(Nq)`（全部为向量化逐点公式计算）；
- 空间复杂度：`O(Nq)`（主要存储 `q` 与分支数组）；
- 默认 `n_q = 801`，在普通 CPU 上毫秒级完成。

## R07

`demo.py` 输出结构：

- 参数字典：包含 `K, a, m1, m2, n_q` 与频率摘要；
- `[monatomic_samples]`：
  - `q_over_pi_per_a`
  - `omega_acoustic_THz`
  - `v_group_km_per_s`
- `[diatomic_samples]`：
  - `omega_acoustic_THz`
  - `omega_optical_THz`
  - `branch_gap_THz`
- `[sanity_checks]`：
  - 小 `q` 线性色散误差
  - 区边界频率误差
  - Gamma 点条件
  - 分支最小间隔

## R08

前置知识：

- 晶格动力学中的谐振近似；
- 布里渊区与色散分支（声学/光学）；
- 基本数值计算与数组操作。

运行环境：

- Python `>=3.10`
- `numpy`
- `pandas`

## R09

适用场景：

- 固体物理课堂上演示“色散关系如何由质量和弹簧常数决定”；
- 快速比较单原子链与双原子链的分支差异；
- 为更复杂模型提供简洁基线（Debye、热输运、电子-声子耦合）。

不适用场景：

- 材料级高精度声子谱预测（缺少真实晶体结构、多维相互作用）；
- 需要非谐效应、温度重整化、声子寿命的工程模型；
- 直接替代第一性原理声子计算。

## R10

正确性直觉与检查逻辑：

1. 单原子链在长波极限必须线性：`omega ~ v_s q`；
2. 单原子链在区边界 `q=pi/a` 必须到达 `omega_max = 2*sqrt(K/M)`；
3. 双原子链在 `Gamma` 点应满足：
   - 声学支 `omega_ac(0)=0`
   - 光学支 `omega_op(0)=sqrt(2K(1/m1+1/m2))`
4. 双原子链全区间内应保持 `omega_op > omega_ac`。

`demo.py` 将这四组规律转成可执行断言，避免“只看图不验算”。

## R11

数值稳定性处理：

- 双原子公式中的根号项使用 `np.maximum(radicand, 0)` 防止浮点负零；
- `omega^2` 结果再做一次 `np.maximum(..., 0)`，避免极小负值传递到开方；
- 线性误差计算分母使用 `np.maximum(omega, 1e-30)` 防止 `0/0`。

这三处保证从 `q=0` 到 `q=pi/a` 的计算稳定。

## R12

默认参数（`DispersionConfig`）：

- `lattice_constant_m = 5.43e-10`
- `spring_constant_n_per_m = 18.0`
- `mono_mass_kg = 4.6637066e-26`
- `mass_a_kg = 4.6637066e-26`
- `mass_b_kg = 1.1623773e-25`
- `n_q = 801`

本地运行 `uv run python demo.py` 的关键输出（参考值）：

- `mono_omega_max_THz = 6.253465663620388`
- `di_optical_gamma_THz = 5.2343065542974205`
- `sound_velocity_m_per_s = 1.066769e+04`
- `small_q_linear_rel_err = 1.481976e-03`
- `min_branch_gap_THz = 1.620967e+00`

## R13

保证类型说明：

- 近似比保证：N/A（非优化问题）；
- 随机成功率保证：N/A（确定性计算）。

本 MVP 的可执行保证：

- 无交互输入；
- 输出固定结构表格；
- 若关键物理规律不满足会触发 `assert` 明确失败。

## R14

常见失败模式：

1. 把质量/弹簧常数单位写错（非 SI）；
2. 使用相等质量却期望显著光学-声学分裂；
3. `n_q` 太小导致低 `q` 线性检验误差大；
4. 忽略浮点误差，根号下出现微小负数引发 `nan`。

脚本已通过参数检查和稳定写法覆盖 3、4；1、2 需建模者自行确保物理输入正确。

## R15

可扩展方向：

1. 从 1D 扩展到 2D/3D 晶格并引入多原胞自由度；
2. 改为显式动力学矩阵 `D(q)` 并做特征值求解（多分支统一框架）；
3. 加入下一近邻相互作用，研究色散弯曲变化；
4. 叠加态密度 `g(omega)` 与热容积分模块；
5. 接入实验/DFT 结果做参数反演。

## R16

相关主题：

- 声子（Phonon）
- 德拜模型（Debye Model）
- 爱因斯坦模型（Einstein Model）
- 晶格动力学与动力学矩阵
- 热输运中的群速度与散射时间

## R17

运行方式（无交互）：

```bash
cd Algorithms/物理-固体物理-0469-声子色散关系_(Phonon_Dispersion)
uv run python demo.py
```

交付核对：

- `README.md` 已完整填写 `R01-R18`；
- `demo.py` 可直接运行并输出表格与校验指标；
- `meta.json` 与任务元数据一致；
- 目录可独立验证。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 创建 `DispersionConfig` 并调用 `validate_config` 执行参数合法性检查。  
2. `build_q_grid` 生成第一布里渊区 `q in [0, pi/a]` 的均匀离散点。  
3. `monatomic_dispersion_1d` 用解析式计算单原子链声学支 `omega_mono(q)`。  
4. `monatomic_group_velocity_1d` 计算 `v_g = d omega / d q` 并用于输出表。  
5. `diatomic_dispersion_1d` 计算双原子链 `A, B, radicand`，再得到声学/光学两条分支。  
6. `build_sample_tables` 抽取 9 个代表 `q` 点，构造单原子与双原子结果表。  
7. `run_sanity_checks` 执行 5 条断言：小 `q` 线性、区边界、Gamma 条件与分支顺序。  
8. `main` 打印参数摘要、两张采样表和检查指标，形成可复核的最小交付。  

第三方库角色说明：`numpy` 仅用于数组计算与向量化公式评估，`pandas` 仅用于表格输出；色散关系、分支构造与物理检查逻辑均在源码中显式实现，不依赖黑盒算法包。

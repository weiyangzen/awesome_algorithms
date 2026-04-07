# 朗道相变理论 (Landau Theory of Phase Transitions)

- UID: `PHYS-0300`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `303`
- 目标目录: `Algorithms/物理-统计力学-0303-朗道相变理论_(Landau_Theory_of_Phase_Transitions)`

## R01

朗道相变理论通过序参量 `m` 的自由能展开来统一描述连续相变与一级相变。  
核心思想是：在临界附近把自由能写成低阶多项式，再用“自由能最小化”确定平衡态。

本目录 MVP 采用均匀序参量模型：
- 用 `f(m,T)=a(T-Tc)m^2 + b m^4 + c m^6 - h m` 建模；
- 显式计算 `df/dm` 与 `d2f/dm2`；
- 先网格找极小值盆地，再用手写 Newton 迭代细化；
- 输出二级相变与一级相变两组参数下的温度扫描结果。

## R02

问题定义（本实现）：
- 输入：
  - 朗道系数 `a,b,c,Tc,h`；
  - 温度数组 `temperatures`；
  - 数值参数 `grid_size/newton_tol/newton_max_iter`；
  - 序参量截断范围 `m in [-m_max, m_max]`。
- 输出：
  - 每个温度的平衡序参量 `m_eq`；
  - 平衡自由能 `f_eq`、曲率 `F''(m_eq)`、估计磁化率 `chi≈1/F''`；
  - 局部极小值个数（用于识别多稳态）；
  - 连续相变近临界解析式对比与一级相变跃迁诊断。

脚本参数写在 `main()`，`uv run python demo.py` 可直接运行。

## R03

核心数学关系：

1. 朗道自由能密度：
   `f(m,T)=a(T-Tc)m^2 + b m^4 + c m^6 - h m`。  
2. 平衡条件（驻点）：
   `df/dm = 2a(T-Tc)m + 4b m^3 + 6c m^5 - h = 0`。  
3. 稳定性条件（局部极小）：
   `d2f/dm2 = 2a(T-Tc) + 12b m^2 + 30c m^4 > 0`。  
4. 连续相变（`b>0,c=0,h=0`）解析序参量：
   `|m| = sqrt(a(Tc-T)/(2b))`（当 `T<Tc`）。  
5. 一级相变（`b<0,c>0,h=0`）共存温度：
   `T* = Tc + b^2/(4ac)`，此处存在非零序参量跳变。

## R04

算法流程（高层）：
1. 校验参数：确保自由能有界（如 `b<0` 时必须 `c>0`）且数值配置合法。  
2. 固定温度 `T`，在 `[-m_max,m_max]` 上均匀采样得到 `f(m,T)`。  
3. 从网格曲线抽取局部极小值种子索引。  
4. 对每个种子执行手写 Newton 迭代求 `df/dm=0`。  
5. 用 `d2f/dm2>0` 过滤掉非极小驻点。  
6. 计算候选极小值自由能并去重。  
7. 选择自由能最小候选作为该温度平衡态。  
8. 汇总温度扫描表，并执行连续/一级相变 sanity checks。

## R05

核心数据结构：
- `LandauParams`（dataclass）：
  - 存储 `a,b,c,tc,h,m_max`。
- `EquilibriumRow`（dataclass）：
  - 存储 `temperature/m_eq/abs_m_eq/free_energy/curvature/susceptibility/phase/local_minima`。
- `list[EquilibriumRow]`：
  - 一个场景（参数组）下的整条温度扫描结果。

## R06

正确性要点：
- 驻点不是终点：必须再检查 `d2f/dm2>0`，避免把极大值或鞍点当稳定态。  
- 使用“全局最小自由能”选平衡分支，防止落入亚稳态。  
- 网格种子 + Newton 细化兼顾全局性与精度。  
- 通过二级相变解析公式和一级相变跳变诊断做结果闭环验证。

## R07

复杂度分析：
- 设温度点数 `N_T`，网格大小 `G`，每温度候选盆地数 `S`，Newton 迭代上限 `K`。
- 每个温度：
  - 网格扫描 `O(G)`；
  - Newton 细化 `O(S*K)`（通常 `S` 很小，常见 1-3）。
- 总时间复杂度：`O(N_T * (G + S*K))`。
- 空间复杂度：
  - 单温度主开销是网格数组 `O(G)`；
  - 输出结果 `O(N_T)`。

## R08

边界与异常处理：
- `a<=0`、`c<0`、或 `b<0 且 c<=0`：抛 `ValueError`。  
- 温度数组非 1D / 为空 / 非有限：抛 `ValueError`。  
- `grid_size` 过小或不是奇数：抛 `ValueError`（保证网格包含 `m=0`）。  
- Newton 出现 `nan/inf` 或 Hessian 过小：该候选判失败并回退网格点。  
- 若细化后无合法极小值，回退到网格全局最小值（保证流程不中断）。

## R09

MVP 取舍：
- 仅做均匀序参量（零维 Landau），不含梯度项 `|∇m|^2` 与空间结构。  
- 不调用 `scipy.optimize.minimize` 等黑盒优化器，保证算法步骤可审计。  
- 输出以命令行表格为主，不依赖绘图库，便于无图形环境验证。  
- 用两组参数直接展示二级与一级相变，优先保证“最小可运行 + 物理可解释”。

## R10

`demo.py` 主要函数职责：
- `validate_inputs`：检查物理参数与数值配置合法性。  
- `free_energy`：计算朗道自由能 `f(m,T)`。  
- `dfdm` / `d2fdm2`：显式给出一阶、二阶导数。  
- `newton_refine`：对 `df/dm=0` 做手写 Newton 迭代。  
- `local_minima_seed_indices`：从网格自由能曲线提取局部极小种子。  
- `deduplicate_candidates`：合并同一势阱的重复候选。  
- `equilibrium_at_temperature`：生成单温度平衡态。  
- `scan_temperatures`：批量温度扫描。  
- `print_table`：格式化输出结果表。  
- `analytic_abs_m_second_order`：连续相变解析解对照。  
- `run_sanity_checks_continuous` / `run_sanity_checks_first_order`：自动验证。  
- `main`：组织两种场景并执行完整流程。

## R11

运行方式：

```bash
cd Algorithms/物理-统计力学-0303-朗道相变理论_(Landau_Theory_of_Phase_Transitions)
uv run python demo.py
```

脚本无交互输入，不需要额外命令行参数。

## R12

输出字段说明：
- `T`：温度。  
- `m_eq`：平衡序参量（按自由能全局最小选枝）。  
- `|m_eq|`：序参量幅值，便于看是否有序。  
- `phase`：`ordered/disordered`。  
- `f_eq`：平衡自由能密度。  
- `F_pp`：`d2f/dm2` 在平衡点的曲率。  
- `chi`：估计易感率 `1/F_pp`（曲率过小视为发散）。  
- `n_min`：该温度识别到的局部极小值数。  
- 附加输出：
  - 连续相变近临界解析对比（数值 `|m|` vs 公式）；
  - 一级相变最大跳变温度区间与理论 `T*`。

## R13

建议最小验证项（脚本已覆盖）：
- 连续相变场景：`T<Tc` 与 `T>Tc` 同时扫描。  
- 连续相变近临界点：与 `|m|=sqrt(a(Tc-T)/(2b))` 做相对误差对比。  
- 一级相变场景：检测温度扫描中的离散跃迁 `|Δm|`。  
- 验证一级相变理论共存温度 `T*=Tc+b^2/(4ac)` 落入扫描区间。  
- 所有扫描在无交互情况下完整运行结束。

可扩展验证：
- 对 `h!=0` 测试偏置与对称性打破。  
- 与 Monte Carlo 或实验数据做参数反演对比。  
- 在更细温度网格上估计临界指数。

## R14

可调参数：
- `a`：控制二次项温度灵敏度。  
- `b,c`：控制势阱形状，决定连续/一级相变类型。  
- `tc`：名义临界温度基准。  
- `h`：外场，控制正负序参量偏置。  
- `m_max`：序参量搜索边界。  
- `grid_size`：全局搜索分辨率。  
- `newton_tol/newton_max_iter`：局部细化精度与收敛预算。

调参建议：
- 一级相变跳变不明显：增大 `grid_size` 或加密 `T` 采样。  
- Newton 偶发失败：增大 `m_max` 或提高 `newton_max_iter`。  
- 近临界数值噪声偏大：收紧 `newton_tol` 并局部加密温度点。

## R15

方法对比：
- 对比 Ising/Monte Carlo：
  - Landau 模型快、解释性强；
  - 但忽略临界涨落与空间关联，定量精度有限。  
- 对比重整化群：
  - 重整化群更适合临界行为高精度分析；
  - Landau 更适合快速定性判断和参数初筛。  
- 对比黑盒优化：
  - 黑盒调用简单；
  - 本实现显式展示导数与迭代路径，更利于教学与审计。

## R16

典型应用场景：
- 铁电、铁磁、结构相变的快速势能地形分析。  
- 复杂模拟前的低成本相图预判。  
- 从实验数据拟合 `a,b,c` 的参数化前处理。  
- 统计物理课程中的“自由能展开 -> 相变类型”教学演示。

## R17

可扩展方向：
- 加入空间梯度项 `k|∇m|^2`，求解畴壁/界面结构。  
- 在 `h-T` 平面扫描，输出更完整相图。  
- 联合 `pandas` 导出 CSV，配套绘图脚本。  
- 用自动微分（如 PyTorch）验证手写导数一致性。  
- 对接拟合流程，从观测序参量反推 Landau 系数。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 定义数值配置（`grid_size/newton_tol/newton_max_iter`）和两组 Landau 系数（连续/一级相变）。  
2. `validate_inputs` 检查参数合法性，尤其是自由能有界条件 `b<0 => c>0`。  
3. `scan_temperatures` 遍历每个温度并调用 `equilibrium_at_temperature`。  
4. `equilibrium_at_temperature` 在 `m` 网格上计算 `f(m,T)`，由 `local_minima_seed_indices` 找到局部极小值种子。  
5. 对每个种子执行 `newton_refine`（显式使用 `dfdm` 与 `d2fdm2`）得到候选驻点。  
6. 用 `d2fdm2>0` 过滤稳定极小值，`deduplicate_candidates` 去除重复盆地候选。  
7. 以自由能最小原则选 `m_eq`，并计算 `|m_eq|`、曲率、`chi≈1/F''`、相态标签。  
8. 输出两套表格；连续相变场景对比解析 `|m|`，一级相变场景报告最大 `|Δm|` 与理论共存温度 `T*`，最后通过 sanity checks 收束。

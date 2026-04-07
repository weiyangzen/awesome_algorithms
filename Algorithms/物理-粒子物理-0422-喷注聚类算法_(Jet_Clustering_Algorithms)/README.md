# 喷注聚类算法 (Jet Clustering Algorithms)

- UID: `PHYS-0403`
- 学科: `物理`
- 分类: `粒子物理`
- 源序号: `422`
- 目标目录: `Algorithms/物理-粒子物理-0422-喷注聚类算法_(Jet_Clustering_Algorithms)`

## R01

喷注聚类算法（Jet Clustering Algorithms）是高能对撞实验里把大量末态粒子重建为少量喷注对象的核心步骤。  
本条目实现一个最小但完整的广义顺序复合（sequential recombination）MVP，并在同一事件上对比三类常见算法：
- `k_t`（`p=+1`）
- `Cambridge/Aachen`（`p=0`）
- `anti-k_t`（`p=-1`）

## R02

问题定义（对应 `demo.py`）：
- 输入：一个事件中的粒子集合 `P={p_i}`，每个粒子有 `(pt, eta, phi, mass)`。
- 输出：每种算法下的喷注集合 `J={j_k}`，以及聚类过程统计（合并次数、beam 声明次数、前若干步 trace）。
- 目标：
  1. 展示三类喷注聚类规则在同一事件上的行为差异；
  2. 明确源码级距离计算、选择最小距离、合并或冻结（beam）全过程；
  3. 保证脚本无交互、可一次运行复现结果。

## R03

物理背景：
- 在 LHC 等实验中，硬散射先产生 parton（夸克/胶子），随后强子化为可探测的稳定粒子。
- 喷注算法将这些离散末态粒子按动力学和几何邻近关系聚成喷注，连接“探测器观测量”与“硬过程尺度”。
- `k_t / C/A / anti-k_t` 都是红外-共线安全（IR/collinear safe）的经典家族，主要差别在距离度量中 `pt` 权重指数 `p` 的选择。

## R04

广义 `k_t` 家族的核心距离定义：

- 粒子-粒子距离：
`d_ij = min(pt_i^(2p), pt_j^(2p)) * (DeltaR_ij^2 / R^2)`

- 粒子-束流距离：
`d_iB = pt_i^(2p)`

- 角距离：
`DeltaR_ij^2 = (eta_i - eta_j)^2 + DeltaPhi_ij^2`

其中 `DeltaPhi` 需做 `[-pi, pi)` 周期折返。  
参数映射：
- `p=+1` -> `k_t`
- `p=0` -> `Cambridge/Aachen`
- `p=-1` -> `anti-k_t`

## R05

输入输出约定（脚本内固定）：
- 输入：`generate_synthetic_event(seed=21)` 生成的 110 个 toy 粒子（两束硬结构 + 一束中等结构 + 软背景）。
- 聚类参数：`R=0.6`, `pt_min=15.0`。
- 输出：
  - 每种算法 top jets 表（`pt/eta/phi/mass/n_const`）；
  - 聚类 trace 前 12 步（`iter, n_active_before, action, best_distance, i, j`）；
  - 汇总表（喷注数量、leading jet `pt`、pair merge/beam 次数等）；
  - 断言通过后打印 `All checks passed.`。

## R06

MVP 算法流程（高层）：
1. 把每个输入粒子转为初始 pseudojet（四动量 + constituent id）。
2. 在活动集合中扫描所有 `d_iB`（beam 距离）。
3. 扫描所有成对 `d_ij`（pair 距离）。
4. 取全局最小距离：
   - 若最小项是 `pair`，按 E-scheme 合并两个 pseudojet；
   - 若最小项是 `beam`，将该 pseudojet 作为最终喷注候选冻结。
5. 活动集合为空后结束，按 `pt_min` 过滤并按 `pt` 降序输出。
6. 对 `k_t / C/A / anti-k_t` 三组 `p` 重复上述流程并做对照。

## R07

复杂度分析（朴素扫描实现）：
- 每轮需计算 `O(n^2)` 个 pair 距离，轮数约 `O(n)`；
- 总时间复杂度约 `O(n^3)`；
- 空间复杂度 `O(n)`（不含输出与 trace 表）。

说明：这版是教学型透明实现，不追求 FastJet 级别的几何加速。

## R08

正确性直觉：
- 每轮都执行“全局最小距离”决策，因此严格符合顺序复合框架定义。
- `p` 的不同决定聚类偏好：
  - `k_t`（`p=+1`）对软粒子权重大，倾向先合并软结构；
  - `C/A`（`p=0`）仅看几何角距离；
  - `anti-k_t`（`p=-1`）强调硬核粒子，通常形成更“圆锥化”的硬喷注。
- E-scheme 四动量相加保证动量守恒式合并。

## R09

核心数据结构：
- `Particle`：输入粒子（`pid, pt, eta, phi, mass`）。
- `PseudoJet`：聚类过程中的可变对象（`p4, constituents`）。
- `Jet`：最终喷注对象（含 `pt/eta/phi/mass` 与 constituent 列表）。
- `ClusterResult`：单算法运行结果（喷注、统计指标、trace 表）。
- `summary_df`：三算法对比汇总表（`p`、喷注数、leading `pt`、迭代统计）。

## R10

边界与数值处理：
- `radius <= 0` 或 `pt_min < 0` 直接报错。
- `pt^(2p)` 计算使用 `safe_pt=max(pt,1e-12)`，避免 `p<0` 时除零/溢出。
- `phi` 差使用周期折返，避免 `-pi/pi` 边界跳变。
- 质量平方 `m2` 使用 `max(m2,0)` 抑制浮点负零。
- 聚类后执行 constituent 完整性校验：最终 constituent multiset 必须与输入 `pid` 一致。

## R11

关键参数与影响：
- `p`：算法类型开关（`+1, 0, -1`）。
- `R`：喷注半径（默认 `0.6`），越大越倾向把更宽角度粒子并入同一喷注。
- `pt_min`：最终喷注筛选阈值（默认 `15.0`）。
- `seed`：toy 事件随机种子（默认 `21`），保证可复现。
- `max_trace_rows`：打印聚类前几步决策，便于审计流程。

## R12

运行方式：

```bash
cd Algorithms/物理-粒子物理-0422-喷注聚类算法_(Jet_Clustering_Algorithms)
uv run python demo.py
```

脚本不读取命令行参数，不请求交互输入。

## R13

输出字段说明：
- `jets above cut`：通过 `pt_min` 的喷注数量。
- `pair_merges`：pair 合并次数。
- `beam_declarations`：beam 冻结次数（最终 pseudojet 数）。
- `rank/jet_id/pt/eta/phi/mass/n_const`：前几大喷注摘要。
- `trace`：早期聚类决策链（每步是 merge 还是 beam 以及对应最小距离）。
- `Summary`：三算法并排比较，便于看 `p` 对聚类行为影响。

## R14

脚本内置验收断言：
1. 每种算法都至少找到 2 个高 `pt` 喷注。  
2. 每种算法的 leading jet `pt` 都高于 `pt_min`。  
3. `iterations == n_particles`，验证“每轮消耗一个活动对象”的计数一致性。  
4. 若全部通过，输出 `All checks passed.`。

## R15

三算法行为差异（本实现可直接观测）：
- `k_t`：通常出现更多软粒子先合并，pair merge 统计可能更高。
- `C/A`：以几何邻近主导，结果常介于 `k_t` 与 `anti-k_t` 之间。
- `anti-k_t`：硬核主导更明显，常形成稳定高 `pt` 主喷注，实验分析中应用最广。

本条目用同一事件同一代码路径对比，避免“数据条件不同导致的伪差异”。

## R16

当前 MVP 局限：
- 仅 toy 事件，不含真实探测器响应、pileup、校准流程。
- 采用朴素 `O(n^3)`，未实现 FastJet 风格高效邻域搜索。
- 未实现 grooming/substructure 指标（如 Soft Drop、N-subjettiness）。

## R17

可扩展方向：
- 接入真实 HepMC/ROOT 事件读取。
- 引入面积校正与 pileup mitigation。
- 增加喷注匹配、真值标注和性能指标（efficiency/purity）。
- 用空间索引或几何分桶将复杂度从教学版 `O(n^3)` 向工程版优化。
- 增加 jet substructure 例程，形成更完整粒子物理分析链路。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `main` 固定 `seed=21` 生成 toy 事件，并设定 `R=0.6`、`pt_min=15.0`。  
2. `run_algorithm_suite` 依次调用 `generalized_kt_cluster`，分别传入 `p=+1,0,-1`。  
3. `generalized_kt_cluster` 将每个 `Particle` 转成 `PseudoJet`（`particle_to_pseudojet` + 四动量变换）。  
4. 每轮先遍历活动对象计算所有 `d_iB`（`generalized_beam_distance`）。  
5. 再双重循环计算所有 `d_ij`（`generalized_pair_distance`，内部显式使用 `DeltaR` 与 `pt^(2p)`）。  
6. 选择全局最小距离：若是 `pair` 则 `merge_pseudojets`（E-scheme 四动量相加），若是 `beam` 则移入最终集合。  
7. 同步记录 trace 行（`iter/n_active_before/action/best_distance/i/j`）用于过程审计。  
8. 活动集合清空后执行计数与 constituent 完整性检查，再映射为 `Jet` 并按 `pt_min` 过滤排序。  
9. `main` 打印每种算法的喷注表和汇总表，执行断言检查并输出 `All checks passed.`。  

说明：本实现没有调用外部喷注库黑盒；`numpy` 负责数值运算，`pandas` 仅用于结果表格展示。

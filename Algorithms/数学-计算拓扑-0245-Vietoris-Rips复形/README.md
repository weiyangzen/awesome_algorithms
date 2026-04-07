# Vietoris-Rips复形

- UID: `MATH-0245`
- 学科: `数学`
- 分类: `计算拓扑`
- 源序号: `245`
- 目标目录: `Algorithms/数学-计算拓扑-0245-Vietoris-Rips复形`

## R01

Vietoris-Rips 复形（简称 VR 复形）是点云拓扑数据分析（TDA）中的核心构造。给定距离空间中的点集与阈值 `epsilon`，把两两距离不超过 `epsilon` 的点连边，再将所有团（clique）提升为高维单纯形，得到一个抽象单纯复形。

本目录提供一个“可运行、可审计”的最小 MVP：
- 手写 VR 复形构建（非黑盒）
- 手写模 2 边界矩阵与高斯消元求秩
- 计算 `beta0/beta1/beta2`，观察滤过下拓扑特征变化

## R02

本条目解决的问题：
- 输入：点云 `X in R^{n x d}`、滤过阈值序列 `epsilon_1 < ... < epsilon_m`、最大维度 `max_dim`
- 输出：每个 `epsilon` 下
  - 单纯形数量（`#V/#E/#T`）
  - Betti 数（`beta0/beta1/beta2`）

`demo.py` 采用固定内置数据，无需任何交互输入：
1. 圆周采样点（展示 1 维环出现与消失）
2. 双簇点云（展示连通分支数随 `epsilon` 下降）

## R03

数学定义（VR 复形）：
- 设顶点集为 `V={1,...,n}`，距离函数为 `d(.,.)`
- 给定阈值 `epsilon >= 0`
- `sigma subseteq V` 是一个单纯形，当且仅当
  - 对任意 `i,j in sigma`，有 `d(i,j) <= epsilon`

等价地，先构造阈值图 `G_epsilon=(V,E_epsilon)`，其中
- `(i,j) in E_epsilon <=> d(i,j) <= epsilon`

然后取其旗复形（clique complex），即图中每个团对应一个单纯形。

## R04

同调计算（模 2）基础关系：
- 链群维数：`dim C_k = n_k`（`k` 维单纯形个数）
- 边界算子：`d_k: C_k -> C_{k-1}`
- 边界矩阵秩：`r_k = rank(d_k)`（在 `GF(2)` 上）

Betti 数：
- `beta_k = dim ker(d_k) - dim im(d_{k+1})`
- 由秩-零空间关系得
  - `beta_k = n_k - r_k - r_{k+1}`

实现里使用 `GF(2)`，因此边界矩阵元素仅为 `0/1`，消元用按位异或完成。

## R05

算法流程（单个 `epsilon`）：
1. 计算点云两两欧氏距离矩阵 `D`
2. 构造邻接条件 `D <= epsilon`
3. 枚举 `0..max_dim` 维候选顶点组合
4. 对每个组合检查是否为 clique，若是则加入对应维度单纯形表
5. 由 `k` 维与 `k-1` 维单纯形构造边界矩阵 `d_k`
6. 在 `GF(2)` 上做高斯消元，得到 `rank(d_k)`
7. 用 `beta_k = n_k - r_k - r_{k+1}` 计算 Betti 数
8. 输出该 `epsilon` 的统计行

## R06

MVP 数据结构：
- `ComplexByDim = Dict[int, List[Simplex]]`
  - `Simplex` 用排序后的顶点 tuple 表示，如 `(1,4,7)`
- `FiltrationRow`
  - 保存 `epsilon, n0, n1, n2, beta0, beta1, beta2`
- 边界矩阵
  - `np.ndarray(dtype=np.uint8)`，元素在 `{0,1}`

这种设计保证实现透明，便于逐函数审计。

## R07

正确性要点：
- VR 判定必须检查组合内“所有”点对距离，而不只是连通即可
- 边界矩阵按“删去一个顶点得到面”构造；在模 2 下符号不影响结果
- `d_{k-1} o d_k = 0` 的代数结构由标准单纯复形边界定义保证
- Betti 公式必须使用同一系数域（本实现统一使用 `GF(2)`）

## R08

复杂度（设点数 `n`，最高维 `p=max_dim`）：
- 距离矩阵：时间 `O(n^2 d)`，空间 `O(n^2)`
- 枚举单纯形：
  - 第 `k` 维候选数 `C(n, k+1)`
  - 每个候选要做 `O((k+1)^2)` 对点检查
- 因而总体近似
  - `O(sum_{k=1..p} C(n,k+1)*(k+1)^2)`

MVP 仅用于小规模演示（例如 `n<=几十`、`p<=2/3`）。

## R09

边界与异常处理：
- 输入点必须是有限实数的二维数组，且样本数 > 0
- `epsilon` 要求 `>=0`，`max_dim` 要求 `>=0`
- 距离矩阵必须方阵
- 若某一维无单纯形，边界矩阵允许空维度（`0 x m` 或 `m x 0`）
- 空矩阵秩按 0 处理，保证 Betti 公式可统一执行

## R10

MVP 取舍说明：
- 没有直接调用 Gudhi/Ripser 等库的一键持久同调接口
- 重点在“从点集到 Betti 数”的可读源代码链路
- 仅实现固定维度（默认到 2 维）与离散阈值，不追求工业级性能
- 依赖最小化，仅使用 `numpy`

## R11

运行方式：

```bash
cd Algorithms/数学-计算拓扑-0245-Vietoris-Rips复形
python3 demo.py
```

脚本不读取命令行参数，不请求用户输入。

## R12

输出字段解释：
- `eps`：当前滤过阈值 `epsilon`
- `#V/#E/#T`：分别为 0/1/2 维单纯形数量
- `beta0`：连通分支数
- `beta1`：1 维环路数（模 2）
- `beta2`：2 维空腔数（模 2）

注意：当只构建到 2 维时，`beta2` 不会被 3 维单纯形边界进一步“填充”，因此在高 `epsilon` 下可能较大，这是截断模型的预期现象。

## R13

内置测试场景：
1. Circle points
- 12 个单位圆均匀采样点
- `epsilon = [0.40, 0.60, 2.05]`
- 在中间阈值通常可观察到 `beta1 = 1`

2. Two clusters
- 两个相距约 2.2 的小簇（每簇 6 点）
- `epsilon = [0.14, 0.30, 2.30]`
- 观察 `beta0` 由多分支逐步降到 1

## R14

关键参数与建议：
- `max_dim`：最大单纯形维度
  - 演示推荐 `2`；若升到 `3+`，组合数会快速增长
- `eps_values`：滤过采样点
  - 建议从小到大选 3-10 个阈值观察拓扑变化
- 点数 `n`
  - 枚举复杂度较高，建议先用 `n<=30` 做验证

## R15

与相关复形对比：
- Čech 复形
  - 几何定义更“精确”（球交非空），但计算代价更高
- Alpha 复形
  - 依赖 Delaunay 结构，几何意义强，维度高时实现更复杂
- Vietoris-Rips 复形
  - 只依赖两两距离，构造简单，最常用于点云 TDA 的实用近似

## R16

典型应用：
- 点云形状分析与流形结构探索
- 时序状态空间重构后的拓扑特征提取
- 图数据的高阶结构近似（经距离/相似度转化后）
- 作为持久同调前处理阶段的基础复形构造

## R17

可扩展方向：
- 接入持久同调（barcode / persistence diagram）
- 使用稀疏邻接和并行枚举优化大样本性能
- 引入 witness / lazy witness 复形降低复杂度
- 增加与 Gudhi/Ripser 的结果对拍脚本做一致性验证

## R18

`demo.py` 源码级流程（8 步）：
1. `main` 生成两个固定点云，并给出各自 `eps_values`。  
2. `run_filtration` 先调用 `pairwise_distance_matrix` 计算距离矩阵。  
3. 对每个 `epsilon`，`build_vietoris_rips_complex` 枚举顶点组合并做 clique 判定，得到各维单纯形表。  
4. `compute_betti_numbers` 逐维调用 `boundary_matrix_mod2` 构造 `d_k`。  
5. `gf2_rank` 对每个 `d_k` 做模 2 高斯消元，得到 `rank(d_k)`。  
6. 按 `beta_k = n_k - rank(d_k) - rank(d_{k+1})` 计算 `beta0/beta1/beta2`。  
7. 每个阈值结果被打包为 `FiltrationRow`，累积成滤过表。  
8. `print_rows` 以表格形式输出两组案例的单纯形数量与 Betti 数变化。  

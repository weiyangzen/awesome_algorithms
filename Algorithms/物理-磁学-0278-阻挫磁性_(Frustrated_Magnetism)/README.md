# 阻挫磁性 (Frustrated Magnetism)

- UID: `PHYS-0275`
- 学科: `物理`
- 分类: `磁学`
- 源序号: `278`
- 目标目录: `Algorithms/物理-磁学-0278-阻挫磁性_(Frustrated_Magnetism)`

## R01

阻挫磁性（frustrated magnetism）指的是：局域相互作用都倾向于某种排布，但几何结构或竞争耦合使这些局域偏好无法同时满足。本文用最小可运行模型展示最经典场景：

- 反铁磁 Ising 在方格晶格上可全部满足（无阻挫）
- 反铁磁 Ising 在三角晶格上不可全部满足（几何阻挫）

## R02

物理直觉：反铁磁耦合希望相邻自旋反向。如果一个三角形三个顶点两两相邻，那么当两条边被“满足”（反向）后，第三条边必然冲突，这就是“不可同时满足”的核心。

- 无阻挫系统通常存在简单长程序（如棋盘格）
- 阻挫系统常有高简并、残余熵、低温复杂态

## R03

本文 Hamiltonian 采用标准反铁磁 Ising 形式：

\[
H = J\sum_{\langle i,j \rangle} s_i s_j,\quad s_i\in\{-1,+1\},\ J>0
\]

- 若 \(s_i s_j=-1\)，该键能量 \(-J\)：满足反铁磁偏好
- 若 \(s_i s_j=+1\)，该键能量 \(+J\)：受挫（未满足）

阻挫比例定义为：

\[
f = \frac{\#\{\langle i,j\rangle: s_i s_j=+1\}}{\#\text{bonds}}
\]

## R04

理论下界对比（最近邻、周期边界）：

- 方格反铁磁：可二分着色，最优时 `f_min = 0`
- 三角反铁磁：每个基本三角形至少 1 条边受挫，最优时 `f_min = 1/3`

因此，仅比较最终 `f` 即可直接识别阻挫效应。

## R05

MVP 目标：在一个脚本里完成“建模 + 求近低能态 + 指标输出”。

- 建立方格与三角晶格最近邻键表
- 用 Metropolis + 几何降温做模拟退火
- 输出总能量、每站点能量、每键能量、受挫键比例
- 明确展示三角晶格的剩余阻挫高于方格晶格

## R06

核心算法为 Metropolis 模拟退火：

1. 随机初始化 `s_i in {-1,+1}`
2. 温度按 `geomspace(T_high -> T_low)` 降低
3. 每个温度做若干 sweeps；每个 sweep 尝试 `N` 次随机单点翻转
4. 单点翻转能量差
   \[
   \Delta E = -2J s_k \sum_{j\in\partial k} s_j
   \]
5. 接受准则：`ΔE <= 0` 必接受；否则按 `exp(-ΔE/T)` 概率接受
6. 每个温度末记录能量和受挫比例

## R07

正确性直觉（而非严格证明）：

- `ΔE` 公式来自仅改变与自旋 `k` 相连的局域键
- Metropolis 接受率满足详细平衡，固定温度下收敛到 Boltzmann 分布
- 缓慢降温会偏向更低能谷
- 方格结果应接近 `f=0`，三角结果应接近 `f≈1/3`，与理论一致

## R08

复杂度（`N=L^2`，`z` 为配位数，方格 `z=4`，三角 `z=6`）：

- 单次翻转复杂度：`O(z)`
- 每个温度一次 sweep：`O(Nz)`
- 总体：`O(n_temps * sweeps_per_temp * N * z)`
- 存储：`O(Nz + Nbonds)`

## R09

demo.py 实现细节：

- `IsingLattice` 数据类封装键表、邻接表、能量函数、阻挫比例函数
- `build_square_af_ising` / `build_triangular_af_ising` 构造周期边界键
- `_build_neighbor_table` 从无向键构造紧凑邻接矩阵与度数
- `metropolis_anneal` 执行退火并记录轨迹
- `summarize` 汇总输出指标

## R10

输入与输出约定：

- 输入：脚本内固定参数（`L, J, 温度范围, sweep 数`）
- 输出：终端文本报告，不需要交互输入
- 随机性：使用固定父种子 `20260407`，并派生两套子种子保证可复现

## R11

运行方式：

```bash
uv run python "Algorithms/物理-磁学-0278-阻挫磁性_(Frustrated_Magnetism)/demo.py"
```

预期行为：脚本在数秒内结束，打印方格与三角两套结果和阻挫差值。

## R12

结果解读指南：

- `energy_per_site` 越低通常代表状态越接近低能构型
- `frustrated bond ratio` 是判定阻挫最直接指标
- 若输出表现为
  - 方格 `f` 接近 `0`
  - 三角 `f` 明显大于方格并接近 `1/3`
  则说明 MVP 成功捕捉阻挫磁性的核心现象

## R13

关键参数与建议：

- `L=18`：兼顾统计稳定性与速度
- `n_temps=40, sweeps_per_temp=14`：足够看到明显差异
- `t_high=4.0, t_low=0.12`：从高温无序过渡到低温近基态

如果需要更平滑结果，可增大 `L`、温度点数或 sweeps。

## R14

局限性：

- 仅为经典 Ising，自旋离散，不含量子涨落
- 仅最近邻耦合，不含次近邻或 Dzyaloshinskii-Moriya 相互作用
- 用单自旋 Metropolis，低温可能存在慢混合
- 结果依赖有限尺寸与退火调度，非严格基态求解器

## R15

可扩展方向：

- 改为 Heisenberg/O(3) 连续自旋模型
- 引入并行回火（parallel tempering）降低低温卡死
- 计算结构因子 `S(q)`、自旋玻璃序参量等更丰富观测量
- 对比 Kagome、pyrochlore 等更强阻挫晶格

## R16

最小验证清单：

- `README.md` 与 `demo.py` 不含未填充占位符
- `uv run python demo.py` 可直接运行
- 输出中方格与三角都给出能量和阻挫比例
- 三角阻挫比例显著高于方格，符合物理预期

## R17

元数据一致性检查：

- `uid`: `PHYS-0275`
- `discipline`: `物理`
- `subcategory`: `磁学`
- `source_number`: `278`
- `name`: `阻挫磁性 (Frustrated Magnetism)`
- `folder_relpath`: `Algorithms/物理-磁学-0278-阻挫磁性_(Frustrated_Magnetism)`

以上与任务描述一致。

## R18

源码级算法流程（非黑箱，8 步）：

1. 在 `build_square_af_ising` / `build_triangular_af_ising` 中枚举周期边界最近邻键，得到无向 `bonds`。
2. `_build_neighbor_table` 统计每个站点度数并填充邻接矩阵 `neighbors`，供 O(z) 局域更新。
3. `metropolis_anneal` 随机初始化自旋数组 `spins`。
4. 生成几何温度序列 `temps = geomspace(t_high, t_low)`。
5. 对每个温度执行 sweeps；每次随机选站点 `k`，读取其邻居并计算局域和 `sum_j s_j`。
6. 用 `ΔE = -2 J s_k sum_j s_j` 计算翻转代价；按 Metropolis 准则决定是否翻转。
7. 每个温度结束后调用 `energy()` 与 `frustrated_ratio()` 记录观测量轨迹。
8. `summarize` 汇总末态指标并在 `main()` 中并排打印方格与三角结果，得到阻挫对比结论。

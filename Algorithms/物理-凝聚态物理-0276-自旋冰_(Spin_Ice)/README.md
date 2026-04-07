# 自旋冰 (Spin Ice)

- UID: `PHYS-0273`
- 学科: `物理`
- 分类: `凝聚态物理`
- 源序号: `276`
- 目标目录: `Algorithms/物理-凝聚态物理-0276-自旋冰_(Spin_Ice)`

## R01

自旋冰（Spin Ice）是几何挫折磁性体系中的代表模型：
在四面体单元上，自旋受局域易轴约束并倾向满足“2-in-2-out”冰规则，
从而形成高度简并的低能态与残余熵，并可在激发态中出现“磁单极子”样缺陷（3-in-1-out / 1-in-3-out）。

本条目给出一个可运行、可校验的最小 MVP：
- 用四面体 Ising 伪自旋模型实现 Metropolis 采样；
- 用 16 态精确枚举做同温度对照；
- 用 `torch` 做能量实现交叉复核。

## R02

本 MVP 的问题设定：

1. 系统由 `n_tetra` 个彼此独立的四面体组成，每个四面体 4 个伪自旋 `sigma_i in {-1,+1}`；
2. 每个自旋沿局域 `<111>` 易轴方向；
3. 哈密顿量采用最近邻有效耦合 + 外场投影：
   `E_tet = J_eff * Σ_{i<j}(sigma_i sigma_j) - h * Σ_i sigma_i (d_i·n)`；
4. 默认参数 `J_eff > 0, h = 0`，低温应偏向 `q = Σ_i sigma_i = 0` 的冰规则态；
5. 温度网格上做无交互批量计算，输出热力学量与缺陷密度。

## R03

关键物理量定义：

- 顶点荷（磁荷代理量）：`q = Σ_i sigma_i`；
- 冰规则态：`q = 0`（2-in-2-out）；
- 单极激发态：`|q| = 2`（3-in-1-out 或 1-in-3-out）；
- 双电荷态：`|q| = 4`（all-in / all-out）。

在 `h=0` 且单四面体下：
- 基态（`q=0`）6 重简并，能量最低；
- 单极激发相对基态能隙约为 `2*J_eff`；
- 低温单极密度满足近似 Arrhenius 关系。

## R04

算法路线分为两条并行路径：

1. 数值路径（MC）：
   - 按温度进行 Metropolis 扫描；
   - 每个 sweep 依次更新 4 个子晶格自旋；
   - 统计 `E, C_v, ice_fraction, monopole_fraction`。
2. 精确路径（Exact）：
   - 穷举单四面体 16 个自旋态；
   - 用配分函数计算精确热力学期望；
   - 与 MC 结果逐温度对比误差。

这样可以避免“只跑蒙特卡洛却无基准”的不透明风险。

## R05

最小工具栈及职责：

- `numpy`：自旋状态、能量与 Metropolis 更新的向量化核心；
- `scipy.special.logsumexp`：稳定计算精确配分函数；
- `pandas`：热力学结果表拼接、打印与导出 CSV；
- `scikit-learn`：
  - `mean_squared_error` 评估 MC vs Exact 误差；
  - `LinearRegression` 拟合单极密度 Arrhenius 斜率；
- `torch`：独立后端复核同一哈密顿量的逐四面体能量。

第三方库只提供数值基础能力，模型构造、更新规则、物理量定义和校验逻辑均在源码中显式实现。

## R06

`demo.py` 的主要产出：

1. 温度-热力学采样表：`energy_per_spin`, `specific_heat`；
2. 缺陷统计：`ice_fraction`, `monopole_fraction`, `double_charge_fraction`；
3. 精确解对照与 RMSE 指标；
4. Arrhenius 拟合得到的激活能估计；
5. 详细结果文件：`thermo_results.csv`。

## R07

正确性校验闭环（脚本内断言）：

1. `energy RMSE (MC vs exact)` 小于阈值；
2. `specific heat RMSE` 小于阈值；
3. `monopole fraction RMSE` 小于阈值；
4. `torch max |E_torch - E_numpy|` 接近机器精度；
5. 低温冰规则占比显著偏高；
6. 高温下 MC 冰规则占比与同温度精确值一致；
7. Arrhenius 激活能接近 `2*J_eff`。

通过后输出 `All checks passed.`。

## R08

默认参数（可直接在 `SpinIceConfig` 中改）：

- `n_tetra = 192`
- `j_eff = 1.0`
- `field_strength = 0.0`
- `field_direction = (0,0,1)`
- `t_min = 0.35`, `t_max = 2.80`, `n_temps = 12`
- `equil_sweeps = 220`, `meas_sweeps = 320`

取舍说明：
- 规模足够稳定又能快速运行；
- 温区覆盖低温冰规则主导到中高温缺陷增多；
- 默认 `h=0` 便于验证经典自旋冰局域约束行为。

## R09

复杂度分析（`N = n_tetra`, `M = n_temps`）：

- 单次子晶格更新：`O(N)`；
- 单个 sweep（4 个子晶格）：`O(N)`；
- 单温度采样：`O((equil_sweeps + meas_sweeps) * N)`；
- 全温区：`O(M * (equil + meas) * N)`；
- 精确 16 态路径：`O(16 * M)`，可忽略。

空间复杂度由自旋矩阵主导：`O(N)`。

## R10

运行方式（无交互）：

```bash
cd Algorithms/物理-凝聚态物理-0276-自旋冰_(Spin_Ice)
uv run python demo.py
```

或在仓库根目录执行：

```bash
uv run python Algorithms/物理-凝聚态物理-0276-自旋冰_(Spin_Ice)/demo.py
```

## R11

输出解读：

1. `Config`：本次模拟参数；
2. `Thermodynamic table sample`：MC 与 Exact 在若干温点的并排值；
3. `Consistency metrics`：三类 RMSE + Torch 复核误差；
4. `Low-T / High-T indicators`：低温冰规则占比、低温残余熵估计；
5. `Arrhenius fit`：激活能估计与 `2*J_eff` 的对照。

## R12

物理解释要点：

- 低温时 `q=0` 态占优，体现冰规则约束；
- 温度升高后 `|q|=2` 单极激发密度上升，系统逐步偏离冰规则流形；
- 比热峰对应缺陷热激发最明显的温区；
- 低温熵接近 `ln(6)/4`（独立四面体模型）反映局域简并性。

## R13

适用范围与局限：

适用：
- 自旋冰局域规则、缺陷统计与热激发的教学/原型验证；
- 快速测试 MC 实现与热力学后处理流程。

局限：
- 本实现采用“独立四面体”近似，不包含真实三维烧绿石晶格的长程约束与环路关联；
- 不含偶极长程相互作用、量子涨落与动力学时间尺度建模；
- 因此不能直接替代真实材料定量模拟。

## R14

常见错误与排查：

1. 把 `q` 统计写错（应是四个 `sigma_i` 求和）；
2. 忘记用同温度精确值对比，导致阈值判定失真；
3. Metropolis `delta_e` 符号错误会导致热力学趋势反常；
4. Arrhenius 拟合温区过宽会拉偏激活能；
5. 将 `h` 与 `d_i·n` 的符号约定混淆会导致磁化方向相反。

## R15

可扩展方向：

1. 从独立四面体扩展到角共享网络（如 pyrochlore / checkerboard ice）；
2. 加入长程偶极相互作用（dipolar spin ice）；
3. 引入单自旋翻转以外的环翻转更新以提升低温采样效率；
4. 加外场扫描，绘制缺陷密度与磁化曲线；
5. 引入并行温度交换（parallel tempering）以改进低温遍历。

## R16

与相关模型关系：

- 与水冰六角网络模型同源：都由“局域规则 + 巨大简并”驱动；
- 与六顶点模型（square ice）在约束结构上相近；
- 与 Ising 模型差异在于几何挫折与局域易轴限制更强；
- 与真实稀土自旋冰材料模型相比，本条目是可审计的低阶教学近似。

## R17

最小可交付清单（本目录已满足）：

1. `README.md`：R01-R18 完整说明；
2. `demo.py`：可直接运行、无需交互；
3. `meta.json`：任务元数据一致；
4. 运行后自动输出校验结果与 `thermo_results.csv`；
5. 不依赖黑盒一键求解，核心流程可在源码逐行追踪。

## R18

`demo.py` 源码级算法流程（8 步）：

1. 读取 `SpinIceConfig`，生成温度网格、局域易轴投影 `d_i·n` 和随机初态自旋。  
2. 对每个温度调用 `simulate_one_temperature()`：先做 `equil_sweeps` 热化，再做 `meas_sweeps` 测量。  
3. 在 `metropolis_sweep()` 中按子晶格 `k=0..3` 顺序向量化更新：显式计算翻转能差 `delta_e` 并按 Metropolis 准则接受/拒绝。  
4. 每次测量通过 `measure_state()` 累积总能量和 `q` 分类占比（冰规则/单极/双电荷）。  
5. 用 `exact_thermo_table()` 穷举 16 态并配合 `logsumexp` 计算精确 `E, C_v, 缺陷密度, 熵`。  
6. 将 MC 表和 Exact 表按温度合并，用 `sklearn.mean_squared_error` 计算 RMSE。  
7. 用 `fit_arrhenius_activation()` 对低温精确单极密度做 `log(rho)` 对 `1/T` 线性回归，得到激活能估计。  
8. 用 `torch_energy_consistency()` 独立复核能量实现，随后执行全部断言并打印 `All checks passed.`。  

说明：主算法（哈密顿量、翻转能差、MC 更新、缺陷统计、精确枚举）均在源码显式实现，第三方库仅承担基础数值与验证工作。

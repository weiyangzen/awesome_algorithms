# 平均场近似

- UID: `MATH-0327`
- 学科: `数学`
- 分类: `统计物理`
- 源序号: `327`
- 目标目录: `Algorithms/数学-统计物理-0327-平均场近似`

## R01

平均场近似（Mean-Field Approximation）用于把多体相互作用问题简化为“单体处在平均背景场中”的自洽问题。  
在统计物理中，最经典场景是 Ising 模型：把邻居自旋的随机涨落用平均磁化强度 `m` 代替，从而把高维耦合系统转成一个标量自洽方程。

本目录 MVP 聚焦铁磁 Ising 平均场：
- 采用自洽方程 `m = tanh(beta * (J*z*m + h))`；
- 手写固定点迭代求解，不调用黑盒求根器；
- 用平均场自由能选择平衡分支，并扫描温度观察相变。

## R02

问题定义（本实现）：
- 输入：
  - 交换常数 `J > 0`；
  - 配位数 `z > 0`；
  - 外场 `h`；
  - 温度序列 `T_list`（要求 `T > 0`）；
  - 固定点迭代参数 `tol, max_iter, damping` 与多组初值 `m0`。
- 输出：
  - 每个温度的平衡磁化 `m_eq`；
  - 对应自由能密度 `f_eq`、磁化率 `chi`、收敛信息；
  - 临界温度附近与 Landau 展开近似的对比。

脚本在 `main()` 内置参数和温度网格，`uv run python demo.py` 可直接运行。

## R03

核心数学关系：

1. 平均场自洽方程（`k_B = 1`）：
   `m = tanh(beta * (J*z*m + h))`, 其中 `beta = 1/T`。  
2. 临界温度：
   `T_c = J*z`（零外场下）。  
3. 平均场自由能密度：
   `f(m) = 0.5*J*z*m^2 - (1/beta)*log(2*cosh(beta*(J*z*m+h)))`。  
4. 磁化率（由自洽方程线性化）：
   `chi = beta*(1-m^2) / (1 - beta*J*z*(1-m^2))`。  
5. `h=0` 且 `T < T_c` 时出现自发磁化双稳分支 `±m`。

## R04

算法流程（高层）：
1. 校验 `J/z/T/tol/max_iter/damping` 合法性。  
2. 遍历每个温度 `T`，计算 `beta=1/T`。  
3. 对每个初值 `m0` 运行阻尼固定点迭代。  
4. 得到候选定点后计算对应自由能 `f(m)`。  
5. 在同温度候选中选取自由能最小的分支作为平衡态。  
6. 计算该平衡态的 `|m|`、`chi`、迭代步数和残差。  
7. 打印温度扫描表，并做临界点附近的近似对比与基本断言。

## R05

核心数据结构：
- `IterationLog`（dataclass）：
  - 记录单次“某温度+某初值”固定点求解结果；
  - 字段包括 `temperature/beta/seed_m0/m_star/free_energy/iterations/residual/converged`。
- `EquilibriumRow`（dataclass）：
  - 记录每个温度最终选中的平衡分支；
  - 附带 `phase`（`ordered/disordered`）与 `susceptibility`。
- `list[IterationLog]`：温度扫描的全部候选轨迹。
- `list[EquilibriumRow]`：按自由能筛选后的平衡态结果。

## R06

正确性要点：
- 自洽映射正确：每轮更新都执行 `m <- tanh(beta*(J*z*m+h))`。  
- 阻尼迭代提高稳定性：`m_next = (1-damping)*m + damping*map(m)`。  
- 物理可行域约束：每轮把 `m` 裁剪到 `[-1, 1]`。  
- 平衡态选择依据自由能最小原则，而非仅依赖单个初值轨迹。  
- 结果做 sanity check：`T>T_c` 的 `|m|` 应小，`T<T_c` 的 `|m|` 应显著非零。

## R07

复杂度分析：
- 设温度点数为 `N_T`，每个温度初值个数为 `N_s`，每次固定点迭代步数上限为 `K`。
- 单步迭代是常数开销（主要是 `tanh` 与若干标量运算）。
- 总时间复杂度：`O(N_T * N_s * K)`。
- 空间复杂度：
  - 若保存全部日志：`O(N_T * N_s)`；
  - 额外计算状态为常数级 `O(1)`。

## R08

边界与异常处理：
- `J <= 0`、`z <= 0`：抛 `ValueError`（本 MVP 只讨论铁磁耦合）。  
- 任一温度 `T <= 0`：抛 `ValueError`。  
- `tol <= 0`、`max_iter <= 0`、`damping` 不在 `(0,1]`：抛 `ValueError`。  
- 迭代中出现非有限数（`nan/inf`）：抛 `RuntimeError`。  
- 若达到 `max_iter` 仍未收敛，记录 `converged=False`，但流程继续，避免单点失败阻断全局扫描。

## R09

MVP 取舍：
- 只做最经典的 Ising 平均场，不加入随机场、反铁磁、长程耦合等扩展。  
- 不做 Monte Carlo 与精确解对照，重点保持最小可运行与可审计。  
- 不用 `scipy.optimize.root` 等黑盒求根，固定点迭代逻辑全部显式实现。  
- 输出以数值表为主，不依赖绘图库，保证命令行环境即跑即得结果。

## R10

`demo.py` 主要函数职责：
- `validate_config`：统一检查输入配置合法性。  
- `mean_field_map`：实现自洽映射 `m -> tanh(beta*(J*z*m+h))`。  
- `free_energy_per_spin`：计算平均场自由能密度。  
- `solve_fixed_point`：执行阻尼固定点迭代并返回收敛信息。  
- `scan_temperatures`：对温度和多初值做批量求解。  
- `mean_field_susceptibility`：计算理论磁化率。  
- `build_equilibrium_rows`：按自由能挑选每个温度的平衡分支。  
- `print_equilibrium_table`：格式化打印结果。  
- `run_sanity_checks`：做高温/低温基本正确性断言。  
- `main`：组织参数、执行扫描、打印汇总。

## R11

运行方式：

```bash
cd Algorithms/数学-统计物理-0327-平均场近似
uv run python demo.py
```

脚本不读取交互输入，不需要命令行参数。

## R12

输出字段说明：
- `T`：温度。  
- `beta`：逆温 `1/T`。  
- `m_eq`：自由能最小分支对应的平衡磁化。  
- `|m_eq|`：用于观察有序程度。  
- `phase`：按 `|m_eq|` 阈值标记 `ordered/disordered`。  
- `f_eq`：平衡自由能密度。  
- `chi`：平均场磁化率（临界处可发散）。  
- `iter`：该平衡分支固定点迭代步数。  
- `residual`：末步残差 `|m_{k+1} - m_k|`。  
- `seed`：选中分支对应初值。  
- `Tc` 与近临界比较：输出 `T_c`、数值 `m`、Landau 近似 `m` 及相对误差。

## R13

建议最小验证项（脚本已覆盖）：
- 温度跨越临界点（`T < T_c` 与 `T > T_c`）的统一扫描。  
- 多初值（负、零、正）检查对称破缺分支。  
- 自由能选枝，避免只看单条轨迹。  
- 临界附近（例如 `T=3.9, T_c=4.0`）与 Landau 近似做量级校对。

建议扩展验证：
- 非零外场 `h != 0` 检查双稳态消失与磁化偏置。  
- 扫描更细温度网格并与解析临界指数做拟合。  
- 与二维 Ising Monte Carlo 结果对比偏差。

## R14

可调参数：
- `J`、`z`：决定临界温度 `T_c=J*z`。  
- `h`：外场强度，控制对称性破缺方向。  
- `temperatures`：扫描区间与分辨率。  
- `seeds`：固定点初值集合，影响候选分支覆盖。  
- `tol`、`max_iter`、`damping`：收敛速度与稳定性折中。

调参建议：
- 临界附近震荡：减小 `damping`（如 `0.6 -> 0.4`）。  
- 收敛过慢：放宽 `tol` 或增大 `max_iter`。  
- 低温支路选取不稳：增加初值数量（如加入 `±0.5`）。

## R15

方法对比：
- 对比精确解/高精度数值：
  - 平均场近似快、可解释，但会高估有序性并忽略涨落。  
- 对比 Monte Carlo（如 Metropolis）：
  - Monte Carlo 代价更高但能保留涨落统计；
  - 平均场更适合做快速基线与参数扫描。  
- 对比重整化群方法：
  - 重整化群能更准确处理临界行为；
  - 平均场实现更简单，适合作为入门与工程近似。

## R16

典型应用场景：
- 铁磁相变的快速近似分析。  
- 神经网络/玻尔兹曼机中的平均场推断启发。  
- 多体系统中“复杂相互作用 -> 自洽方程”的建模范式教学。  
- 在复杂模拟前做参数区间预筛选。

## R17

可扩展方向：
- 增加外场扫描，画出平均场磁滞回线。  
- 增加时间依赖动力学（如 `m_{t+1}` 动力系统分析）。  
- 改为矢量自旋（Heisenberg）或多子晶格平均场。  
- 引入随机耦合实现简化自旋玻璃平均场。  
- 对接 `pandas`/`matplotlib` 输出 CSV 与相图。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 固定模型参数 `J,z,h`、温度网格和迭代配置，并计算 `T_c = J*z`。  
2. `scan_temperatures` 对每个温度 `T` 和每个初值 `m0` 调用 `solve_fixed_point`。  
3. `solve_fixed_point` 内部反复执行 `mean_field_map`，用阻尼更新 `m_next=(1-d)*m+d*map(m)` 并裁剪到 `[-1,1]`。  
4. 每轮迭代计算残差 `|m_next-m|`，若低于 `tol` 则判定收敛并返回。  
5. 收敛后用 `free_energy_per_spin` 计算候选分支自由能，形成 `IterationLog`。  
6. `build_equilibrium_rows` 按温度聚合候选，选择自由能最小者作为平衡态，并计算 `phase` 与 `susceptibility`。  
7. `print_equilibrium_table` 输出每个温度的 `m_eq/f_eq/chi/iter/residual`，并统计候选收敛率。  
8. `main` 对临界温度附近点计算 Landau 近似 `m ~ sqrt(3*(T_c-T)/T_c)` 并做误差对比，再执行 `run_sanity_checks` 结束。

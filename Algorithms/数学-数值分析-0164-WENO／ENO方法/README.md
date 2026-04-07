# WENO/ENO方法

- UID: `MATH-0164`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `164`
- 目标目录: `Algorithms/数学-数值分析-0164-WENO／ENO方法`

## R01

本条目实现 WENO/ENO 思想在一维线性对流方程上的最小可运行 MVP：

- 方程：`u_t + a u_x = 0`（周期边界）；
- 空间离散：基于单元值的界面重构；
- 时间推进：`SSP-RK3`；
- 对比方法：`ENO3-like` 与 `WENO5-JS`。

目标是把“非振荡高阶重构”完整落地，而不是调用现成黑盒 PDE 求解器。

## R02

问题定义（离散计算版本）：

- 输入：
  - 初值函数 `u(x,0)=u0(x)`；
  - 网格数 `N`、区间长度 `L`、速度 `a`；
  - CFL 参数 `cfl`、终止时间 `T`；
  - 重构方法 `method in {eno3, weno5}`。
- 输出：
  - 终时刻数值解 `u^n`；
  - 与解析平移解的 `L1/Linf` 误差；
  - 总变差 `TV` 与数值范围 `min/max`（用于观察振荡）。

## R03

核心数学公式：

1) 守恒离散形式

`du_i/dt = -(F_{i+1/2} - F_{i-1/2}) / dx`，其中线性通量 `F=a*u`。

2) WENO5-JS 左界面重构（`a>0`）

- 三个三阶候选多项式值：`q0, q1, q2`；
- 平滑指标：`beta0, beta1, beta2`；
- 理想权重：`d=(0.1, 0.6, 0.3)`；
- 非线性权重：
  `alpha_k = d_k / (eps + beta_k)^p`，`omega_k = alpha_k / sum(alpha)`；
- 重构值：`u_{i+1/2}^- = sum_k omega_k q_k`。

3) ENO3-like（本 MVP 版本）

- 使用与 WENO 同一组 `q0,q1,q2` 与 `beta0,beta1,beta2`；
- 直接选择 `beta` 最小对应的候选多项式作为 `u_{i+1/2}^-`；
- 这是教学友好的 ENO 风格简化，不是完整分差树版 ENO。

4) 时间离散

采用三阶 SSP-RK3：

- `u^(1) = u^n + dt L(u^n)`
- `u^(2) = 3/4 u^n + 1/4 (u^(1) + dt L(u^(1)))`
- `u^(n+1) = 1/3 u^n + 2/3 (u^(2) + dt L(u^(2)))`

## R04

算法流程（MVP）：

1. 在周期网格上生成初值向量 `u0`。
2. 根据 CFL 设定时间步长 `dt = cfl * dx / |a|`。
3. 每个 RK 子步调用空间离散算子 `L(u)`。
4. `L(u)` 内部先做界面重构：`ENO3-like` 或 `WENO5`。
5. 用重构得到的 `u_{i+1/2}^-` 形成通量 `F_{i+1/2}=a*u_{i+1/2}^-`。
6. 用通量差更新各单元导数。
7. 迭代到终止时间 `T`。
8. 与解析平移解比较并输出误差与总变差。

## R05

核心数据结构：

- `numpy.ndarray`：
  - 解向量 `u`（长度 `N`）；
  - 候选重构 `q0/q1/q2`；
  - 平滑指标 `beta0/beta1/beta2`；
  - 界面通量 `flux_iphalf`。
- `SimulationSummary`（`dataclass`）：
  - `method`、`case_name`、`n_cells`、`cfl`、`final_time`；
  - `l1_error`、`linf_error`、`total_variation`、`u_min`、`u_max`。

## R06

正确性要点：

- 守恒型更新使用“界面通量差”，满足离散守恒结构；
- WENO 在平滑区趋近高阶线性组合，在间断附近自动降低高振荡模板权重；
- ENO 风格选择局部最平滑模板，避免跨越间断造成强振荡；
- 周期边界通过 `np.roll` 实现，索引一致且无外插误差。

## R07

复杂度分析：

- 单次空间算子调用：`O(N)`；
- RK3 每步调用 3 次空间算子，单时间步 `O(N)`；
- 时间步数约为 `O(T/dt)`，总体 `O(N * T/dt)`；
- 额外内存：若干与 `N` 同长度数组，空间 `O(N)`。

## R08

边界与异常处理：

- `cfl <= 0`、`dx <= 0`、`T < 0` 时抛出 `ValueError`；
- 未知方法名时抛出 `ValueError`；
- 速度 `a=0` 时直接返回初值（解不演化）；
- 设置 `max_steps` 防止参数异常导致无限循环。

## R09

MVP 取舍说明：

- 采用 `numpy` + 标准库，避免重型框架；
- 只实现正速度 `a>0` 的左偏重构路径，保持代码短而清晰；
- ENO 选用“最小 beta 候选”简化版，直观展示“自适应选模板”思想；
- 未加入通用黎曼求解器、源项、多维网格等扩展。

## R10

`demo.py` 函数职责：

- `weno5_left_state`：WENO5-JS 界面重构；
- `eno3_like_left_state`：ENO3-like 界面重构；
- `spatial_operator`：组装通量差得到半离散算子 `L(u)`；
- `rk3_step`：执行一次 SSP-RK3 更新；
- `solve_advection`：推进到终止时间；
- `run_case`：单个样例求解与误差统计；
- `main`：批量运行平滑/间断样例并打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0164-WENO／ENO方法
python3 demo.py
```

脚本无需交互输入，会自动输出两类初值下 ENO 与 WENO 的误差和变差指标。

## R12

输出解读：

- `L1` / `Linf`：终时刻对解析平移解误差；
- `TV`：总变差，间断问题中可用于观察数值耗散与振荡倾向；
- `min/max`：用于观察是否出现明显超调或欠调；
- 一般可观察到：平滑样例里 WENO5 精度优于 ENO3-like。

## R13

最小测试集（已内置）：

- `smooth-sine`：`sin(x)+0.5 sin(2x)`，检验高阶精度；
- `discontinuous`：方波初值，检验非振荡能力；
- 两组均在一个周期后对比解析平移解（`T=2π`，`a=1`）。

## R14

关键参数建议：

- `n_cells`：默认 `240`，可提高到 `400+` 观察更明显阶数差异；
- `cfl`：默认 `0.45`，若出现不稳定可下调到 `0.3`；
- `eps`（WENO）：默认 `1e-6`，用于避免除零并调节权重敏感度；
- `power`（WENO）：默认 `2`，控制平滑区与间断区权重分离强度。

## R15

方法对比：

- ENO：
  - 优点：思路直接，模板选择明确；
  - 缺点：选择离散跳变，光滑区误差常比 WENO 大。
- WENO：
  - 优点：通过连续权重融合模板，平滑区精度和稳健性更好；
  - 缺点：计算量更高，参数（`eps`、`p`）会影响表现。

## R16

应用场景：

- 可压缩流体中的激波捕捉（高阶有限体积/差分框架）；
- 对流主导方程的高分辨率无振荡离散；
- 数值 PDE 教学中“ENO 到 WENO”的方法演化演示。

## R17

可扩展方向：

- 增加负速度/通用通量分裂（Lax-Friedrichs）与双向重构；
- 实现经典分差树 ENO3 以替代当前 ENO-like 简化版；
- 增加 Burgers 方程等非线性通量与黎曼通量；
- 增加收敛阶实验（网格加密）与自动化单元测试。

## R18

源码级算法流（`demo.py`，8 步）：

1. `main` 设置网格、CFL、终止时间，并组织 `smooth-sine` 与 `discontinuous` 两个样例。  
2. `run_case` 生成周期网格 `x` 与初值 `u0`，调用 `solve_advection` 执行时间推进。  
3. `solve_advection` 根据 `dt = cfl*dx/|a|` 循环推进，并在每步调用 `rk3_step`。  
4. `rk3_step` 依次计算三次 `L(u)`，按 SSP-RK3 组合得到下一时刻解。  
5. `spatial_operator` 根据 `method` 选择 `eno3_like_left_state` 或 `weno5_left_state` 重构 `u_{i+1/2}^-`。  
6. 重构后形成界面通量 `F_{i+1/2}=a*u_{i+1/2}^-`，再由差分 `-(F_{i+1/2}-F_{i-1/2})/dx` 得到 `L(u)`。  
7. 推进结束后，`run_case` 用 `exact_periodic_shift` 构造解析参考解并计算 `L1/Linf/TV/min/max`。  
8. `print_summary` 按方法输出结果，形成 ENO 与 WENO 在平滑/间断样例上的可比较报告。  

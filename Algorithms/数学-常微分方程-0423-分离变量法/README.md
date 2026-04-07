# 分离变量法

- UID: `MATH-0423`
- 学科: `数学`
- 分类: `常微分方程`
- 源序号: `423`
- 目标目录: `Algorithms/数学-常微分方程-0423-分离变量法`

## R01

分离变量法用于一阶可分离常微分方程：

`dy/dx = f(x) g(y)`。

核心思路是把未知量按变量拆开：

`(1/g(y)) dy = f(x) dx`

并积分得到隐式关系：

`∫(1/g(y))dy = ∫f(x)dx + C`。

本目录实现最小可运行 MVP：用数值积分和单调反函数插值恢复 `y(x)`，再与 `scipy.solve_ivp` 与解析解对照验证。

## R02

本实现的问题定义：
- 输入：
  - 右端分解函数 `f(x)`、`g(y)`；
  - 初值 `y(x0)=y0`；
  - 严格递增网格 `x_grid`；
  - `y` 可行区间 `y_bounds`（用于构建 `H(y)` 映射）。
- 输出：
  - 分离变量法构造的轨迹 `y_sep(x)`；
  - SciPy 参考解 `y_scipy(x)`；
  - 解析参考 `y_exact(x)`（示例内置）；
  - ODE 残差与误差统计。

`demo.py` 采用无交互固定样例，直接执行即可。

## R03

数学基础（可分离方程）：

1. 原方程：`y' = f(x)g(y)`。
2. 变量分离：`(1/g(y)) dy = f(x) dx`。
3. 对初值点 `x0` 到 `x` 积分：
   `∫_{y0}^{y(x)} (1/g(s)) ds = ∫_{x0}^{x} f(t) dt`。
4. 定义
   `H(y)=∫_{y_ref}^{y} (1/g(s)) ds`，
   则有 `H(y(x)) = H(y0) + I(x)`，其中
   `I(x)=∫_{x0}^{x} f(t)dt`。
5. 若 `g(y)` 在区间内不变号且不为 0，则 `H` 单调，可通过反函数恢复 `y(x)`。

## R04

算法流程（高层）：
1. 检查 `x_grid` 合法性（1D、严格递增、有限值）。  
2. 在网格上计算 `f(x)`，用累积梯形积分得到 `I(x)=∫f`。  
3. 在 `y_bounds` 上离散化 `y`，计算 `g(y)`。  
4. 构造 `H(y)=∫(1/g(y))dy` 的离散表（同样用累积梯形积分）。  
5. 用初值 `y0` 插值得到 `H(y0)`，构造目标 `target(x)=H(y0)+I(x)`。  
6. 通过单调插值求反函数 `y_sep(x)=H^{-1}(target(x))`。  
7. 数值微分得到 `y'_sep`，检查残差 `y'_sep - f(x)g(y_sep)`。  
8. 用 `solve_ivp` 得参考轨迹，统计误差并输出 PASS/FAIL。

## R05

核心数据结构：
- `SeparableCase`（`dataclass`）：
  - `name`、`f`、`g`、`y0`、`x_grid`；
  - `y_bounds`；
  - `exact_solution`（用于额外核对）。
- `SeparationResult`（`dataclass`）：
  - 轨迹：`x, y_sep, y_scipy, y_exact`；
  - 误差：`abs_error_to_scipy, abs_error_to_exact`；
  - 诊断：`residual_numeric`、`max_abs_residual`；
  - 指标：`relative_l2_error_to_scipy`、`relative_l2_error_to_exact`、
    `mse_to_scipy`、`torch_max_abs_error_to_scipy`。

抽样展示由 `pandas.DataFrame` 输出，便于人工核查。

## R06

正确性保障：
- 理论层：严格依据 `∫dy/g(y)=∫f(x)dx` 的分离积分公式。  
- 结构层：`H(y)` 只在 `g` 不变号区域构造，保证单调反解有效。  
- 数值层：
  - 与 `solve_ivp` 对比相对 L2 误差；
  - 与已知解析解对比相对 L2 误差；
  - 检查离散残差 `y' - f(x)g(y)` 的最大绝对值。
- 多样例层：覆盖指数型、Logistic 型、反三角型三类可分离结构。

## R07

复杂度（单样例）：
- 记 `n = len(x_grid)`，`m = y_map_points`（默认 50001）。  
- 计算 `I(x)`：`O(n)`。  
- 构造 `H(y)`：`O(m)`。  
- 反插值恢复 `y(x)`：`O(n)`。  
- 残差与误差统计：`O(n)`。  
- 总时间复杂度：`O(n + m)`；空间复杂度：`O(n + m)`。

## R08

边界与异常处理：
- `x_grid` 非法（非 1D、非递增、含 `nan/inf`）抛 `ValueError`。  
- `y_bounds` 非法或 `y0` 不在区间内抛 `ValueError`。  
- `g(y)` 在区间内过小或变号导致 `1/g(y)` 不稳定时抛 `ValueError`。  
- `H(y)` 非严格单调（不可逆）时抛 `ValueError`。  
- `target(x)` 超出 `H` 映射范围时抛 `ValueError`。  
- `solve_ivp` 失败时抛 `RuntimeError`。  
- 指标未达阈值时抛 `RuntimeError`，避免静默错误。

## R09

MVP 取舍：
- 保留：
  - 分离变量法本体（不是调用黑盒直接求解）；
  - 数值积分 + 反插值的可解释实现；
  - SciPy/解析双重校验。
- 不做：
  - 自动符号积分与符号反函数；
  - 事件检测、自适应映射区间扩张；
  - 高维系统、非可分离方程处理。

目标是“小而诚实”的可复现算法演示。

## R10

`demo.py` 函数职责：
- `_check_grid`：统一检查网格输入。  
- `_build_monotone_integral_map`：在 `y` 轴构造单调映射 `H(y)`。  
- `_first_derivative`：计算数值一阶导。  
- `solve_by_separation`：主入口，串联分离积分、反函数恢复和诊断。  
- `make_report_table`：抽样输出关键点。  
- `build_cases`：构造 3 个固定测试样例。  
- `main`：执行全部样例、打印汇总并给出 PASS。

## R11

运行方式（无交互）：

```bash
cd Algorithms/数学-常微分方程-0423-分离变量法
uv run python demo.py
```

脚本会直接输出每个样例的误差表与最终汇总。

## R12

输出字段说明：
- 单样例输出：
  - `y range (separated)`：分离变量法结果区间；
  - 抽样表列：
    - `x`；
    - `y_sep`（分离法结果）；
    - `y_scipy`（参考数值解）；
    - `y_exact`（解析参考）；
    - `abs_err_scipy`、`abs_err_exact`；
    - `residual`（离散方程残差）。
- 汇总输出：
  - `rel_l2_scipy`、`rel_l2_exact`；
  - `max_abs_residual`；
  - `mse_scipy`（`sklearn` 计算）；
  - `torch_max_abs`（`torch` 计算）；
  - `PASS`。

## R13

内置测试场景：
1. Variable-rate exponential：`y' = x y, y(0)=1`，解析解 `exp(x^2/2)`。  
2. Logistic growth：`y' = y(1-y), y(0)=0.2`，解析解 `1/(1+4e^{-x})`。  
3. Arctan/tangent pair：`y' = (1+x^2)(1+y^2), y(0)=0`，
   解析解 `tan(x + x^3/3)`。

这三个案例分别覆盖线性增长、有界增长、非线性三角映射三种典型可分离结构。

## R14

关键参数：
- `y_map_points=50001`：`H(y)` 离散映射密度。  
- `g_floor=1e-10`：防止 `1/g` 爆炸的阈值。  
- 各样例 `x_grid`：
  - `[0,1.2]` / 601 点；
  - `[0,4.0]` / 801 点；
  - `[0,0.7]` / 501 点。  
- 验收阈值（`main`）：
  - `rel_l2_scipy < 2e-4`；
  - `rel_l2_exact < 3e-4`；
  - `max_abs_residual < 3e-3`。

## R15

方法对比：
- 对比直接数值积分（RK45）：
  - RK45 通用性更强；
  - 分离变量法在可分离结构下可直接利用方程结构、可解释性更强。  
- 对比一阶线性积分因子法：
  - 积分因子法针对线性方程；
  - 分离变量法适用于 `f(x)g(y)` 结构，不要求线性。  
- 对比隐式欧拉等离散法：
  - 离散法更偏通用近似；
  - 当前方法先构造连续积分关系，再反解，结构信息利用更充分。

## R16

典型应用：
- 人口/生态 Logistic 增长模型分析。  
- 反应动力学中一阶可分离速率方程。  
- 课堂中“从微分形式到积分形式”的方法演示。  
- 作为通用 ODE 求解器之前的可解释基线校验模块。

## R17

可扩展方向：
- 增加自动 `y_bounds` 扩展策略，减少手工配置。  
- 增加分段映射与奇异点邻域处理（如 `g(y)=0` 附近）。  
- 支持批量样例并行执行与 CSV 报告导出。  
- 与符号工具（如 SymPy）结合，自动识别可闭式积分情形。  
- 对误差做网格收敛性分析（`x` 与 `y` 双网格 refinement）。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 调用 `build_cases` 生成 3 组可分离 ODE（含 `f,g,y0,x_grid,y_bounds`）。  
2. 每个样例进入 `solve_by_separation`，先做 `x_grid` 和初值/区间合法性检查。  
3. 在 `x_grid` 上计算 `f(x)`，用 `cumulative_trapezoid` 得到右侧积分 `I(x)=∫f`。  
4. 调 `_build_monotone_integral_map`：在 `y_bounds` 上离散 `g(y)`，计算
   `H(y)=∫(1/g)dy`，并强制单调。  
5. 用插值求 `H(y0)`，构造 `target(x)=H(y0)+I(x)`，再反插值得到 `y_sep(x)`。  
6. 通过 `_first_derivative` 计算 `y'_sep`，组装残差 `y'_sep-f(x)g(y_sep)`。  
7. 调 `solve_ivp` 得参考数值解 `y_scipy`，并用内置解析解得 `y_exact`，计算 L2 误差、
   `sklearn` MSE、`torch` 最大绝对误差。  
8. `main` 打印抽样表和汇总表，按阈值输出 `PASS`；若未通过则抛 `RuntimeError`。

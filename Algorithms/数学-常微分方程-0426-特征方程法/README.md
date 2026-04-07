# 特征方程法

- UID: `MATH-0426`
- 学科: `数学`
- 分类: `常微分方程`
- 源序号: `426`
- 目标目录: `Algorithms/数学-常微分方程-0426-特征方程法`

## R01

特征方程法用于求解二阶线性常系数齐次常微分方程：

`a y'' + b y' + c y = 0, a != 0`

核心思想是先求代数方程 `a r^2 + b r + c = 0` 的根，再依据根的类型直接写出解析解。  
本目录 MVP 不做符号系统依赖，只做数值可运行实现，并对结果做残差与参考解校验。

## R02

本实现的问题定义：
- 输入：
  - 常系数 `a, b, c`（`a != 0`）；
  - 初值 `y(0)=y0, y'(0)=dy0`；
  - 一维严格递增网格 `x_grid`。
- 输出：
  - 由特征方程法构造的离散解析解 `y(x)`；
  - 数值导数与 ODE 残差；
  - 与 `scipy.integrate.solve_ivp` 参考解的误差指标。

`demo.py` 内置 3 个非交互测试场景（实根、重根、共轭复根）。

## R03

数学基础（判别式 `Δ = b^2 - 4ac`）：

1. 特征方程：`a r^2 + b r + c = 0`。
2. 若 `Δ > 0`（两个不等实根 `r1, r2`）：
   `y = C1 e^(r1 x) + C2 e^(r2 x)`。
3. 若 `Δ = 0`（重根 `r`）：
   `y = (C1 + C2 x)e^(r x)`。
4. 若 `Δ < 0`（`r = α ± iβ`）：
   `y = e^(αx) [C1 cos(βx) + C2 sin(βx)]`。
5. 使用初值 `y(0), y'(0)` 可以唯一确定 `C1, C2`。

## R04

算法流程（高层）：
1. 校验网格：一维、递增、有限值、长度足够。  
2. 计算判别式 `Δ`，判定根型。  
3. 按根型选用对应解析模板，结合初值反推出 `C1, C2`。  
4. 在 `x_grid` 上批量计算 `y(x), y'(x)`。  
5. 用数值微分得到 `y''(x)`，构造残差 `a y'' + b y' + c y`。  
6. 用 `solve_ivp` 求同一初值问题的参考轨迹，计算相对 L2 误差。  
7. 汇总每个场景的误差与残差，给出 PASS/FAIL。

## R05

核心数据结构：`CharacteristicResult`（`dataclass`）
- 方程参数：`a, b, c`；
- 网格与解：`x, y, y_prime, y_second_numeric`；
- 验证量：`residual_numeric, y_scipy, abs_error_to_scipy`；
- 指标：`relative_l2_error, max_abs_residual`；
- 解释性字段：`root_case, roots, c1, c2`。

表格展示使用 `pandas.DataFrame`，便于肉眼检查离散点行为。

## R06

正确性要点：
- 解析构造正确性来自标准特征方程法分支公式。  
- 初值一致性：每个分支都显式使用 `y(0), y'(0)` 求 `C1, C2`。  
- 方程一致性：`a y'' + b y' + c y` 的离散残差应接近 0。  
- 数值一致性：与 `solve_ivp` 参考解在同网格上误差应足够小。  
- 三种根型全覆盖，避免只在单一场景“看起来正确”。

## R07

复杂度（单场景、网格点数 `n`）：
- 解析解逐点评估：`O(n)`；
- 数值一二阶导：`O(n)`；
- 残差与误差统计：`O(n)`；
- 总时间复杂度：`O(n)`；
- 额外空间复杂度：`O(n)`。

## R08

边界与异常处理：
- `a = 0`：方程退化，抛 `ValueError`。  
- `x_grid` 非法（非一维/非递增/存在非有限值/点数不足）：抛 `ValueError`。  
- `root_tol <= 0`：抛 `ValueError`。  
- 参考解求解失败：抛 `RuntimeError`。  
- 误差指标未过阈值：抛 `RuntimeError`，防止静默失败。

## R09

MVP 取舍：
- 仅实现二阶、常系数、齐次方程；不覆盖高阶与非齐次。  
- 解析主流程手写，不把“求解本身”交给黑盒库。  
- `solve_ivp` 只用于验证，不参与特征方程求解。  
- 只做终端文本输出，不做绘图或文件落盘，保持最小可运行。

## R10

`demo.py` 函数职责：
- `_check_grid`：网格合法性检查。  
- `_first_second_derivative`：离散一、二阶导计算。  
- `_solve_distinct_real / _solve_repeated_real / _solve_complex_conjugate`：
  三种根型的解析解与导数构造。  
- `_solve_reference_with_scipy`：生成参考数值解。  
- `solve_by_characteristic_equation`：主入口，串联根型判断、求解、诊断。  
- `make_report_table`：抽样输出核心数据。  
- `main`：运行三个场景并汇总 PASS/FAIL。

## R11

运行方式（无交互）：

```bash
cd Algorithms/数学-常微分方程-0426-特征方程法
uv run python demo.py
```

脚本将直接打印每个场景的根型、常数、抽样点误差与汇总指标。

## R12

输出说明：
- 单场景输出：
  - 方程与根型（`distinct_real/repeated_real/complex_conjugate`）；
  - 特征根、常数 `c1, c2`；
  - 抽样表格字段：
    - `x`：采样点；
    - `y_char`：特征方程解析值；
    - `y_scipy`：参考数值解；
    - `abs_err`：绝对误差；
    - `residual`：ODE 残差。
- 汇总输出：
  - `relative_l2_error`；
  - `max_abs_residual`；
  - 最终 `PASS` 布尔值。

## R13

内置测试场景：
1. `y'' - 3y' + 2y = 0`（不等实根 `1, 2`）。  
2. `y'' - 2y' + y = 0`（重根 `1`）。  
3. `y'' + 2y' + 5y = 0`（共轭复根 `-1 ± 2i`）。

每个场景都设置 `y(0), y'(0)` 并在同一网格上做解析/数值对照。

## R14

可调参数：
- `x = np.linspace(0.0, 3.0, 601)`：积分区间与采样密度。  
- `root_tol`：判定“重根”阈值（默认 `1e-12`）。  
- `solve_ivp` 容差：`rtol=1e-10, atol=1e-12`。  
- 判定阈值：
  - `relative_l2_error < 1e-5`；
  - `max_abs_residual < 2e-2`（内点）。

## R15

方法对比：
- 对比“纯数值法（如 RK）”：
  - 数值法通用性更强；
  - 特征方程法在常系数齐次场景下能直接给解析表达，更可解释。  
- 对比“降阶法”：
  - 降阶法需先已知一个解；
  - 特征方程法不需先验解，但仅适用于常系数线性齐次结构。  
- 对比“拉普拉斯变换”：
  - 拉普拉斯更系统但步骤更重；
  - 当前场景特征方程法更轻量直接。

## R16

典型应用：
- 常系数振动/阻尼模型的解析求解与快速验证。  
- ODE 课程中“判别式决定解形态”的教学演示。  
- 数值求解器基准测试：用解析解做回归对照。  
- 工程模型参数扫描前的快速可解释基线。

## R17

可扩展方向：
- 扩展到 `n` 阶常系数线性齐次方程（多项式高阶特征根）。  
- 加入非齐次项与特解构造（待定系数/常数变易）。  
- 增加自动绘图与 CSV 报告导出。  
- 引入批量参数网格，输出稳定性区域统计。  
- 增加对复重根、近重根数值敏感性的鲁棒分析。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构造统一网格和 3 个测试方程（分别覆盖三种根型）。  
2. 每个场景调用 `solve_by_characteristic_equation`，先做 `a`、`x_grid`、`root_tol` 的输入检查。  
3. 计算判别式 `Δ=b^2-4ac`，并分支到 `_solve_distinct_real`、`_solve_repeated_real` 或 `_solve_complex_conjugate`。  
4. 在对应分支内，先由初值方程解出 `c1,c2`，再逐点计算解析 `y(x), y'(x)`。  
5. 用 `_first_second_derivative` 对解析轨迹做离散求导得到 `y''`，组装残差 `a y''+b y'+c y`。  
6. 调用 `_solve_reference_with_scipy`（`solve_ivp`）获得同初值问题的参考数值解 `y_scipy`。  
7. 计算 `|y-y_scipy|`、相对 L2 误差和内点最大残差，写入 `CharacteristicResult`。  
8. `main` 打印每个场景抽样表与汇总表，按阈值输出 `PASS`；若失败抛 `RuntimeError` 终止。

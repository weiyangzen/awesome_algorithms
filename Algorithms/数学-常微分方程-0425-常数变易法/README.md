# 常数变易法

- UID: `MATH-0425`
- 学科: `数学`
- 分类: `常微分方程`
- 源序号: `425`
- 目标目录: `Algorithms/数学-常微分方程-0425-常数变易法`

## R01

常数变易法（Variation of Parameters）用于线性非齐次 ODE。  
本目录聚焦二阶标准型：

`y'' + p(x) y' + q(x) y = g(x)`。

若已知齐次方程 `y'' + p y' + q y = 0` 的基解 `y1, y2`，可令齐次解中的常数变成函数，构造特解 `y_p`，再与齐次部分叠加得到满足初值的总解。`demo.py` 给出最小可运行数值版并与 SciPy/解析解交叉验证。

## R02

本实现的问题定义：
- 输入：
  - 系数函数 `p(x), q(x), g(x)`；
  - 齐次基解与导数 `y1, y1', y2, y2'`；
  - 初值 `y(x0)=y0, y'(x0)=dy0`；
  - 严格递增网格 `x_grid`。
- 输出：
  - 常数变易法轨迹 `y_var(x)`；
  - 参考数值解 `y_scipy(x)`；
  - 解析参考 `y_exact(x)`（示例内置）；
  - 残差与误差指标（L2、MSE、torch 最大绝对误差）。

脚本是非交互执行：`uv run python demo.py`。

## R03

数学公式（标准型）：

1. Wronskian：`W = y1*y2' - y1'*y2`。
2. 常数变易导数：
   - `u1' = -(y2*g)/W`
   - `u2' =  (y1*g)/W`
3. 取 `u1(x0)=u2(x0)=0`，做定积分得到 `u1(x), u2(x)`。
4. 特解：`y_p = u1*y1 + u2*y2`。
5. 总解：`y = c1*y1 + c2*y2 + y_p`。
6. 在 `x0` 由线性方程
   `[[y1,y2],[y1',y2']] [c1,c2]^T = [y0,dy0]^T`
   求得 `c1,c2`。

## R04

算法流程（高层）：
1. 校验 `x_grid`（一维、有限值、严格递增）。
2. 在网格上评估 `p,q,g,y1,y1',y2,y2'`。
3. 计算 `W`，检查是否接近 0（防止线性相关/病态）。
4. 根据 `u1',u2'` 用累积梯形积分得到 `u1,u2`。
5. 组装 `y_p`，并用初值解出齐次系数 `c1,c2`。
6. 得到总解 `y=y_h+y_p`。
7. 数值求导构造 ODE 残差 `y''+p y'+q y-g`。
8. 调 `solve_ivp` 得参考解，计算误差并汇总 PASS/FAIL。

## R05

核心数据结构：
- `VariationCase`（输入定义）：
  - `p,q,g`；`y1,y1_prime,y2,y2_prime`；
  - `y0,dy0,x_grid`；`exact_solution`。
- `VariationResult`（输出与诊断）：
  - `y, y_h, y_p, y_scipy, y_exact`；
  - `wronskian, c1, c2`；
  - `abs_error_to_scipy, abs_error_to_exact`；
  - `residual_numeric` 与误差指标。

`pandas.DataFrame` 用于抽样打印关键点，便于人工核查。

## R06

正确性保障：
- 理论正确性：直接使用常数变易法标准公式。  
- 可逆性保障：要求 `|W|` 不低于阈值。  
- 初值一致性：显式解 `c1,c2`，而非隐式调整。  
- 方程一致性：检查离散残差 `y''+p y'+q y-g`。  
- 参考一致性：与 `solve_ivp` 结果对比；并与解析解对比。  
- 多场景覆盖：指数共振、三角共振、重根基解三类。

## R07

复杂度（单场景，`n=len(x_grid)`）：
- 函数评估与 Wronskian：`O(n)`。
- `u1,u2` 累积积分：`O(n)`。
- 组装解与残差：`O(n)`。
- 参考求解与误差统计（按 `t_eval` 网格计）：`O(n)` 量级。
- 总体：时间 `O(n)`，空间 `O(n)`。

## R08

边界与异常处理：
- `x_grid` 非法（维度、排序、有限性）抛 `ValueError`。  
- 任一函数输出形状不匹配或出现非有限值抛 `ValueError`。  
- `wronskian_floor<=0` 抛 `ValueError`。  
- `|W|` 过小抛 `ValueError`。  
- 初值点基解矩阵近奇异抛 `ValueError`。  
- `solve_ivp` 失败抛 `RuntimeError`。  
- 误差指标不达阈值抛 `RuntimeError`，避免静默失败。

## R09

MVP 取舍：
- 保留：
  - 常数变易法核心积分过程（`u1',u2' -> u1,u2 -> y_p`）；
  - 初值系数线性求解；
  - SciPy/解析双校验。
- 不做：
  - 自动寻找基解（本实现假设基解已知）；
  - 符号推导与闭式积分；
  - 边值问题与高阶系统泛化。

目标是“小而诚实”的可复现实验。

## R10

`demo.py` 主要函数：
- `_check_grid`：检查网格输入。  
- `_eval_on_grid`：统一做函数评估、形状和有限性校验。  
- `_first_second_derivative`：离散一/二阶导。  
- `_solve_reference_with_scipy`：建立一阶系统并调用 RK45。  
- `solve_by_variation_of_parameters`：常数变易法主流程。  
- `make_report_table`：抽样输出结果表。  
- `build_cases`：构造 3 个固定测试场景。  
- `main`：执行、汇总、阈值判定。

## R11

运行方式：

```bash
cd Algorithms/数学-常微分方程-0425-常数变易法
uv run python demo.py
```

输出包含每个案例的常数项、Wronskian 范围、抽样误差表以及最终 PASS。

## R12

输出字段说明：
- 抽样表列：
  - `x`：采样点；
  - `y_var`：常数变易法结果；
  - `y_scipy`：`solve_ivp` 参考；
  - `y_exact`：解析参考；
  - `abs_err_scipy`、`abs_err_exact`；
  - `residual`：离散 ODE 残差。
- 汇总指标：
  - `rel_l2_scipy`、`rel_l2_exact`；
  - `max_abs_residual`；
  - `mse_scipy`（`sklearn`）；
  - `torch_max_abs`（`torch`）。

## R13

内置三组测试：
1. Exponential resonance：`y''-y=e^x`。  
2. Sine forcing with resonance：`y''+y=sin(x)`。  
3. Repeated-root homogeneous basis：`y''+2y'+y=x`。

三组分别覆盖：指数强迫、三角强迫、重根齐次基底，能验证公式与实现在不同基解结构下的稳定性。

## R14

关键参数：
- 网格：
  - Case1/2：`[0,2.0]` 上 801 点；
  - Case3：`[0,2.2]` 上 901 点。
- 稳定性阈值：`wronskian_floor=1e-10`。
- 参考求解器容差：`rtol=1e-10, atol=1e-12`。
- 通过阈值（`main`）：
  - `rel_l2_scipy < 5e-5`；
  - `rel_l2_exact < 8e-5`；
  - `max_abs_residual < 6e-2`。

## R15

方法对比：
- 对比待定系数法：
  - 待定系数法仅适配有限类型强迫项；
  - 常数变易法更通用，只要已知齐次基解即可。  
- 对比 Green 函数法：
  - Green 函数更抽象、构造更重；
  - 常数变易法更直接，可逐步落到数值实现。  
- 对比纯数值法（RK）：
  - RK 通用但不显式利用基解结构；
  - 常数变易法提供可解释的“齐次+特解”分解。

## R16

典型应用：
- 线性振动/电路模型中的非齐次激励响应分析。  
- ODE 课程里“从齐次基解构造非齐次解”的教学演示。  
- 数值求解器回归测试：使用解析/结构化解做对照。  
- 需要解释“外源项如何累积进入解”的工程建模场景。

## R17

扩展方向：
- 支持高阶线性方程与矩阵形式基解。  
- 自动基解生成（与特征方程法或数值基解联动）。  
- 引入自适应网格以降低残差峰值。  
- 批量参数扫描与 CSV/图形导出。  
- 增加近奇异 Wronskian 的正则化策略。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 调 `build_cases` 生成 3 个二阶非齐次 ODE 场景，包含 `p,q,g` 与齐次基解 `y1,y2`。  
2. 每个场景进入 `solve_by_variation_of_parameters`，先用 `_check_grid` 和 `_eval_on_grid` 完成输入合法性检查。  
3. 在网格上计算 `W=y1*y2'-y1'*y2`，并检查 `|W|>=wronskian_floor`。  
4. 按常数变易公式计算 `u1'=-y2*g/W`、`u2'=y1*g/W`，再用 `scipy.integrate.cumulative_trapezoid` 积分成 `u1,u2`。  
5. 组装特解 `y_p=u1*y1+u2*y2`，在初值点用 2x2 线性系统解出齐次系数 `c1,c2`，得到总解 `y=y_h+y_p`。  
6. 通过 `_first_second_derivative` 求离散 `y',y''`，构造残差 `y''+p y'+q y-g` 评估方程满足程度。  
7. 调 `_solve_reference_with_scipy`（`solve_ivp`）求参考轨迹，并与 `y`、`y_exact` 计算 L2、MSE（`sklearn`）与最大绝对误差（`torch`）。  
8. `main` 打印抽样表和汇总表，按阈值生成 `PASS`，若失败抛异常终止。

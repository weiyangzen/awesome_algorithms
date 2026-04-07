# 改进Euler方法/Heun方法

- UID: `MATH-0147`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `147`
- 目标目录: `Algorithms/数学-数值分析-0147-改进Euler方法／Heun方法`

## R01

本条目实现常微分方程初值问题的一步法：改进 Euler 方法（Heun 方法）的最小可运行版本，并与显式 Euler 做对照。

目标：
- 给出 `y' = f(t, y), y(t0)=y0` 的数值近似流程；
- 展示 Heun 通过“预测+校正”提升精度的机制；
- 通过已知解析解样例验证误差与收敛阶。

## R02

问题定义（有限精度计算）：
- 输入：
  - 右端函数 `f(t, y)`；
  - 区间 `[t0, t_end]`（`t_end > t0`）；
  - 初值 `y0`；
  - 步数 `N >= 1`（步长 `h = (t_end - t0)/N`）。
- 输出：
  - 网格点 `t_n` 上的数值解 `y_n`；
  - 若有解析解 `y(t)`，输出误差指标（如最大绝对误差）。

## R03

数学基础：

1. 初值问题
- `y' = f(t, y), y(t0)=y0`。

2. 显式 Euler（对照组）
- `y_{n+1} = y_n + h f(t_n, y_n)`。

3. Heun / 改进 Euler（显式梯形法）
- 预测器：`y_{n+1}^{(p)} = y_n + h f(t_n, y_n)`；
- 校正器：`y_{n+1} = y_n + (h/2) [ f(t_n, y_n) + f(t_{n+1}, y_{n+1}^{(p)}) ]`。

4. 精度结论
- Heun 的局部截断误差为 `O(h^3)`，全局误差为 `O(h^2)`；
- 显式 Euler 全局误差为 `O(h)`。

## R04

算法总览（MVP）：
1. 校验输入区间和步数合法性。
2. 构造均匀时间网格 `t_0,...,t_N`。
3. 用 Euler 方法在网格上推进，得到 `y_n^Euler`。
4. 用 Heun 方法在同一网格上推进，得到 `y_n^Heun`。
5. 使用解析解（若提供）计算两种方法误差。
6. 在多组步数下重复，比较误差衰减并估计收敛阶。

## R05

核心数据结构：
- `numpy.ndarray`：
  - `t_values`：时间网格；
  - `y_values`：数值解序列；
  - `exact_values`：解析解序列（演示中使用）。
- `ODESolution`（`dataclass`）：
  - `method`：方法名；
  - `t_values`、`y_values`、`exact_values`；
  - `max_abs_error`：`max |y_n - y(t_n)|`。

## R06

正确性要点：
- 两种方法都由微分方程积分形式
  `y(t_{n+1}) = y(t_n) + \int_{t_n}^{t_{n+1}} f(t,y(t)) dt`
  的数值积分近似得到；
- Euler 用左端矩形近似积分；
- Heun 先用 Euler 预测终点斜率，再用端点斜率均值（梯形思想）校正；
- 对足够光滑且满足 Lipschitz 条件的 `f`，Heun 具有二阶全局精度。

## R07

复杂度分析：
- 时间复杂度：
  - Euler：`O(N)` 次函数评估；
  - Heun：`O(2N)` 次函数评估，仍是 `O(N)`。
- 空间复杂度：
  - 存完整轨线为 `O(N)`；
  - 若只保留当前步可降为 `O(1)`（本 MVP 为可解释性保留全轨线）。

## R08

边界与异常处理：
- `N < 1` 抛出 `ValueError`；
- `t_end <= t0` 抛出 `ValueError`；
- `t0/t_end/y0` 非有限数抛出 `ValueError`；
- 若右端函数返回 `nan/inf`，在推进中抛出 `ValueError`。

## R09

MVP 取舍：
- 仅依赖 `numpy` + 标准库，避免引入重框架；
- 不调用黑盒 ODE 求解器，手写一步更新公式，保证教学可追溯；
- 选取有解析解的标量问题，便于定量验证收敛行为。

## R10

`demo.py` 函数职责：
- `check_inputs`：参数合法性检查；
- `rhs`：示例方程右端 `f(t,y)`；
- `exact_solution`：对应解析解；
- `euler_method`：显式 Euler 推进；
- `heun_method`：改进 Euler（Heun）推进；
- `build_solution`：封装求解结果与误差；
- `estimate_order`：根据 `log(error)-log(h)` 拟合实验收敛阶；
- `main`：组织实验并输出结果。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0147-改进Euler方法／Heun方法
python3 demo.py
```

脚本无需任何交互输入。

## R12

输出解读：
- `N`：步数；
- `h`：步长；
- `Euler_max_err`：Euler 最大绝对误差；
- `Heun_max_err`：Heun 最大绝对误差；
- `Estimated order`：基于多组网格误差拟合的实验阶，理论上 Heun 接近 2、Euler 接近 1。

## R13

建议最小测试集：
- 主样例：`y' = y - t^2 + 1, y(0)=0.5, t in [0,2]`（解析解已知）；
- 参数测试：`N=1` 的极粗网格；
- 异常测试：`N=0`、`t_end<=t0`、`y0=nan`；
- 鲁棒性测试：更大步数（如 `N=320`）观察误差继续下降。

## R14

可调参数：
- `step_list`：步数序列（默认 `[10, 20, 40, 80, 160]`）；
- `t0, t_end, y0`：初值问题参数；
- 可替换 `rhs/exact_solution` 为其他带解析解问题。

## R15

方法对比：
- 相比 Euler：Heun 每步多一次函数评估，但通常可显著降低误差；
- 相比 RK4：Heun 实现更简单、开销更低，但精度阶数更低；
- 相比隐式梯形法：Heun 显式、无需非线性方程求解，但稳定性弱于隐式法。

## R16

应用场景：
- 教学中展示从一阶方法到二阶方法的改进路径；
- 中低精度实时仿真（对实现复杂度敏感）；
- 作为更高阶 Runge-Kutta 方法的入门基线。

## R17

后续扩展方向：
- 加入向量状态 `y in R^d` 与批量 ODE；
- 增加自适应步长（局部误差控制）；
- 与 `scipy.integrate.solve_ivp` 做系统交叉验证；
- 增加单元测试与收敛性自动检查。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 设置样例 ODE、区间、初值和步数列表。  
2. 对每个 `N`，`check_inputs` 校验 `N`、区间和初值有效。  
3. `euler_method` 构造网格后逐步执行 `y_{n+1}=y_n+h f(t_n,y_n)`。  
4. `heun_method` 在每步先算 `k1=f(t_n,y_n)`，得到预测值 `y_pred=y_n+h k1`。  
5. 同一步再算 `k2=f(t_{n+1},y_pred)`，用 `y_{n+1}=y_n+(h/2)(k1+k2)` 完成校正。  
6. `build_solution` 将数值轨线与解析解逐点比较，得到 `max_abs_error`。  
7. 汇总各 `N` 对应 `h` 与误差，`estimate_order` 在对数坐标拟合实验收敛阶。  
8. 打印误差表、阶数估计和样例轨线片段，形成可验证的最小闭环。  

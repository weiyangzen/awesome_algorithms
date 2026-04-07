# Runge-Kutta-Fehlberg方法

- UID: `MATH-0149`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `149`
- 目标目录: `Algorithms/数学-数值分析-0149-Runge-Kutta-Fehlberg方法`

## R01

Runge-Kutta-Fehlberg（常写作 RKF45）是一种带有内嵌误差估计的显式 Runge-Kutta 方法。它在一次步进中同时给出 4 阶和 5 阶近似，并利用二者差值作为局部截断误差估计，从而自动调整步长。相比固定步长 RK4，RKF45 在同等精度需求下通常能显著减少函数评估次数。

## R02

目标问题是常微分方程初值问题：

\[
\frac{dy}{dt} = f(t, y), \quad y(t_0)=y_0, \quad t\in[t_0,t_{end}]
\]

其中 `y` 可为标量或向量。我们希望在给定绝对误差容忍 `atol` 和相对误差容忍 `rtol` 下，自动选择合适步长 `h`，在稳定推进到 `t_end` 的同时控制局部误差。

## R03

RKF45 的核心思想是“同一步内双解估计”：

1. 用同一组阶段斜率 `k1..k6` 组合出 4 阶解 `y4` 和 5 阶解 `y5`。
2. 误差估计 `e = y5 - y4`。
3. 若归一化误差 `err_ratio <= 1`，接受步长并前进；否则拒绝该步并缩小步长重算。
4. 用 `h_new = h * safety * err_ratio^(-1/5)`（限幅）更新下一个步长。

## R04

Fehlberg(4,5) 常用系数（Butcher 形式）如下：

- 阶段节点：`c = [0, 1/4, 3/8, 12/13, 1, 1/2]`
- 斜率组合：
  - `k1 = h f(t, y)`
  - `k2 = h f(t + 1/4 h, y + 1/4 k1)`
  - `k3 = h f(t + 3/8 h, y + 3/32 k1 + 9/32 k2)`
  - `k4 = h f(t + 12/13 h, y + 1932/2197 k1 - 7200/2197 k2 + 7296/2197 k3)`
  - `k5 = h f(t + h, y + 439/216 k1 - 8 k2 + 3680/513 k3 - 845/4104 k4)`
  - `k6 = h f(t + 1/2 h, y - 8/27 k1 + 2 k2 - 3544/2565 k3 + 1859/4104 k4 - 11/40 k5)`
- 4 阶解：
  - `y4 = y + 25/216 k1 + 1408/2565 k3 + 2197/4104 k4 - 1/5 k5`
- 5 阶解：
  - `y5 = y + 16/135 k1 + 6656/12825 k3 + 28561/56430 k4 - 9/50 k5 + 2/55 k6`

## R05

误差控制采用“绝对+相对”混合标尺：

\[
scale = atol + rtol\cdot \max(\|y_n\|_\infty, \|y_{n+1}\|_\infty)
\]
\[
err\_ratio = \frac{\|y_5 - y_4\|_\infty}{scale}
\]

- `err_ratio <= 1`：接受步。
- `err_ratio > 1`：拒绝步，缩小步长重试。

步长更新使用安全因子并限幅：

\[
h_{new} = h \cdot \mathrm{clip}(safety\cdot err\_ratio^{-1/5},\ fac_{min},\ fac_{max})
\]

## R06

伪代码（简化）：

```text
initialize t=t0, y=y0, h=h0
while t < t_end:
    if t+h > t_end: h = t_end - t
    y4, y5, err = rkf45_step(f, t, y, h)
    scale = atol + rtol * max(norm_inf(y), norm_inf(y5))
    err_ratio = err / scale

    if err_ratio <= 1:
        accept: t <- t+h, y <- y5, record point
    else:
        reject: keep t,y unchanged

    factor = safety * err_ratio^(-1/5) (with zero-error guard)
    factor <- clip(factor, fac_min, fac_max)
    h <- clip(h*factor, h_min, h_max)
```

## R07

本目录 `demo.py` 的 MVP 设计：

- 自实现 `rkf45_step`，不依赖黑盒 ODE 求解器。
- 自实现 `integrate_rkf45`，包含：
  - 误差归一化
  - 接受/拒绝步统计
  - 最小/最大步长边界
- 测试方程选用有解析解的标量 ODE：
  - `y' = y - t^2 + 1, y(0)=0.5, t∈[0,2]`
  - 解析解：`y(t) = (t+1)^2 - 0.5*exp(t)`

## R08

运行方式：

```bash
python3 Algorithms/数学-数值分析-0149-Runge-Kutta-Fehlberg方法/demo.py
```

脚本会输出：步数统计、末端数值与解析解误差、若干采样点对比。

## R09

输出字段说明：

- `accepted_steps`：被接受并写入轨迹的步数。
- `rejected_steps`：因误差超标被丢弃的步数。
- `final_time`：积分末端时间（应为 `2.0`）。
- `final_value`：数值解在 `t_end` 的结果。
- `exact_value`：解析解在 `t_end` 的结果。
- `abs_error`：`|final_value - exact_value|`。
- `max_abs_error_on_grid`：在记录网格上的最大绝对误差。

## R10

复杂度分析（设最终接受步数为 `N`，拒绝步数为 `R`）：

- 时间复杂度：`O((N+R) * C_f)`，其中每次尝试步需要 6 次 `f` 评估，`C_f` 为单次 `f` 成本。
- 空间复杂度：
  - 若保存完整轨迹：`O(N * d)`，`d` 为状态维数。
  - 若仅需末值：可降为 `O(d)`。

## R11

数值性质简述：

- 5 阶主解、4 阶嵌入解用于误差估计。
- 对平滑非刚性问题效果好，且自适应步长能在“平稳区大步、剧烈区小步”。
- 显式方法对刚性问题通常不理想，可能出现极小步长和效率下降。

## R12

与固定步长 RK4 对比：

- RK4：每步 4 次函数评估，但无内置误差估计，步长需要人工调参。
- RKF45：每次尝试 6 次评估，带自动误差控制；总体上通常以更少的总尝试数达到目标精度。
- 当问题局部难度变化明显时，RKF45 的优势更明显。

## R13

参数建议：

- 一般工程任务：`rtol=1e-6, atol=1e-9` 可作为起点。
- 初始步长 `h0` 不宜过大，推荐占积分区间的 `1%~5%`。
- `safety` 常取 `0.8~0.95`。
- `fac_min/fac_max` 需要限制过激变化，避免步长振荡。

## R14

常见实现陷阱：

- 误差尺度只用绝对误差，忽略量纲和量级变化。
- 拒步后仍推进 `t`（应保持原位重算）。
- 忽略 `t+h > t_end` 的末步截断。
- 未限制 `h_min`，在困难问题上可能死循环。
- 向量问题中直接把标量误差逻辑照搬，未使用范数统一判据。

## R15

可扩展方向：

- 向量 ODE：当前实现已支持 `numpy.ndarray` 状态。
- 事件检测：可在每个接受步后检查符号变化并做根定位。
- 与隐式法切换：针对刚性区间可混合 BDF/Radau 类方法。
- 批量求解：可将 `f` 向量化，提升参数扫描效率。

## R16

MVP 预期结果（不同机器会有微小差异）：

- 能稳定到达 `t=2`。
- 末端绝对误差通常在 `1e-7 ~ 1e-10` 量级（取决于容忍参数）。
- 拒绝步数量应较少，但在初始步长偏激进时会出现少量拒步，这是正常现象。

## R17

最小验证清单：

- 运行 `python3 demo.py` 无异常退出。
- 输出中的 `final_time` 等于目标终点。
- `abs_error` 与 `max_abs_error_on_grid` 足够小并与容忍参数同量级。
- 手动放宽/收紧 `rtol` 后，步数与误差呈合理变化（精度高 -> 步数增加）。

## R18

`demo.py` 中算法流程（源码级，8 步）：

1. `main()` 定义测试方程、解析解、积分区间与误差容忍参数。
2. `main()` 调用 `integrate_rkf45(...)`，进入自适应积分主循环。
3. 每轮循环先裁剪步长，确保末步不越过 `t_end`。
4. 调用 `rkf45_step(f, t, y, h)` 计算 `k1..k6`，并组合得到 `y4`、`y5` 与 `err_inf`。
5. `integrate_rkf45` 用 `atol + rtol * max(norm_inf(y), norm_inf(y5))` 计算尺度，得到 `err_ratio`。
6. 若 `err_ratio <= 1`：接受该步，更新 `t,y` 并记录轨迹；否则只计数拒步，不更新时间状态。
7. 基于 `err_ratio` 计算缩放因子 `factor = safety * err_ratio^(-1/5)`，经 `fac_min/fac_max` 限幅后更新 `h`。
8. 循环结束后返回轨迹和统计信息，`main()` 再与解析解比对并打印误差摘要。

# Verlet积分

- UID: `MATH-0156`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `156`
- 目标目录: `Algorithms/数学-数值分析-0156-Verlet积分`

## R01

本条目实现二阶常微分方程 `x'' = a(x, t)` 的最小可运行 Verlet 积分（采用 velocity-Verlet 形式），重点展示三件事：
- 离散更新公式如何从动力学方程落地到代码；
- 该方法在简谐振子上的二阶精度表现；
- 与显式 Euler 对比时，长期能量行为的差异。

MVP 选用单位质量简谐振子：
- 方程：`x'' = -ω^2 x`
- 初值：`x(0)=1, v(0)=0`
- 默认参数：`ω=1`

## R02

问题定义（固定步长）：
- 输入：加速度函数 `a(x, t)`、初始状态 `(t0, x0, v0)`、步长 `h`、步数 `N`
- 输出：离散轨迹 `t_n, x_n, v_n (n=0..N)`
- 目标：在给定时间网格上近似真实轨道，并评估误差与能量漂移

本 MVP 不做自适应步长，仅保留固定步长版本，确保算法结构可直接审查。

## R03

velocity-Verlet 核心公式：

设 `a_n = a(x_n, t_n)`，则

`x_{n+1} = x_n + h v_n + 0.5 h^2 a_n`

`a_{n+1} = a(x_{n+1}, t_{n+1})`

`v_{n+1} = v_n + 0.5 h (a_n + a_{n+1})`

性质要点：
- 全局误差阶：`O(h^2)`（在光滑条件下）
- 属于辛积分器家族（对哈密顿系统长期结构保持更好）
- 时间可逆（对称形式）

## R04

算法流程：
1. 校验标量输入：`h > 0`、`steps > 0`、所有初值有限。
2. 分配长度 `steps+1` 的 `t/x/v` 数组并写入初值。
3. 计算当前加速度 `a_n = a(x_n, t_n)`。
4. 用位置更新式得到 `x_{n+1}`。
5. 在新位置上计算 `a_{n+1}`。
6. 用速度更新式得到 `v_{n+1}`。
7. 推进时间并写入数组。
8. 重复直到完成 `N` 步，返回轨迹。

## R05

核心数据结构（`numpy.ndarray`）：
- `t: shape (N+1,)` 时间网格
- `x: shape (N+1,)` 位置轨迹
- `v: shape (N+1,)` 速度轨迹
- `results: list[tuple]` 收敛实验汇总（步长、步数、终点误差、最大误差）

选择一维数组是因为当前演示问题是一维动力系统，便于阅读和验证。

## R06

正确性与数值行为验证：
- `demo.py` 提供解析解 `x(t), v(t)`，可直接计算 `|x_num - x_exact|`。
- 对 `h = 0.2, 0.1, 0.05, 0.025` 进行收敛测试，经验阶应接近 2。
- 与显式 Euler 比较长期积分时的相对能量漂移：
  - Verlet 的能量误差通常有界振荡；
  - 显式 Euler 往往出现系统性漂移（常见为发散增大）。

## R07

复杂度分析：
- 单次积分（`N` 步）
  - 时间复杂度：`O(N)`
  - 空间复杂度：`O(N)`（保存全轨迹）
- 若仅保留当前状态，空间可降到 `O(1)`，但不利于误差分析与可视化。

## R08

边界与异常处理：
- `h <= 0` 或 `steps <= 0`：抛 `ValueError`
- `t0/x0/v0/h` 非有限值：抛 `ValueError`
- 加速度计算得到非有限值：抛 `RuntimeError`
- `t_end / h` 不是近整数（在固定网格实验中）：抛 `ValueError`
- `ω <= 0`（简谐模型参数非法）：抛 `ValueError`

## R09

MVP 取舍：
- 仅使用 `numpy` + 标准库，不依赖 `scipy.integrate` 等黑盒求解器。
- 只做一维简谐振子，换取“可解析对照 + 能量诊断”的完整闭环。
- 保留一个显式 Euler 对照实现，用最小代价说明 Verlet 的长期稳定性优势。

## R10

`demo.py` 函数职责：
- `validate_scalar_inputs`：基础参数校验
- `require_integer_steps`：由 `(t_end - t0)/h` 推断并校验步数
- `sho_acceleration_factory`：生成简谐振子加速度函数
- `exact_sho_state`：给出解析解 `(x(t), v(t))`
- `velocity_verlet`：核心 Verlet 积分器
- `explicit_euler_second_order`：二阶系统的显式 Euler 对照
- `sho_energy`：计算总能量
- `run_verlet_accuracy_case`：单组步长误差实验
- `estimate_orders`：相邻步长估计经验收敛阶
- `run_long_horizon_comparison`：长期能量与误差对照
- `print_trajectory_sample`：打印轨迹样例
- `main`：组织输出

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0156-Verlet积分
python3 demo.py
```

脚本无交互输入，直接输出收敛表、经验阶、长期稳定性对照与轨迹样例。

## R12

输出字段解释：
- `h`：时间步长
- `steps`：积分步数
- `final_abs_error`：终点位置绝对误差
- `max_abs_error`：全程位置最大绝对误差
- `estimated order p`：相邻步长下的经验阶估计
- `max_rel_energy_drift`：全程最大相对能量漂移
- `final_rel_energy_drift`：终点相对能量漂移

期望现象：Verlet 的 `p` 约为 2，且长期能量漂移明显优于显式 Euler。

## R13

建议最小测试集：
- 收敛测试：`h = {0.2, 0.1, 0.05, 0.025}`
- 参数异常：`h=0`、`h<0`、`steps=0`
- 非法数值：`x0=nan`、`v0=inf`
- 网格异常：`t_end=20, h=0.3`（非整数步）
- 模型异常：`omega <= 0`

## R14

关键可调参数：
- `h_values`：控制收敛实验精度与成本
- `t_end`：收敛实验时间范围
- `long_h` 与 `long_steps`：长期稳定性实验分辨率和总时长
- `omega`：系统频率

通常先用较大 `h` 快速检查，再逐步减小 `h` 验证二阶收敛。

## R15

方法对比：
- 显式 Euler：一阶，最简单，但长期能量行为较差
- Verlet（本条目）：二阶、辛结构、长期轨道质量好
- RK4：局部高精度（四阶），但非辛，长时间哈密顿系统中不一定保结构

在分子动力学、天体动力学等长期积分场景，Verlet 常作为基础选项。

## R16

典型应用：
- 分子动力学（粒子位置/速度时间推进）
- 天体力学（轨道长期演化）
- 物理仿真引擎中的保结构时间积分
- 教学中用于展示“精度 vs 稳定性 vs 结构保持”差异

## R17

后续扩展方向：
- 推广到多维向量状态 `x in R^d`
- 增加阻尼/外力项，构建更一般的 `x'' = a(x, v, t)`
- 引入自适应步长与事件检测
- 与 Leapfrog、Stormer-Verlet、Symplectic Euler 进行系统对照

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 设定 `omega`、步长列表、长期实验参数，并进入收敛实验循环。  
2. 每个步长由 `run_verlet_accuracy_case` 通过 `require_integer_steps` 推出 `steps`。  
3. `velocity_verlet` 先校验输入并初始化 `t/x/v` 数组与初始加速度 `a0`。  
4. 每一步先计算 `x_{n+1} = x_n + h v_n + 0.5 h^2 a_n`，再计算新加速度 `a_{n+1}`。  
5. 使用 `v_{n+1} = v_n + 0.5 h (a_n + a_{n+1})` 更新速度并写入轨迹。  
6. 积分结束后，`exact_sho_state` 在同一网格上给出解析解，计算终点/全局误差。  
7. `estimate_orders` 用相邻步长误差比估计经验阶，验证二阶收敛。  
8. `run_long_horizon_comparison` 分别调用 Verlet 与显式 Euler，对比能量漂移与长期位置误差。  

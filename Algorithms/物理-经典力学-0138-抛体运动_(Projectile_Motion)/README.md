# 抛体运动 (Projectile Motion)

- UID: `PHYS-0138`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `138`
- 目标目录: `Algorithms/物理-经典力学-0138-抛体运动_(Projectile_Motion)`

## R01

抛体运动是经典力学中最基础的二维运动模型之一：

- 水平方向做匀速运动；
- 竖直方向做匀加速度运动（加速度为 `-g`）；
- 忽略空气阻力时，轨迹为抛物线。

本条目给出一个可运行的最小闭环：

1. 按解析公式生成轨迹；
2. 用数值方法求落地时刻；
3. 从带噪观测反演初速度与重力加速度；
4. 交叉验证 `numpy` 与 `torch` 结果一致。

## R02

问题定义（本 MVP）：

- 已知发射参数 `x0, y0, v0, theta, g`；
- 计算飞行时间、射程、最高点与离散轨迹；
- 在加入观测噪声后，仅根据 `(t, x, y)` 数据反推 `x0, y0, vx0, vy0, g`；
- 输出结构化诊断，并用断言保证结果在可解释误差范围内。

## R03

无阻力抛体运动核心方程：

1. 速度分解：
   - `vx0 = v0 * cos(theta)`
   - `vy0 = v0 * sin(theta)`
2. 位置解析解：
   - `x(t) = x0 + vx0 * t`
   - `y(t) = y0 + vy0 * t - 0.5 * g * t^2`
3. 速度解析解：
   - `vx(t) = vx0`
   - `vy(t) = vy0 - g * t`
4. 最高点时刻（若 `vy0 > 0`）：
   - `t_peak = vy0 / g`
5. 落地时刻 `t_flight`：满足 `y(t_flight)=0, t_flight>0`。

## R04

算法设计由三块组成：

- **轨迹生成**：直接使用解析公式在离散时间网格上计算 `x,y,vx,vy`；
- **落地时间求解**：把 `y(t)=0` 作为一维根问题，使用 `scipy.optimize.brentq` 在正区间求根；
- **参数反演**：
  - `x(t)` 用线性回归拟合 `x = b0 + b1*t`；
  - `y(t)` 用二次基函数回归拟合 `y = c0 + c1*t + c2*t^2`；
  - 根据 `c2=-g/2` 得到 `g_est=-2*c2`。

## R05

设轨迹采样点数为 `N`、观测点数为 `M`：

- 解析轨迹生成：时间 `O(N)`，空间 `O(N)`；
- 一维根求解（Brent）：时间近似 `O(K)`（`K` 为迭代次数，通常很小）；
- 线性回归拟合：时间 `O(M)`（特征维数固定），空间 `O(M)`。

总体是线性复杂度，适合教学和批量验证。

## R06

`demo.py` 包含三个非交互演示：

- **Demo A: Forward simulation**
  - 计算落地时刻、射程、最高点；
  - 生成轨迹 DataFrame 并输出关键统计。
- **Demo B: Inverse estimation from noisy observations**
  - 对采样点注入高斯噪声；
  - 用 `scikit-learn` 回归反推 `vx0, vy0, g`、发射角与发射速度。
- **Demo C: Torch consistency check**
  - 用 `torch` 复算 `x(t), y(t)`；
  - 与 `numpy` 结果做最大绝对误差对比。

## R07

优点：

- 物理公式与代码一一对应，透明度高；
- 同时覆盖正向仿真与反向参数估计；
- 引入多库交叉校验，便于发现实现错误。

局限：

- 仅适用于真空近似（无阻力、无风）；
- 仅做二维平面运动；
- 反演模型假设噪声较小且时间戳准确。

## R08

前置知识：

- 牛顿第二定律与匀加速运动；
- 三角函数与二次函数；
- 最小二乘回归基础。

运行依赖（本仓库已声明）：

- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`
- `torch`

## R09

适用场景：

- 经典力学课程中的抛体运动教学演示；
- 检验传感器轨迹反演流程是否合理；
- 作为更复杂弹道模型（含阻力）开发前的基线。

不适用场景：

- 中远程弹道工程（需考虑阻力、地球曲率、自旋）；
- 复杂环境风场与湍流条件；
- 高精度武器级制导仿真。

## R10

正确性直觉：

1. 解析模型保证 `x(t)` 线性、`y(t)` 二次；
2. 落地时刻通过 `y(t)=0` 正根确定，射程即 `x(t_flight)-x0`；
3. 回归反演本质是把物理参数映射到线性可估系数；
4. `torch` 与 `numpy` 独立实现一致，能降低单实现偏差风险。

## R11

数值稳定策略：

- 使用 `float64`；
- `brentq` 采用有符号区间夹逼，避免牛顿法初值敏感；
- 回归使用固定随机种子生成噪声，保证可复现；
- 通过能量漂移与参数误差双指标断言结果可靠。

## R12

默认参数（见 `demo.py`）：

- `x0=0.0`, `y0=1.2`
- `speed=22.0 m/s`
- `angle_deg=37.0`
- `gravity=9.81 m/s^2`
- 轨迹采样点 `num_samples=240`

调参建议：

- 提高 `num_samples` 可获得更平滑轨迹；
- 噪声较大时可增加观测点或采用稳健回归；
- 若断言失败，先降低噪声标准差再检查实现。

## R13

- 近似比保证：N/A（非组合优化问题）。
- 随机化成功率：核心物理计算为确定性；仅观测噪声包含固定种子随机项。

在默认配置下，本实现保证：

- 飞行时间与射程均为正；
- 能量漂移接近机器精度；
- 参数反演误差在预设阈值内。

## R14

常见失效模式：

1. 角度单位混淆（把度误当弧度）；
2. `g<=0` 导致模型物理意义错误；
3. 观测噪声过大造成反演偏差超阈值；
4. `brentq` 夹逼区间未覆盖根（会报错）。

## R15

可扩展方向：

- 引入线性/二次空气阻力，改用 ODE 数值积分；
- 将二维扩展到三维并加入风场；
- 用卡尔曼滤波替换静态回归做在线估计；
- 用真实传感器数据替换合成噪声数据。

## R16

相关条目与方法：

- 牛顿运动定律；
- 匀加速直线运动；
- 最小二乘法与线性回归；
- 一维方程数值求根（Brent 方法）。

## R17

`demo.py` 的 MVP 功能清单：

- `ProjectileParams`：统一管理发射参数并校验合法性；
- `initial_velocity_components`：计算 `vx0, vy0`；
- `flight_time_to_ground`：用 `scipy.optimize.brentq` 求正根；
- `simulate_trajectory_numpy`：生成 `pandas` 轨迹表；
- `estimate_parameters_from_observations`：用 `sklearn` 从带噪观测反演参数；
- `torch_trajectory`：`torch` 版本轨迹计算，用于一致性检查。

运行方式：

```bash
cd "Algorithms/物理-经典力学-0138-抛体运动_(Projectile_Motion)"
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（9 步）：

1. 在 `main` 中构建 `ProjectileParams` 并调用 `validate()` 检查 `speed>0`、`g>0`。
2. `initial_velocity_components` 把 `speed` 与 `angle_deg` 转成 `vx0, vy0`。
3. `flight_time_to_ground` 构造 `y(t)`，先用解析上界估计正根区间，再用 `brentq` 求 `y(t)=0` 的正根。
4. `simulate_trajectory_numpy` 在 `[0, t_flight]` 建立时间网格，向量化计算 `x,y,vx,vy`，并生成 `pandas.DataFrame`。
5. `compute_summary_metrics` 从轨迹中提取 `range`、`max_height`、`t_peak` 和机械能漂移。
6. `build_noisy_observations` 以固定随机种子对轨迹子采样点加噪，形成拟合输入。
7. `estimate_parameters_from_observations` 分别拟合 `x=b0+b1*t` 与 `y=c0+c1*t+c2*t^2`，再映射回 `x0,y0,vx0,vy0,g,v0,theta`。
8. `torch_trajectory` 用 `torch` 重算同一时间序列的 `x,y`，与 `numpy` 结果计算最大绝对误差。
9. `main` 打印正向仿真、反向估计和一致性指标，并执行断言，确保脚本一次运行可自动验证通过。

该实现没有把第三方库当黑箱：核心物理方程、参数映射与验证指标都在源码中显式展开，库仅用于数值求根、表格与线性回归等通用基础能力。

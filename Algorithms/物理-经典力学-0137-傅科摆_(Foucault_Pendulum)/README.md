# 傅科摆 (Foucault Pendulum)

- UID: `PHYS-0137`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `137`
- 目标目录: `Algorithms/物理-经典力学-0137-傅科摆_(Foucault_Pendulum)`

## R01

傅科摆用于展示地球自转：在地面参考系看，摆动平面会以恒定角速度缓慢旋转。

本条目给出一个可运行最小模型（MVP）：
- 用局部东-北平面（EN）小角近似建立动力学；
- 数值积分得到 `x(t), y(t), vx(t), vy(t)`；
- 从转折点轨迹估计“摆动平面进动角速度”；
- 与理论值 `Omega * sin(latitude)` 做误差对比。

## R02

建模前提（MVP 级）：
- 摆角小，近似 `sin(theta) ≈ theta`；
- 只保留水平面位移，不显式积分竖直方向；
- 地球角速度在局部坐标中取常量；
- 忽略空气阻力（默认 `damping_per_s=0`）。

定义：
- `w0 = sqrt(g/L)`（摆的固有角频率）；
- `sigma = Omega_earth * sin(phi)`（傅科进动理论角速度）；
- 状态量 `s=[x, y, vx, vy]`，其中 `x` 向东、`y` 向北。

## R03

本实现采用线性耦合方程：

`x_ddot + gamma*x_dot + w0^2*x - 2*sigma*y_dot = 0`

`y_ddot + gamma*y_dot + w0^2*y + 2*sigma*x_dot = 0`

转换成一阶系统：
- `dx/dt = vx`
- `dy/dt = vy`
- `dvx/dt = -w0^2*x + 2*sigma*vy - gamma*vx`
- `dvy/dt = -w0^2*y - 2*sigma*vx - gamma*vy`

这是 `demo.py` 的 `foucault_rhs` 直接实现形式。

## R04

推导直觉（简版）：
1. 小角摆在水平面两个正交方向都是“简谐振子”，给出 `w0^2*x`、`w0^2*y` 回复项。
2. 在地球自转参考系，科里奥利项为 `-2*Omega×v`。
3. 取局部 EN 坐标并保留主导耦合后，得到与速度成正比的交叉项 `±2*sigma*v`。
4. 两个方向因此不再独立，摆动主轴会缓慢旋转。
5. 一阶 ODE 交给通用积分器求解，便于稳定得到长时间轨迹。

## R05

默认参数（`demo.py`）：
- `latitude_deg = 45.0`
- `length_m = 20.0`
- `gravity_mps2 = 9.81`
- `damping_per_s = 0.0`
- 初值：`x0=0.18 m, y0=0, vx0=0, vy0=0`
- 仿真区间：`0 ~ 21600 s`（6 小时）
- 采样点：`43200`（约 `0.5 s` 间隔）
- 积分器：`solve_ivp(method="DOP853")`

选择 6 小时是为了让进动角累计足够明显，便于估计斜率。

## R06

MVP 代码结构：
- `FoucaultConfig`：集中管理物理参数与数值参数
- `natural_frequency`：计算 `w0`
- `coriolis_coupling_rate`：计算 `sigma`
- `foucault_rhs`：构造 ODE 右端
- `mechanical_energy_like`：计算线性模型能量型不变量
- `_pick_local_maxima`：提取半周期转折点
- `estimate_precession_rate`：基于转折点方向拟合进动速率
- `simulate`：积分并汇总指标
- `main`：打印 summary 与轨迹样本并执行断言

## R07

伪代码：

```text
input: latitude, L, g, gamma, initial state
w0 = sqrt(g/L)
sigma = Omega * sin(latitude)

integrate s' = f(s; w0, sigma, gamma)

from trajectory:
  energy(t) = 0.5*(vx^2 + vy^2) + 0.5*w0^2*(x^2 + y^2)
  pick turning points of r(t)=sqrt(x^2+y^2)
  angle_k = atan2(y_k, x_k)
  orientation_k = 0.5 * unwrap(2*angle_k)
  fit orientation_k vs t_k by linear regression
  slope -> precession_rate_est

compare |precession_rate_est| with |sigma|
report errors and diagnostics
```

## R08

复杂度（`N = num_points`）：
- 时间复杂度：`O(N)`（低维 RHS，积分评估与后处理都线性）
- 空间复杂度：`O(N)`（保留整段轨迹和诊断序列）

该问题维度固定为 4，计算成本主要由仿真时长与采样密度决定。

## R09

数值稳定性与可信度策略：
- 用高阶自适应积分器 `DOP853` + 严格容差；
- 输出 `max_rel_energy_drift` 评估无阻尼场景数值守恒；
- 进动率估计不依赖单点，而是大量转折点线性拟合；
- 当可用转折点不足时返回 `nan` 并触发断言。

若误差偏大，优先调大 `num_points` 或延长 `t_end_s`。

## R10

与相关模型对比：
- 本模型：二维线性小角傅科摆，适合算法验证与教学。
- 完整三维摆绳约束模型：更真实，但状态和约束处理更复杂。
- 仅给理论公式 `Omega*sin(phi)`：没有轨迹级可审计证据。

本条目取中间路线：保留明确动力学积分与可复核估计流程，同时保持代码小而可读。

## R11

参数建议：
- 想更快看到进动：增大 `|sin(latitude)|`（离赤道更远）。
- 想减少小角误差：减小初始位移 `x0_m, y0_m`。
- 想测试耗散：设置 `damping_per_s > 0`。
- 保持 `length_m > 0`、`gravity_mps2 > 0`、`num_points >= 200`。

## R12

实现细节说明：
- `estimate_precession_rate` 通过 `r(t)=sqrt(x^2+y^2)` 的局部峰值定位每个半周期端点；
- 端点角度取 `atan2(y, x)`，再用 `0.5*unwrap(2*angle)` 处理“同一直线相差 `pi`”的不连续；
- 最后 `np.polyfit(t, orientation, 1)` 得到进动角速度估计值；
- `mechanical_energy_like` 体现“科里奥利项不做功”这一性质，在无阻尼时应近似守恒。

## R13

运行方式：

```bash
cd "Algorithms/物理-经典力学-0137-傅科摆_(Foucault_Pendulum)"
uv run python demo.py
```

或在仓库根目录运行：

```bash
uv run python Algorithms/物理-经典力学-0137-傅科摆_(Foucault_Pendulum)/demo.py
```

脚本无交互输入，直接打印结果。

## R14

输出解读：
- `natural_period_s`：摆的快振荡周期（秒级）
- `sigma_theory_rad_s`：理论进动角速度 `Omega*sin(phi)`
- `precession_rate_est_rad_s`：由轨迹拟合得到的估计角速度
- `precession_period_*_h`：对应进动一周所需小时数
- `precession_rate_abs_rel_error`：理论与估计的相对误差（绝对值）
- `max_rel_energy_drift`：能量型守恒指标
- `turning_points_used`：用于拟合的转折点数量

重点看两项：
- 误差是否较小；
- 能量漂移是否保持在低量级。

## R15

常见问题排查：
- `ODE integration failed`：检查参数是否有限、区间是否异常。
- `Failed to estimate precession rate`：仿真时长不足或采样太粗，导致峰值点太少。
- 误差偏大：
  - 提高 `num_points`；
  - 延长 `t_end_s`；
  - 降低初始振幅以满足小角近似。
- 能量漂移偏大：适当收紧 `rtol/atol`。

## R16

可扩展方向：
- 改为非线性三维摆绳约束模型，比较线性模型误差；
- 加入空气阻尼/支点摩擦，拟合真实实验数据；
- 扫描纬度，验证 `precession_rate ~ sin(phi)`；
- 引入可视化（轨迹与主轴角随时间曲线）；
- 在同一脚本中加入“无科里奥利项”对照组。

## R17

适用边界与限制：
- 仅适用于小角、局部平面近似；
- 忽略地球曲率变化、振幅过大时的非线性项；
- 当前进动估计基于转折点拟合，适合教学与算法验证，不是高精度实验反演流程；
- 输出主要用于说明机制与验证实现，不替代完整工程仿真。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main()` 创建 `FoucaultConfig`，固定纬度、摆长、初值、积分区间与容差。  
2. `simulate(cfg)` 做参数合法性检查，并创建 `t_eval` 时间网格和初始状态向量。  
3. `simulate` 调用 `solve_ivp`，将 `foucault_rhs` 作为 RHS 回调执行数值积分。  
4. `foucault_rhs` 每次回调先算 `w0` 与 `sigma`，再按耦合方程返回 `[vx, vy, ax, ay]`。  
5. 积分结束后，`mechanical_energy_like` 逐时刻计算能量型量，并给出 `max_rel_energy_drift`。  
6. `estimate_precession_rate` 先通过 `_pick_local_maxima` 抽取 `r(t)` 的转折点，再计算 `atan2(y, x)` 端点角。  
7. 同函数对 `2*angle` 做 `unwrap` 后除以 2，得到连续主轴方向，并用 `np.polyfit` 拟合斜率作为进动率。  
8. `main` 输出理论值、估计值、相对误差、轨迹头尾，并用断言校验误差与守恒指标，形成完整“建模-积分-估计-验证”闭环。  

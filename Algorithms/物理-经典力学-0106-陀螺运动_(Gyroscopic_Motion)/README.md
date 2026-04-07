# 陀螺运动 (Gyroscopic Motion)

- UID: `PHYS-0106`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `106`
- 目标目录: `Algorithms/物理-经典力学-0106-陀螺运动_(Gyroscopic_Motion)`

## R01

问题定义：研究重力场中“定点支撑对称陀螺”（heavy symmetric top）的经典力学运动。

本条目把陀螺状态用欧拉角 `theta`（倾角）、`phi`（进动角）、`psi`（自旋角）描述，目标是数值模拟并解释：
- 快速自旋下的稳定进动（precession）；
- 倾角的小幅振荡（nutation）；
- 能量与守恒广义动量的稳定性。

## R02

物理背景：
- 陀螺在重力力矩 `tau = r x mg` 作用下并不立即倒下，而会绕竖直方向慢速进动；
- 当初值不是“完美稳态”时，倾角会出现小幅周期摆动（章动/摆振）；
- 对称刚体（`I1=I2=I_perp, I3=I_spin`）是经典可解析建模入口。

工程上，陀螺模型常用于：姿态稳定、航天器角动量轮、惯性导航概念演示。

## R03

本 MVP 的计算任务：
1. 给定陀螺参数和初始欧拉角速度；
2. 用拉格朗日方程导出的 ODE 做时间积分；
3. 输出核心指标：
   - `theta` 振幅；
   - 平均进动角速度；
   - 快速自旋近似公式对比；
   - 能量与两个守恒动量漂移量。

## R04

建模假设（最小可运行而非全物理细节）：
- 刚体为对称陀螺：`I_perp` 与 `I_spin` 常数；
- 支点固定，无摩擦耗散；
- 重力恒定、质心到支点距离为 `l`；
- 不考虑空气阻尼、轴承损耗、弹性形变。

因此这是“保守系统 + 非线性耦合角运动”的标准教学模型。

## R05

核心方程（`demo.py` 直接实现）：

拉格朗日量：
`L = 0.5*I_perp*(theta_dot^2 + phi_dot^2*sin^2(theta)) + 0.5*I_spin*(psi_dot + phi_dot*cos(theta))^2 - m*g*l*cos(theta)`

循环坐标对应守恒量：
- `p_psi = I_spin*(psi_dot + phi_dot*cos(theta)) = B`
- `p_phi = I_perp*sin^2(theta)*phi_dot + p_psi*cos(theta) = A`

由此可解：
- `phi_dot = (A - B*cos(theta)) / (I_perp*sin^2(theta))`
- `psi_dot = B/I_spin - phi_dot*cos(theta)`

并得到 `theta` 动力学：
`I_perp*theta_ddot = I_perp*phi_dot^2*sin(theta)*cos(theta) - B*phi_dot*sin(theta) + m*g*l*sin(theta)`

## R06

数值算法流程：
1. 由初值计算常数 `A, B`；
2. 将状态设为 `y=[theta, theta_dot, phi, psi]`；
3. 在 `rhs` 中根据当前 `theta` 与 `A, B` 反求 `phi_dot, psi_dot`；
4. 用上式计算 `theta_ddot`；
5. 通过 `scipy.integrate.solve_ivp(DOP853)` 做高精度积分；
6. 后处理中计算能量、守恒量、进动均值和章动幅值。

## R07

复杂度（设积分采样点数为 `N`）：
- 时间复杂度：`O(N)`，每步仅常数次三角函数与代数运算；
- 空间复杂度：`O(N)`，保存时序状态用于诊断输出。

本问题瓶颈不是矩阵分解，而是 ODE 采样长度与精度设置。

## R08

数值稳定性策略：
- 近极点奇异保护：`sin^2(theta)` 分母使用 `max(sin^2, EPS)`；
- 高精度积分器：`DOP853 + rtol=1e-9, atol=1e-11`；
- 诊断守恒量：能量和 `p_phi/p_psi` 相对漂移用于校验数值稳定。

如果初值接近 `theta=0` 或 `pi`，欧拉角参数化会更敏感，建议改用四元数版本。

## R09

适用场景：
- 经典力学课程中的陀螺进动/章动可视化前置计算；
- 惯性姿态控制概念验证（无耗散理想化）；
- 与解析近似公式（快旋进动）做 sanity check。

不适合直接用于：
- 强阻尼、强外扰、碰撞接触等真实工程细节；
- `theta` 穿越极点的鲁棒长时仿真（应切换姿态参数化）。

## R10

正确性检查点（脚本中已落地）：
1. ODE 求解成功，否则抛 `RuntimeError`。
2. `energy_rel_drift` 需足够小（MVP 阈值 `5e-4`）。
3. `mean_precession > 0`，验证存在正向进动。
4. `nutation_amplitude_deg` 不应退化到近 0（避免完全平凡轨道）。
5. 输出 `p_phi/p_psi` 漂移，检查守恒结构未被数值破坏。

## R11

默认参数（`GyroscopeParams`）：
- 转动惯量：`I_perp=0.02`, `I_spin=0.04`
- 质量与几何：`m=0.5 kg`, `l=0.1 m`
- 初始角速率：`theta_dot0=0.25`, `phi_dot0=0.068`, `psi_dot0=180`
- 仿真：`t_end=8 s`, `num_steps=2000`

参数意义：
- `psi_dot0` 越大，通常越接近“快速稳定进动”；
- `theta_dot0` 决定章动扰动强弱；
- `I_spin` 越大，同等重力下进动角速度近似更小。

## R12

一次实测输出（默认参数，`uv run python demo.py`）：
- `theta_range_deg = [19.960, 20.040]`
- `nutation_amplitude_deg = 0.080`
- `mean_precession_rad_s = 0.068120`
- `fast_spin_formula_rad_s = 0.068101`
- `energy_rel_drift = 2.12e-13`

说明：数值积分与快速自旋近似公式高度一致，且守恒量漂移极小。

## R13

理论保证（本 MVP 层级）：
- 该模型在无耗散条件下属于保守系统，理论上存在能量守恒与循环坐标守恒量；
- 数值方法不提供严格“全局误差上界证明”，但通过高精度容差与守恒诊断进行工程验证；
- 因此本条目是“物理一致性 + 数值一致性”校核，而非形式化证明器。

## R14

常见失败模式与修复：
- 失败：`theta` 过近 0 或 `pi` 导致欧拉角奇异。
  - 处理：远离极点初值，或改四元数表示。
- 失败：步长/容差过松，能量漂移放大。
  - 处理：收紧 `rtol/atol` 或缩短步长。
- 失败：初值设成完美稳态，章动几乎为零。
  - 处理：加入小 `theta_dot0` 扰动。
- 失败：参数尺度不合理导致角速度过大。
  - 处理：先做无量纲估算，再调参。

## R15

工程实践建议：
- 把“守恒量漂移”当成第一质量指标，而不是只看轨迹曲线好看；
- 记录 `theta/phi/psi` 与 `phi_dot`，避免仅凭角度终值误判；
- 参数扫描时优先改变 `psi_dot0` 与 `theta_dot0`，可快速观察稳态进动与章动分界；
- 需要更长时间仿真时，建议同时输出采样稀释版日志，控制内存。

## R16

相关方法脉络：
- 经典解析：拉格朗日顶、欧拉方程、有效势分析；
- 数值积分：`RK45`/`DOP853`/辛积分（长期守恒更好）；
- 更稳姿态表示：方向余弦矩阵、四元数；
- 拓展模型：加入阻尼、控制力矩、随机扰动。

## R17

本目录 `demo.py` 的交付内容：
- 使用 `numpy + scipy + pandas` 的最小可运行脚本；
- 明确实现了对称重陀螺的欧拉角动力学；
- 无需交互输入，直接打印参数与诊断表。

运行方式：

```bash
cd Algorithms/物理-经典力学-0106-陀螺运动_(Gyroscopic_Motion)
uv run python demo.py
```

成功运行后会输出章动范围、平均进动率、快速自旋公式对比、守恒漂移量。

## R18

`demo.py` 源码级算法流拆解（8 步，非黑盒）：
1. `GyroscopeParams` 定义物理参数和初始条件，形成可复现实验配置。
2. `conserved_momenta` 从初始欧拉角速度计算守恒常数 `A=p_phi, B=p_psi`。
3. `euler_rates` 在每个时刻用 `A,B,theta` 显式反解 `phi_dot` 与 `psi_dot`。
4. `rhs` 按拉格朗日方程计算 `theta_ddot`，组装一阶系统导数 `[theta_dot, theta_ddot, phi_dot, psi_dot]`。
5. `simulate` 调用 `solve_ivp(DOP853)` 做时间积分，得到全轨迹 `theta/phi/psi`。
6. `total_energy` 与动量重构模块计算 `E(t), p_phi(t), p_psi(t)`，并统计相对漂移。
7. 同时计算物理可解释指标：`mean_precession`、`steady_precession`（快旋近似）、`nutation_amplitude_deg`。
8. `print_report` 用 `pandas.DataFrame` 输出结果，`main` 执行断言保证最小物理一致性。

该流程未调用第三方“陀螺黑盒求解器”；核心动力学方程和诊断都在源码中逐步可追踪。

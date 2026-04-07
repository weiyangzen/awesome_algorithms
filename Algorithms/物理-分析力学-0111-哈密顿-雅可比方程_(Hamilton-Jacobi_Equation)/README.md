# 哈密顿-雅可比方程 (Hamilton-Jacobi Equation)

- UID: `PHYS-0111`
- 学科: `物理`
- 分类: `分析力学`
- 源序号: `111`
- 目标目录: `Algorithms/物理-分析力学-0111-哈密顿-雅可比方程_(Hamilton-Jacobi_Equation)`

## R01

问题目标：把哈密顿-雅可比方程从理论表达落到可运行、可核验的最小实现。

本条目采用 1D 简谐振子作为 MVP 场景，验证三件事：
- 时间无关 HJ 方程 `H(q, dW/dq)=E` 的数值残差接近 0；
- 数值积分得到的作用函数 `W(q)` 与闭式表达一致；
- 由 HJ 完全积分恢复出的轨迹，与高精度常微分方程积分解一致。

## R02

本目录任务定义：
- 输入（脚本内固定参数）：质量 `m`、角频率 `omega`、总能量 `E`、空间采样密度、时间采样密度。
- 系统模型：
  - `H(q,p)=p^2/(2m)+0.5*m*omega^2*q^2`；
  - 势能 `V(q)=0.5*m*omega^2*q^2`。
- 输出：
  - `W(q)` 的数值解与闭式解误差；
  - HJ PDE 残差；
  - HJ 轨迹与 `solve_ivp` 参考轨迹的 `q/p` 最大误差；
  - 能量漂移、周期等诊断指标表。

脚本无需交互输入，`uv run python demo.py` 可直接运行。

## R03

核心数学关系（时间无关 HJ）：

1. 方程：
   - `H(q, dW/dq) = E`。
2. 对简谐振子代入：
   - `(1/(2m))*(dW/dq)^2 + 0.5*m*omega^2*q^2 = E`。
3. 得到动量分支：
   - `p(q)=dW/dq=+sqrt(2m(E-V(q)))`（本 MVP 取正分支）。
4. 作用函数定义：
   - `W(q)=int_0^q p(q') dq'`。
5. 闭式解（设 `A=sqrt(2E/(m*omega^2))`）：
   - `W(q)=0.5*m*omega*(q*sqrt(A^2-q^2)+A^2*arcsin(q/A))`，且 `W(0)=0`。
6. 轨迹恢复：
   - 由完整积分 `S=W-Et` 可恢复 `q(t)=A*sin(omega*t)`、`p(t)=m*A*omega*cos(omega*t)`（取 `q(0)=0,p(0)>0` 分支）。

## R04

算法流程（高层）：
1. 校验参数合法性（正值、采样规模、允许区间）。
2. 计算转折点振幅 `A`，构造 `q in [0, q_max]` 网格（`q_max < A`）。
3. 计算 `p(q)`，用累积梯形积分构造数值 `W_num`。
4. 计算闭式 `W_closed`，比较两者误差。
5. 对 `W_num` 做数值微分得到 `dW/dq`，计算 HJ 残差。
6. 用 HJ 理论直接生成轨迹 `(q_hj(t), p_hj(t))`。
7. 用 `scipy.solve_ivp(DOP853)` 求同初值参考轨迹 `(q_ref, p_ref)`。
8. 输出误差指标和质量门槛断言。

## R05

核心数据结构：
- `HJParams`（`dataclass`）：统一管理物理与离散参数。
- `q_grid: np.ndarray`：作用函数求积的坐标网格。
- `w_num / w_closed: np.ndarray`：数值与闭式作用函数。
- `report: pandas.DataFrame`：最终诊断指标表（统一打印和回归比较）。

## R06

正确性检查点：
- 数学一致性：`dW/dq` 与 `p(q)` 应近似一致。
- PDE 一致性：`(dW/dq)^2/(2m)+V-E` 的最大绝对值应很小。
- 动力学一致性：HJ 恢复轨迹应与高精度 ODE 轨迹重合。
- 守恒一致性：无耗散系统中能量相对漂移应接近 0。

脚本内置断言：
- `max_abs_HJ_residual <= 1e-2`
- `max_abs_q_error_vs_scipy <= 1e-8`
- `max_abs_p_error_vs_scipy <= 1e-8`

## R07

复杂度分析：
- 作用函数计算（`n_q` 网格）：
  - 时间 `O(n_q)`，空间 `O(n_q)`。
- 轨迹对照（`n_t` 采样）：
  - HJ 解析轨迹生成 `O(n_t)`；
  - 参考 ODE 求解整体近似随采样点线性增长，常数较大。
- 总体为线性规模，适合作为教学和基准 MVP。

## R08

边界与异常处理：
- `mass/omega/energy <= 0`：抛 `ValueError`。
- `q_fraction_max` 不在 `(0,1)`：抛 `ValueError`。
- 采样点过少（`n_q` 或 `n_t < 32`）：抛 `ValueError`。
- `q` 超出经典允许区（`E < V(q)` 且超过容差）：抛 `ValueError`。
- `solve_ivp` 失败：抛 `RuntimeError`。

## R09

MVP 取舍说明：
- 只做 1 自由度、可分离、时间无关情形。
- 只演示正动量分支，不覆盖多值拼接与转折点跨越细节。
- 不引入高阶辛算法或复杂边值问题求解器。
- 优先保留“方程-代码-诊断”闭环，避免大而全框架。

## R10

`demo.py` 函数职责：
- `check_params`：参数合法性校验。
- `turning_point_amplitude`：转折点振幅 `A`。
- `potential`：势能函数 `V(q)`。
- `momentum_positive_branch`：按能量壳计算 `p(q)`。
- `reduced_action_numeric`：数值积分得到 `W_num`。
- `reduced_action_closed_form`：闭式 `W_closed`。
- `hj_residual_from_w`：计算 HJ PDE 残差。
- `canonical_rhs`：正则方程右端。
- `hj_trajectory_from_constant_beta`：HJ 恢复轨迹。
- `scipy_reference_trajectory`：高精度参考轨迹。
- `build_report`：汇总全部指标。
- `main`：执行并打印，触发质量门槛。

## R11

运行方式：

```bash
cd Algorithms/物理-分析力学-0111-哈密顿-雅可比方程_(Hamilton-Jacobi_Equation)
uv run python demo.py
```

脚本不读取命令行参数，也不请求用户输入。

## R12

本地实测输出（2026-04-07）：

```text
=== Hamilton-Jacobi Equation MVP (1D Harmonic Oscillator) ===
params: {'mass': 1.0, 'omega': 2.0, 'energy': 2.0, 'q_fraction_max': 0.98, 'n_q': 1500, 'n_t': 1200, 't_end': 6.0}
                        metric     value
     turning_point_amplitude_A  1.000000
max_abs_W_numeric_minus_closed 3.292e-07
         max_abs_dW_dq_minus_p 2.260e-05
           max_abs_HJ_residual 9.560e-06
      max_abs_q_error_vs_scipy 1.809e-12
      max_abs_p_error_vs_scipy 3.679e-12
      energy_rel_drift_HJ_traj 2.220e-16
    energy_rel_drift_scipy_ref 3.901e-12
               analytic_period  3.141593
               simulated_t_end  6.000000
```

结论：HJ 方程残差与轨迹误差都在非常小量级，MVP 验证通过。

## R13

建议测试集：
- 基线：当前默认参数（`m=1, omega=2, E=2`）。
- 高频情形：增大 `omega`，观察同 `n_t` 下轨迹误差变化。
- 低能量情形：减小 `E`，检查靠近转折点时数值微分稳定性。
- 网格敏感性：降低 `n_q`，确认残差与 `dW/dq` 误差按预期变大。

异常测试建议：
- `q_fraction_max >= 1`（应报错）；
- `energy <= 0`（应报错）；
- 人为构造 `q` 超出允许区（应报错）。

## R14

可调参数：
- `mass, omega, energy`：物理尺度与周期。
- `q_fraction_max`：离转折点的安全距离。
- `n_q`：作用函数积分与微分精度控制。
- `n_t, t_end`：轨迹对照窗口与分辨率。

调参建议：
- 若 `max_abs_HJ_residual` 偏大，优先提高 `n_q`。
- 若轨迹误差偏大，优先提高 `n_t` 或缩短 `t_end` 做局部验证。
- 若靠近转折点不稳，降低 `q_fraction_max`（例如 `0.95`）。

## R15

方法对比：
- 对比直接 ODE 积分：
  - ODE 法直接推进状态；
  - HJ 法先求作用函数，再由常数积分恢复轨迹，更接近“积分不变量”视角。
- 对比拉格朗日方程数值积分：
  - 拉格朗日通常处理 `(q, q_dot)`；
  - HJ 在正则变量中更强调正则变换与可分离结构。
- 对比黑盒 PDE 求解：
  - 本实现把 `p(q)`、`W(q)`、残差检查全部显式展开，便于教学审计。

## R16

典型应用场景：
- 分析力学课程中讲解“正则方程与 HJ 统一框架”。
- 从守恒量出发构造轨迹的教学示例。
- 更复杂可分离系统（中心力场、作用角变量）实现前的最小模板。

## R17

扩展方向：
- 扩展到双势阱或非线性势能，比较多值分支拼接。
- 增加时间依赖哈密顿量，验证 `∂S/∂t + H = 0` 的全方程离散。
- 增加二维可分离系统，展示多自由度动作变量。
- 引入辛积分对照，比较“HJ 恢复轨迹 vs 辛积分轨迹”长期误差。

## R18

`demo.py` 源码级算法流（8 步，非黑盒）：
1. `main` 初始化 `HJParams` 并调用 `check_params` 做参数合法性门禁。  
2. `build_report` 调用 `turning_point_amplitude` 计算振幅 `A`，据此构造 `q` 网格并限制在允许区内部。  
3. `reduced_action_numeric` 调 `momentum_positive_branch` 得到 `p(q)`，再用 `cumulative_trapezoid` 实现 `W_num(q)=∫p dq`。  
4. `reduced_action_closed_form` 计算闭式 `W_closed`，并与 `W_num` 做逐点误差对比。  
5. `hj_residual_from_w` 用 `np.gradient` 求 `dW/dq`，显式代回 `((dW/dq)^2)/(2m)+V-E` 得到 PDE 残差。  
6. `hj_trajectory_from_constant_beta` 按 HJ 完全积分构造解析轨迹 `(q_hj, p_hj)`。  
7. `scipy_reference_trajectory` 用同初值调用 `solve_ivp(DOP853)` 得到参考轨迹 `(q_ref, p_ref)`，然后计算 `q/p` 最大偏差与能量漂移。  
8. `build_report` 聚合为 `DataFrame`，`main` 打印指标并执行三条阈值断言，保证脚本运行即可自动验收。  

第三方库仅承担通用数值算子（数组、积分器、表格输出），哈密顿-雅可比建模、残差验证和轨迹重建逻辑均在源码中显式实现。

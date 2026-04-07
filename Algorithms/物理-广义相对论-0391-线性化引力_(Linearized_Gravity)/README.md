# 线性化引力 (Linearized Gravity)

- UID: `PHYS-0372`
- 学科: `物理`
- 分类: `广义相对论`
- 源序号: `391`
- 目标目录: `Algorithms/物理-广义相对论-0391-线性化引力_(Linearized_Gravity)`

## R01

本条目实现线性化引力的最小可运行 MVP：在 TT（横向无迹）规范下，数值求解真空引力波的两种物理偏振 `h_+` 与 `h_×`。  
目标是把“弱场方程 -> 有限差分离散 -> 与解析解对照 -> 数值一致性诊断”完整打通，而不是只给公式。

## R02

物理背景与问题定义：

- 弱场展开：`g_{mu nu} = eta_{mu nu} + h_{mu nu}`，且 `|h_{mu nu}| << 1`。
- 迹反演变量：`bar(h)_{mu nu} = h_{mu nu} - 1/2 * eta_{mu nu} * h`。
- Lorenz 规范下真空线性化爱因斯坦方程：`□ bar(h)_{mu nu} = 0`。
- 对沿 `z` 方向传播的 TT 引力波，只剩两个自由度 `h_+`、`h_×`，都满足 1D 波动方程：
  - `d2 h / dt2 - c^2 d2 h / dz2 = 0`。

MVP 输出：
- 数值解与解析解误差（RMSE、最大绝对误差）；
- 离散 PDE 残差；
- 离散能量漂移；
- TT 无迹条件检查。

## R03

本实现的最小建模假设：

- 时空背景为 Minkowski + 小扰动，不处理非线性反馈；
- 真空传播（`T_{mu nu}=0`），无源项；
- 只做 1D 传播与周期边界，聚焦算法正确性；
- 偏振采用 TT 规范（`h_xx=+h_+`, `h_yy=-h_+`, `h_xy=h_×`）。

因此该条目是“线性引力波传播核心链路”的教学与验证实现，不是全 GR 数值相对论求解器。

## R04

核心算法（`demo.py`）：

1. 在环形网格 `z in [0, L)` 上生成高斯脉冲初值；
2. 令初始速度满足右行波条件 `dh/dt = -c * dh/dz`；
3. 用二阶 Taylor 初始化 `t=-dt` 层；
4. 用 leapfrog 更新：
   - `h^{n+1} = 2h^n - h^{n-1} + (c*dt)^2 * Delta_z h^n`；
5. 对 `h_+` 与 `h_×` 独立演化；
6. 用解析右行高斯解做终态对照并输出诊断指标。

## R05

离散算子与边界：

- 空间二阶导数：中心差分 + 周期边界（`np.roll`）。
- 空间一阶导数：中心差分 + 周期边界。
- 稳定性条件：`CFL = c*dt/dz <= 1`（本实现默认 `0.80`）。

这样可避免额外边界反射处理，使验证重点集中于线性化引力波方程本身。

## R06

正确性验证设计：

- 解析对照：右行高斯脉冲 `h(z,t)=f(z-ct)`；
- 误差指标：
  - `relative_rmse`
  - `relative_max_abs_error`
- 方程满足性：
  - `pde_residual_relative = RMS(d2t h - c^2 d2z h) / RMS(c^2 d2z h)`
- 守恒性近似检查：
  - 离散能量 `E ~ integral((dh/dt)^2 + c^2(dh/dz)^2)/2 dz` 的相对漂移；
- 规范检查：
  - `max|h_xx + h_yy|`（TT 无迹条件）。

## R07

复杂度分析（单个偏振）：

- 设网格点数 `Nz`、时间步数 `Nt`；
- 时间复杂度：`O(Nz * Nt)`；
- 空间复杂度：`O(Nz * Nt)`（本实现保留全时序以便诊断；若只保留最近三层可降至 `O(Nz)`）。

两种偏振独立求解，总体量级仅是常数倍。

## R08

`demo.py` 的主要函数职责：

- `periodic_laplacian` / `periodic_gradient`：离散导数算子；
- `gaussian_on_ring`：周期域高斯脉冲初值；
- `initialize_previous_step`：leapfrog 二阶初始化；
- `simulate_single_polarization`：单偏振演化与误差/残差/能量诊断；
- `run_linearized_gravity_mvp`：组装 `h_+` 与 `h_×` 的完整实验；
- `run_checks`：阈值断言；
- `main`：无交互输出总报告与探测点采样表。

## R09

默认参数（可在 `run_linearized_gravity_mvp` 修改）：

- `nz=600`
- `domain_length_m=4.0e7`
- `cfl=0.80`
- `n_steps=220`
- `amp_plus=1.0e-4`
- `amp_cross=0.7e-4`
- `width_m=1.2e6`

这些参数保证：
- 脉冲在仿真时长内不会绕周期域回卷污染对照；
- 线性近似始终成立（`|h| << 1`）；
- 误差与稳定性检查可稳定通过。

## R10

运行方式（无交互）：

```bash
cd Algorithms/物理-广义相对论-0391-线性化引力_(Linearized_Gravity)
uv run python demo.py
```

脚本将打印：
- 网格与 CFL 信息；
- 两个偏振的误差/残差/能量漂移表；
- 一个固定探测点的时间采样对照表。

## R11

输出指标解释：

- `rel_rmse`：终态相对均方根误差；
- `rel_max_abs_error`：终态相对最大误差；
- `residual_rel`：离散 PDE 相对残差；
- `energy_drift`：离散能量相对漂移；
- `peak_abs_strain`：仿真期间的最大 `|h|`；
- `trace_hij`：`h_xx+h_yy`，TT 规范下应接近 0。

## R12

MVP 取舍说明：

- 选择 1D TT 波而非 3D 张量演化，是为了最小成本展示线性化引力核心算法；
- 不调用 GR 专用黑盒库，所有离散步骤在源码中显式展开；
- 使用 `numpy + pandas + scipy.constants` 的轻量栈，保证可读性与可运行性平衡。

## R13

适用场景：

- 广义相对论课程中的线性化引力/引力波数值演示；
- 更复杂数值相对论代码前的基准验证；
- 教学中解释 TT 规范、偏振自由度与波动方程联系。

不适用场景：

- 强场非线性并合（需要全 Einstein 方程）；
- 含源动力学（`T_{mu nu} != 0`）；
- Kerr 背景与曲率散射等高阶问题。

## R14

常见失败模式与处理：

1. `CFL > 1` 导致不稳定发散；
2. 仿真时间过长导致周期回卷，解析对照失真；
3. 初始速度 `dh/dt` 设错方向，右行波变成混合行波；
4. 忘记二阶初始化，导致首步误差放大；
5. 把 `h` 设得过大，超出线性近似物理边界。

代码中对 `CFL`、网格规模、仿真时长与峰值应变均有显式检查。

## R15

与替代方法对比：

- 解析法：最干净，但只能处理少量可解初值；
- 频域谱法：高精度但实现复杂度更高；
- 本实现（二阶显式 FDTD）：
  - 优点：实现短、可解释、可逐行追踪；
  - 缺点：时间步受 CFL 约束，长期误差控制不如高阶谱法。

## R16

可扩展方向：

1. 加入源项 `□ bar(h)_{mu nu} = -16 pi G T_{mu nu}/c^4`（如简化四极矩驱动）；
2. 从 1D 扩展到 2D/3D 张量网格；
3. 引入吸收边界（PML）替代周期边界；
4. 与干涉仪响应模型耦合，直接输出应变通道；
5. 用 `torch` 或 `numba` 做大规模并行加速。

## R17

交付检查（本目录）：

- `README.md`：R01-R18 全部填写完成；
- `demo.py`：可直接运行，无占位符；
- `meta.json`：UID、学科、分类、源序号、目录路径与任务元数据保持一致；
- 无需任何交互输入即可完成验证输出。

## R18

`demo.py` 源码级算法流（8 步）：

1. `main` 调用 `run_linearized_gravity_mvp`，设置网格、CFL、步数和两种偏振幅值。  
2. `run_linearized_gravity_mvp` 构造 `z/t` 网格，并检查 `CFL<=1` 与“传播距离不回卷”的时长约束。  
3. 对每个偏振调用 `simulate_single_polarization`：先用 `gaussian_on_ring` 生成 `h(z,0)`，再由 `dh/dt=-c*dh/dz` 设定右行波初速度。  
4. `initialize_previous_step` 用二阶 Taylor 公式构造 `t=-dt` 层，保证 leapfrog 启动仍是二阶精度。  
5. 在时间循环中用离散更新 `h^{n+1}=2h^n-h^{n-1}+(c*dt)^2 Delta_z h^n` 推进全历史。  
6. 终态调用 `analytic_right_moving` 计算解析解，得到 `relative_rmse` 与 `relative_max_abs_error`。  
7. 用最后三层场值构造离散 PDE 残差，并通过 `energy_series` 计算离散能量漂移，形成稳定性诊断。  
8. `run_checks` 对误差、残差、能量漂移、线性幅值和 TT 无迹条件做断言，最后 `main` 打印诊断表和探测点采样表。  

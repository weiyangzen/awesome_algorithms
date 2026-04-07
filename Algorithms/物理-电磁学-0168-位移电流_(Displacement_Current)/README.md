# 位移电流 (Displacement Current)

- UID: `PHYS-0167`
- 学科: `物理`
- 分类: `电磁学`
- 源序号: `168`
- 目标目录: `Algorithms/物理-电磁学-0168-位移电流_(Displacement_Current)`

## R01

位移电流是 Maxwell 为修正安培环路定律而引入的项，定义为：
\[
I_d = \epsilon_0 \frac{d\Phi_E}{dt}
\]
它不是自由电荷跨越介质间隙的传导运动，而是时变电场引起的等效电流项。该条目使用一个最小可运行 RC 电容充电模型，数值验证 `I_d` 与导线传导电流 `I_c` 的一致性。

## R02

核心方程组：
\[
\oint \mathbf{B}\cdot d\mathbf{l} = \mu_0\left(I_c + \epsilon_0\frac{d\Phi_E}{dt}\right)
\]
平行板电容器（真空）几何关系：
\[
C = \epsilon_0\frac{A}{d},\quad \Phi_E = E A = \frac{V_c}{d}A
\]
RC 阶跃响应：
\[
V_c(t)=V_s\left(1-e^{-t/RC}\right),\quad I_c(t)=\frac{V_s}{R}e^{-t/RC}
\]
因此有
\[
I_d=\epsilon_0\frac{d\Phi_E}{dt}=C\frac{dV_c}{dt}=I_c
\]
在理想模型中严格成立。

## R03

MVP 问题设定：
- 电路：`R + 真空平行板电容 C` 串联，输入阶跃电压 `V_s`。
- 几何：板半径 `a`、板间距 `d`，面积 `A=πa²`。
- 数值任务：
  1. 用解析 RC 公式生成 `V_c(t)` 与 `I_c(t)`；
  2. 用 `Phi_E(t)=E(t)A` 计算电通量并做数值求导，得到 `I_d=ε0 dPhi_E/dt`；
  3. 比较 `I_d` 与 `I_c` 误差；
  4. 在回路半径 `r=a` 处比较 `B(I_c)` 与 `B(I_d)`，验证安培-麦克斯韦一致性。

## R04

离散实现公式：
- 时间离散：`t = linspace(0, t_end, n_time)`
- 电场与电通量：
\[
E(t)=\frac{V_c(t)}{d},\quad \Phi_E(t)=E(t)A
\]
- 通量导数（数值）：
\[
\frac{d\Phi_E}{dt}\Big|_{t_i} \approx \texttt{gradient}(\Phi_E, t)_i
\]
- 位移电流：
\[
I_d(t_i)=\epsilon_0\frac{d\Phi_E}{dt}\Big|_{t_i}
\]
- 安培回路磁场（取 `r=a`）：
\[
B_c(t)=\frac{\mu_0 I_c(t)}{2\pi a},\quad B_d(t)=\frac{\mu_0 I_d(t)}{2\pi a}
\]

## R05

数值注意事项：
- `np.gradient` 在边界点精度较低，误差评估时剔除少量端点样本（代码中 `trim=4`）。
- `n_time` 越大，`dPhi_E/dt` 数值误差越小。
- 当前模型忽略边缘场（fringing）与介质损耗，验证目标是 Maxwell 修正项一致性，不是工程级场分布复现。

## R06

预期数值行为：
- `I_c(t)` 与 `I_d(t)` 在全时域应高度一致；
- `B(I_c)` 与 `B(I_d)` 在回路 `r=a` 处同样一致；
- 提高 `n_time` 时，`L2/Linf` 与相对误差继续下降。

## R07

复杂度分析：
- 时间复杂度：`O(n_time)`（全部操作是逐点向量计算与一次梯度）。
- 空间复杂度：`O(n_time)`（存储 `t, Vc, Ic, Phi, Id` 等数组）。
- 无迭代求解器、无稀疏矩阵组装，MVP 计算开销很低。

## R08

核心数据结构：
- `DisplacementConfig`：输入参数（电压、电阻、几何、时间采样）。
- `simulate_rc_step` 返回字典：`t, vc, ic, e_field, flux_e, area, capacitance, tau, t_end`。
- `current_metrics`：`Id` 对 `Ic` 的 `L2/Linf/RelL2`。
- `b_metrics`：`B(Id)` 对 `B(Ic)` 的 `L2/Linf/RelL2`。
- `pandas.DataFrame`：输出关键时刻快照表。

## R09

伪代码：

```text
read config
validate positive physical parameters
A <- pi * a^2
C <- epsilon0 * A / d
tau <- R * C

build uniform time grid t
compute Vc(t), Ic(t) from RC step response
compute E(t)=Vc/d and PhiE(t)=E*A
compute Id(t)=epsilon0 * gradient(PhiE, t)

compute error(Id, Ic)
compute Bc=mu0*Ic/(2*pi*a), Bd=mu0*Id/(2*pi*a)
compute error(Bd, Bc)
print metrics + snapshot table
```

## R10

`demo.py` 默认参数：
- `v_supply = 12 V`
- `resistance = 50 Ω`
- `plate_radius = 0.05 m`
- `plate_gap = 1e-3 m`
- `t_end_factor = 6`（即 `t_end = 6τ`）
- `n_time = 4000`

这些参数让指数衰减过程充分展开，且数值求导足够稳定。

## R11

输出解读：
- 第一段打印物理常量和几何、电容、时间常数。
- 第二段 `Current consistency`：检查 `Id` 与 `Ic`。
- 第三段 `Ampere-Maxwell consistency`：检查 `B(Id)` 与 `B(Ic)`。
- 最后一段 `Time snapshots` 给出若干时刻 `Vc/Ic/Id/B` 值，便于人工 spot-check。

## R12

代码模块划分：
- `validate_config`：参数约束检查。
- `compute_geometry`：`A, C, tau` 计算。
- `simulate_rc_step`：构造 `t, Vc, Ic, Phi_E`。
- `displacement_current_from_flux`：由通量导数得到 `Id`。
- `compute_error_metrics`：误差范数评估。
- `ampere_maxwell_fields`：`B(Ic)` / `B(Id)` 计算。
- `build_snapshot_table`：生成输出数据表。
- `main`：组织执行与打印。

## R13

运行方式：

```bash
uv run python demo.py
```

在目录 `Algorithms/物理-电磁学-0168-位移电流_(Displacement_Current)` 中执行即可，无需交互输入。

## R14

常见错误：
- 忘记乘 `epsilon0`，把 `dPhi_E/dt` 直接当作 `Id`。
- 几何量单位混用（`mm` 当 `m`）导致电容和时间常数数量级错误。
- 用很小 `n_time` 做数值导数，导致 `Id` 抖动并误判模型错误。
- 在 `r!=a` 时直接比较 `B(Ic)` 与 `B(Id)` 而不考虑包围通量面积比例。

## R15

最小验证步骤：
1. 直接运行默认参数，确认 `Relative L2 error` 很小。
2. 将 `n_time` 降到 `500`，观察误差变大；再提到 `8000`，观察误差下降。
3. 修改 `plate_gap` 和 `plate_radius`，检查 `C=ε0A/d` 与 `tau=RC` 的变化是否符合直觉。
4. 把 `v_supply` 改为其他值，确认相对误差基本保持同量级。

## R16

适用范围与局限：
- 适用：位移电流概念验证、教学演示、代码基线回归。
- 局限：
  - 仅真空平行板理想模型；
  - 未模拟边缘场、辐射效应、导体寄生参数；
  - 未做 2D/3D 全波电磁场离散（如 FDTD/FEM）。

## R17

可扩展方向：
- 将阶跃激励扩展为正弦稳态激励并做频域对比。
- 加入介质相对介电常数 `epsilon_r`，分析材料对 `Id` 的影响。
- 引入含损耗模型（并联电导）区分传导电流与位移电流的相位特性。
- 使用 2D 网格近似场分布，验证非均匀电场下的通量积分与 `Id`。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构造 `DisplacementConfig`，调用 `validate_config` 确保电压、电阻、几何、采样点数均为合法正值。
2. `simulate_rc_step` 先调用 `compute_geometry` 计算 `A=πa²`、`C=ε0A/d`、`tau=RC`。
3. 在 `simulate_rc_step` 中生成均匀时间网格 `t`，并按 RC 阶跃解析式计算 `Vc(t)` 与 `Ic(t)`。
4. 将 `Vc` 转成电场 `E=Vc/d`，再得到电通量 `Phi_E=E*A`。
5. `displacement_current_from_flux` 对 `Phi_E(t)` 执行 `np.gradient` 求时间导数，再乘 `epsilon0` 得到 `Id(t)`。
6. `compute_error_metrics` 比较 `Id` 和 `Ic`，输出 `L2/Linf/RelL2`；并在边界做 `trim` 避免导数端点误差主导。
7. `ampere_maxwell_fields` 计算 `B_c=mu0*Ic/(2πa)` 与 `B_d=mu0*Id/(2πa)`，再用 `compute_error_metrics` 比较二者一致性。
8. `build_snapshot_table` 选取 8 个时刻输出 `t, Vc, Ic, Id, B_c, B_d`，`main` 汇总打印全部指标，形成可复现的最小验证报告。

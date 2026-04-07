# 光学定理 (Optical Theorem)

- UID: `PHYS-0227`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `228`
- 目标目录: `Algorithms/物理-量子力学-0228-光学定理_(Optical_Theorem)`

## R01

光学定理（Optical Theorem）给出散射振幅前向虚部与总截面的关系：

- `sigma_tot = (4*pi/k) * Im f(theta=0)`

其中 `k` 是入射波数，`f(theta)` 是散射振幅。它本质上来自 S 矩阵幺正性，是“概率守恒”在散射问题中的可观测表达。

## R02

本条目的目标是构建一个最小但可审计的数值实验，验证光学定理在纯弹性散射中成立。  
选用硬球势（hard sphere）作为模型，因为它的分波相移可直接写出，避免把复杂势场求解器当黑盒。

## R03

MVP 任务定义：

- 模型：半径 `a` 的硬球散射；
- 输入：若干 `k` 值；
- 通过分波展开计算 `f(0)` 与 `sigma_tot`；
- 独立计算角分布积分 `sigma_el = 2*pi * ∫ |f(theta)|^2 d(cos theta)`；
- 验证：`sigma_tot(phase shift)`、`sigma_tot(optical theorem)`、`sigma_el` 三者一致。

## R04

核心公式：

1. 硬球边界条件 `u_l(r=a)=0` 导致
`tan(delta_l) = j_l(ka) / y_l(ka)`。

2. 分波散射振幅
`f(theta) = (1/k) * sum_l (2l+1) e^{i delta_l} sin(delta_l) P_l(cos theta)`。

3. 前向振幅（`P_l(1)=1`）
`f(0) = (1/k) * sum_l (2l+1) e^{i delta_l} sin(delta_l)`。

4. 相移表达的总截面
`sigma_tot = (4*pi/k^2) * sum_l (2l+1) sin^2(delta_l)`。

5. 光学定理
`sigma_tot = (4*pi/k) * Im f(0)`。

## R05

设最大分波截断为 `lmax`，角积分高斯点数为 `Q`：

- 相移计算：`O(lmax)`；
- 前向振幅：`O(lmax)`；
- 角网格振幅积分：`O(lmax * Q)`；
- 空间复杂度：`O(lmax + Q)`。

本 MVP 的代价主要在角积分（用于独立校验），而非光学定理本身。

## R06

`demo.py` 的实验流程：

- 固定 `a=1`，扫描 `k={0.5, 1.0, 2.0, 4.0}`；
- 每个 `k` 自适应选择 `lmax`；
- 输出 `sigma_phase`、`sigma_optical`、`sigma_elastic` 与相对误差；
- 自动断言误差阈值和 `|S_l|=1` 幺正性。

运行时不需要任何交互输入。

## R07

优点：

- 用最少代码呈现“幺正性 -> 光学定理”的完整数值链路；
- 同时用三条路径算截面，避免单一路径自洽假阳性；
- 物理可解释性强，每个中间量都能单独检查。

局限：

- 仅覆盖中心势、单通道、纯弹性散射；
- 未涉及吸收道（复势）或多通道耦合；
- `lmax` 采用经验截断，不是严格误差控制器。

## R08

前置知识：

- 分波展开与勒让德多项式；
- 球贝塞尔函数 `j_l`、`y_l`；
- S 矩阵幺正性与截面定义。

环境依赖：

- Python `>=3.10`
- `numpy`
- `scipy`

## R09

适用场景：

- 量子散射教学中的光学定理演示；
- 为更复杂散射程序做单元校验基准；
- 快速检查数值分波实现是否保持概率守恒。

不适用场景：

- 非中心势和各向异性散射；
- 多通道或有吸收的反应散射；
- 需要实验拟合级高精度误差控制的生产仿真。

## R10

正确性直觉：

1. 分波幺正散射满足 `S_l = e^{2i delta_l}`，因此 `|S_l|=1`；
2. `Im[e^{i delta} sin delta] = sin^2 delta`，直接把前向虚部映射到总截面求和；
3. 对纯弹性散射，`sigma_tot = sigma_el`，角分布积分提供了独立闭环；
4. 三条路径一致，说明“公式、实现、数值积分”三环都工作正常。

## R11

数值稳定性策略：

- 使用 `arctan2(j_l, y_l)` 计算相移，减少象限歧义；
- 采用高斯-勒让德积分而非均匀角步长，降低角积分误差；
- `lmax` 使用 `ka + buffer` 的保守截断，减少高角动量漏项；
- 对相对误差用 `max(sigma, 1e-15)` 防止除零放大。

## R12

关键参数：

- `sphere_radius`：散射长度尺度；
- `k_values`：波数采样点；
- `l_buffer` / `min_lmax`：分波截断强度；
- `quad_order`：角积分精度。

调参建议：

- 若 `rel_err_elastic` 偏大，先增大 `quad_order`；
- 再增大 `l_buffer`，检查高 `l` 截断误差；
- 若只想快速 smoke test，可减少 `k` 点数和积分阶数。

## R13

本算法不是优化问题，没有近似比定义；可给出的保证是数值一致性保证：

- 对固定配置，输出可复现；
- `sigma_tot(phase)` 与 `sigma_tot(optical)` 在双精度内一致；
- `sigma_elastic` 与 `sigma_tot` 在给定离散误差阈值内一致；
- `|S_l|=1` 的断言验证模型幺正性未被实现破坏。

## R14

常见失效模式：

1. `lmax` 过小导致高角动量尾项被截断，光学定理看似失配；
2. 角积分阶数不足，`sigma_elastic` 对振荡结构分辨不够；
3. 在极高 `ka` 仍用低 `l_buffer`，前向振幅求和收敛慢；
4. 把 `delta_l` 当作连续可微相位直接差分会引入分支跳跃问题。

## R15

可扩展方向：

- 有限深势阱：数值解径向方程再抽取 `delta_l`；
- 含吸收道：复相移下验证广义光学定理；
- 多通道散射：把标量 `S_l` 扩展为通道矩阵；
- 不确定性分析：对截断与积分误差做系统外推。

## R16

相关主题：

- 分波方法（Partial Wave Analysis）；
- Born 近似与前向散射幅；
- 反应截面与非弹性散射；
- S 矩阵理论与幺正约束。

## R17

`demo.py` MVP 功能清单：

- 计算硬球模型分波相移 `delta_l`；
- 构造前向散射振幅 `f(0)`；
- 分别通过相移公式与光学定理计算总截面；
- 通过角分布积分计算弹性总截面并做交叉验证；
- 自动断言误差阈值并打印表格结果。

运行方式：

```bash
cd Algorithms/物理-量子力学-0228-光学定理_(Optical_Theorem)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（8 步）：

1. `choose_lmax` 根据 `ka` 选择分波截断，确定后续求和维度。  
2. `hard_sphere_phase_shifts` 调用 `scipy.special.spherical_jn/spherical_yn` 计算 `j_l(ka), y_l(ka)`，再用 `arctan2` 得到每个 `delta_l`。  
3. `partial_wave_coefficients` 把 `delta_l` 显式变换成系数 `c_l=(2l+1)e^{i delta_l}sin(delta_l)`，没有隐藏库逻辑。  
4. `forward_scattering_amplitude` 用 `f(0)=sum_l c_l / k` 计算前向振幅。  
5. `sigma_total_from_phase_shifts` 逐 `l` 累加 `(2l+1)sin^2(delta_l)` 得到 `sigma_tot`（相移路径）。  
6. `sigma_optical_theorem` 直接计算 `(4*pi/k)*Im f(0)` 得到 `sigma_tot`（光学定理路径）。  
7. `scattering_amplitude_on_mu_grid + sigma_elastic_angle_integral` 在高斯点上重建 `f(theta)` 并积分 `|f|^2` 得 `sigma_elastic`（角分布路径）。  
8. `run_case/main` 汇总三条路径、计算相对误差与 `|S_l|-1`，并用断言执行自动验收。  

说明：第三方库只负责特殊函数值与高斯节点生成；物理算法（相移、振幅、截面、光学定理校验）全部在源码中逐步展开。

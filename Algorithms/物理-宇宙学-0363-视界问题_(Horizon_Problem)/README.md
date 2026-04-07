# 视界问题 (Horizon Problem)

- UID: `PHYS-0345`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `363`
- 目标目录: `Algorithms/物理-宇宙学-0363-视界问题_(Horizon_Problem)`

## R01

视界问题是标准大爆炸宇宙学中的核心初始条件问题之一：
- 在不含暴胀的标准热史下，退耦时刻（`z ~ 1089`）的粒子视界对应天空上仅约 `~1` 度量级。
- 这意味着 CMB 天空中大范围区域在退耦前彼此不在因果接触内。
- 但观测上 CMB 在远大于 1 度的尺度上仍高度各向同性（温度起伏仅 `~10^-5`），形成“为何本应互不通信的区域温度几乎一致”的张力。

## R02

本目录 MVP 目标：用一个可运行、可审计的最小数值模型，把“标准宇宙学下的因果补丁不足”与“暴胀如何放大因果补丁”量化出来。

实现闭环：
1. 在平直 `LambdaCDM` 近似下计算退耦时刻共动粒子视界 `chi_hor(a_dec)`。
2. 计算今天到最后散射面的共动距离 `chi_LSS`。
3. 估计标准模型下可因果连通角尺度 `theta_std = chi_hor/chi_LSS` 及全空独立补丁数。
4. 用极简 `theta_eff = theta_std * exp(N)` 模型给出几何意义的 `N` 下界。

## R03

参数与符号（见 `demo.py` 中 `HorizonParams`）：
- `H0`：哈勃常数，默认 `67.4 km/s/Mpc`。
- `Omega_m, Omega_r, Omega_Lambda`：物质、辐射、暗能量密度参数（平直条件下 `Omega_Lambda = 1 - Omega_m - Omega_r`）。
- `z_dec`：退耦红移，默认 `1089`。
- `a_dec = 1/(1+z_dec)`：退耦尺度因子。
- `chi_hor(a_dec)`：退耦时共动粒子视界。
- `chi_LSS`：从退耦到今天的共动径向距离。
- `theta_std`：标准热史可因果连通角尺度。

## R04

MVP 采用的核心关系式：

1. 维度化哈勃函数
`H(a) = H0 * E(a)`

2. 平直 `LambdaCDM` 的 `E(a)`
`E(a) = sqrt(Omega_r/a^4 + Omega_m/a^3 + Omega_Lambda)`

3. 共动距离
`chi(a1,a2) = (c/H0) * int_{a1}^{a2} da / (a^2 E(a))`

4. 粒子视界（退耦时）
`chi_hor(a_dec) = chi(a_min, a_dec)`

5. 最后散射面共动距离
`chi_LSS = chi(a_dec, 1)`

6. 角尺度与补丁数近似
`theta_std ~= chi_hor/chi_LSS`
`N_patch ~= 4/theta_std^2`

7. 暴胀放大补丁角尺度（教学级简化）
`theta_eff(N) = theta_std * exp(N)`

## R05

本任务的“算法输出”包括：
- `chi_hor_dec`：退耦时共动视界（Mpc）。
- `chi_to_lss`：到最后散射面的共动距离（Mpc）。
- `theta_std_deg`：标准模型下视界角（度）。
- `patches_std`：全空独立因果补丁估计数。
- `N_min`（两种几何判据）：使补丁覆盖达到目标角尺度的最小 e-folds。
- `N` 扫描表：`theta_eff_deg`、`causal_patches`、`single_patch_sky`。

## R06

`demo.py` 计算流程：
1. 根据 `HorizonParams` 得到 `a_dec` 与 `Omega_Lambda`。
2. 用 `comoving_distance_mpc` 数值积分得到 `chi_hor_dec` 与 `chi_to_lss`。
3. 计算 `theta_std` 和 `patches_std`，展示标准视界问题的量级。
4. 以 `theta_target=2`（面积判据）和 `theta_target=pi`（反点判据）计算 `N_min`。
5. 构建 `N=0..8` 的扫描表，观察补丁数随 `N` 指数下降。
6. 对 `z_dec` 做灵敏度扫描，验证结论稳定性。

## R07

正确性依据：
- 物理层：视界问题本质是“退耦前可因果连通尺度不足以覆盖 CMB 大角尺度”。
- 数学层：`theta_eff(N)` 随 `N` 单调指数增长，`causal_patches ~ exp(-2N)` 单调下降。
- 程序层：所有核心量都由显式公式和显式积分得到，结果可逐项复核。

## R08

复杂度（设数值积分自适应采样点数为 `q`，`N` 扫描点数为 `m`）：
- 单次共动距离计算约 `O(q)`。
- 主结果需要常数次积分，复杂度 `O(q)`。
- `N` 扫描表构造 `O(m)`。
- 空间复杂度 `O(m)`（主要来自表格）。

## R09

数值稳定性与工程处理：
- 积分下限使用 `a_min=1e-8`，避免直接触碰 `a=0` 的端点奇异。
- 对积分区间做参数合法性检查：要求 `0 < a_start < a_end <= 1`。
- `scipy.integrate.quad` 使用较严格容差（`epsabs=1e-10`, `epsrel=1e-8`）。
- 对角度、补丁数与 e-fold 输入都设置正值校验，避免静默错误。

## R10

代码模块划分：
- `HorizonParams`：统一参数入口；
- `omega_lambda` / `e_of_a`：背景膨胀模型；
- `comoving_distance_mpc`：通用共动距离积分器；
- `particle_horizon_mpc`：退耦视界特化；
- `horizon_angle_rad` / `causal_patch_count`：角尺度与补丁数映射；
- `required_efolds`：按目标角尺度反求 `N`；
- `build_efold_table`：批量生成 `N` 扫描结果；
- `run_demo`：组织实验并输出报告。

## R11

最小依赖栈：
- `numpy`：指数、角度换算和数组运算；
- `scipy`：`integrate.quad` 自适应数值积分；
- `pandas`：结果表格化展示。

未引入 Boltzmann 求解器或大型宇宙学管线，保持可读、可验证、可复现的最小 MVP。

## R12

运行方式（仓库根目录）：

```bash
uv run python "Algorithms/物理-宇宙学-0363-视界问题_(Horizon_Problem)/demo.py"
```

或切换目录后运行：

```bash
cd "Algorithms/物理-宇宙学-0363-视界问题_(Horizon_Problem)"
uv run python demo.py
```

脚本无交互输入。

## R13

输出字段解释：
- `Comoving particle horizon at decoupling`：退耦时能因果通信的最大共动半径；
- `Comoving distance to last-scattering surface`：观察者到 LSS 的共动距离；
- `Standard-horizon angular scale`：标准模型下的因果补丁角半径；
- `Estimated independent causal patches`：CMB 天空最粗略“独立初值区域”数量；
- `N_min`：达到指定覆盖角判据所需的几何下界 e-folds；
- 表格 `single_patch_sky`：在该简化判据下是否达到“单补丁覆盖全空”量级。

## R14

自检建议：
1. 把 `z_dec` 改大（更早退耦），应看到 `theta_std` 变小、补丁数增大。
2. 把 `Omega_r` 提大，早期视界增长更慢，`theta_std` 通常变小。
3. 把 `N` 扫描上限从 `8` 改到 `10`，应看到补丁数继续按 `exp(-2N)` 下降。
4. 对比两种阈值（`2 rad` 与 `pi rad`），后者所需 `N` 应更高。

## R15

模型边界与局限：
- 这是教学级几何 MVP，不是精密宇宙学参数反演模型。
- `theta_eff = theta_std * exp(N)` 是“可解释下界”近似，未显式追踪再加热与慢滚细节。
- 未纳入声学视界、再电离和微扰转移函数等高精度效应。
- 所得 `N_min` 是“几何可连通性下界”，通常小于文献中 `50-60` 的完整动力学估计。

## R16

可扩展方向：
- 引入更完整早期宇宙热史（再加热方程状态、`g_*` 演化）。
- 把“模式出入视界”条件与 `k=aH` 显式结合，计算更真实的 `N_k`。
- 加入观测参数后验（Planck/BAO）做不确定性传播。
- 对接 CMB 初始条件模块，比较“有/无暴胀”的温度相关函数差异。

## R17

与暴胀宇宙学的关系：
- 视界问题、平坦性问题、磁单极子问题是暴胀的经典动机。
- 在本实现中，核心机制被压缩为：暴胀把原本很小的因果补丁指数放大。
- 因此只要 `N` 足够大，退耦时大片天空就可追溯到暴胀前同一因果连通区域，从而解释 CMB 大尺度均匀性。

## R18

本实现的源码级算法流（8 步）：
1. `run_demo` 构造 `HorizonParams`，并计算 `a_dec = 1/(1+z_dec)`。
2. `omega_lambda` 在平直约束下求 `Omega_Lambda = 1 - Omega_m - Omega_r`，并做正值检查。
3. `e_of_a` 显式实现 `E(a)`，不把背景模型隐藏在第三方黑箱内。
4. `comoving_distance_mpc` 构造被积函数 `1/(a^2 E(a))`；`scipy.integrate.quad` 对该函数做自适应 Gauss-Kronrod 采样并返回积分值。
5. 通过两次积分得到 `chi_hor_dec = chi(a_min,a_dec)` 与 `chi_to_lss = chi(a_dec,1)`。
6. `horizon_angle_rad` 与 `causal_patch_count` 映射出 `theta_std` 和 `N_patch = 4/theta^2`，定量展示标准视界问题。
7. `required_efolds` 显式反解 `theta_std * exp(N) >= theta_target`，给出面积判据与反点判据的 `N_min`。
8. `build_efold_table` 逐点计算 `N` 网格上的 `theta_eff`、`causal_patches` 与布尔判定，`run_demo` 最后打印主结果和 `z_dec` 灵敏度扫描。

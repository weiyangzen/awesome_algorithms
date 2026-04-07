# 波导理论 (Waveguide Theory)

- UID: `PHYS-0178`
- 学科: `物理`
- 分类: `电磁学`
- 源序号: `179`
- 目标目录: `Algorithms/物理-电磁学-0179-波导理论_(Waveguide_Theory)`

## R01

本条目给出一个可运行的波导理论最小实现（MVP），聚焦**矩形金属波导的主模 TE10**：
- 计算截止频率 `f_c`；
- 在频率扫描上区分“可传播区”和“截止下倏逝区”；
- 输出色散相关量：`beta`、`lambda_g`、`v_phase`、`v_group`、`Z_TE`；
- 导出 `waveguide_dispersion.csv` 与 `te10_profile.csv`，便于后续画图和核查。

## R02

问题定义（本 MVP）：
- 输入参数：波导宽边 `a`、窄边 `b`、模式索引 `(m,n)`、介质参数 `(eps_r, mu_r)`、频率扫描区间、工作频率 `f0`、传播长度 `L`。
- 输出结果：
  - 截止频率 `f_c`；
  - 传播常数 `beta`（仅 `f>f_c`）；
  - 截止下衰减常数 `alpha_ev`（仅 `f<f_c`）；
  - 导行波长 `lambda_g`、相速度 `v_phase`、群速度 `v_group`、TE 波阻抗 `Z_TE`；
  - 传播传递因子 `H(L)=exp(-j beta L)`（理想无损近似）。

## R03

核心公式（矩形波导）：

1. 截止频率

`f_c = (v/2) * sqrt((m/a)^2 + (n/b)^2)`，其中 `v = c / sqrt(eps_r * mu_r)`。

2. 波数与归一化比值

`k = 2pi f / v`，`r = (f_c/f)^2`。

3. 可传播区（`f > f_c`）

`beta = k * sqrt(1-r)`

`lambda_g = 2pi / beta`

`v_phase = omega / beta = v / sqrt(1-r)`

`v_group = v * sqrt(1-r)`

`Z_TE = eta / sqrt(1-r)`，其中 `eta = sqrt(mu/epsilon)`。

4. 截止下（`f < f_c`）倏逝衰减

`alpha_ev = k * sqrt(r-1)`。

5. 无损传播传递因子

`H(L) = exp(-j beta L)`。

## R04

目标与边界：
- 目标：把“截止 -> 色散 -> 传播”这一波导理论主链路落成可执行脚本。
- 边界：
  - 采用理想 PEC 无损模型，不做导体损耗/介质损耗精细建模；
  - 不求解复杂激励耦合、弯折、台阶不连续、法兰反射等工程细节；
  - 以解析公式 + 数值表格为主，不使用全波仿真器替代。

## R05

复杂度分析（`N = sweep_points`）：
- 截止频率计算：`O(1)`；
- 频率扫描色散计算：逐点向量化，时间复杂度 `O(N)`；
- 结果表构建与导出：`O(N)`；
- 空间复杂度：`O(N)`。

## R06

数值示例（默认参数，WR-90 近似尺寸 `a=22.86 mm, b=10.16 mm`，空气介质）：
- 模式 `TE10` 截止频率约 `6.557 GHz`；
- 在 `f0=10 GHz`：
  - `beta ≈ 158.24 rad/m`；
  - `lambda0 ≈ 29.98 mm`，`lambda_g ≈ 39.71 mm`；
  - `v_phase/c ≈ 1.324`，`v_group/c ≈ 0.755`；
  - `Z_TE ≈ 498.97 Ohm`；
  - 且 `v_phase * v_group ≈ c^2`（无损色散关系）。

## R07

为何该条目重要：
- 波导是微波工程、雷达、卫星通信、粒子加速器射频系统的基础传输结构；
- 截止频率决定“能否传输”的硬约束；
- 色散关系决定相位延迟、群时延和脉冲传输特性；
- 对后续学习谐振腔、滤波器、耦合器和模式匹配有直接基础价值。

## R08

理论假设：
- 时谐场（单频稳态）框架；
- 波导为理想均匀直波导，截面不随 `z` 变化；
- 理想 PEC 边界；
- 介质均匀各向同性；
- 主示例使用 TE10 模式，场分布轮廓使用 `Ey(x) ~ sin(pi x/a)` 归一化示意。

## R09

适用范围与局限：
- 适用：教学演示、参数扫描、早期工程估算、算法验证。
- 局限：
  - 未覆盖导体粗糙度、表面电阻导致的插损；
  - 未覆盖高阶模多模耦合与模式转换；
  - 未覆盖不连续结构引起的反射系数 `S11/S21` 计算；
  - 不能替代 FDTD/FEM/MoM 等全波建模。

## R10

正确性依据（实现链路）：
1. 由矩形波导解析式计算 `f_c`；
2. 对每个频率点判断 `f > f_c` 与否；
3. 对可传播点计算 `beta, lambda_g, v_phase, v_group, Z_TE`；
4. 对截止下点计算 `alpha_ev`；
5. 在工作点使用 `H(L)=exp(-j beta L)` 得到传播相位变化。

该链路与经典微波教材中的 TE 模式色散关系一致。

## R11

误差与偏差来源：
- 数值误差：浮点除法与接近截止处的条件数变差；
- 模型误差：理想 PEC/无损近似低估真实损耗；
- 工程误差：真实连接器、弯折、加工公差会引入附加反射与损耗。

因此本 MVP 结果应解释为“理论基线”，而不是实测替代值。

## R12

实现取舍：
- 工具栈保持最小：`numpy + scipy.constants + pandas`；
- 不调用黑盒电磁仿真库，全部核心量由源码显式公式计算；
- 输出 CSV 方便后续在任意环境（Python/Excel/Matlab）复核。

## R13

可验证性（`demo.py` 内置检查）：
- `f_c` 需落在 WR-90 TE10 常见范围（约 `6.4~6.7 GHz`）；
- 工作点应满足 `v_phase > c`、`0 < v_group < c`；
- 验证 `v_phase * v_group ≈ c^2`；
- 验证 `10 GHz` 下 `lambda_g` 数值区间合理；
- 脚本结束输出 `All checks passed.`。

## R14

鲁棒性处理：
- 对 `a,b,eps_r,mu_r` 与频率输入做正值检查；
- 禁止 `(m,n)=(0,0)` 的非法模式索引；
- 在截止附近使用显式掩码分流（传播区 vs 倏逝区），避免对负数开方；
- 用 `NaN` 标记不适用物理量（如截止下 `beta`），避免误解释。

## R15

代码结构（见 `demo.py`）：
- `WaveguideConfig`：参数容器；
- `medium_speed` / `medium_impedance`：介质基础量；
- `cutoff_frequency_rectangular`：截止频率；
- `te_mode_dispersion`：频率向量上的色散主计算；
- `dominant_te10_profile`：TE10 横向场形状采样；
- `build_dispersion_table`：组织扫描表；
- `evaluate_operating_point`：工作点关键指标与传递因子；
- `main`：串联执行、导出 CSV、打印结果和断言。

## R16

与更高阶方法关系：
- 更高保真：可扩展到有损波导（导体/介质损耗）并引入复传播常数 `gamma = alpha + j beta`；
- 更完整结构：可继续做台阶不连续的模式匹配法（Mode Matching）；
- 更复杂几何：可用 FEM/FDTD 处理任意截面和三维结构。

本条目定位是“波导理论主方程可运行骨架”，便于向工程模型演进。

## R17

运行方式：

```bash
cd Algorithms/物理-电磁学-0179-波导理论_(Waveguide_Theory)
uv run python demo.py
```

运行后将：
- 终端打印截止频率、工作点色散参数与抽样表；
- 生成 `waveguide_dispersion.csv`（频率扫描主表）；
- 生成 `te10_profile.csv`（TE10 横向场轮廓）；
- 通过内置检查后输出 `All checks passed.`。

## R18

源码级算法流程拆解（`demo.py`，8 步）：
1. `main` 创建 `WaveguideConfig`，确定波导尺寸、模式索引、扫描频段和工作点。  
2. `build_dispersion_table` 调 `cutoff_frequency_rectangular`，按 `(a,b,m,n,eps_r,mu_r)` 计算 `f_c`。  
3. 生成频率网格 `f_hz`，送入 `te_mode_dispersion` 做向量化计算。  
4. `te_mode_dispersion` 内部先算 `k` 与比值 `r=(f_c/f)^2`，再用掩码把 `f>f_c` 与 `f<f_c` 分流。  
5. 对可传播点计算 `beta/lambda_g/v_phase/v_group/Z_TE`；对截止下点计算 `alpha_ev`，其余量填 `NaN`。  
6. `evaluate_operating_point` 在单频 `f0` 读取对应色散量，并计算 `H(L)=exp(-j beta L)` 的幅相。  
7. `dominant_te10_profile` 生成 `Ey(x)=sin(pi x/a)` 归一化横向分布，构建 `te10_profile.csv`。  
8. `main` 导出两份 CSV、打印关键指标，并执行物理一致性断言（如 `v_phase*v_group≈c^2`）。

说明：未将核心计算交给黑盒库；截止、色散、传播传递全由源码中的显式公式逐步展开。

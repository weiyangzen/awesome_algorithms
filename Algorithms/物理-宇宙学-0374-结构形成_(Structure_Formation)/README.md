# 结构形成 (Structure Formation)

- UID: `PHYS-0356`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `374`
- 目标目录: `Algorithms/物理-宇宙学-0374-结构形成_(Structure_Formation)`

## R01

本条目实现一个“最小但诚实”的宇宙学结构形成 MVP：
- 线性增长因子 `D(a)` 的数值求解（平直 `ΛCDM`）。
- 线性功率谱 `P(k,z)=D(z)^2 P0(k)` 的演化。
- `Press-Schechter` 质量函数 `dn/dlnM` 的计算。
- 用 `scikit-learn` 和 `PyTorch` 做可解释诊断，而不是黑箱调用。

## R02

问题对象是“从早期密度涨落到晚期暗晕统计”的简化链路。MVP 不追求高精度宇宙学拟合，而追求：
- 方程可追踪。
- 数值流程可复现。
- 每一步都能在源码里定位。

## R03

核心方程：
- 背景膨胀：`E(a)=H(a)/H0 = sqrt(Ωm0 a^-3 + ΩΛ0)`。
- 线性增长 ODE：`D'' + [2 + dlnH/dlna]D' - 3/2 Ωm(a)D = 0`。
- 线性谱演化：`P(k,z)=D(z)^2 P0(k)`。
- 方差：`σ^2(R)=∫ dk k^2 P(k) W^2(kR)/(2π^2)`，`W` 为球形 top-hat。
- Press-Schechter：`dn/dlnM ∝ (ρm/M) ν |dlnσ/dlnM| exp(-ν^2/2)`，`ν=δc/σ(M)`。

## R04

`demo.py` 输入是脚本内固定参数（非交互），输出是终端报表：
- 增长历史下采样摘要：`z, D(z), sigma8(z), P(k=0.1,z)`。
- 拟合诊断：`n_eff` 与增长指数 `gamma`。
- `z=0` 的质量函数样例表。

## R05

默认参数（`FlatLambdaCDM`）：
- `h=0.67`
- `Ωm0=0.315`
- `ΩΛ0=0.685`
- `n_s=0.965`
- `k_damp=0.35 (h/Mpc)`
- `δc=1.686`
- `sigma8_target=0.81`

## R06

实现策略：
- 用 `scipy.solve_ivp` 解增长方程，得到 `a,z,D,f` 网格。
- 用 toy 形状 `P0(k) ∝ k^ns exp[-(k/kd)^2]` 表示线性初始谱。
- 用 `scipy.root_scalar` 标定振幅，使 `σ8(z=0)` 达到目标。
- 用积分与导数计算 `Press-Schechter` 质量函数。

## R07

时间复杂度（主导项）：
- 增长 ODE：`O(N_a)`。
- `σ(M)` 计算：`O(N_M * N_k)`。
- 质量函数后处理：`O(N_M)`。

以默认 `N_k=1024, N_M=60`，CPU 上可在秒级完成。

## R08

数值稳定设计：
- `x->0` 处 top-hat 使用泰勒极限，避免 `0/0`。
- `σ(M)` 下限裁剪到 `1e-10`，避免 `log(0)`。
- ODE 使用严格容差：`rtol=1e-8, atol=1e-10`。
- 幅度标定采用有括区根求解，避免发散迭代。

## R09

边界与失败条件：
- 若增长 ODE 失败，抛出 `RuntimeError`。
- 若 `sigma8` 振幅求根不收敛，抛出 `RuntimeError`。
- 本实现只覆盖平直 `ΛCDM`（`Ωk=0`），不含辐射与中微子。

## R10

MVP 的“诚实范围”：
- 是教学/算法演示，不是 Boltzmann 求解器替代品。
- 功率谱采用 toy 形状，不含 CAMB/CLASS 级转移函数细节。
- 质量函数只实现 `Press-Schechter`，未含 Sheth-Tormen 修正。

## R11

运行方式：

```bash
uv run python Algorithms/物理-宇宙学-0374-结构形成_(Structure_Formation)/demo.py
```

脚本无交互输入，直接打印结果。

## R12

输出读法：
- `D(z)` 与 `sigma8(z)` 随 `z` 上升而下降，体现线性增长冻结。
- `P(k=0.1,z)` 随 `D(z)^2` 缩放。
- `dn/dlnM` 在高质量端指数抑制，反映稀有峰统计。

## R13

最小验证清单：
- `D(z=0) ≈ 1`（归一化定义）。
- 拟合的 `gamma` 接近常见经验值 `~0.55`。
- `sigma8(z=0)` 等于目标值（由振幅标定保证）。
- README 与 demo 不包含占位符文本。

## R14

结果解释建议：
- 把该 MVP 视为“结构形成主链路的可执行骨架”。
- 用它比较参数变化方向性（例如 `Ωm0` 增大时增长更快）。
- 不把绝对数值当作精密观测拟合结论。

## R15

当前局限：
- 缺少真实转移函数与 BAO 特征。
- 未实现非线性修正（Halofit / N-body 标定）。
- 质量函数无并合历史与偏置建模。

## R16

可扩展方向：
- 接入 CAMB/CLASS 的线性谱替换 toy `P0(k)`。
- 增加 Sheth-Tormen / Tinker 质量函数。
- 加入红移空间畸变、偏置与观测投影模块。

## R17

工具栈与职责：
- `numpy`: 数组网格、窗口函数、数值导数。
- `scipy`: ODE、积分、根求解。
- `pandas`: 表格化结果输出。
- `scikit-learn`: 拟合有效谱斜率 `n_eff`。
- `PyTorch`: 对增长指数 `gamma` 做自动微分拟合。

## R18

`demo.py` 的源码级算法流程（8 步）：
1. `solve_growth_history` 调 `solve_ivp` 解 `D(a)` 二阶 ODE，产出 `a,z,D,f`。
2. `toy_primordial_power` 在 `k` 网格构建无振幅的 `P0` 形状。
3. 在 `main` 中定义 `sigma8_minus_target(A)`，调用 `root_scalar` 求振幅 `A`。
4. 用 `linear_power_at_z` 把 `P0` 传播到多个红移，得到 `P(k,z)` 摘要。
5. 用 `LinearRegression` 在 `0.02<k<0.2` 上拟合 `logP-logk`，得到 `n_eff`。
6. `fit_growth_index_torch` 以 `f(a)` 为监督，优化 `gamma` 使 `f≈Ωm(a)^gamma`。
7. `press_schechter_mass_function` 通过 `σ(M)` 与 `dlnσ/dlnM` 计算 `dn/dlnM`。
8. `main` 汇总为三段报表打印，形成可直接验证的最小端到端结果。

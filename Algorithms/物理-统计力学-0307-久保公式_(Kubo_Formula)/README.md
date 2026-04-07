# 久保公式 (Kubo Formula)

- UID: `PHYS-0304`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `307`
- 目标目录: `Algorithms/物理-统计力学-0307-久保公式_(Kubo_Formula)`

## R01

久保公式（Kubo Formula）是线性响应理论的核心结果，用于把“外场下的输运系数”写成“平衡态涨落相关函数”的时间积分。  
本目录的 MVP 选择 1D Langevin/Ornstein-Uhlenbeck 速度过程，演示最常见的 Green-Kubo 关系：

- 扩散系数：`D = ∫_0^∞ <v(0)v(t)> dt`
- 电导率（单位体积、单粒子原型）：`σ = β ∫_0^∞ <J(0)J(t)> dt`, 其中 `J=qv`, `β=1/(k_B T)`

## R02

问题定义（本实现）：

- 输入：一组动力学参数 `γ, m, kBT, q, dt, n_steps`。
- 过程：生成平衡态速度时间序列 `v_t`，估计自相关 `C_vv(τ)` 与 `C_jj(τ)`，做数值积分。
- 输出：`D_estimate, σ_estimate` 及其与理论值的相对误差。

理论闭式解（OU 过程）：

- `D_theory = kBT / (mγ)`
- `σ_theory = q^2 / (mγ)`

## R03

久保公式适用前提（线性响应近似）：

- 系统初始在热平衡附近；
- 外场足够弱，响应可以线性化；
- 相关函数衰减使积分收敛。

本 demo 不直接施加外场，而是用平衡涨落反推输运系数，这正是 Kubo 思路的数值化版本。

## R04

离散化建模：

- 时间网格：`t_n = n * dt`
- OU 递推：`v_{n+1} = a v_n + η_n`
- 系数：`a = exp(-γ dt)`
- 噪声：`η_n ~ N(0, (kBT/m)(1-a^2))`

该离散形式保证平衡方差逼近 `Var(v)=kBT/m`，便于直接用于 Kubo/Green-Kubo 积分。

## R05

数值相关函数定义（无偏估计）：

- `C_xx(τ) = (1/(N-τ)) * Σ_{n=0}^{N-τ-1} x_n x_{n+τ}`
- `x=v` 得速度自相关 `C_vv`
- `x=J=qv` 得电流自相关 `C_jj`
- 先做去均值再求相关，以降低有限样本长时偏置。

积分采用梯形公式：

- `∫_0^∞ C(t)dt ≈ ∫_0^{t_max} C(t)dt`
- `t_max = max_lag * dt`，以有限截断代替无限上限。

## R06

实现中的关键工程选择：

- 自相关用 FFT 计算（`O(N log N)`），避免直接双循环的 `O(N^2)`；
- 设 `cap_lag = max_lag_fraction * N`，并在该上限内取首个零交叉点作为 `max_lag`，降低长时噪声积分污染；
- 固定随机种子保证可复现实验；
- 输出 JSON 便于后续自动验证脚本解析。

## R07

算法伪代码：

1. 读取配置参数；
2. 用 OU 递推生成 `v[0..N-1]`；
3. 计算 `C_vv = autocorr_unbiased(v)`；
4. 设 `J=qv`，计算 `C_jj = autocorr_unbiased(J)`；
5. 在 `0..max_lag` 上对 `C_vv` 做梯形积分得 `D_est`；
6. 在 `0..max_lag` 上对 `C_jj` 做梯形积分并乘 `β` 得 `σ_est`；
7. 计算理论值 `D_theory, σ_theory` 与相对误差；
8. 打印结构化结果。

## R08

正确性直觉：

- OU 过程是可解析模型，其关联函数为指数衰减；
- 速度关联积分应给出 `kBT/(mγ)`；
- 电流只是速度乘常数 `q`，因此电导率与扩散系数保持解析比例关系；
- 当 `n_steps` 增大时，估计值应收敛到理论值。

## R09

复杂度分析：

- OU 仿真：时间 `O(N)`，空间 `O(N)`；
- FFT 自相关：时间 `O(N log N)`，空间 `O(N)`；
- 梯形积分与误差统计：时间 `O(N)`（或 `O(max_lag)`）。

总复杂度由 FFT 主导：`O(N log N)`。

## R10

误差来源与稳定性：

- 有限样本误差：`N` 不够大时长时间滞后处噪声大；
- 积分截断误差：`t_max` 太小会低估积分；
- 离散时间误差：`dt` 太大时 OU 连续极限近似变差；
- 随机波动：单次轨迹会偏离理论值。

缓解办法：增大 `n_steps`、适当减小 `dt`、调节 `max_lag_fraction` 并做多次平均。
此外，使用“零交叉截断”比固定长窗口更稳健，通常能明显降低长时噪声带来的系统误差。

## R11

`demo.py` 模块说明：

- `KuboConfig`：集中管理模型与数值参数；
- `simulate_ou_velocity`：生成平衡态速度轨迹；
- `autocorr_unbiased`：基于 FFT 的无偏自相关；
- `integrate_trapezoid`：统一积分接口；
- `run_kubo_demo`：执行完整估计流程并返回指标；
- `main`：打印 JSON 结果（无交互输入）。

## R12

运行方式：

```bash
uv run python Algorithms/物理-统计力学-0307-久保公式_(Kubo_Formula)/demo.py
```

预期行为：终端输出一段 JSON，包含配置、估计值、理论值、相对误差与少量诊断量（如 `Cvv_0`）。

## R13

输出字段解释（核心）：

- `D_estimate` / `D_theory`：扩散系数估计与理论值；
- `sigma_estimate` / `sigma_theory`：电导率估计与理论值；
- `D_rel_error` / `sigma_rel_error`：相对误差；
- `cap_lag`：允许积分的最大滞后上限（硬上限）；
- `max_lag`：零交叉策略选出的实际积分滞后；
- `Cvv_0`、`Cjj_0`：零时刻相关值，用于快速 sanity check。

## R14

边界与失败场景：

- `n_steps < 2` 时无法计算相关函数；
- `max_lag < 2` 时积分无意义；
- `kBT <= 0` 或 `mass <= 0` 或 `gamma <= 0` 时物理参数不合法；
- 极端小样本下，误差可能较大，不应据此否定公式本身。

## R15

可扩展方向：

- 从单粒子扩展到多粒子电流 `J = Σ_i q_i v_i`；
- 计算频域响应 `σ(ω)`（对相关函数做傅里叶变换）；
- 加入 block averaging / bootstrap 误差条；
- 扩展到剪切黏度、热导率等其他 Green-Kubo 输运系数。

## R16

最小验证清单：

1. 代码可直接运行且无交互；
2. 输出 JSON 可解析；
3. `D_rel_error` 与 `sigma_rel_error` 在合理统计范围（通常几个百分点到十几个百分点，随样本长度变化）；
4. 增大 `n_steps` 后误差总体下降（统计意义上）。

## R17

参考知识点（概念层）：

- R. Kubo, “Statistical-Mechanical Theory of Irreversible Processes.”
- Green-Kubo relations in equilibrium statistical mechanics.
- Langevin equation 与 Ornstein-Uhlenbeck process 的解析相关函数。

本目录不依赖外部专用物理库，重点是把公式映射成可执行、可验证的最小数值流程。

## R18

`demo.py` 源码级算法流（非黑箱，8 步）：

1. 读取 `KuboConfig`，确定 `N, dt, γ, m, kBT, q`。  
2. 在 `simulate_ou_velocity` 中计算 `a=exp(-γdt)` 和噪声标准差 `sqrt((kBT/m)(1-a^2))`。  
3. 用 for 循环递推生成 `v_{n+1}=a v_n + noise`，得到平衡速度序列 `v`。  
4. 调用 `autocorr_unbiased(v)`：先去均值，再 `rfft -> 功率谱 -> irfft` 得到自相关并除以 `(N-τ)` 做无偏归一化。  
5. 在 `cap_lag` 内寻找首个零交叉点作为 `max_lag`，对 `C_vv` 积分到 `max_lag` 得 `D_estimate`。  
6. 构造 `J=qv`，重复第 4-5 步得到 `∫C_jj dt`，并乘 `β=1/kBT` 得 `sigma_estimate`。  
7. 根据 OU 解析式计算 `D_theory=kBT/(mγ)` 与 `sigma_theory=q^2/(mγ)`。  
8. 计算相对误差，汇总为 JSON 打印，形成可复现实验结果。  

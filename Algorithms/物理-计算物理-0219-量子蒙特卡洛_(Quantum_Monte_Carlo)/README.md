# 量子蒙特卡洛 (Quantum Monte Carlo)

- UID: `PHYS-0218`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `219`
- 目标目录: `Algorithms/物理-计算物理-0219-量子蒙特卡洛_(Quantum_Monte_Carlo)`

## R01

本条目实现的是量子蒙特卡洛中的一个最小可运行分支: **变分量子蒙特卡洛 (Variational Monte Carlo, VMC)**。目标是用随机采样近似求解量子体系基态能量, 并展示完整的“采样-估计-选参”闭环, 而不是只调用黑盒库。

## R02

演示问题选择一维谐振子:

- 哈密顿量: `H = -1/2 d^2/dx^2 + 1/2 x^2`
- 单位制: `m = omega = hbar = 1`
- 精确基态能量: `E0 = 0.5`

这是 QMC 的标准教学入口, 因为可验证性强, 同时保留了真实的马尔可夫链采样和统计误差问题。

## R03

VMC 采用带参数试探波函数:

- `psi_alpha(x) = exp(-alpha x^2)`, `alpha > 0`
- 采样目标分布与 `|psi_alpha|^2` 成正比: `pi(x) propto exp(-2 alpha x^2)`

通过对 `alpha` 扫描, 选择使能量期望最小的参数, 由变分原理得到基态能量上界。

## R04

局域能定义为:

- `E_L(x) = (H psi_alpha)(x) / psi_alpha(x)`

对于本问题可解析化简为:

- `E_L(x) = alpha + (1/2 - 2 alpha^2) x^2`

VMC 用马尔可夫链样本 `x_i` 估计能量:

- `E(alpha) ≈ (1/N) sum_i E_L(x_i)`

## R05

采样器采用 Metropolis-Hastings 随机游走:

- 提议: `x_new = x + Normal(0, proposal_std)`
- 对数接受率: `log r = -2 alpha (x_new^2 - x^2)`
- 接受规则: 若 `log r >= 0` 则接受, 否则以 `exp(log r)` 的概率接受

实现中包含 `burn-in` 和 `thin`, 以减小初值偏差与自相关影响。

## R06

统计估计包含三项:

- 能量均值 `mean(E_L)`
- 标准误 `SEM`
- 积分自相关时间 `tau_int` 的正序截断估计

`SEM` 通过有效样本数 `N_eff = N / tau_int` 修正, 比直接 `std/sqrt(N)` 更接近真实误差。

## R07

参数优化策略采用“粗扫描 + 精修”:

- 粗扫描: 在 `alpha in [0.3, 0.8]` 网格上做短链评估
- 选最优: 取蒙特卡洛估计能量最小的 `alpha*`
- 精修: 在 `alpha*` 上做更长链, 输出最终能量与误差

这是小规模 MVP 常用方案, 复杂度低、可解释性高。

## R08

时间与空间复杂度:

- 单条链时间复杂度: `O(n_steps)`
- 单次局域能评估: `O(1)`
- 多参数扫描总复杂度: `O(n_alpha * n_steps)`
- 空间复杂度: `O(n_saved_samples)`

对本仓库的教学规模, 该实现能在秒级完成。

## R09

主要误差来源与控制手段:

- 统计涨落: 增加采样步数, 或多链平均
- 自相关偏大: 调整 `proposal_std` 与 `thin`
- 热化不足: 增大 `burn_in`
- 变分偏差: 改进试探波函数族

当接受率过低(<0.2)或过高(>0.9)时, 往往意味着步长不合适。

## R10

`demo.py` 代码结构:

- `VMCConfig`: 采样参数容器
- `local_energy`: 局域能公式
- `analytic_variational_energy`: 本试探族的解析能量
- `metropolis_vmc`: Metropolis 主循环
- `estimate_mean_sem_tau`: 均值/误差/相关时间估计
- `run_alpha`: 单参数执行
- `main`: 网格扫描 + 精修 + 结果打印

## R11

关键“数学-代码”对应关系:

- `pi(x) propto exp(-2 alpha x^2)` 对应 `log_ratio = -2*alpha*(x_new^2 - x^2)`
- `E_L(x)` 公式对应 `local_energy(...)`
- `E(alpha)` 的样本平均对应 `energies.mean()`
- 有效样本修正对应 `estimate_mean_sem_tau(...)` 中 `tau_int` 与 `N_eff`

这保证了实现不是库封装调用, 而是从算法定义到数值流程逐段可追踪。

## R12

运行方式:

```bash
uv run python Algorithms/物理-计算物理-0219-量子蒙特卡洛_(Quantum_Monte_Carlo)/demo.py
```

脚本无交互输入, 直接在终端输出扫描表和最终估计。

## R13

预期输出应包含:

- 标题与模型说明
- `alpha` 网格扫描表: `E_MC +/- SEM`, `E_analytic`, `accept_rate`, `tau_int`
- 最优 `alpha*` 的长链结果
- 与精确基态 `0.5` 的误差 `|E_MC - E_exact|`

合理结果通常在 `alpha ≈ 0.5` 附近取得最小能量。

## R14

可调超参数建议:

- `proposal_std`: 控制游走步长, 先从 `1.0` 起调
- `n_steps`: 不足时优先加大
- `burn_in`: 至少覆盖若干个相关时间尺度
- `thin`: 当存储压力或强相关明显时增大
- `alpha` 扫描区间: 若先验更强可收窄范围提升效率

## R15

与其他 QMC 分支的关系:

- 本实现: VMC, 简单、稳定、易验证
- Diffusion Monte Carlo (DMC): 能进一步逼近基态, 但实现更复杂
- Path Integral Monte Carlo (PIMC): 侧重有限温度路径积分

因此该 MVP 适合作为后续 DMC/PIMC 的基础脚手架。

## R16

常见问题排查:

- 输出 `nan` 或异常大误差: 检查 `alpha > 0`、`proposal_std > 0`
- 接受率很低: 缩小 `proposal_std`
- 接受率过高但收敛慢: 适当增大 `proposal_std`
- 结果漂移大: 提高 `n_steps`, 或多次不同随机种子复现实验

## R17

可扩展方向:

- 多电子体系与 Slater-Jastrow 试探波函数
- 引入自动微分优化变分参数(例如用 PyTorch 训练参数)
- 使用重配置/随机重整化降低优化噪声
- 并行多链和分布式采样
- 升级到 DMC, 引入行走者权重与分支过程

## R18

`demo.py` 的源码级算法流程可分为 8 步:

1. 在 `main()` 中固定随机种子并构造 `scan_config`。
2. 生成 `alpha` 网格, 对每个 `alpha` 调用 `run_alpha()`。
3. `run_alpha()` 内部调用 `metropolis_vmc()` 构建马尔可夫链。
4. `metropolis_vmc()` 每步先做高斯提议 `x_new`, 再按 `log_ratio` 执行接受拒绝。
5. 链在 `burn_in` 后按 `thin` 规则收集样本, 并用 `local_energy()` 计算 `E_L` 序列。
6. `run_alpha()` 将 `E_L` 传入 `estimate_mean_sem_tau()`, 得到能量均值、标准误和 `tau_int`。
7. `main()` 从扫描结果中选取能量最小的 `alpha*`, 用更长的 `refine_config` 再跑一条链。
8. 最后打印扫描表与精修结果, 并报告与精确基态能量 `0.5` 的绝对误差。

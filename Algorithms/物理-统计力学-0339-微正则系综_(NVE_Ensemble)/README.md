# 微正则系综 (NVE Ensemble)

- UID: `PHYS-0332`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `339`
- 目标目录: `Algorithms/物理-统计力学-0339-微正则系综_(NVE_Ensemble)`

## R01

微正则系综（NVE）描述的是孤立体系：粒子数 `N`、体积 `V`、总能量 `E` 固定，不与外界交换热或功。  
它是统计力学里最“封闭”的平衡系综，也是理解能量守恒动力学（如哈密顿系统、无温控分子动力学）的基础。

## R02

本任务的 MVP 目标是用最小可运行代码复现 NVE 的三个核心特征：
- 总能量在数值积分过程中近似守恒；
- 动能与势能时间平均满足均分趋势（谐系统下 `⟨K⟩ ≈ ⟨U⟩`）；
- 由动能反推出的微正则温度与总能量尺度一致。

实现选择为“1D 周期谐振子链 + velocity-Verlet 辛积分”，重点在机制透明而不是大规模高精度模拟。

## R03

微正则分布可写为：
- 相空间概率密度：`ρ(q, p) ∝ δ(H(q,p) - E)`
- 态密度：`Ω(E, N, V)`
- 熵：`S(E, N, V) = k_B ln Ω(E, N, V)`
- 温度定义：`1/T = (∂S/∂E)_{N,V}`

在经典动力学视角中，若轨道在能量壳层上充分遍历（遍历性假设），时间平均可近似系综平均。

## R04

`demo.py` 采用的一维周期谐链模型：
- 位置：`x_i`，速度：`v_i`
- 质量：`m`，弹簧常数：`k`
- 哈密顿量：
  - `K = (m/2) Σ_i v_i^2`
  - `U = (k/2) Σ_i (x_{i+1} - x_i)^2`
  - `H = K + U`
- 周期边界：`x_{N} = x_0`

力学方程由 `F_i = -∂U/∂x_i = k(x_{i+1} + x_{i-1} - 2x_i)` 给出。

## R05

算法选择：velocity-Verlet（辛积分）

单步更新：
1. `x(t+dt) = x(t) + v(t)dt + 0.5 a(t) dt^2`
2. 用新位置计算 `a(t+dt)`
3. `v(t+dt) = v(t) + 0.5 [a(t)+a(t+dt)] dt`

相比显式欧拉法，辛积分在长期能量稳定性上更适合 NVE 场景。

## R06

初态构造策略（保证初始能量精确落在目标能量壳附近）：
- 先随机生成位置并去掉质心平移（`mean(x)=0`）；
- 将位置缩放到目标势能 `U_target = 0.4 E_target`；
- 随机生成速度并去掉总动量（`mean(v)=0`）；
- 将速度缩放到目标动能 `K_target = 0.6 E_target`；
- 因而初始 `K + U = E_target`。

这样可以避免“从任意初态慢慢靠近目标能量”带来的额外复杂度。

## R07

统计流程：
- 演化 `n_steps` 步，每隔 `sample_every` 记录一次样本；
- 每个样本记录：`K`、`U`、`E_total`、`T_est`；
- 结束后计算：
  - 最大相对能量漂移 `max|ΔE/E0|`
  - `⟨K⟩/⟨U⟩`（均分检查）
  - 温度估计误差（`T_est` vs `E/(N-1)`）
  - 速度分布统计量（均值、方差、超额峰度）

## R08

代码结构（`demo.py`）：
- `NVEConfig`：集中管理系统规模、时间步长、采样参数、随机种子。
- 物理函数：
  - `kinetic_energy`
  - `potential_energy`
  - `spring_forces`
- 数值积分：`velocity_verlet_step`
- 初值构造：`initialize_state`
- 主流程：`run_simulation`
- 入口：`main`（打印样本表 + 汇总指标 + sanity check）

## R09

默认参数（兼顾速度与可读性）：
- `n_particles = 32`
- `target_total_energy = 32.0`
- `dt = 0.02`
- `n_steps = 8000`
- `sample_every = 160`
- `seed = 20260407`

该配置在普通 CPU 上可秒级运行，并能稳定展示 NVE 能量守恒与均分趋势。

## R10

伪代码：

```text
config <- defaults
(x, v) <- initialize_state(config)
F <- spring_forces(x)

for step in [0 .. n_steps]:
    if step is sampled:
        record K(v), U(x), E=K+U, T_est=2K/(N-1)
    if step == n_steps:
        break
    (x, v, F) <- velocity_verlet_step(x, v, F)

postprocess records:
    energy drift, equipartition ratio, temperature error, velocity moments
print table and summary
```

## R11

可复现性：
- 使用固定随机种子；
- 所有初值由同一 `numpy.random.default_rng(seed)` 生成；
- 无交互输入、无外部文件依赖；
- 每次运行应给出同构的统计趋势。

## R12

输出字段解释：
- `step`：采样时刻步数；
- `kinetic` / `potential`：样本时刻动能与势能；
- `total_energy`：应围绕初值小幅波动；
- `temperature_est`：按 `T = 2K/(N-1)` 的瞬时估计。

摘要指标：
- `max_abs_relative_drift` 越小越符合 NVE；
- `equipartition_ratio_K_over_U` 接近 1 表示谐系统均分行为成立；
- `temperature_relative_error` 反映动能温度与能量尺度的一致性。

## R13

数值注意事项：
- `dt` 过大会导致显著能量漂移；
- 总步数太短会让时间平均不稳定；
- 谐链是可积/近可积系统，统计混合速度可能受模态结构影响；
- 本例是教学级 MVP，不等同于大规模分子动力学生产计算。

## R14

与替代方案对比：
- 当前：velocity-Verlet
  - 优点：简单、显式、能量长期稳定性好；
  - 缺点：仍有离散化误差，`dt` 需调参。
- 显式欧拉
  - 优点：实现最短；
  - 缺点：能量漂移明显，不适合 NVE。
- 高阶辛方法（如 Yoshida）
  - 优点：精度更高；
  - 缺点：实现复杂度提高，不利于最小示例。

## R15

可扩展方向：
1. 升级到二维/三维 Lennard-Jones 粒子体系；
2. 引入径向分布函数 `g(r)`、速度自相关函数等观测量；
3. 多组 `dt` 做能量漂移收敛实验；
4. 增加约束算法（如 SHAKE）与更真实势函数；
5. 对比 NVE 与 NVT（加入 thermostat）下的统计差异。

## R16

运行方式（无交互）：

```bash
uv run python Algorithms/物理-统计力学-0339-微正则系综_(NVE_Ensemble)/demo.py
```

若当前目录已在该算法目录下：

```bash
uv run python demo.py
```

## R17

交付核对：
- `README.md`：`R01-R18` 全部填充完毕；
- `demo.py`：可直接运行，输出 NVE 采样表与摘要；
- `meta.json`：任务元信息保持一致；
- 目录自包含，不依赖外部交互，不修改共享文件。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `NVEConfig` 定义 `N, m, k, E_target, dt, n_steps, sample_every, seed`。  
2. `initialize_state` 生成随机 `x, v`，分别去除均值并按目标 `U`/`K` 比例缩放，使初始 `K+U=E_target`。  
3. `spring_forces` 按离散拉普拉斯形式计算每个粒子的弹簧力。  
4. `run_simulation` 进入时间循环，每逢采样点调用 `kinetic_energy` 与 `potential_energy` 记录 `K/U/E/T`。  
5. 循环内每一步调用 `velocity_verlet_step` 完成“位置半步-新力-速度半步”的辛更新。  
6. 演化结束后计算 `relative_drift = (E - E0)/|E0|`，提取能量守恒指标（最大漂移、标准差）。  
7. 用样本均值给出 `⟨K⟩/⟨U⟩` 与 `T_est`，并与 `T_theory = E0/(N-1)` 对比得到相对误差。  
8. 汇总速度样本计算均值/方差/超额峰度，在 `main` 中打印全表与 `PASS/CHECK_MANUALLY` 结论。  

实现未调用第三方 NVE 黑盒：哈密顿量、力、积分器、统计量均在源码中逐步显式展开。

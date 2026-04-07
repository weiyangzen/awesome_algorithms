# 多铁性材料 (Multiferroics)

- UID: `PHYS-0276`
- 学科: `物理`
- 分类: `凝聚态物理`
- 源序号: `279`
- 目标目录: `Algorithms/物理-凝聚态物理-0279-多铁性材料_(Multiferroics)`

## R01

本条目把“多铁性材料”实现为一个可运行的最小算法闭环：

- 用耦合 Landau 自由能模型描述铁电极化 `P` 与铁磁磁化 `M` 的共同演化；
- 通过 SciPy 求解平衡态 `(P*, M*)`，并生成带噪声的合成测量数据；
- 通过 `scikit-learn` 估计磁电耦合线性响应系数；
- 通过 PyTorch 反演耦合参数 `gamma`，验证“正向模拟 -> 逆向识别”一致性。

## R02

多铁性（multiferroics）核心点在于：同一材料中至少两类序参量（常见为极化与磁化）可共存且可耦合。

在算法视角下，关键问题是：

1. 如何给出一个透明、可求解、可校验的耦合模型；
2. 如何把该模型转成“从实验可观测量到参数识别”的最小数据链路；
3. 如何在脚本中量化耦合强度与响应质量，而不是只做概念说明。

## R03

`demo.py` 中的物理模型采用双序参量 Landau 展开：

`F(P,M) = 1/2 a_p (T-Tc_p) P^2 + 1/4 b_p P^4 + 1/2 a_m (T-Tc_m) M^2 + 1/4 b_m M^4 - gamma P M - E P - H M`

平衡条件来自驻点方程：

- `dF/dP = a_p (T-Tc_p) P + b_p P^3 - gamma M - E = 0`
- `dF/dM = a_m (T-Tc_m) M + b_m M^3 - gamma P - H = 0`

其中 `E/H` 是外电场与外磁场，`gamma` 是待识别的磁电耦合强度。

## R04

本 MVP 的简化假设：

- 空间均匀（0D）序参量，不解偏微分场方程；
- 只保留到四阶项，忽略高阶各向异性项；
- 只拟合单一耦合系数 `gamma`，其它参数视为已知；
- 用合成数据演示算法闭环，不直接对接真实实验噪声谱。

这些假设保证脚本短小透明，适合基础验证与教学。

## R05

实现层对应函数：

- `free_energy`：显式计算 `F(P,M;T,E,H)`；
- `stationarity_residual`：构造驻点方程残差向量；
- `solve_equilibrium_scipy`：先 `scipy.optimize.root`，失败后 `minimize` 回退；
- `generate_synthetic_dataset`：在温度网格上做 `H` 扫描和 `E` 扫描并加噪；
- `estimate_me_slopes`：用线性回归提取 `alpha_me=dP/dH`, `beta_me=dM/dE`；
- `torch_newton_equilibrium`：可微牛顿迭代求平衡态；
- `fit_gamma_torch`：Adam 训练反演 `gamma`。

## R06

端到端流程：

1. 固定 Landau 真值参数与仿真配置；
2. 生成 `T x field` 网格上的平衡态真值；
3. 叠加高斯噪声形成观测 `P_obs/M_obs`；
4. 按温度分组做线性拟合，得到 `alpha_me(T)` 和 `beta_me(T)`；
5. 在近零偏置下扫温度，标注 `FE/FM/multiferroic` 区间；
6. 以观测数据为监督信号，用 PyTorch 反演 `gamma`；
7. 汇总关键指标并执行断言门槛。

## R07

复杂度估计（记温度点 `N_T`，场点 `N_F`，Torch 迭代 `E`，牛顿步 `K`）：

- SciPy 平衡求解样本数：`2 * N_T * N_F`；
- 每个样本求解成本近似常数（小维 2x2），总体约 `O(N_T N_F)`；
- 回归成本：`O(N_T N_F)`；
- Torch 前向+反向：`O(E * K * N_T N_F)`。

默认参数（`N_T=12`, `N_F=9`, `E=250`, `K=35`）在本地可秒级完成。

## R08

数值稳定与工程策略：

- 平衡求解先用 `root`，再用 `minimize` 兜底，减少局部失败；
- 多初值候选后按自由能最小值选物理解；
- `E/H` 扫描围绕正偏置，避免序参量分支频繁翻转；
- Torch 中 `gamma = softplus(raw)+eps` 强制非负；
- 牛顿迭代中对 `det(J)` 做小阈值保护，防止除零。

## R09

适用场景：

- 多铁性入门课程中的“模型到数据再到参数反演”示例；
- 研究前期验证磁电耦合估计算法可行性；
- 对后续高维模型（相场/第一性原理代理）做原型对照。

不适用场景：

- 需要真实晶体对称性、畴壁动力学、应力耦合的高保真研究；
- 直接替代实验拟合软件进行发表级参数推断。

## R10

脚本内置质量门槛：

1. 样本量一致性：`n_samples == n_temp * n_field * 2`；
2. 驻点残差均值：`mean_residual < 2e-6`；
3. 线性响应回归质量：`mean_alpha_r2 > 0.95`, `mean_beta_r2 > 0.95`；
4. 耦合参数反演：`|gamma_fit - gamma_true| < 0.02`；
5. Torch 拟合误差：`final_loss < 2e-4`；
6. 多铁性温区至少包含 15 个温度采样点。

## R11

默认参数（`LandauParams`, `SimulationConfig`）：

- `a_p=0.018, b_p=1.0, Tc_p=360`
- `a_m=0.030, b_m=1.1, Tc_m=280`
- `gamma_true=0.22`
- 温度：`250 -> 380`，`12` 点
- 偏置场：`E0=0.09, H0=0.09`
- 扫描增量：`delta in [-0.08, 0.08]`，`9` 点
- 噪声：`sigma=5e-4`
- Torch：`epochs=250, lr=0.05, newton_steps=35`

## R12

本地运行（`uv run python demo.py`）结果：

- `n_samples = 216`
- `mean_stationarity_residual = 0.00000000`
- `mean_alpha_r2 = 0.97649958`
- `mean_beta_r2 = 0.95318341`
- `gamma_true = 0.22000000`
- `gamma_fit = 0.21990135`
- `gamma_abs_error = 0.00009865`
- `torch_final_loss = 0.00000043`

说明该 MVP 在当前参数下可稳定恢复耦合强度，且线性响应拟合质量可接受。

## R13

结果解释：

- `gamma` 反演误差约 `1e-4`，说明可微求解器与合成数据机制一致；
- `alpha_me` 与 `beta_me` 的 `R^2` 在低温端略低、在中高温端更高，体现了非线性背景强弱随温度变化；
- 相图尾部显示 `T` 接近 `Tc_p/Tc_m` 后序参量快速衰减，多铁性区间收缩。

## R14

常见失败模式与修复：

- 平衡求解失败：扩大初值集合或提高回退优化容错；
- 拟合 `R^2` 过低：减小噪声或增大场扫描窗口；
- `gamma` 反演偏移：增加 Torch 迭代轮数、调小学习率；
- 牛顿不稳：提高阻尼、加大 `det` 保护阈值。

## R15

工程化建议：

- 保留“真值列 + 观测列”双轨数据，方便回归测试；
- 每次改模型项都同步更新断言阈值，避免静默退化；
- 对真实数据接入前，先做参数敏感性扫描（噪声、场幅、温度窗）；
- 若要批量实验，建议把 `generate/fit/report` 拆成独立模块。

## R16

可扩展方向：

- 加入应变 `u` 与磁弹/电弹耦合项；
- 扩展到时域动力学（Landau-Khalatnikov）研究开关过程；
- 用贝叶斯反演替代点估计，给出 `gamma` 置信区间；
- 引入空间项 `|grad P|^2, |grad M|^2` 做相场模拟。

## R17

交付清单：

- `demo.py`：可运行的多铁性 MVP（`numpy + scipy + pandas + scikit-learn + torch`）；
- `README.md`：R01-R18 全部完成；
- `meta.json`：任务元数据与目录信息保持一致。

运行方式：

```bash
cd Algorithms/物理-凝聚态物理-0279-多铁性材料_(Multiferroics)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流（8 步）：

1. `LandauParams/SimulationConfig` 定义物理参数与数值超参数。  
2. `stationarity_residual` 把 `dF/dP=0, dF/dM=0` 显式写成二维非线性方程。  
3. `solve_equilibrium_scipy` 对每个 `(T,E,H)` 先 root 求根，再最小化回退，并以最低自由能解为最终平衡态。  
4. `generate_synthetic_dataset` 构造两类扫描（`H` 扫描和 `E` 扫描），输出 `P_true/M_true` 与带噪 `P_obs/M_obs`。  
5. `estimate_me_slopes` 对每个温度分组调用 `LinearRegression`，分别提取 `alpha_me=dP/dH` 与 `beta_me=dM/dE`。  
6. `torch_newton_equilibrium` 将同一驻点方程写成可微牛顿迭代，显式构造雅可比矩阵并迭代更新 `(P,M)`。  
7. `fit_gamma_torch` 仅把 `gamma` 设为可训练参数，用 Adam 最小化 `P/M` 观测误差，得到 `gamma_fit`。  
8. `main` 汇总指标、打印表头，并通过断言门槛完成自动验收。

第三方库仅负责数值求解与优化器；模型方程、残差、雅可比、流程拆解均在源码中显式实现，非黑盒调用。

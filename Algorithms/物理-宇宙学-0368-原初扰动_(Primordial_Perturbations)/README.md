# 原初扰动 (Primordial Perturbations)

- UID: `PHYS-0350`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `368`
- 目标目录: `Algorithms/物理-宇宙学-0368-原初扰动_(Primordial_Perturbations)`

## R01

原初扰动通常指暴胀阶段产生的早期小振幅涨落，可用曲率扰动场 `zeta(x)` 描述。其统计性质（功率谱斜率、是否高斯）直接影响后续 CMB 各向异性与大尺度结构。该 MVP 聚焦两个最小可验证量：

1. 标量谱倾斜 `n_s` 的恢复；
2. 局域型非高斯参数 `f_NL` 的近似恢复。

## R02

本条目的实现目标是构建可复现、无交互、可审计的最小闭环：

1. 合成三维原初高斯场 `zeta_g`；
2. 以局域型模型注入非高斯性 `zeta = zeta_g + f_NL(zeta_g^2-<zeta_g^2>)`；
3. 用壳平均估计各向同性 `P_zeta(k)`；
4. 用 `scikit-learn` 与 `PyTorch` 分别估计谱倾斜；
5. 用三阶矩估计 `f_NL` 并与高斯参考场对照。

## R03

MVP 的物理与数值假设：

- 统计均匀、统计各向同性；
- 周期立方盒边界；
- 原初谱模型采用简化幂律加指数截断；
- 非高斯性只考虑局域型单参数 `f_NL`；
- 不含辐射转移函数、再组合细节、观测窗口效应。

## R04

`demo.py` 对应的核心公式：

1. 原初谱模型：
`P_zeta(k)=A_s*(k/k0)^(n_s-1)*exp(-(k/k_cut)^2)`

2. 高斯场着色：
`zeta_g(k)=white(k)*sqrt(P_zeta(k))`

3. 局域型非高斯注入：
`zeta(x)=zeta_g(x)+f_NL*(zeta_g(x)^2-<zeta_g^2>)`

4. 离散功率谱估计：
`P_hat(k_grid)=(L^3/N^6)*|FFT[zeta(x)]|^2`

5. 截断校正后的对数线性拟合：
`log P_hat + (k/k_cut)^2 = log A_eff + (n_s-1)log(k/k0)`

6. 三阶矩估计器（弱非高斯近似）：
`f_NL_hat ~= <zeta^3> / (6*Var(zeta)^2)`

## R05

算法流程（MVP）：

1. 构建三维 `|k|` 网格；
2. 生成白噪声并按目标 `P_zeta(k)` 着色，得到 `zeta_g`；
3. 把 `zeta_g` 缩放到目标方差，便于控制非高斯强度；
4. 注入局域型 `f_NL` 得到 `zeta`；
5. 对 `zeta` 的 Fourier 模式做壳平均得到一维 `P_hat(k)`；
6. 用加权线性回归估计 `n_s`；
7. 用 PyTorch 梯度下降独立拟合 `A_s,n_s`；
8. 计算 `f_NL_hat`、偏度、正态性检验并输出结果表。

## R06

脚本输出包含：

1. 全局摘要：网格、盒尺度、种子、真值 `n_s/f_NL`；
2. 谱拟合汇总表：`sklearn` 与 `torch` 的 `n_s_hat`、`A_eff_hat`、`log_rmse`；
3. 非高斯诊断表：`f_NL_hat`、`skewness`、`normaltest_p`；
4. 若干 `k` 采样点上的 `Pk`、模型值和比值。

同时脚本内置断言，若关键恢复指标异常会直接报错。

## R07

优点：

- 公式链路与代码实现一一对应；
- 同时覆盖二点统计（谱）和三点统计代理（三阶矩）；
- 固定随机种子，便于回归测试。

局限：

- `f_NL_hat` 使用弱非高斯近似，不替代完整双谱估计；
- 没有引入 CMB/LSS 实测系统误差；
- 非高斯注入仅演示局域型，不含等边型/正交型模板。

## R08

前置知识：

- 傅里叶变换与功率谱定义；
- 高斯随机场与壳平均；
- 偏度与三阶矩的统计意义。

运行依赖：

- Python `>=3.10`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `torch`

## R09

适用场景：

- 原初扰动教学中的最小数值演示；
- 非高斯注入与检测的单元测试样例；
- 后续复杂参数反演管线前的基础验证。

不适用场景：

- 精密宇宙学参数约束；
- 真实巡天数据生产级处理；
- 需要完整双谱模板拟合的研究级分析。

## R10

正确性直觉：

1. 如果谱着色正确，`P_hat(k)` 的形状应与输入模型一致；
2. 对 `log P` 做截断项校正后，斜率应恢复为 `n_s-1`；
3. 非高斯场应比高斯参考场有更高偏度与更显著正态性偏离；
4. 用三阶矩估计得到的 `f_NL_hat` 应接近注入真值。

## R11

数值稳定性策略：

- `k=0` 模式不参与谱计算和拟合；
- 分箱要求最小模式数，避免小样本波动主导；
- 低 `k` 拟合窗口避开高 `k` 截断段；
- 对数拟合只在 `P>0` 模式上进行；
- PyTorch 使用 `float64` 以降低小参数拟合误差。

## R12

关键参数（`PrimordialConfig`）：

- `n_grid`, `box_size_mpc`：空间分辨率与盒尺度；
- `amplitude`, `spectral_index`, `k_pivot_mpc_inv`, `k_cut_mpc_inv`：原初谱模型参数；
- `f_nl_local`, `sigma_g_target`：非高斯注入强度与基准方差；
- `n_k_bins`：壳平均分箱数量；
- `torch_steps`, `torch_lr`：梯度拟合步数与学习率。

调参建议：

- 增大 `n_grid` 可改善谱估计平滑性；
- 若 `f_NL_hat` 偏差过大，可降低 `sigma_g_target` 使弱非高斯近似更稳；
- 若 torch 拟合波动，可增加 `torch_steps` 或降低 `torch_lr`。

## R13

保证类型说明：

- 近似比保证：N/A（非组合优化问题）；
- 概率保证：N/A（固定 seed 下流程确定性可复现）。

工程保证：

- 无交互输入，单命令运行；
- 输出结构稳定，适合自动化检查；
- 关键指标异常会触发断言而非静默通过。

## R14

常见失败模式：

1. 忽略 FFT 归一化导致谱幅度失真；
2. 拟合区间覆盖过高 `k` 导致 `n_s` 偏移；
3. `f_NL` 太大使弱非高斯近似失效；
4. 分箱过细导致每壳模式数太少、噪声放大；
5. 未固定随机种子导致回归测试不可复验。

## R15

可扩展方向：

1. 从三阶矩估计扩展到双谱模板拟合；
2. 添加多次 realization 的误差条与协方差估计；
3. 接入 CLASS/CAMB 的线性理论谱替代简化模型；
4. 增加张量扰动或等曲率扰动通道；
5. 与 CMB/LSS 观测前向模型联动。

## R16

相关主题：

- 暴胀模型与慢滚参数；
- 标量谱指数 `n_s` 与 running；
- 双谱与非高斯性模板（local/equilateral/orthogonal）；
- 原初扰动到 CMB 各向异性的转移函数；
- 大尺度结构初始条件生成。

## R17

运行方式（无交互）：

```bash
cd Algorithms/物理-宇宙学-0368-原初扰动_(Primordial_Perturbations)
uv run python demo.py
```

完成判据：

- `README.md` 的 `R01-R18` 已全部填充；
- `demo.py` 可直接运行并输出结果；
- `meta.json` 与任务元数据一致。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `main` 初始化 `PrimordialConfig`，明确 `A_s,n_s,f_NL`、网格和随机种子。  
2. `synthesize_primordial_field` 先生成白噪声，再调用 `primordial_power_spectrum` 逐点计算 `P_zeta(k)`，并用 `sqrt(P)` 对 Fourier 模式着色得到高斯场 `zeta_g`。  
3. 同一函数内把 `zeta_g` 缩放到 `sigma_g_target`，随后按 `zeta=zeta_g+f_NL(zeta_g^2-<zeta_g^2>)` 注入局域型非高斯修正。  
4. `estimate_isotropic_power_spectrum` 使用 `numpy.fft.fftn` 计算三维离散谱，按几何 `k` 壳进行分箱平均，得到一维 `P_hat(k)`。  
5. `fit_spectral_index_sklearn` 构建 `x=log(k/k0), y=log P + (k/k_cut)^2`，调用 `LinearRegression.fit(..., sample_weight=...)` 求解 `n_s` 与有效振幅 `A_eff`。  
6. `fit_spectral_index_torch` 不把优化器当黑箱：显式定义 `y_hat=lnA+(n_s-1)x-(k/k_cut)^2`，再用 `Adam` 最小化加权平方残差，得到独立参数估计。  
7. `estimate_local_fnl` 显式计算方差和三阶矩，按 `f_NL_hat=<zeta^3>/(6 Var(zeta)^2)` 得到近似估计，并用 `scipy.stats.skew/normaltest` 量化非高斯性。  
8. `main` 汇总 `pandas` 表格、执行断言（`n_s` 误差、回归质量、`f_NL` 偏差、Gaussian 对照）并打印可复验报告。  

第三方库分工：`numpy` 负责随机场与 FFT；`scipy` 负责统计检验；`scikit-learn` 负责加权线性回归；`torch` 负责显式目标函数的梯度优化；`pandas` 负责结构化输出。核心物理公式、归一化、分箱与判据均在源码中手写实现。

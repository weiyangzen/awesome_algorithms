# 因子化定理 (Factorization Theorem)

- UID: `PHYS-0401`
- 学科: `物理`
- 分类: `QCD`
- 源序号: `420`
- 目标目录: `Algorithms/物理-QCD-0420-因子化定理_(Factorization_Theorem)`

## R01

因子化定理（Factorization Theorem）在 QCD 中的核心表述是：
高能可观测量可写成“短程可微扰部分 × 长程非微扰部分”的卷积，并带有幂次抑制余项。

本条目给出一个可运行的最小 MVP：构造 toy PDF、部分子亮度卷积、硬散射核与 `1/Q^2` 幂修正，生成伪数据并反演系数，最后检查因子化尺度稳定性。

## R02

目标与范围：

1. 用显式公式实现 `sigma = H * L * (1 + c_np/Q^2)`；
2. 在 `(tau, Q)` 网格上生成伪观测并拟合 `(c0, c_np)`；
3. 比较 `mu_F = {Q/2, Q, 2Q}` 下的残余尺度依赖；
4. 给出和规则、拟合指标、Torch 一致性与自动微分检查。

不覆盖：

1. 真实 NLO/NNLO 系数函数与严格 DGLAP 卷积核；
2. 多味道阈值、喷注算法、实验系统协方差矩阵；
3. 生产级全局 PDF 拟合。

## R03

`demo.py` 中的数学模型：

1. 因子化观测量（toy 形式）
   - `sigma(tau,Q,mu_F) = H(Q,mu_F) * L(tau,mu_F) * (1 + c_np / Q^2)`
2. 亮度卷积
   - `L(tau,mu_F) = int_tau^1 [dx/x] f_q(x,mu_F) f_qbar(tau/x,mu_F)`
3. PDF 参数化与尺度演化
   - `f_i(x,mu_F)=N_i x^{a_i}(1-x)^{b_i} * exp(lambda_i ln(mu_F/mu0) (1-x))`
4. 动量归一化（解析 Beta 函数）
   - `N_i = m_i / B(a_i+2, b_i+1)`
5. 硬核
   - `H = 1 + alpha_s [ c0 + c1 ln(Q^2/mu_F^2) ]`
6. `c1` 的估计策略
   - 先数值估计 `d ln L / d ln mu_F` 的平均斜率，再令 `d ln H / d ln mu_F` 近似抵消该斜率。

## R04

物理直觉：

1. `L(tau,mu_F)` 承担长程结构（强子内部部分子分布）；
2. `H(Q,mu_F)` 承担短程硬散射信息；
3. 理想因子化下，`mu_F` 人为分割不应改变可观测量；
4. 有限阶截断后会残留尺度依赖，本脚本通过硬核对数项实现部分补偿；
5. `c_np/Q^2` 表征高 twist 或非微扰尾项，随能标增大衰减。

## R05

正确性关注点：

1. PDF 归一化使用解析 Beta 函数而非数值凑参数；
2. `momentum_sum_rule_diagnostics` 验证 `m_q + m_qbar + m_g = 1`；
3. 拟合残差按 `sigma_err` 加权，`chi2/dof` 可解释；
4. 对照 `c1=0` 与 `c1!=0` 的尺度变异，验证“补偿项有效”；
5. 用 PyTorch 复算前向与梯度，避免仅依赖单后端实现。

## R06

复杂度（`N_tau` 个 `tau` 点，`N_Q` 个 `Q` 点，积分网格 `K`）：

1. 单次亮度计算：`O(K)`；
2. 全网格亮度：`O(N_tau * N_Q * K)`；
3. 拟合维度 `P=2`，一次残差评估：`O(N_tau * N_Q)`（亮度已预计算）；
4. 迭代拟合总成本约 `O(I * N_tau * N_Q)`；
5. 额外尺度扫描 `3` 个比率，复杂度同阶。

默认配置下可在秒级完成。

## R07

主流程（与 `main` 一致）：

1. 校验配置合法性；
2. 构建缓存化亮度计算器；
3. 估计补偿系数 `c1`；
4. 生成伪数据集 `sigma_obs`；
5. 用 `least_squares` 拟合 `(c0, c_np)`；
6. 计算拟合指标与动量和规则；
7. 统计有/无硬核对数项时的尺度变异；
8. 做 Torch 前向一致性与梯度检查；
9. 执行断言并输出表格与摘要。

## R08

最小工具栈：

1. `numpy`：向量化与基础数值；
2. `scipy.integrate.simpson`：卷积积分；
3. `scipy.optimize.least_squares`：参数反演；
4. `scipy.special.beta`：解析归一化；
5. `pandas`：数据集与报告表；
6. `scikit-learn`：`RMSE` 与 `R^2`；
7. `torch`：前向一致性和自动微分验证。

## R09

核心函数接口：

1. `build_luminosity_getter(cfg) -> Callable[[tau, muF], float]`
2. `estimate_compensating_c1(cfg, get_luminosity, c0_ref) -> (c1, mean_slope)`
3. `build_synthetic_dataset(cfg, get_luminosity, c1) -> pd.DataFrame`
4. `sigma_factorized_from_luminosity(...) -> np.ndarray`
5. `residual_vector(theta, df, c1, cfg) -> np.ndarray`
6. `fit_factorized_coefficients(df, c1, cfg) -> FitSummary`
7. `momentum_sum_rule_diagnostics(cfg) -> dict`
8. `scale_variation_table(cfg, get_luminosity, c0, c_np, c1) -> pd.DataFrame`
9. `torch_consistency_and_gradients(df, c0, c_np, c1, cfg) -> dict`

## R10

验收策略（脚本内断言）：

1. `chi2/dof < 2.4`；
2. `R^2 > 0.97`；
3. 拟合参数接近真值：
   - `|c0_fit-c0_true| <= 30%`
   - `|c_np_fit-c_np_true| <= 40%`
4. 动量和规则误差 `|total-1| <= 3e-4` 且推断胶子动量为正；
5. 有补偿时平均尺度变异小于无补偿；
6. Torch 与 NumPy 前向最大差异 < `5e-11`，梯度范数为正。

## R11

边界与异常处理：

1. 配置检查：`tau`、`Q`、`mu0`、积分点数、尺度比率；
2. 卷积积分避开端点：`x in [tau+1e-6, 1-1e-6]`；
3. 残差中若参数导致幂修正项非物理（如 `1+c_np/Q^2` 过小），返回惩罚向量；
4. 对不合法预测值（`nan/inf`）同样做惩罚；
5. `sigma_err` 设下限，避免除零。

## R12

与标准 QCD 因子化工作的关系：

1. 保留了核心结构：卷积 `L` 与硬核 `H` 的分离；
2. 保留了“因子化尺度残留依赖需要高阶修正抵消”的思想；
3. 使用 `1/Q^2` 项体现幂次修正；
4. 省略真实 splitting kernel、高阶匹配系数和实验系统学，因此它是教学/工程样机，不是精密预测器。

## R13

默认数值配置：

1. `tau` 取 10 个点：`0.015` 到 `0.290`；
2. `Q` 取 5 个点：`20, 30, 50, 80, 120 GeV`；
3. 总数据点：`50`；
4. 积分网格：`400`；
5. `alpha_s = 0.22`，`mu0 = 2 GeV`；
6. 伪数据真值：`c0_true=1.35`, `c_np_true=18.0`；
7. 噪声模型：`sigma_err = 0.018*|sigma_true| + 8e-5`。

## R14

工程实现注意点：

1. 亮度计算带缓存，避免同一 `(tau,mu_F)` 重复积分；
2. 拟合时把亮度视为已知输入，降低优化成本；
3. 断言门槛写在代码中，防止“能跑但退化”；
4. 报告同时给出参数恢复、尺度变异、Torch 诊断；
5. 所有输出来自脚本内部配置，无需交互输入。

## R15

预期输出特征：

1. 打印估计得到的补偿系数 `c1`；
2. 输出拟合摘要（`c0_fit`, `c_np_fit`, `chi2/dof`, `RMSE`, `R^2`）；
3. 输出动量和规则诊断；
4. 输出有/无对数补偿时的尺度变异均值与最大值；
5. 输出 Torch 前向差异与梯度范数；
6. 打印部分样本行并以 `All checks passed.` 收尾。

## R16

可扩展方向：

1. 用离散卷积核替换 toy 演化项，接近 DGLAP；
2. 扩展到多 flavor 与重味阈值；
3. 将硬核拓展到 NLO/NNLO 系数函数；
4. 引入实验协方差矩阵与系统参数；
5. 用 Torch 端到端自动微分做联合拟合。

## R17

本条目交付：

1. `README.md`：R01-R18 完整说明；
2. `demo.py`：可直接运行的因子化定理最小 MVP；
3. `meta.json`：保持与任务元数据一致。

运行方式：

```bash
cd Algorithms/物理-QCD-0420-因子化定理_(Factorization_Theorem)
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，9 步）：

1. `validate_config` 对能标、网格、尺度比率和噪声参数做合法性检查。  
2. `build_luminosity_getter` 构建缓存函数；内部用 `simpson` 计算 `L(tau,mu_F)` 卷积。  
3. `estimate_compensating_c1` 在 `mu_F=Q*exp(±eps)` 处做有限差分，估计 `d ln L/d ln mu_F`，并反推补偿 `c1`。  
4. `build_synthetic_dataset` 用真值 `(c0_true,c_np_true)` 生成 `sigma_true`，叠加噪声得到 `sigma_obs`。  
5. `residual_vector` 定义加权残差 `(pred-obs)/sigma_err`，`fit_factorized_coefficients` 用 `least_squares` 拟合 `(c0,c_np)`。  
6. `momentum_sum_rule_diagnostics` 在 `mu0` 上积分检查 `m_q+m_qbar+m_g=1`。  
7. `scale_variation_table` 分别计算 `c1!=0` 与 `c1=0` 的 `mu_F` 变分，量化残余尺度依赖。  
8. `torch_consistency_and_gradients` 用 Torch 复算前向并反向传播，输出前向差与梯度范数。  
9. `run_quality_checks` 汇总所有门槛，全部通过后在 `main` 打印报告并输出 `All checks passed.`。  

说明：第三方库仅承担基础数值积分/优化/指标计算；因子化主干（PDF 参数化、卷积亮度、硬核补偿、尺度变异诊断）均在源码里显式展开，没有把物理流程交给黑箱 API。

# 布赖特-维格纳公式 (Breit-Wigner Formula)

- UID: `PHYS-0229`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `230`
- 目标目录: `Algorithms/物理-量子力学-0230-布赖特-维格纳公式_(Breit-Wigner_Formula)`

## R01

布赖特-维格纳公式用于描述散射截面在单一不稳定共振态附近的能量分布。常见 Lorentz 形式为：

`sigma(E) = sigma_bg + A * (Gamma/2)^2 / ((E - E_r)^2 + (Gamma/2)^2)`

其中：

- `E_r` 是共振能量（峰中心）
- `Gamma` 是共振宽度（与寿命成反比）
- `A` 是峰强度系数
- `sigma_bg` 是平滑背景项

## R02

典型应用场景：

- 核反应与高能散射中的截面峰拟合
- 原子/分子共振谱线参数提取
- 从实验数据反推共振质量（`E_r`）和寿命相关量（`Gamma`）
- 对比理论散射振幅与观测谱线形状

## R03

本条目采用“可拟合”的最小数学模型：

1. 数据层：给定离散能量点 `E_i` 与观测值 `y_i`。
2. 结构层：`y_i = sigma(E_i; theta) + epsilon_i`。
3. 参数向量：`theta = [E_r, Gamma, A, sigma_bg]`。
4. 目标函数（加权最小二乘）：
   `min_theta sum_i ((sigma(E_i;theta)-y_i)/s_i)^2`，其中 `s_i` 为观测标准差估计。

`demo.py` 中使用 `scipy.optimize.least_squares` 求解该非线性反问题。

## R04

直观理解：

- 共振态像“短寿命中间态”，在 `E_r` 附近显著增强散射概率。
- `Gamma` 越大，峰越宽，表示态寿命越短（`tau ~ hbar/Gamma`）。
- 远离峰中心时，Lorentz 项按 `1/(E-E_r)^2` 衰减，背景项主导。
- 拟合的本质是把“峰位置、宽度、强度、背景”从噪声数据中拆分出来。

## R05

正确性与可检验性质：

1. 半高全宽性质：当 `E = E_r ± Gamma/2` 时，峰项降到最大值的一半，因此 FWHM 等于 `Gamma`。
2. 非负性：当 `A >= 0` 且 `Gamma > 0` 时，峰项非负。
3. 峰值位置：模型最大值发生在 `E = E_r`。
4. 参数可解释：`E_r` 控制中心、`Gamma` 控制宽度、`A` 控制峰高、`sigma_bg` 控制基线。
5. 数值层面：若优化收敛且残差接近噪声规模，说明模型与数据匹配合理。

## R06

复杂度分析（`n` 为采样点数，`k` 为优化迭代轮数）：

- 单次模型评估：`O(n)`
- 单次解析 Jacobian 评估：`O(n)`
- 每轮迭代构造与求解局部子问题（4 维参数）：约 `O(n * p + p^3)`，`p=4`
- 总体复杂度近似 `O(k*n)`（常数项较小）

因此该 MVP 能快速处理几百到几千点谱线数据。

## R07

标准实现流程：

1. 准备 `E` 与观测 `y`，并估计噪声尺度 `sigma`。
2. 构造初值：峰位用 `argmax(y)`，背景用分位数估计。
3. 定义 Breit-Wigner 模型函数。
4. 定义加权残差向量 `r(theta)`。
5. 推导并实现解析 Jacobian `dr/dtheta`。
6. 设定参数边界（尤其 `Gamma > 0`, `A > 0`）。
7. 用 `least_squares(method="trf")` 迭代求解。
8. 输出参数、误差指标和样本点拟合对比。

## R08

`demo.py` 的 MVP 设计：

- 依赖：`numpy`、`scipy`、`pandas`、`scikit-learn`
- 数据：脚本内生成带噪声的合成共振数据（固定随机种子，可复现）
- 求解：带边界的加权非线性最小二乘（解析 Jacobian）
- 评估：`RMSE`、加权 `RMSE`、`R2`、参数回收误差
- 运行方式：`uv run python demo.py`（无交互输入）

## R09

`demo.py` 核心接口：

- `breit_wigner_cross_section(energy, e_r, gamma, amplitude, background)`
- `build_synthetic_dataset(rng, n_points=260)`
- `residual_vector(theta, energy, observed, sigma)`
- `residual_jacobian(theta, energy, sigma)`
- `initial_guess(energy, observed)`
- `fit_breit_wigner(energy, observed, sigma)`
- `print_report(...)`
- `main()`

## R10

测试策略：

- 可运行性测试：`uv run python demo.py` 直接完成全流程。
- 回归测试：已知真值参数下，拟合参数应接近真值。
- 指标测试：`R2` 应显著高于 0，`RMSE` 应与噪声量级一致。
- 稳健性测试：不同随机种子下参数估计应保持同量级稳定。
- 边界测试：极小 `Gamma`、低信噪比时检查失败信息是否清晰。

## R11

边界条件与异常处理：

- `Gamma` 通过边界约束为正，避免除零和非物理宽度。
- 振幅 `A` 约束为正，避免与背景项发生退化符号抵消。
- 若优化器未收敛，抛出 `RuntimeError` 并保留错误消息。
- 若 `J^T J` 近奇异，标准误差回退为 `NaN`，但主拟合结果仍可报告。

## R12

与相关谱线模型关系：

- 与纯 Lorentz 线型：Breit-Wigner 是其共振散射语境下的物理解读。
- 与高斯线型：高斯多用于仪器展宽主导；Breit-Wigner 强调寿命展宽。
- 与 Voigt 线型：Voigt 是 Lorentz 与 Gaussian 卷积，适合同时存在两类展宽。
- 与 Fano 线型：Fano 处理连续态-离散态干涉导致的不对称峰。

## R13

示例参数设置（`demo.py`）：

- 能量范围：`E in [0.6, 1.4]`
- 真值：`E_r=1.01`, `Gamma=0.085`, `A=8.5`, `sigma_bg=0.45`
- 采样点：`n=260`
- 噪声：`sigma_i = 0.05 + 0.015*sqrt(max(clean_i,0))`
- 随机种子：`20260407`

这组参数能稳定生成“单峰 + 噪声”的可拟合数据。

## R14

工程实现注意点：

- 用解析 Jacobian 替代差分 Jacobian，可减少函数调用并提升收敛稳定性。
- 采用加权残差，避免高截面区主导全部损失。
- 初值非常重要：峰位初值直接影响是否收敛到正确局部极小。
- 通过参数边界把优化域限制在物理可解释区域。

## R15

最小示例输出解读：

- `Parameter recovery` 表显示真值、拟合值和绝对误差。
- `stderr_approx` 来源于局部线性化协方差近似，只作量级参考。
- `Sample predictions` 给出若干能量点的观测与拟合对照。
- 当 `R2` 高且参数误差小，表示 Breit-Wigner 对该数据是有效近似。

## R16

可扩展方向：

- 多共振峰叠加：`sum_j BW_j(E)`
- 能量依赖宽度：`Gamma(E)` 替代常数宽度
- 引入干涉项，升级到复振幅拟合而非仅拟合截面
- 与仪器响应卷积，构造 Voigt/Breit-Wigner 混合模型
- 将拟合迁移到贝叶斯推断，给出更稳健的不确定性区间

## R17

本条目交付内容：

- `README.md`：完成 R01-R18，覆盖理论、实现、测试与工程说明
- `demo.py`：可直接运行的 Breit-Wigner 参数拟合 MVP
- `meta.json`：保持与任务元数据一致（UID/学科/分类/源序号/目录）

运行命令：

```bash
uv run python Algorithms/物理-量子力学-0230-布赖特-维格纳公式_(Breit-Wigner_Formula)/demo.py
```

或在目录内：

```bash
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，8 步）：

1. **生成可控数据**
   `build_synthetic_dataset` 先按真值参数生成 `clean` 曲线，再加入异方差高斯噪声，得到 `observed` 与 `sigma`。

2. **构造可解释初值**
   `initial_guess` 用 `argmax(observed)` 估计 `E_r`，用低分位数估计背景，再给出宽度与振幅初值。

3. **定义前向模型**
   `breit_wigner_cross_section` 计算
   `background + amplitude * (Gamma/2)^2 / ((E-E_r)^2 + (Gamma/2)^2)`。

4. **定义优化目标与解析导数**
   `residual_vector` 返回加权残差 `(model-observed)/sigma`；
   `residual_jacobian` 显式写出对 `E_r/Gamma/A/background` 的偏导并加权。

5. **进入 SciPy `least_squares` 主循环（TRF）**
   在 `fit_breit_wigner` 中调用 `least_squares(method="trf")`：
   每轮用当前参数计算残差和 Jacobian，并施加边界约束。

6. **TRF 子问题求解（非黑箱解释）**
   SciPy 在每轮构造局部线性近似 `r(theta + p) ≈ r + Jp`，
   在信赖域与边界反射机制下求步长 `p`（近似 Gauss-Newton/正则化子问题），
   再根据目标下降情况调整信赖域半径并更新参数。

7. **收敛后统计不确定性**
   取最终 `J` 构造 `J^T J`，按 `cov ≈ s^2 (J^T J)^(-1)` 估计参数标准误差 `stderr_approx`。

8. **结果验收与报告**
   `print_report` 计算 `RMSE`、加权 `RMSE`、`R2`，并输出参数回收表与采样点残差，形成可审计的最小证据链。

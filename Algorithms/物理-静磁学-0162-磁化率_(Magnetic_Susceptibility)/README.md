# 磁化率 (Magnetic Susceptibility)

- UID: `PHYS-0161`
- 学科: `物理`
- 分类: `静磁学`
- 源序号: `162`
- 目标目录: `Algorithms/物理-静磁学-0162-磁化率_(Magnetic_Susceptibility)`

## R01

磁化率 `χ` 用于描述材料在外加磁场 `H` 下产生磁化强度 `M` 的能力。在各向同性、线性静磁近似中：

\[
M=\chi H.
\]

本条目给出一个最小可运行 MVP：用合成测量数据 `(H, M)` 回归估计 `χ`，并输出 `μ_r=1+χ` 与材料类型的简化判断。

## R02

关键方程（SI 制）：

\[
M=\chi H + M_0,\quad B=\mu_0(H+M),\quad \mu_r=1+\chi.
\]

- `M0`：小偏置项（仪器零点偏差、背景项）。
- `μ0 = 4π×10^{-7} H/m`：真空磁导率。
- 当 `χ>0` 通常为顺磁响应，`χ<0` 通常为抗磁响应；铁磁材料在小信号线性区可用“有效 χ”近似。

## R03

MVP 问题设定：

- 已知一组离散测量点 `(H_i, M_i)`；
- 假设数据位于线性磁化区间；
- 目标估计 `χ` 与 `M0`；
- 结果用多种实现交叉验证：`numpy`、`scipy`、`scikit-learn`，并提供可选 `PyTorch` 梯度法对照。

## R04

算法选择为一元线性回归（带截距）：

\[
\min_{\chi,M_0}\sum_i\left(M_i-(\chi H_i+M_0)\right)^2.
\]

该问题是标准最小二乘，可直接通过线性代数闭式求解，也可用统计回归接口或梯度下降近似求解。

## R05

数值与物理前提：

- 仅针对线性区，未覆盖饱和磁化与磁滞回线；
- `H` 样本应覆盖足够动态范围，否则 `χ` 与 `M0` 易混叠；
- 噪声默认加在 `M` 上，近似高斯白噪声；
- 本 MVP 侧重参数估计与可复现实验流程，不做复杂材料本构识别。

## R06

数据生成（`demo.py` 内置）：

- `H` 在 `[−2e5, 2e5] A/m` 线性采样；
- 真实参数：`χ_true=3.2e-3`，`M0_true=12 A/m`；
- 在 `M` 上叠加高斯噪声（`noise_std=8 A/m`）；
- 同时计算 `B=μ0(H+M)`，用于额外的 `B-H` 公式校验。

## R07

复杂度（样本数 `n`）：

- `numpy` 最小二乘：时间 `O(n)`（一元回归常数维），空间 `O(n)`；
- `scipy.linregress`：时间 `O(n)`，空间 `O(n)`；
- `sklearn.LinearRegression`：时间 `O(n)`，空间 `O(n)`；
- `torch` 训练：时间 `O(n * steps)`，空间 `O(n)`。

## R08

核心数据结构：

- `SusceptibilityConfig`：数据规模、真实参数、噪声、优化步数等；
- `pandas.DataFrame`：存储 `H_A_per_m`, `M_A_per_m`, `B_T`；
- `dict[str, dict[str, float]]`：每种估计器输出 `chi/m0/rmse/r2`；
- 汇总表 `DataFrame`：按 `|χ_est-χ_true|` 排序比较结果。

## R09

伪代码：

```text
input config
generate synthetic H, M, B
for each estimator in {numpy, scipy, sklearn, (optional torch)}:
    fit M = chi * H + M0
    compute rmse and r2
build summary table and rank by |chi_est - chi_true|
compute mu_r = 1 + best_chi
compute chi_from_b = mean(B/(mu0*H)-1) for |H|>1
print dataset preview and all metrics
```

## R10

默认参数（`demo.py`）：

- `n_samples = 80`
- `h_min = -2.0e5 A/m`
- `h_max = 2.0e5 A/m`
- `true_chi = 3.2e-3`
- `true_m0 = 12.0 A/m`
- `noise_std = 8.0 A/m`
- `torch_steps = 1200`, `torch_lr = 0.08`

## R11

脚本输出包括：

- 合成数据前 5 行；
- 各估计器的 `χ`、`M0`、`RMSE`、`R²`；
- 与真实 `χ_true` 的绝对/相对误差；
- 基于最优 `χ` 的材料类型简化判断；
- `μ_r=1+χ` 和 `B-H` 直接公式估计 `χ`。

## R12

`demo.py` 函数划分：

- `generate_synthetic_dataset`：构造 `H/M/B` 数据；
- `estimate_with_numpy`：闭式最小二乘；
- `estimate_with_scipy`：统计回归接口；
- `estimate_with_sklearn`：机器学习接口；
- `estimate_with_torch`：梯度下降拟合；
- `classify_material`：按 `χ` 符号与量级做粗分类；
- `format_summary_table`：整理结果；
- `main`：执行全流程并打印。

## R13

运行方式（在当前算法目录）：

```bash
uv run python demo.py
```

无需交互输入，直接打印结果。

## R14

常见错误与规避：

- 用超出线性区的数据回归，得到的 `χ` 物理意义变弱；
- `H` 取值过窄会导致斜率估计不稳定；
- 忽略偏置项 `M0` 可能把系统误差误判为 `χ`；
- 直接用 `B/(μ0H)-1` 时若 `H≈0` 会出现数值放大，故示例中做了 `|H|>1` 过滤。

## R15

最小验证策略：

1. 运行默认参数，确认 `numpy/scipy/sklearn` 的 `χ` 接近一致；
2. 提高 `n_samples`（如 200），观察估计方差下降；
3. 降低噪声到 `noise_std=2`，应看到误差进一步减小；
4. 把 `true_chi` 改为负值（如 `-1e-4`）验证抗磁分类分支。

## R16

适用范围与局限：

- 适用：线性静磁近似、参数估计教学、仪器标定流程原型；
- 局限：
  - 不建模磁滞、饱和、各向异性与温度依赖；
  - 不包含时变电磁耦合；
  - 真实实验中仍需误差模型、单位校准和不确定度传播分析。

## R17

可扩展方向：

- 引入分段线性或 `tanh` 饱和模型，覆盖更宽 `H` 区间；
- 对真实实验数据增加异常点鲁棒回归（Huber/RANSAC）；
- 增加贝叶斯估计输出 `χ` 的置信区间；
- 联合温度变量建立 Curie/Curie-Weiss 类型拟合流程。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 创建 `SusceptibilityConfig`，调用 `generate_synthetic_dataset` 生成 `H/M/B`。
2. `generate_synthetic_dataset` 按 `M=χ_true H+M0_true+noise` 生成观测值，并用 `B=μ0(H+M)` 计算磁感应强度。
3. `estimate_with_numpy` 构建设计矩阵 `[H, 1]`，调用 `np.linalg.lstsq` 解出 `(χ, M0)`，再计算 `RMSE/R²`。
4. `estimate_with_scipy` 调用 `stats.linregress` 取得 `slope/intercept`，并同样回算误差指标。
5. `estimate_with_sklearn` 调用 `LinearRegression.fit` 学得系数与截距，再计算预测质量。
6. `estimate_with_torch`（可选）把 `H,M` 转为张量，构建 `nn.Linear(1,1)`，用 `Adam` 迭代最小化 MSE，提取参数。
7. `format_summary_table` 汇总所有估计器，按 `|χ_est-χ_true|` 排序，得到最优估计。
8. `main` 进一步计算 `μ_r=1+χ_best` 与 `χ_from_b=mean(B/(μ0H)-1)`（过滤 `|H|<=1`），并打印完整结果。

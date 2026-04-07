# 正电子湮灭 (Positron Annihilation)

- UID: `PHYS-0458`
- 学科: `物理`
- 分类: `凝聚态物理`
- 源序号: `481`
- 目标目录: `Algorithms/物理-凝聚态物理-0481-正电子湮灭_(Positron_Annihilation)`

## R01

正电子湮灭是凝聚态材料表征中的关键探针：注入材料的正电子最终与电子湮灭，产生 `511 keV` 伽马光子。湮灭前的寿命和动量分布会携带缺陷、空位、孔隙等微观信息。

本条目给出一个最小可运行 MVP，聚焦 **正电子湮灭寿命谱（PALS）**：

- 用多指数衰减描述不同湮灭通道；
- 用高斯仪器响应函数模拟探测系统展宽；
- 用泊松噪声模拟计数统计；
- 用受约束的非线性最小二乘反演寿命与强度参数。

## R02

典型应用场景：

- 金属与半导体中的空位/位错缺陷评估
- 多孔材料中正电子素（positronium）寿命通道分析
- 聚合物自由体积定量（长寿命分量）
- 退火前后材料微观缺陷演化对比

## R03

本目录采用的数学模型：

1. 内禀湮灭率模型（`K` 分量）：
   `r(t) = sum_{k=1..K} I_k * exp(-t/tau_k) / tau_k`

2. 约束条件：
   `tau_k > 0`, `I_k >= 0`, `sum_k I_k = 1`

3. 仪器响应函数（IRF）采用高斯形式：
   `g(t) ~ exp(-(t-mu)^2 / (2 sigma^2))`

4. 探测到的谱形为卷积：
   `s(t) = (r * g)(t)`

5. 每个时间 bin 的期望计数：
   `lambda_i = N_signal * s_i + b`

6. 实际观测计数：
   `y_i ~ Poisson(lambda_i)`

## R04

物理直觉：

- 短寿命分量通常对应致密、电子密度高的湮灭环境；
- 长寿命分量通常指向空位团簇、孔洞或正电子素相关态；
- 强度 `I_k` 表示对应通道在总湮灭事件中的占比；
- 仪器时间分辨率（`sigma`）会把原本尖锐的衰减特征“抹平”，必须在拟合中显式建模。

## R05

正确性要点：

1. 通过 `softmax` 参数化强度，保证 `I_k` 非负且和为 1。
2. 通过 `exp` 参数化寿命和计数尺度，保证 `tau_k` 与背景等参数非负。
3. 卷积后做归一化，保证谱形是概率意义上的分布。
4. 目标函数采用 Pearson 风格加权残差，兼顾泊松计数方差随计数变化的特性。
5. 输出 `R2`、`MAE`、`reduced_chi2` 与参数误差，做可审计的拟合质量验证。

## R06

复杂度分析（`N` 个时间 bin，`K` 个寿命分量，迭代 `M` 次）：

- 单次模型评估：
  - 多指数构造：`O(KN)`
  - FFT 卷积：`O(N log N)`
- 单次残差评估约为：`O(KN + N log N)`
- 总体拟合开销约：`O(M * (KN + N log N))`
- 空间复杂度：`O(N)`（主要为谱数组与中间卷积结果）

在本 MVP（`K=3`, `N=450`）下，运行成本很低，重点在可解释性与稳健性。

## R07

算法流程（高层）：

1. 构造时间网格与真值参数（寿命、强度、IRF、计数尺度）。
2. 生成无噪声期望谱并加入泊松噪声得到观测谱。
3. 将拟合变量映射为受物理约束参数（`exp` + `softmax`）。
4. 构建加权残差函数。
5. 调用有界非线性最小二乘（TRF）迭代最小化。
6. 反解得到拟合寿命、强度、背景和信号总计数。
7. 计算并打印参数偏差与拟合质量指标。

## R08

`demo.py` MVP 设计选择：

- 依赖栈：`numpy + scipy + pandas + scikit-learn`
- 不依赖黑盒 PALS 专用包，核心模型和参数化全部显式实现
- 合成数据 + 回归反演同脚本完成，便于验证算法闭环
- 固定随机种子，确保复现性
- 无命令行参数、无交互输入，直接执行即可得到结果

## R09

`demo.py` 函数接口：

- `stable_softmax(logits) -> np.ndarray`
- `gaussian_irf(time_ns, sigma_ns, center_ns) -> np.ndarray`
- `intrinsic_annihilation_rate(time_ns, lifetimes_ns, intensities) -> np.ndarray`
- `convolved_shape(time_ns, lifetimes_ns, intensities, sigma_ns, center_ns) -> np.ndarray`
- `expected_counts(..., signal_counts, background_per_bin) -> np.ndarray`
- `simulate_spectrum(...) -> tuple[np.ndarray, np.ndarray]`
- `unpack_parameters(params, n_components) -> tuple[...]`
- `residual_vector(params, time_ns, observed_counts, ...) -> np.ndarray`
- `fit_lifetime_spectrum(...) -> dict[str, ...]`

## R10

测试策略：

- 形状与参数约束检查：输入维度、非负性、归一化条件
- 拟合可行性检查：优化器返回 `success=True`
- 数据拟合质量检查：`R2` 接近 1、`MAE` 合理、`reduced_chi2` 不异常
- 参数可恢复性检查：拟合寿命/强度接近合成真值（允许统计波动）

## R11

边界条件与异常处理：

- `sigma_ns <= 0`：拒绝并抛 `ValueError`
- 时间轴长度不足或非递增：抛 `ValueError`
- 寿命非正、强度非归一：抛 `ValueError`
- 观测计数存在负数：抛 `ValueError`
- 卷积归一化失败（极端异常输入）：抛 `RuntimeError`

## R12

与其他正电子谱技术的关系：

- 本实现是时间域寿命谱（PALS）路线；
- 另一类常见方法是多普勒展宽谱（DBS），侧重湮灭电子动量分布（`S/W` 参数）；
- 二者可互补：PALS 更敏感于寿命通道，DBS 更直接反映动量与化学环境。

## R13

`demo.py` 的示例参数：

- 时间网格：`N=450`, `dt=0.02 ns`（总窗约 `9 ns`）
- 真值寿命：`[0.125, 0.420, 1.950] ns`
- 真值强度：`[0.58, 0.30, 0.12]`
- 仪器响应：`sigma=0.080 ns`, `center=0.180 ns`
- 信号总计数：`180000`
- 平坦背景：`3 counts/bin`

这组参数能同时展示短/中/长寿命分量，且在噪声下仍可稳定回归。

## R14

工程实现注意点：

- 多指数分量存在“标签交换”现象，输出前按寿命排序便于解释
- 计数问题属于异方差噪声，残差加权比无权最小二乘更稳健
- 强背景或低计数会显著降低可辨识性，需要更长采集时间或先验约束
- 卷积使用 FFT 以降低计算量，适合后续扩展到更大时间窗

## R15

最小示例的解读方式：

- 若拟合结果中长寿命分量升高，通常意味着材料中更开放的缺陷体积增加；
- 若短寿命分量占比增大，通常意味着湮灭更偏向致密环境；
- `reduced_chi2` 可作为是否“过拟合/欠拟合”与噪声一致性的粗指标。

## R16

可扩展方向：

- 支持源项修正（source correction）和多背景项
- 支持已知 IRF 测量数据输入，而非固定高斯 IRF
- 引入模型选择（AIC/BIC）自动比较分量数 `K`
- 引入贝叶斯后验估计（MCMC）给出参数置信区间
- 扩展到温度/退火序列批处理分析

## R17

本条目交付说明：

- `README.md`：R01-R18 全部完成
- `demo.py`：包含合成、拟合、评估的可运行 MVP
- `meta.json`：保持与任务元信息一致

运行方式：

```bash
cd Algorithms/物理-凝聚态物理-0481-正电子湮灭_(Positron_Annihilation)
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，8 步）：

1. **参数物理化映射**
   `unpack_parameters` 把无界优化变量拆成寿命、强度、信号计数、背景，并用 `exp/softmax` 显式施加物理约束。

2. **构造内禀湮灭率**
   `intrinsic_annihilation_rate` 逐分量累加 `I_k * exp(-t/tau_k)/tau_k`，形成未展宽的时间域速率曲线。

3. **构造仪器响应并卷积**
   `gaussian_irf` 生成归一化高斯响应，`convolved_shape` 通过 `fftconvolve` 计算 `r*g`，得到探测器展宽后的谱形。

4. **计数域前向模型**
   `expected_counts` 把归一化谱形缩放为 `signal_counts * shape + background_per_bin`，得到每个 bin 的理论计数。

5. **噪声建模与样本生成**
   `simulate_spectrum` 对理论计数做泊松采样，生成观测谱，模拟真实计数统计涨落。

6. **残差与权重计算**
   `residual_vector` 计算 `(model-observed)/sqrt(max(observed,1))`，使高计数 bin 不会在优化中不成比例地主导。

7. **TRF 有界非线性最小二乘迭代**
   `fit_lifetime_spectrum` 调用 `scipy.optimize.least_squares(method="trf")`，在边界约束下迭代更新参数，内部通过雅可比近似和信赖域步长控制收敛。

8. **结果排序与质量审计输出**
   `main` 对拟合寿命排序后与真值对齐，输出组件级误差表和 `MAE/R2/reduced_chi2`，完成从物理模型到数值反演的闭环验证。

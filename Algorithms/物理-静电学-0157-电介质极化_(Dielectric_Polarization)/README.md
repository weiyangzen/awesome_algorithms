# 电介质极化 (Dielectric Polarization)

- UID: `PHYS-0156`
- 学科: `物理`
- 分类: `静电学`
- 源序号: `157`
- 目标目录: `Algorithms/物理-静电学-0157-电介质极化_(Dielectric_Polarization)`

## R01

本条目实现“静电场下线性各向同性电介质极化”的最小可运行版本（MVP），目标是把核心物理关系与可执行计算流程打通：
- 根据关系 `P = ε0 χe E` 生成或解释极化强度；
- 由含噪观测 `(E, P_measured)` 反推介电极化率 `χe`；
- 同时给出电位移 `D = ε0 E + P`，并输出拟合误差指标（RMSE、`R^2`）。

## R02

问题定义（MVP 范围）：
- 输入：
  - 静电场采样 `E_i`（单位 `V/m`）；
  - 对应极化观测 `P_i`（单位 `C/m^2`，可带测量噪声）。
- 输出：
  - 极化率估计 `χe_hat`；
  - 拟合极化 `P_fit`；
  - 电位移 `D_measured` 与 `D_fit`；
  - 拟合质量指标（RMSE、`R^2`、相对误差）。
- 约束：
  - 本实现假设材料线性、各向同性、静态，不处理频散/各向异性/强非线性饱和极化。

## R03

数学模型（SI 制）：
1. 线性介质本构关系：`P = ε0 χe E`。  
2. 相对介电常数：`εr = 1 + χe`。  
3. 电位移定义：`D = ε0 E + P = ε0 (1 + χe) E = ε E`。  
4. 若观测模型为 `P_measured = ε0 χe E + noise`，在“过原点线性回归”下有闭式估计：
`χe_hat = (E^T P_measured) / (ε0 E^T E)`。  
5. 拟合值 `P_fit = ε0 χe_hat E`，并据此计算残差与误差统计量。

## R04

算法流程（MVP）：
1. 生成（或读取）电场采样向量 `E`。  
2. 按真值 `χe_true` 计算 `P_true = ε0 χe_true E`。  
3. 叠加高斯噪声得到 `P_measured`（用于模拟实验误差）。  
4. 用最小二乘闭式公式估计 `χe_hat`。  
5. 计算 `P_fit`，再计算 `D_measured = ε0E + P_measured`、`D_fit = ε0E + P_fit`。  
6. 输出 `χe` 相对误差、RMSE、`R^2` 和样本表格。  
7. 执行阈值断言，确保模型恢复质量符合预期。

## R05

核心数据结构：
- `numpy.ndarray`：
  - `E`、`P_true`、`P_measured`、`P_fitted`、`D_measured`、`D_fitted`。
- `PolarizationResult`（`dataclass`）：
  - 保存全部数组和标量诊断信息：`chi_true`、`chi_est`、`rmse`、`r2`。
- `pandas.DataFrame`：
  - 仅用于终端友好预览（非算法依赖核心）。

## R06

正确性要点：
- 在假设 `P = ε0 χe E + noise` 下，目标是最小化平方误差 `||P_measured - ε0 χe E||^2`。
- 对标量参数 `χe` 求导并令导数为零，可得闭式解  
  `χe_hat = (E^T P_measured)/(ε0 E^T E)`，这正是 `demo.py` 的估计公式。
- 得到 `χe_hat` 后，`P_fit` 与 `D_fit` 均由定义直接构造，物理量纲一致。
- 在噪声较小且 `E` 覆盖充分时，`χe_hat` 应接近 `χe_true`，`R^2` 接近 1。

## R07

复杂度分析：
- 时间复杂度：`O(n)`，其中 `n` 是采样点数（点积、逐元素计算均线性）。
- 空间复杂度：`O(n)`，主要用于存储 `E/P/D` 向量。
- 该问题是单参数估计，计算开销远小于通用矩阵回归。

## R08

边界与异常处理：
- `chi_true <= 0` -> `ValueError`（本 MVP 限定普通介质）。
- `n_points < 5` -> `ValueError`（样本过少不稳定）。
- `e_min >= e_max` 或非有限值 -> `ValueError`。
- `noise_std < 0` -> `ValueError`。
- `E` 与 `P` 维度不一致、存在 `nan/inf` -> `ValueError`。
- `sum(E^2) == 0`（退化输入）-> `ValueError`。

## R09

MVP 取舍说明：
- 选择“可解释 + 可复现”的最小模型：一维电场、线性介质、闭式估计。
- 不调用 `scikit-learn` 回归器，避免黑盒；直接实现公式便于源码追踪。
- 不引入有限元网格、泊松方程边界值求解、分子极化微观模型等复杂层次。
- 先保证演示链路可靠，再考虑高阶扩展。

## R10

`demo.py` 模块职责：
- `generate_synthetic_dataset`：生成线性介质观测样本。
- `estimate_chi_linear`：实现过原点最小二乘估计 `χe`。
- `evaluate_fit`：计算 RMSE 和 `R^2`。
- `run_dielectric_polarization_mvp`：串联数据生成、参数估计、`D` 计算。
- `run_checks`：执行质量断言（`R^2`、相对误差、残差上界）。
- `preview_table`：生成终端可读的样本预览表。
- `main`：无交互入口，直接运行并输出报告。

## R11

运行方式：

```bash
cd Algorithms/物理-静电学-0157-电介质极化_(Dielectric_Polarization)
uv run python demo.py
```

脚本不需要交互输入，会自动完成数据生成、估计、检查并打印结果。

## R12

输出字段解读：
- `chi_true`：用于生成合成数据的真实极化率；
- `chi_est`：由观测反推得到的估计值；
- `chi_relative_error`：估计相对误差；
- `RMSE(P)`：极化拟合均方根误差；
- `R^2(P fit)`：拟合优度；
- `max|P_measured-P_fitted|` 与 `max|D_measured-D_fitted|`：最大逐点残差；
- 表格列：`E`、`P_true`、`P_measured`、`P_fitted`、`D_measured`、`D_fitted`。

## R13

建议最小测试集：
- 正常场景：`chi_true=2.7`、`noise_std=8e-8`，应满足高 `R^2` 与低相对误差。
- 无噪声场景：`noise_std=0`，理论上应恢复到机器精度级误差。
- 高噪声场景：提高 `noise_std`，观察 `RMSE` 上升、`R^2` 下降。
- 异常输入：
  - `n_points < 5`；
  - `e_min >= e_max`；
  - 人工构造全零 `E`（退化分母）。

## R14

关键可调参数：
- `chi_true`：介质强度（示例默认 `2.7`）；
- `n_points`：采样点数（默认 `61`）；
- `e_min/e_max`：电场扫描范围；
- `noise_std`：观测噪声标准差；
- `seed`：随机种子（保证可复现）。

经验：若噪声变大，可通过增大 `n_points` 改善估计稳定性。

## R15

方法对比：
- 对比“黑盒线性回归”：
  - 本实现直接使用解析解，便于物理解释与调试；
  - 黑盒库更通用，但对本单参数问题属于过度抽象。
- 对比“非线性极化模型（如含 `E^3` 项）”：
  - 线性模型简单稳健，适合低到中等场强；
  - 非线性模型更贴近强场材料，但参数更多、辨识更复杂。

## R16

应用场景：
- 电介质实验教学：从 `P-E` 数据回推材料极化率；
- 材料初筛：比较不同样品在静电场下的极化响应；
- 电磁仿真前处理：给出 `χe/εr` 的基线估计；
- 传感器标定：建立静态响应线性近似模型。

## R17

可扩展方向：
- 各向异性介质：`χ` 从标量扩展为张量；
- 非线性极化：`P = ε0(χ1 E + χ3 E^3 + ...)`；
- 频域介电响应：`χ(ω)` 与复介电常数；
- 与泊松/拉普拉斯方程耦合，处理空间分布 `E(x)` 与边界条件；
- 用真实实验 CSV 数据替换合成数据流程。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 调用 `run_dielectric_polarization_mvp`，设置 `χe_true`、采样点数、场强范围、噪声和随机种子。  
2. `generate_synthetic_dataset` 生成均匀分布电场向量 `E`，按 `P_true = ε0 χe_true E` 构造理想极化，并叠加高斯噪声得到 `P_measured`。  
3. `estimate_chi_linear` 对 `E` 与 `P_measured` 做有限性/形状检查，计算 `denom = E^T E` 与 `numer = E^T P`。  
4. 同函数中用闭式解 `χe_hat = numer / (ε0 * denom)` 得到极化率估计，再计算 `P_fitted = ε0 χe_hat E`。  
5. `evaluate_fit` 计算残差向量 `P_measured - P_fitted`，输出 `RMSE` 与 `R^2`。  
6. `run_dielectric_polarization_mvp` 按定义生成 `D_measured = ε0E + P_measured` 与 `D_fitted = ε0E + P_fitted`，并封装成 `PolarizationResult`。  
7. `run_checks` 对 `R^2`、`χe` 相对误差、`D` 最大残差做阈值断言，保证演示结果可信。  
8. `preview_table` 生成样本预览，`main` 打印关键统计量与表格，最后输出 `All checks passed.`。  

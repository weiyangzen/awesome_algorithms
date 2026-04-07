# 电子衍射 (Electron Diffraction)

- UID: `PHYS-0255`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `258`
- 目标目录: `Algorithms/物理-量子力学-0258-电子衍射_(Electron_Diffraction)`

## R01

电子衍射是“电子具有波动性”的直接证据之一。若电子被加速电压 `V` 加速，其德布罗意波长为 `lambda=h/p`，电子束通过周期结构（晶面阵列、薄膜、光栅等）后会出现角度选择性的干涉峰。

本目录给出一个可运行 MVP，包含两件事：
- 正向：由 `V -> lambda -> 衍射强度 I(theta)` 生成含噪观测；
- 反向：由观测峰位与全曲线拟合反推出 `lambda`，再映射回 `V`。

## R02

本 MVP 使用的物理简化模型：

- 电子波长（相对论修正）：
  `lambda(V) = h / sqrt(2 m e V (1 + eV/(2mc^2)))`
- 晶面间距：`d`（已知常量）
- 有限晶面数：`N`（表示有限阵列）
- 衍射角：`theta`（扫描范围 `[-45°,45°]`）

强度采用有限阵列干涉因子（并乘平滑包络）：
- 相位：`phi = 2*pi*d*sin(theta)/lambda`
- 振幅比例：`A ~ sin(N*phi/2)/sin(phi/2)`
- 强度：`I(theta) ~ (A/N)^2 * envelope(theta)`

## R03

问题目标（demo 内部自洽）：

1. 设定真实加速电压 `V_true=3000 V`，计算真实 `lambda_true`。
2. 在角度网格上生成干涉图样并加入高斯噪声，得到“实验观测”。
3. 从正角度侧提取 1~3 级峰位，回归 `sin(theta_m)=m*lambda/d` 得到 `lambda_est`。
4. 用 PyTorch 对整条曲线做 MSE 拟合，进一步细化 `lambda`。
5. 由 `lambda_est` 反解电压 `V_est`，比较与 `V_true` 的一致性。

## R04

为什么选这个建模方式：

- 电子衍射核心不在复杂几何，而在“波长-相位-干涉峰”链路；
- 有限阵列模型比无限晶体更接近可计算 MVP，且能直接看到主峰/旁瓣结构；
- 峰位回归对应经典实验读图法；
- 全曲线 PyTorch 拟合对应现代数值反演法。

## R05

`demo.py` 中的正向计算主线：

1. `electron_wavelength_relativistic`：由电压算德布罗意波长；
2. `finite_array_intensity`：按有限阵列干涉公式计算 `I(theta)`；
3. `simulate_observation`：加噪并归一化，得到模拟观测曲线。

这三步构成“从物理参数到可观测数据”的最小闭环。

## R06

峰位法反演（解析结构）：

- 衍射级次满足近似关系 `d*sin(theta_m)=m*lambda`；
- 用 `m` 作为自变量、`sin(theta_m)` 作为因变量做过原点线性回归；
- 回归斜率 `slope=lambda/d`，因此 `lambda=slope*d`。

该步骤由：
- `detect_positive_peaks`
- `estimate_wavelength_from_peaks`
完成。

## R07

电压反解步骤：

- 已知目标波长 `lambda_target`，求解 `electron_wavelength_relativistic(V)-lambda_target=0`；
- 使用 `scipy.optimize.brentq` 在 `[1, 5e5]` 伏区间内求根；
- 输出 `V_est` 并与真实 `V_true` 对比。

函数对应：`voltage_from_wavelength`。

## R08

PyTorch 曲线拟合步骤：

- 参数：仅拟合一个标量 `lambda`；
- 目标：最小化 `MSE(I_model(theta,lambda), I_obs(theta))`；
- 约束：用 `softplus(raw_lambda)` 保证波长始终为正；
- 优化器：`Adam`，默认 700 步。

函数对应：`refine_wavelength_with_torch` 与 `finite_array_intensity_torch`。

## R09

为什么不是“第三方一键黑盒”：

- 峰检测只做几何提取，不直接输出波长；
- 回归只拟合线性关系，不替代物理公式；
- Torch 只负责最小化误差，衍射模型公式仍由源码显式实现；
- 电压反解通过显式方程求根。

即：物理关系、数值流程、反演步骤全部可审计。

## R10

运行方式（无交互）：

```bash
cd Algorithms/物理-量子力学-0258-电子衍射_(Electron_Diffraction)
uv run python demo.py
```

脚本会打印：
- 真实与估计波长（pm）；
- 真实与估计加速电压（V）；
- 峰位回归 `R^2`；
- Torch 曲线拟合损失；
- 检测到的正侧峰位表。

## R11

输出结果应满足的定性特征：

- `lambda_est_regression` 与 `lambda_est_torch` 都接近 `lambda_true`；
- `V_est` 接近 `V_true`（允许噪声导致小偏差）；
- 峰位随级次单调增大；
- 回归 `R^2` 接近 1；
- Torch 的最终 MSE 较小。

## R12

复杂度（MVP 量级）：

- 正向计算：`O(G)`，`G` 为角度网格点数；
- 峰检测：`O(G)`；
- 线性回归：`O(K)`，`K` 为峰数（本例 `K=3`）；
- Torch 拟合：`O(T*G)`，`T` 为迭代步数（本例 700）。

整体计算量很小，普通 CPU 下可快速完成。

## R13

数值稳定性处理：

- `sin(phi/2)` 近零处使用极限替代（`ratio -> N`）避免除零爆炸；
- 强度裁剪为非负并归一化；
- 波长拟合使用正值参数化（`softplus`）；
- 电压反解使用有根区间和 Brent 有界求根。

## R14

MVP 的假设与边界：

- 使用 1D 有限晶面阵列，不覆盖真实 3D 晶体的所有几何因子；
- 假设晶面间距 `d` 已知；
- 噪声模型为独立高斯噪声；
- 只拟合单一参数 `lambda`，未同时反演 `d`、仪器展宽等系统误差参数。

这些限制是刻意的，目的是保持“最小但诚实”。

## R15

常见失败模式与排查：

- 峰数不足：调小噪声或放宽 `find_peaks` 参数；
- 回归异常：检查峰序号与角度是否一一对应；
- Torch 拟合不收敛：减小学习率或增大迭代步；
- 电压反解失败：检查 `lambda_est` 是否落在可物理解释范围内。

## R16

可扩展方向：

1. 把 1D 阵列扩展到 2D 晶格（引入晶向和方位角）；
2. 显式加入原子散射因子与 Debye-Waller 因子；
3. 同时反演 `lambda` 与 `d`，并给出不确定性区间；
4. 采用贝叶斯或 MCMC 做参数后验估计；
5. 与真实实验数据文件对接，替代合成噪声数据。

## R17

依赖说明：

- `numpy`：向量化数值计算；
- `scipy`：峰检测与一维求根；
- `pandas`：峰位结果表格；
- `scikit-learn`：线性回归估计 `lambda/d`；
- `torch`：全曲线参数拟合。

该组合覆盖“信号处理 + 物理拟合 + 机器学习回归 + 自动微分优化”四类最小能力。

## R18

`demo.py` 的源码级算法流程（9 步）：

1. 在 `run_electron_diffraction_mvp` 中设置 `V_true, d, N, theta_grid`。  
2. 调用 `electron_wavelength_relativistic(V_true)`，得到真实电子波长 `lambda_true`。  
3. 调用 `simulate_observation`：内部先用 `finite_array_intensity` 生成无噪声干涉图，再叠加高斯噪声并归一化。  
4. 调用 `detect_positive_peaks`，在 `theta>0` 区域用 `find_peaks` 找到前 3 个衍射峰并构建峰位表。  
5. 调用 `estimate_wavelength_from_peaks`，对 `sin(theta_m)=m*lambda/d` 做过原点线性回归，得到 `lambda_est_regression` 和 `R^2`。  
6. 调用 `voltage_from_wavelength(lambda_est_regression)`，通过 Brent 求根反解电压。  
7. 调用 `refine_wavelength_with_torch`：把 `lambda` 作为可学习参数，最小化模型曲线与观测曲线 MSE，得到 `lambda_est_torch`。  
8. 再次调用 `voltage_from_wavelength(lambda_est_torch)` 得到 Torch 版本电压估计，并在 `run_checks` 中验证误差、单调性与损失阈值。  
9. `main` 打印波长/电压估计、回归指标和峰位表，完成“正向仿真 + 反向识别”的闭环演示。  

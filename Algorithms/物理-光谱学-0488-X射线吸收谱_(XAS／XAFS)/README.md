# X射线吸收谱 (XAS/XAFS)

- UID: `PHYS-0465`
- 学科: `物理`
- 分类: `光谱学`
- 源序号: `488`
- 目标目录: `Algorithms/物理-光谱学-0488-X射线吸收谱_(XAS／XAFS)`

## R01

XAS/XAFS 的核心目标是：
从吸收系数 `mu(E)` 随能量 `E` 的变化中，提取局域原子结构信息。

在吸收边以上，振荡项 `chi(k)`（EXAFS）携带邻近配位壳层距离、无序度与配位数等信息。
本条目聚焦最小可运行链路：`mu(E) -> chi(k) -> |FT(R)| -> 壳层峰位`。

## R02

本 MVP 要解决的问题：

1. 对一条带噪声的 XAS 光谱，自动估计吸收边 `E0`；
2. 完成 pre-edge 扣除与 edge-step 归一化，得到 `chi(E)`；
3. 将 `E` 转为 `k`，构建均匀 `chi(k)`；
4. 做 `k^2` 加权与窗函数后，变换到 `R` 空间并检测主峰；
5. 与已知合成真值比较，给出 PASS/FAIL。

## R03

建模范围与假设：

- 使用合成 Fe-like K-edge 谱线（`E0_true=7112 eV`）验证算法流程；
- 采用两壳层 EXAFS 模型（约 `R1=1.95 A`, `R2=3.10 A`）；
- 背景采用线性 pre-edge + 软阶跃边；
- 不实现完整 XANES 多重散射求解，也不做真实实验仪器响应反卷积。

这保证实现“小而诚实”：可跑通核心算法，同时不过度承诺真实实验精度。

## R04

`demo.py` 使用的关键方程：

- 波数变换：
  `k = sqrt((E - E0) / 3.80998212)`（单位：`k` 为 `A^-1`，`E` 为 `eV`）
- EXAFS 合成模型：
  `chi(k) = sum_j [ Nj*Aj/(k*Rj^2) * exp(-2*sigma_j^2*k^2) * exp(-2*Rj/lambda(k)) * sin(2*k*Rj + phi_j) ]`
- 归一化：
  `mu_norm(E) = (mu(E) - mu_pre(E)) / edge_step`
- 振荡项：
  `chi(E) = mu_norm(E) - 1`
- R 空间变换（离散近似）：
  `FT(R) = | sum_k [ w(k) * k^2 * chi(k) * exp(2i*k*R) * dk ] |`

## R05

脚本输出：

- `E0` 估计误差；
- 两个壳层峰位估计值与绝对误差；
- `k` 空间样例表（含 `chi(k)` 与窗后信号）；
- `R` 空间强峰样例；
- 阈值检查与 `Validation: PASS/FAIL`。

## R06

实现策略：

- 用 `dataclass` 固化参数（合成参数、流程参数、验证报告）；
- 用 `savgol_filter + gradient` 估计 `E0`；
- 用 pre/post 线性拟合实现背景与步高估计；
- 用 `interp1d` 将非均匀 `k` 重采样到均匀网格；
- 用显式矩阵 `exp(2i*k*R)` 做直接傅里叶变换；
- 用 `find_peaks` 仅负责峰检测，不替代物理建模。

## R07

优点：

- 端到端覆盖 XAFS 常见核心步骤；
- 可解释性强，所有公式都在源码显式实现；
- 带定量阈值检查，方便自动验证。

局限：

- 数据是合成谱，不代表真实同步辐射实验全部复杂性；
- 未做相位校正导致的精细壳层校准；
- 未包含多重散射路径拟合与参数反演（如完整 FEFF/拟合器工作流）。

## R08

前置知识：

- XAS / EXAFS 的 `mu(E)`、`E0`、`chi(k)`、`R` 空间峰物理含义；
- 傅里叶变换与窗函数基础；
- 一维信号平滑与峰检测。

依赖环境：

- Python 3.10+
- `numpy`
- `scipy`
- `pandas`

## R09

设：

- `N_E` 为能量采样点数；
- `N_k` 为均匀 `k` 点数；
- `N_R` 为 `R` 网格点数。

复杂度：

- 平滑、梯度、拟合、插值、峰检测：均为线性量级，约 `O(N_E + N_k + N_R)`；
- 显式矩阵傅里叶 `exp(2i*k*R)`：`O(N_k * N_R)`；
- 总体主导项为 `O(N_k * N_R)`，空间复杂度约 `O(N_k * N_R)`（矩阵相位项）。

## R10

数值稳定性处理：

- `k` 分母使用 `safe_k = max(k, 1e-6)`，避免除零；
- pre/post 拟合要求最小采样点数，防止退化拟合；
- `edge_step` 必须正值，否则直接报错；
- `k` 窗函数支持长度不足时主动报错；
- 峰检测使用 prominence 约束，减少噪声假峰。

## R11

默认参数（见 `demo.py`）：

- 能量范围：`7050 ~ 7600 eV`，`2400` 点；
- 真实边位：`E0_true = 7112 eV`；
- 噪声：高斯 `std=0.002`；
- `k` 网格：`2.5 ~ 12.5 A^-1`，步长 `0.05`；
- 窗函数区间：`3.0 ~ 11.0 A^-1`；
- `R` 网格：`0.5 ~ 4.5 A`，`801` 点。

## R12

内置验证阈值：

1. `|E0_est - E0_true| < 6 eV`
2. `|R1_est - R1_true| < 0.30 A`
3. `|R2_est - R2_true| < 0.35 A`

三项全部满足输出 `Validation: PASS`，否则输出 `FAIL` 并以非零状态退出。

## R13

保证类型说明：

- 该算法是“信号处理 + 物理近似”流程验证，不是严格反演最优解保证；
- 在固定随机种子下结果可复现（确定性）；
- 保证体现在“满足阈值的回归检查”，不是对真实实验数据的普适精度承诺。

## R14

常见失效模式：

1. 噪声过大导致导数法误判 `E0`；
2. pre/post 拟合区间选取不当，造成 edge-step 偏差；
3. `k` 范围太窄导致 R 空间分辨率不足；
4. 只看幅值峰而不做相位校正时，壳层位置可能系统偏移；
5. 单位混用（eV、A、A^-1）会直接破坏结果数量级。

## R15

可扩展方向：

- 加入 spline 背景（`mu0(E)`）替代线性背景；
- 引入 phase correction 与 backscattering 振幅库；
- 做多壳层参数拟合（`N, R, sigma^2, dE0`）与置信区间估计；
- 接入真实实验数据文件（`csv/h5`）并输出报告图。

## R16

相关主题：

- XANES 与 EXAFS 分区处理；
- Debye-Waller 因子与热无序；
- 窗函数对 R 分辨率与旁瓣的影响；
- EXAFS 正向模型与反演拟合。

## R17

运行方式：

```bash
cd Algorithms/物理-光谱学-0488-X射线吸收谱_(XAS／XAFS)
uv run python demo.py
```

预期输出：

- `Edge and normalization` 统计行；
- `R-space shell validation` 两条壳层误差；
- `k-space` 与 `R-space` 预览表；
- 最终 `Validation: PASS`。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `generate_synthetic_xas` 根据两壳层 `chi(k)`、线性背景和软阶跃边生成带噪声 `mu(E)`。  
2. `estimate_e0` 对 `mu(E)` 做 Savitzky-Golay 平滑，再取一阶导最大值作为 `E0_est`。  
3. `normalize_xas` 在 pre/post 区间做线性拟合，计算 `edge_step`，得到 `mu_norm` 和 `chi(E)`。  
4. `energy_to_k` 把 `E>E0+offset` 区域转成 `k`，并用线性插值构造均匀 `chi(k)`。  
5. `make_hanning_window` 在指定 `k` 区间构造 Hanning 窗，抑制截断振铃。  
6. `exafs_fourier_transform` 计算 `k^2*chi(k)` 后，显式构造 `exp(2ikR)` 矩阵做离散积分，得到 `|FT(R)|`。  
7. `pick_shell_peaks` 在 `|FT(R)|` 上做峰检测，按 prominence 选主峰并输出估计壳层位置。  
8. `main` 组装阈值检查（`E0` 与两壳层误差），打印诊断表并给出 `PASS/FAIL`。

第三方库职责拆解：

- `numpy`：数组计算、指数相位矩阵、数值积分；
- `scipy.signal`：平滑与峰检测；
- `scipy.interpolate`：非均匀 `k` 到均匀 `k` 插值；
- `pandas`：仅用于预览表输出。

核心算法逻辑（背景扣除、归一化、`k` 变换、加权、窗、傅里叶、验收阈值）均在源码中显式实现，而非黑盒调用。

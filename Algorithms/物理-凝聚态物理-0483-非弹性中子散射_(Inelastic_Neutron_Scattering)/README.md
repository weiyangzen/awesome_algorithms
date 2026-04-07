# 非弹性中子散射 (Inelastic Neutron Scattering)

- UID: `PHYS-0460`
- 学科: `物理`
- 分类: `凝聚态物理`
- 源序号: `483`
- 目标目录: `Algorithms/物理-凝聚态物理-0483-非弹性中子散射_(Inelastic_Neutron_Scattering)`

## R01

非弹性中子散射（INS）关注的是散射前后中子能量发生变化的过程，用来反演材料中的激发谱（声子、磁振子、自旋涨落等）。算法核心是从实验强度图 `S(Q, ω)` 中提取色散关系 `ω(Q)`，并拟合模型参数（如能隙 `Δ`、群速度或有效耦合参数）。

本目录的 MVP 给出一个完整可运行流程：
- 合成带噪声的 `S(Q, ω)`。
- 在每个 `Q` 切片上做峰检测 + 局部线型拟合。
- 对提取出的峰位做全局色散拟合。
- 用 SciPy 与 PyTorch 两种优化路径交叉验证。

## R02

物理背景（简化版）：
- 入射/出射中子满足动量与能量守恒，实验结果常写成 `S(Q, ω)`。
- 当材料存在准粒子激发时，`S(Q, ω)` 在对应 `(Q, ω)` 处出现峰。
- 峰位随 `Q` 的轨迹即色散关系，可映射到材料微观参数。

MVP 采用一个带隙的一维风格色散模型：

`ω(Q) = sqrt(Δ^2 + (v sin(πQ))^2)`

其中 `Δ` 是能隙，`v` 是色散尺度（速度/耦合的等效参数）。

## R03

输入与输出定义：
- 输入（本 MVP）：程序内部生成的 `Q` 网格、`ω` 网格和模拟参数。
- 中间量：二维强度图 `S(Q, ω)`，以及每个 `Q` 的候选峰位。
- 输出：
  - `peak_table.csv`：逐 `Q` 提取出的峰参数（峰位、半宽、背景等）。
  - `fit_summary.csv`：SciPy 与 PyTorch 的全局拟合参数和误差指标。
  - `ins_map_preview.csv`：强度图的预览切片（便于快速检查）。

## R04

数学建模与线型：

1. 色散模型：
`ω0(Q; Δ, v) = sqrt(Δ^2 + (v sin(πQ))^2)`

2. 每个 `Q` 切片的谱线模型（洛伦兹线型 + 常数背景）：
`L(ω) = A * γ^2 / ((ω-ωc)^2 + γ^2) + B`

3. 合成数据：
`S(Q, ω) = L(ω; A(Q), ω0(Q), γ, B(Q)) + weak_mode + noise`

4. 全局拟合目标：
最小化 `sum_i [ω_peak(Q_i) - ω0(Q_i; Δ, v)]^2`（SciPy）
或 Huber 损失（PyTorch）以提高鲁棒性。

## R05

算法流程（高层伪代码）：

```text
Generate q_grid, omega_grid
S = simulate_INS_map(q_grid, omega_grid)
S_proc = preprocess(S)
peaks = []
for each q_i:
    detect candidate peaks in S_proc[q_i, :]
    select dominant peak
    local fit with Lorentzian -> refined omega_peak
    append record
fit (Δ, v) by nonlinear regression on {q_i, omega_peak_i}
fit (Δ, v) again by PyTorch autodiff
export CSV artifacts and metrics
```

## R06

复杂度分析（`NQ = len(Q)`, `NW = len(ω)`）：
- 合成数据：`O(NQ * NW)`
- 预处理（标准化 + 高斯平滑）：`O(NQ * NW)`
- 峰检测：`O(NQ * NW)`
- 局部线型拟合：约 `O(NQ * k * iters)`，`k` 为局部窗口点数
- 全局参数拟合：`O(NQ * iters_global)`

总体瓶颈通常在逐 `Q` 的局部非线性拟合与全局拟合迭代上。

## R07

数值稳定性与鲁棒性策略：
- 先标准化再平滑，降低不同 `Q` 切片的幅度偏置。
- `find_peaks` 使用 `prominence` 过滤弱噪声峰。
- 局部 `curve_fit` 设置参数边界，避免无物理意义解。
- 若拟合失败则丢弃该点，不强行插值。
- PyTorch 路径使用 `softplus` 保证 `Δ, v > 0`，并使用 Huber 损失抑制离群值。

## R08

关键假设与适用范围：
- 仅考虑单主峰分支（每个 `Q` 选一个最显著峰）。
- 线型用洛伦兹函数近似，忽略仪器分辨函数卷积。
- 示例是合成数据，真实实验常需做背景建模、分辨率去卷积、多分支分离。
- `Q` 和 `ω` 维度都按均匀网格处理。

## R09

MVP 设计取舍：
- 不引入大型框架，直接使用 `numpy/scipy/pandas/sklearn/torch`。
- 用可复现随机种子生成数据，保证每次运行结果接近。
- 重点保留“从强度图到色散参数”的最短闭环，而不是追求实验级完整处理链。

## R10

实现细节（数据生成）：
- `q_grid = linspace(0.05, 0.95, 70)`
- `omega_grid = linspace(0.2, 12.0, 260)`
- 真值参数：`Δ=2.0, v=8.0, γ=0.55`
- 强度由主色散峰 + 弱平带特征 + 高斯噪声组成。

这样可以构造“足够像真实图谱”的场景，同时保持计算量小。

## R11

实现细节（峰提取与拟合）：
- 逐 `Q` 切片先做 `find_peaks(prominence=0.25)`。
- 选最显著峰后，在 `±2 meV` 窗口内做局部洛伦兹拟合。
- 局部拟合产出 `omega_peak`、`gamma`、`background`。
- 再对所有 `omega_peak(Q)` 做全局 `Δ, v` 拟合。
- 用两种优化器：
  - SciPy: `curve_fit` 最小二乘。
  - PyTorch: Adam + autodiff + Huber loss。

## R12

运行方式：

```bash
uv run python Algorithms/物理-凝聚态物理-0483-非弹性中子散射_(Inelastic_Neutron_Scattering)/demo.py
```

无需任何交互输入。运行后会在当前目录下生成 3 个 CSV 结果文件。

## R13

结果解读：
- `fit_summary.csv` 中 `scipy_curve_fit` 与 `pytorch_autodiff` 两行应接近真值（`ground_truth` 行）。
- `MAE` 越小、`R2` 越接近 1，说明色散拟合质量越高。
- 若两种方法差异大，通常是峰提取阶段存在离群点或窗口参数不合适。

## R14

最小验证清单：
- 脚本可直接运行，不抛异常。
- `peak_table.csv` 至少有 20 个有效点。
- `fit_summary.csv` 包含 3 行：`scipy_curve_fit`、`pytorch_autodiff`、`ground_truth`。
- 拟合得到的 `Δ, v` 为正值且数量级合理。

## R15

可扩展方向：
- 多分支峰追踪（声子支 + 磁支）。
- 加入仪器分辨函数并执行卷积拟合。
- 使用 2D 全图联合拟合替代逐 `Q` 切片。
- 使用贝叶斯方法给出参数置信区间。
- 对接真实 `NeXus` 或仪器原始格式数据。

## R16

常见失败模式：
- 噪声太大时，`find_peaks` 选到假峰。
- 峰间距过近时，单峰模型会偏置中心。
- 色散模型设错（如实际并非 `sin` 结构）会导致系统误差。
- 局部拟合窗口太小/太大都可能导致不稳定。

调参优先顺序建议：`prominence` -> 局部窗口宽度 -> 平滑强度 -> 损失函数。

## R17

与凝聚态研究流程的对应关系：
- 本 MVP 对应“实验初筛 + 参数初估”阶段。
- 在论文级分析中，通常还需：
  - 对温度、极化、晶向做批量比较；
  - 联合理论模型（线性自旋波、DFT 声子谱等）；
  - 进行系统误差和仪器函数评估。

因此本实现定位为教学/原型，而非最终科学结论工具。

## R18

源码级算法拆解（非黑盒，8 步）：

1. `simulate_ins_map` 先调用 `dispersion_model` 生成每个 `Q` 的理论峰位 `ω0(Q)`。
2. 在每个 `Q` 上，调用 `lorentzian` 构造主峰，并叠加弱平带与高斯噪声形成 `S(Q, ω)`。
3. `preprocess_map` 使用 `StandardScaler.fit_transform` 做全图标准化，再用 `gaussian_filter1d` 沿 `ω` 轴平滑。
4. `extract_peak_centers` 对每个 `Q` 切片调用 `find_peaks`，按 `prominence` 选主峰索引。
5. 在主峰附近截取局部窗口，调用 `scipy.optimize.curve_fit` 拟合 `lorentzian`，得到精修 `omega_peak`。
6. 汇总所有 `omega_peak(Q)` 后，`fit_dispersion_scipy` 再次调用 `curve_fit` 拟合全局参数 `(Δ, v)`。
7. `fit_dispersion_torch` 用 PyTorch 建立同一解析公式，通过 `softplus` 约束参数为正，并以 Adam 最小化 Huber 损失。
8. `summarize_fit` 用 `mean_absolute_error` 和 `r2_score` 评估拟合，最后由 `main` 写出 `peak_table.csv`、`fit_summary.csv`、`ins_map_preview.csv`。

# 自旋霍尔效应 (Spin Hall Effect)

- UID: `PHYS-0196`
- 学科: `物理`
- 分类: `自旋电子学`
- 源序号: `197`
- 目标目录: `Algorithms/物理-自旋电子学-0197-自旋霍尔效应_(Spin_Hall_Effect)`

## R01

问题定义：将“自旋霍尔效应（SHE）”做成可运行的最小数值 MVP，展示“纵向电荷电流 `Jc` 通过自旋轨道耦合产生横向自旋流与边界自旋积累”的完整计算闭环。

本条目在 `demo.py` 中实现四段链路：
- 正向：1D 自旋扩散方程 + 无自旋流边界，得到 `mu_s(y)` 与 `j_s(y)`；
- 反演 A（`scipy`）：厚度扫描下拟合 `theta_sh` 与 `lambda_sf`；
- 反演 B（`sklearn`）：ISHE 电压-电流线性拟合估计 `theta_sh`；
- 反演 C（`PyTorch`）：对带噪剖面做联合参数拟合。

## R02

物理核心采用重金属薄膜中常见的简化漂移-扩散图像：
- 电荷流 `Jc` 沿 `x` 方向驱动，SHE 产生沿 `y` 方向的自旋流源项；
- 稳态下自旋化学势 `mu_s(y)` 满足二阶扩散方程，特征长度为 `lambda_sf`；
- 薄膜上下表面取开路自旋边界：`j_s(+-t/2)=0`；
- 边界处形成正负号相反的自旋积累，中心附近接近零。

该 MVP 强调“透明公式 + 可计算 + 可校验”，不追求器件级多物理细节。

## R03

计算任务拆解：
1. 给定 `theta_sh, lambda_sf, sigma, t`，离散厚度坐标 `y`；
2. 计算 `mu_s(y)` 与 `j_s(y)`，并组织为 `pandas.DataFrame`；
3. 生成厚度扫描下的边界信号并加噪；
4. 用 `scipy.optimize.curve_fit` 拟合 `theta_sh`、`lambda_sf`；
5. 生成 ISHE 电压-电流数据并加噪；
6. 用 `sklearn.linear_model.LinearRegression` 拟合斜率与 `R^2`；
7. 用 `torch.optim.Adam` 对带噪剖面联合拟合 `theta_sh`、`lambda_sf`；
8. 执行数值与物理断言门槛，输出 summary。

## R04

模型假设（有意简化）：
- 采用 1D 厚度方向模型，不显式求解横向/纵向二维分布；
- `sigma`、`theta_sh`、`lambda_sf` 视为常数，忽略温度依赖；
- 自旋边界条件取理想开路 `j_s=0`，未加入界面自旋混合电导；
- ISHE 电压使用平均自旋流近似，不含接触电阻与几何畸变。

## R05

`demo.py` 使用的关键公式：

1. 边界自旋积累模型（厚度扫描）：
`mu_edge(t) = 2*lambda*theta*Jc/sigma * tanh(t/(2*lambda))`

2. 剖面解（`y in [-t/2, t/2]`）：
`mu_s(y) = [2*lambda*theta*Jc/sigma] * sinh(y/lambda)/cosh(t/(2*lambda))`

3. 自旋流构成关系：
`j_s(y) = theta*Jc - (sigma/2) * d(mu_s)/dy`

4. 边界条件：
`j_s(+-t/2)=0`

5. 平均自旋流衰减因子：
`eta = 1 - tanh(x)/x`, `x=t/(2*lambda)`

6. ISHE 线性关系（MVP 近似）：
`V_ishe = [L*rho*theta^2*eta] * Jc`

## R06

算法流程：
1. 构建 `SHEParams`（材料参数、几何、扫描范围、噪声和优化超参）。
2. 在 `simulate_spin_profiles` 中扫描 `Jc` 与 `y`，显式生成 `mu_s, j_s` 表格。
3. 在 `fit_edge_scan_with_scipy` 中构造厚度扫描合成观测并用 `curve_fit` 反演 `theta_sh, lambda_sf`。
4. 在 `fit_ishe_line_with_sklearn` 中构造 `V_ishe-Jc` 线性数据并拟合 `R^2` 与斜率。
5. 在 `torch_fit_profiles` 中把剖面观测加噪后做端到端参数拟合。
6. 在 `validate` 中检查边界条件、反演误差和数值门槛。
7. 汇总 summary 与样例表，脚本直接退出（无交互输入）。

## R07

复杂度估计（`N_j=n_j`, `N_y=n_y`, `N_t=n_thickness_scan`, `E=torch_epochs`）：
- 正向剖面生成：`O(N_j * N_y)`；
- `scipy` 曲线拟合：约 `O(N_t * I_cf)`（`I_cf` 为迭代评估次数）；
- `sklearn` 线性拟合：`O(N_j)`；
- `torch` 联合拟合：`O(E * N_j * N_y)`。

默认配置下脚本通常 1-3 秒可完成，适合 MVP 校验。

## R08

数值稳定策略：
- 参数拟合统一到 `uV` 量纲，避免 `1e-6` 量级导致优化停滞；
- `curve_fit` 使用正参数边界（`theta>0, lambda>0`）；
- `torch` 参数通过 `softplus` 约束为正；
- 反演与验证都使用固定随机种子，保证可复现；
- 除法位置都加 `max(..., eps)` 防止极端输入导致除零。

## R09

适用场景：
- 自旋电子学入门课程的 SHE 机制与参数反演演示；
- “正向模拟 + 逆向拟合”流程原型验证；
- 算法回归测试中需要快速稳定、可解释的合成基准。

不适用场景：
- 需要界面散射、温度耦合、磁化动力学等高保真器件建模；
- 论文级实验反演（含系统误差、器件非理想和复杂噪声模型）；
- 需处理非线性强驱动或时变脉冲激励的精细输运。

## R10

脚本内置质量门槛：
1. 边界无自旋流条件：`boundary_max_rel < 5e-11`；
2. 剖面反对称性：`mu_s` 必须同时有正值与负值；
3. 中心点近零：`center_mu_abs_mean_uV < 2e-12`；
4. `scipy` 反演：`theta`/`lambda` 相对误差均 `< 0.20`，且 `edge_mae_uV < 0.020`；
5. `sklearn` 线性拟合：`R^2 > 0.995`，`theta` 相对误差 `< 0.14`；
6. `torch` 联合拟合：`theta`/`lambda` 相对误差均 `< 0.12`，`loss < 1.5e-3`。

## R11

默认参数（`SHEParams`）：
- 电导率 `conductivity_S_per_m = 3.6e6`；
- 电阻率 `resistivity_ohm_m = 1/sigma`；
- `theta_sh = 0.11`，`lambda_sf_nm = 1.7`；
- 薄膜厚度 `8.0 nm`，器件长度 `20 um`；
- 剖面网格：`n_j=9`，`n_y=121`，`Jc` 范围 `2e10 ~ 1e11 A/m^2`；
- 厚度扫描：`1.5 ~ 18 nm`，`26` 点；
- 噪声：`edge 0.012 uV`，`ISHE 0.015 uV`，`profile 0.020 uV`；
- Torch：`epochs=650`，`lr=0.04`，`seed=11`。

## R12

本地实测（命令：`uv run python demo.py`）输出摘要：

- `theta_sh_true = 0.11`
- `lambda_sf_true_nm = 1.7`
- `n_profile_rows = 1089`
- `edge_theta_fit = 0.1102305468`
- `edge_lambda_fit_nm = 1.695656854`
- `edge_mae_uV = 0.002894221304`
- `ishe_r2 = 0.9999999998`
- `ishe_theta_est = 0.1099997112`
- `torch_theta_fit = 0.1099942276`
- `torch_lambda_fit_nm = 1.698794755`
- `torch_final_loss_V2 = 0.0004041617667`
- `boundary_max_rel = 1.362391881e-17`

样例剖面（前 10 行）显示 `y<0` 区域 `mu_s_uV` 为负，符合边界自旋积累符号结构。

## R13

结果解释：
- 三条反演链路（`scipy` / `sklearn` / `torch`）都回到接近真值的 `theta_sh` 与 `lambda_sf`；
- `R^2` 接近 1，说明 ISHE 线性近似在该参数区间内数值自洽；
- `boundary_max_rel` 近机器精度，说明边界条件实现正确；
- `torch` 拟合损失落在噪声量级，说明端到端反演没有明显欠拟合。

## R14

常见失败模式与修复：
- 失败：`curve_fit` 卡在初值或误差很大。
  - 修复：检查量纲（优先使用 `uV` 量纲拟合）、减小观测噪声、调整初值。
- 失败：`R^2` 偏低。
  - 修复：减少 `noise_ishe_uV` 或增加 `Jc` 采样点。
- 失败：Torch 收敛慢。
  - 修复：降低 `torch_lr`，增加 `torch_epochs`，或把观测规模做标准化。
- 失败：边界条件断言失败。
  - 修复：检查 `j_s` 的符号约定与 `d(mu_s)/dy` 系数是否一致。

## R15

工程实践建议：
- 将“正向生成 + 三种反演 + 断言门槛”作为统一回归用例，避免只看单一指标；
- 把几何与材料参数放在 dataclass，便于后续批量扫描；
- 保持 SI 单位与展示单位分离（内部 SI，输出 `uV/nm/um`），降低维护风险；
- 若后续接入实验数据，可直接替换合成观测数组并复用拟合函数。

## R16

可扩展方向：
- 引入界面自旋混合电导，加入非零边界自旋泵浦项；
- 增加温度依赖和多层结构（HM/FM 双层）耦合；
- 从 1D 扩展到 2D/3D 网格，处理非均匀电流分布；
- 使用贝叶斯反演输出 `theta_sh, lambda_sf` 的置信区间；
- 加入频域激励，扩展到时变自旋输运。

## R17

本目录交付物：
- `README.md`：R01-R18 完整说明；
- `demo.py`：可运行 MVP（`numpy + scipy + pandas + scikit-learn + torch`）；
- `meta.json`：保持任务元数据一致（UID、学科、分类、源序号、目录路径）。

运行方式：

```bash
cd Algorithms/物理-自旋电子学-0197-自旋霍尔效应_(Spin_Hall_Effect)
uv run python demo.py
```

脚本无交互输入，运行后直接打印 summary 和样例表。

## R18

`demo.py` 源码级算法流拆解（8 步，非黑盒）：
1. `SHEParams` 定义材料常数、几何尺寸、扫描范围和优化超参。
2. `simulate_spin_profiles` 调用 `spin_profile_V_and_current`，按解析式显式计算 `mu_s(y)` 与 `j_s(y)`，生成剖面数据表。
3. `fit_edge_scan_with_scipy` 先用 `edge_accumulation_model_V` 生成厚度-边界信号，再加噪并用 `curve_fit` 求 `theta_sh`、`lambda_sf`。
4. `fit_ishe_line_with_sklearn` 基于 `eta=1-tanh(x)/x` 构造 `V_ishe-Jc` 数据，用线性回归拿斜率和 `R^2`，再反推 `theta_sh`。
5. `torch_fit_profiles` 将 `theta`、`lambda` 设为可训练参数，使用 `softplus` 保证正值，并以剖面 MSE 为损失做 Adam 优化。
6. `validate` 对边界条件、剖面对称性、三类拟合误差及损失阈值执行断言。
7. `build_summary` 汇总真值、拟合值和关键质量指标。
8. `main` 串联以上步骤并打印输出，完成“模拟-反演-校验”闭环。

第三方库仅用于数值拟合与优化；SHE 物理关系、边界条件和指标判定逻辑都在源码中展开实现。

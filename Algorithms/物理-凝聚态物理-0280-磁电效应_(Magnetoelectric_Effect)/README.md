# 磁电效应 (Magnetoelectric Effect)

- UID: `PHYS-0277`
- 学科: `物理`
- 分类: `凝聚态物理`
- 源序号: `280`
- 目标目录: `Algorithms/物理-凝聚态物理-0280-磁电效应_(Magnetoelectric_Effect)`

## R01

问题定义：将“磁电效应（Magnetoelectric Effect）”从概念说明转成可运行的最小算法闭环。

本条目交付的 MVP 聚焦线性磁电耦合：
- 正向关系：`P = P0 + alpha H`（磁场诱导极化）；
- 对偶关系：`M = M0 + alpha^T E`（电场诱导磁化）；
- 温度依赖：`alpha(T) = alpha0 * s(T)`，其中 `s(T)` 由临界型函数给出。

脚本在同一次运行中完成：
1. 合成含噪观测数据；
2. 逐温度反演 `alpha(T)`；
3. 拟合临界温度行为（`T_N`, `beta`）；
4. 用 PyTorch 做全局联合参数精修。

## R02

磁电效应的核心是电与磁序参量的线性耦合。在张量形式下：

- `P_i = P0_i + sum_j alpha_ij H_j`
- `M_i = M0_i + sum_j alpha_ji E_j`

若系统满足互易约束（在本 MVP 的简化设定中），由 `P~H` 与 `M~E` 反演得到的张量应近似满足：

- `alpha_PH ~= alpha_ME^T`

因此，本实现不仅估计耦合强度，也显式计算互易误差 `||alpha_PH - alpha_ME^T||_F` 作为物理一致性指标。

## R03

计算任务拆解：
1. 在温度网格上构造序参量尺度函数 `s(T)`；
2. 用 `alpha(T)=alpha0*s(T)` 生成多温度下的 `(E,H,P,M)` 合成观测；
3. 每个温度点分别对 `P~H`、`M~E` 做多输出线性回归，得到 `alpha_PH` 与 `alpha_ME`；
4. 对称化得到 `alpha_sym = 0.5*(alpha_PH + alpha_ME^T)`，并计算 `alpha_eff` 与互易误差；
5. 用 `scipy.optimize.least_squares` 拟合 `alpha_eff(T)` 的临界型包络，估计 `(A, T_N, beta, c)`；
6. 用 `torch` 在全数据上联合拟合 `alpha0, P0, M0, T_N, beta`，形成正向-逆向闭环验证。

## R04

模型假设（有意简化）：
- 采用二维张量（`2x2`）描述耦合，不覆盖完整三维晶体对称性分类；
- 假设线性响应主导，忽略高阶项（如 `E^2H`、`EH^2`）；
- 温度依赖统一压缩为单尺度 `s(T)=max(1-T/T_N,0)^beta`；
- 噪声采用独立高斯噪声，不考虑实验相关噪声与漂移；
- 不引入畴结构、磁滞回线、频率色散等复杂效应。

## R05

`demo.py` 关键公式：

1. 临界尺度函数：
`s(T) = max(1 - T/T_N, 0)^beta`

2. 温度依赖耦合张量：
`alpha(T) = alpha0 * s(T)`

3. 磁致极化：
`P = P0 + alpha(T) H + epsilon_P`

4. 电致磁化：
`M = M0 + alpha(T)^T E + epsilon_M`

5. 有效耦合标量（用于温度包络拟合）：
`alpha_eff = ||alpha_sym||_F / sqrt(4)`

6. 临界包络拟合模型：
`alpha_eff(T) = A * max(1-T/T_N,0)^beta + c`

## R06

算法流程：
1. `check_params` 校验网格、噪声、温区与优化超参。
2. `simulate_dataset` 生成多温度、多样本合成数据集（含真实 `alpha_scale_true`）。
3. `estimate_tensor_by_temperature` 在每个温度上做两次线性回归：`P~H` 与 `M~E`。
4. 计算 `alpha_sym`、`alpha_eff`、`reciprocity_fro` 与局部回归 `R^2`。
5. `fit_critical_envelope` 用 `least_squares` 回归 `alpha_eff(T)`，得到 `T_N` 和 `beta`。
6. `torch_global_refinement` 构建可微前向模型并用 Adam 联合优化全局参数。
7. 输出 summary、温度表头与拟合张量，并执行质量断言。

## R07

复杂度估计（`Nt` 温度点数，`Ns` 每温度样本数，总样本 `N=Nt*Ns`，Torch 轮数 `E`）：

- 合成数据：`O(N)`；
- 分温度线性回归：`O(N)`（每温度常数维度小矩阵）；
- 临界包络 least-squares：`O(I_ls * Nt)`；
- PyTorch 联合优化：`O(E * N)`；
- 空间复杂度：`O(N)`。

在默认设置（`Nt=24`, `Ns=36`, `E=500`）下，`uv run python demo.py` 可在秒级完成。

## R08

数值稳定策略：
- `order_parameter_scale` 对负区间显式截断（`clip`），避免临界函数出现复数或非法值；
- 临界拟合采用边界约束（`T_N`、`beta`、`A`、`c`）防止非物理解；
- PyTorch 中 `T_N` 与 `beta` 通过 `softplus` 参数化保证正值；
- 联合优化中加入轻微 `L2` 正则抑制张量参数发散；
- 输出互易误差与双通道 `R^2`，避免只看单一损失导致误判。

## R09

适用场景：
- 磁电耦合教学演示（张量、互易性、临界温度拟合）；
- 算法原型验证（从观测反演 `alpha(T)`、`T_N`、`beta`）；
- 为真实实验数据分析提供可替换的最小骨架。

不适用场景：
- 需要严格材料定量预测的第一性原理任务；
- 存在强非线性、强磁滞、频率依赖或多相共存的样品；
- 需要完整晶体点群约束与三维张量分解的高精度分析。

## R10

脚本内置正确性门槛：
1. 有序相（`alpha_scale_true > 0.08`）的平均回归质量：
   - `mean_r2_P_given_H_ordered > 0.995`
   - `mean_r2_M_given_E_ordered > 0.995`
2. 互易误差：`mean_reciprocity_fro < 0.25`
3. 临界包络拟合：`critical_fit_r2 > 0.90`
4. PyTorch 全局反演：
   - `tn_abs_error_K < 18`
   - `beta_abs_error < 0.25`
   - `torch_joint_mse < 0.020`
   - `torch_r2_P > 0.97`, `torch_r2_M > 0.97`
   - `alpha0_mae < 0.35`

## R11

默认参数（`MEParams`）：
- 温度网格：`120~350 K`，`24` 点
- 每温度样本数：`36`（总计 `864` 条）
- 真值临界参数：`T_N=282 K`, `beta=0.46`
- 场强范围：`E in [-2.0, 2.0]`, `H in [-1.6, 1.6]`
- 真值耦合张量：
  `alpha0=[[2.80,-1.00],[0.60,2.20]]`
- 偏置：`P0=(0.030,-0.020)`, `M0=(0.012,0.018)`
- 噪声：`noise_P=0.018`, `noise_M=0.015`
- Torch 优化：`epochs=500`, `lr=0.05`

## R12

本地实测（命令：`uv run python demo.py`）：

Summary：
- `n_samples_total = 864`
- `mean_r2_P_given_H = 0.727396`
- `mean_r2_M_given_E = 0.723114`
- `mean_r2_P_given_H_ordered = 0.999370`
- `mean_r2_M_given_E_ordered = 0.999739`
- `mean_reciprocity_fro = 0.008545`
- `critical_fit_r2 = 0.999997`
- `critical_tn_fit_K = 281.933399`
- `critical_beta_fit = 0.460066`
- `torch_tn_fit_K = 278.532074`
- `torch_beta_fit = 0.431630`
- `torch_joint_mse = 0.007711`
- `torch_r2_P = 0.997532`
- `torch_r2_M = 0.997933`
- `alpha0_mae = 0.027578`
- `tn_abs_error_K = 3.467926`
- `beta_abs_error = 0.028370`

注：全温区平均 `R^2` 偏低是因为接近/高于 `T_N` 时耦合接近 0，信号被噪声主导；因此脚本对“有序相区间”单独设定回归门槛。

## R13

结果一致性解释：
- 互易误差均值仅 `0.008545`，说明 `P~H` 与 `M~E` 两条反演链路高度一致；
- 临界包络拟合几乎重建真值（`T_N` 误差小于 `0.1 K`，`beta` 误差约 `6.6e-5`）；
- Torch 全局拟合在噪声存在时仍恢复 `T_N` 与 `beta`（误差分别 `3.47 K` 和 `0.028`）；
- `alpha0_mae` 仅 `0.0276`，说明张量主结构恢复良好。

## R14

常见失败模式与修复：
- 失败：温区完全高于 `T_N`，导致有效信号接近 0。
  - 修复：下调温区下限或增加低温采样点。
- 失败：场强扫描范围过窄，回归病态。
  - 修复：增大 `e_field_max_MVm` 与 `h_field_max_T`。
- 失败：噪声过大导致 `alpha_eff(T)` 抖动，临界拟合不稳。
  - 修复：增加 `samples_per_temp` 或降低噪声参数。
- 失败：Torch 收敛慢或震荡。
  - 修复：降低 `torch_lr`、增加 `torch_epochs`、适度提高正则。

## R15

工程实践建议：
- 保留“逐温度回归 + 全局可微反演”双链路，便于回归测试与故障定位；
- 始终报告“有序相指标”和“全温区指标”，防止对噪声主导区间过拟合解读；
- 把 `alpha_eff(T)`、`reciprocity_fro(T)` 存档用于批次比较；
- 真实实验接入时，可直接替换 `simulate_dataset` 为数据读取，后续反演模块可复用。

## R16

可扩展方向：
- 扩展到三维 `3x3` 张量与晶体点群约束；
- 引入非线性项（如 `P = P0 + alpha H + eta |H|^2 H`）；
- 将 `T_N` 拓展为应力/掺杂函数，做多条件联合拟合；
- 用贝叶斯推断输出 `alpha0, T_N, beta` 置信区间；
- 接入真实实验流程中的漂移校正、基线漂移与系统误差建模。

## R17

本目录交付内容：
- `demo.py`：可运行 MVP（`numpy + scipy + pandas + scikit-learn + torch`）；
- `README.md`：R01-R18 完整说明；
- `meta.json`：任务元数据与目录信息。

运行方式：

```bash
cd Algorithms/物理-凝聚态物理-0280-磁电效应_(Magnetoelectric_Effect)
uv run python demo.py
```

无需交互输入，程序会直接输出 summary、温度估计表头与反演张量。

## R18

`demo.py` 源码级算法流拆解（8 步，非黑盒）：
1. `MEParams` 与 `check_params` 固定并验证温区、噪声、场强和优化超参。
2. `order_parameter_scale` 计算 `s(T)=max(1-T/TN,0)^beta`，作为耦合强度温度缩放。
3. `simulate_dataset` 显式生成 `alpha(T)=alpha0*s(T)`，并据此构造 `(E,H)->(P,M)` 合成观测加噪数据。
4. `estimate_tensor_by_temperature` 在每个温度点分别拟合 `P~H` 和 `M~E`，得到 `alpha_PH`、`alpha_ME` 与偏置项。
5. 同函数内进一步构造 `alpha_sym`、`alpha_eff`、`reciprocity_fro` 与局部 `R^2`，形成可审计统计表。
6. `fit_critical_envelope` 使用 `scipy.optimize.least_squares` 对 `alpha_eff(T)` 拟合 `A, T_N, beta, c`。
7. `torch_global_refinement` 把同一物理方程写成可微图，通过 Adam 同时优化 `alpha0, P0, M0, T_N, beta`。
8. `main` 汇总 summary、打印关键表并执行质量断言，完成“正向仿真 -> 反演估计 -> 一致性校验”闭环。

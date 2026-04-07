# 昂萨格倒易关系 (Onsager Reciprocal Relations)

- UID: `PHYS-0305`
- 学科: `物理`
- 分类: `非平衡统计`
- 源序号: `308`
- 目标目录: `Algorithms/物理-非平衡统计-0308-昂萨格倒易关系_(Onsager_Reciprocal_Relations)`

## R01

本条目把“昂萨格倒易关系”落地为一个可运行的最小算法闭环：

- 在近平衡线性响应框架下建立 `J = L X`；
- 在 `B=0` 时检验 `L(0) = L(0)^T`（Onsager 对称性）；
- 在 `B!=0` 时检验 `L(B) = L(-B)^T`（Onsager-Casimir 关系，偶宇称变量简化形态）；
- 用同一批数据分别通过 `scikit-learn`、`SciPy`、`PyTorch` 三条路线估计输运矩阵并交叉验证。

脚本目标不是复杂材料建模，而是提供“可解释、可验证、可扩展”的算法骨架。

## R02

理论背景（线性不可逆热力学）可写为：

- 通量向量：`J = (J1, J2)^T`
- 力向量：`X = (X1, X2)^T`
- 线性关系：`J_i = sum_j L_ij X_j`

在时间反演偶变量且外磁场为零时，昂萨格倒易关系要求：

- `L_ij(0) = L_ji(0)`

在存在磁场 `B` 时，Casimir 修正给出：

- `L_ij(B) = L_ji(-B)`

本 MVP 将 `L(B)` 分解为：

- `L(B) = L_sym + B * L_asym`
- 其中 `L_sym^T = L_sym`，`L_asym^T = -L_asym`

这样可以直接构造并检验 `L(B) = L(-B)^T`。

## R03

`demo.py` 的输入输出（无交互）如下。

输入（脚本内固定参数）：
1. 磁场取值 `B in {-1, 0, +1}`。
2. 每个磁场下随机生成热力学力 `X1, X2`。
3. 真值输运矩阵 `L_sym_true` 与霍尔型反对称系数 `hall_coeff_true`。
4. 高斯噪声标准差 `noise_std`。

输出：
1. 每个 `B` 上的回归估计矩阵 `L_hat(B)`；
2. `B=0` 的 SciPy 对称约束拟合矩阵；
3. 全数据 PyTorch 联合拟合得到的 `L_sym` 与 `hall_coeff`；
4. 倒易关系误差（Frobenius 范数）、`R^2`、`MAE`、`MSE`；
5. 断言通过后输出 `All checks passed.`。

## R04

建模假设（有意最小化）：

1. 仅考虑二维通量-力耦合（`2x2` 矩阵）。
2. 仅考虑线性响应，不含高阶项（如 `X^2`、耦合饱和等）。
3. 截距固定为 0（`fit_intercept=False`），即 `X=0` 时 `J≈0`。
4. 噪声为独立同分布高斯噪声。
5. 变量宇称取偶变量情形，Casimir 关系简化为转置关系。

## R05

脚本中的核心公式：

1. 线性响应：
`J = L(B) X`

2. 对称-反对称分解：
`L(B) = L_sym + B * L_asym`

3. 反对称矩阵参数化：
`L_asym = [[0, k], [-k, 0]]`

4. `B=0` 的昂萨格检验：
`E_0 = ||L(0) - L(0)^T||_F`

5. `B!=0` 的 Casimir 检验：
`E_C = ||L(+1) - L(-1)^T||_F`

6. 拟合质量：
`R^2 = 1 - SS_res / SS_tot`，并报告 `MAE/MSE`。

## R06

算法流程：

1. `simulate_dataset` 生成三组磁场数据，构造带噪 `X -> J` 样本。
2. `estimate_l_by_field` 对每个 `B` 独立做多输出线性回归，得到 `L_hat(B)`。
3. 计算回归版倒易误差：`E_0` 与 `E_C`。
4. `fit_symmetric_b0_scipy` 在 `B=0` 数据上做对称约束最小二乘，得到 `L_sym_fit`。
5. `torch_joint_fit` 在全数据上联合拟合 `L_sym` 与 `k`。
6. 输出三路结果并执行质量门槛断言。

## R07

设每个磁场样本数为 `N`，磁场数为 `M=3`，维度固定 `d=2`。

- 数据生成：`O(MN)`；
- 三组线性回归：`O(MN)`（`d` 固定，常数很小）；
- SciPy 约束最小二乘：`O(I_s * N)`；
- PyTorch 联合优化：`O(E * M * N)`；
- 空间复杂度：`O(MN)`。

默认参数下（`N=480`, `E=700`）运行为秒级。

## R08

数值稳定与可解释性策略：

1. 参数合法性检查（样本数、噪声、磁场集合、训练轮次）。
2. 使用 `fit_intercept=False` 保持物理约束 `X=0 -> J=0`。
3. SciPy 先用回归结果做初值，减少迭代不稳定。
4. PyTorch 加轻量 `L2` 正则，避免参数漂移。
5. 同时输出 `R^2 / MAE / reciprocity error`，防止单一指标误判。

## R09

适用场景：

1. 非平衡统计课程中倒易关系的可计算演示。
2. 线性输运矩阵估计与互易误差基准测试。
3. 后续替换成实验数据前的算法联调模板。

不适用场景：

1. 强非线性或远离近平衡区的输运现象。
2. 需要严格材料参数预测的高精度研究任务。
3. 存在显著记忆效应、时变系数或多物理场强耦合但未建模的场景。

## R10

`demo.py` 的质量门槛（断言）：

1. 三个磁场下最小 `R^2 > 0.985`。
2. 零磁场对称误差 `||L(0)-L(0)^T||_F < 0.090`。
3. Casimir 误差 `||L(+1)-L(-1)^T||_F < 0.100`。
4. SciPy 的 `B=0` 对称拟合 `R^2 > 0.985`。
5. PyTorch 联合拟合 `R^2 > 0.985`。
6. 霍尔系数恢复误差 `|k_hat-k_true| < 0.060`。

## R11

默认参数（`OnsagerParams`）：

- 随机种子：`305`
- 每个磁场样本：`480`（总样本 `1440`）
- 力范围：`X1,X2 in [-1.25, 1.25]`
- 噪声：`noise_std = 0.035`
- 磁场：`{-1, 0, +1}`
- 真值对称矩阵：
  `L_sym_true = [[1.35, 0.52], [0.52, 1.05]]`
- 真值反对称系数：`hall_coeff_true = 0.36`
- PyTorch：`epochs=700`, `lr=0.04`, `weight_decay=3e-4`

## R12

本地实测（命令：`uv run python demo.py`）核心输出：

- `r2_B=-1 = 0.998636`
- `r2_B=0 = 0.998657`
- `r2_B=+1 = 0.998743`
- `reg_zero_symmetry_fro = 0.003971`
- `reg_casimir_fro = 0.006577`
- `reg_l0_mae_vs_true = 0.001505`
- `scipy_r2_B0 = 0.998656`
- `scipy_l0_mae_vs_true = 0.000973`
- `torch_joint_r2 = 0.998678`
- `torch_joint_mse = 0.001220`
- `torch_zero_symmetry_fro = 0.000000`
- `torch_casimir_fro = 0.000000`
- `torch_l0_mae_vs_true = 0.000169`
- `hall_abs_err = 0.000117`

并成功输出 `All checks passed.`。

## R13

结果解释：

1. 三个磁场下 `R^2` 都接近 `0.999`，说明线性模型和数据生成机制匹配良好。
2. 回归版 `E_0` 与 `E_C` 均约 `1e-3` 到 `1e-2` 量级，满足倒易关系。
3. SciPy 在 `B=0` 强制对称后仍保持高 `R^2`，说明“对称约束”没有破坏数据拟合。
4. PyTorch 全局联合拟合几乎精确恢复 `k_true`，且自动满足结构性对称/反对称约束。

## R14

常见失败模式与修复：

1. 失败：`R^2` 下降明显。
   修复：增大 `n_samples_per_field` 或降低 `noise_std`。

2. 失败：`E_0` 偏大。
   修复：检查 `fit_intercept` 是否误设为 `True`，并确认数据是否含系统偏置项。

3. 失败：`E_C` 偏大。
   修复：检查 `B=+1/-1` 样本数量是否失衡，或 `hall_coeff_true` 是否与数据生成逻辑一致。

4. 失败：Torch 不收敛。
   修复：降低 `torch_lr`（如 `0.02`），或提高 `torch_epochs`。

## R15

工程实践建议：

1. 保留“分组回归 + 约束拟合 + 可微拟合”三链路，便于故障定位。
2. 将 `E_0`、`E_C` 与 `R^2` 一起做回归测试指标。
3. 若接入真实实验数据，优先加入单位归一化与异常值过滤。
4. 先在 `B=0` 做对称性 sanity check，再扩展到多磁场数据。

## R16

可扩展方向：

1. 从 `2x2` 扩展到 `n x n` 多通道耦合。
2. 引入奇宇称变量，使用完整 Casimir 符号因子 `epsilon_i epsilon_j`。
3. 在模型中加入温度依赖 `L(T,B)` 并做多条件联合拟合。
4. 增加贝叶斯估计输出参数置信区间。
5. 用真实时间序列替代静态 i.i.d. 样本并建模时滞效应。

## R17

目录交付清单：

- `README.md`：R01-R18 完整说明；
- `demo.py`：可直接运行的最小 MVP；
- `meta.json`：与任务元数据一致。

运行方式：

```bash
cd Algorithms/物理-非平衡统计-0308-昂萨格倒易关系_(Onsager_Reciprocal_Relations)
uv run python demo.py
```

程序无需任何交互输入。

## R18

`demo.py` 源码级算法流拆解（8 步，非黑盒）：

1. `OnsagerParams` 与 `check_params` 固定并校验样本规模、噪声、磁场集合和优化超参。
2. `simulate_dataset` 显式构造 `L(B)=L_sym+B*L_asym`，生成 `X` 并计算 `J=LX+noise`，写入 `pandas.DataFrame`。
3. `estimate_l_by_field` 按 `B` 分组调用 `sklearn.linear_model.LinearRegression`，得到每个磁场的 `L_hat(B)` 与 `R^2/MAE`。
4. 在 `main` 中直接由矩阵运算计算 `E_0=||L(0)-L(0)^T||_F` 与 `E_C=||L(+1)-L(-1)^T||_F`。
5. `fit_symmetric_b0_scipy` 用 `scipy.optimize.least_squares` 在 `B=0` 上拟合参数 `(a,b,c)`，并组装对称矩阵 `[[a,b],[b,c]]`。
6. `torch_joint_fit` 用 `theta=(a,b,c,k)` 参数化 `L_sym` 与 `L_asym`，在全数据上通过 Adam 最小化联合 `MSE + L2`。
7. 由 Torch 拟合参数重建 `L(B)`，再次计算全局 `R^2/MSE` 和倒易误差，形成第三条验证链路。
8. `main` 汇总三条链路指标并执行断言门槛，完成“合成数据 -> 估计反演 -> 倒易关系校验”的闭环。

说明：`scikit-learn`、`SciPy`、`PyTorch` 仅承担数值优化器/回归器角色；倒易关系的物理结构、矩阵分解和误差定义均在源码中显式实现。

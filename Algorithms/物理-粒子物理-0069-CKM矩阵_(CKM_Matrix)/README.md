# CKM矩阵 (CKM Matrix)

- UID: `PHYS-0069`
- 学科: `物理`
- 分类: `粒子物理`
- 源序号: `69`
- 目标目录: `Algorithms/物理-粒子物理-0069-CKM矩阵_(CKM_Matrix)`

## R01

CKM（Cabibbo-Kobayashi-Maskawa）矩阵描述弱相互作用中下型夸克味态与质量本征态之间的线性变换。若记上型夸克为 `u,c,t`，下型夸克为 `d,s,b`，则带电流耦合项可写成：

`J_mu^+ ~ (u,c,t)_L * gamma_mu * V_CKM * (d,s,b)_L`

其中 `V_CKM` 是 `3x3` 复矩阵，满足幺正条件 `V V^dagger = I`。`demo.py` 给出标准参数化构造、数值一致性检查与参数反演的最小实现。

## R02

典型应用场景：

- 标准模型味物理中 `W` 介导衰变振幅计算
- `B` 介子、`K` 介子衰变中的 CKM 元素提取
- CP 破坏大小（Jarlskog 不变量）量化
- 全局拟合中对 `|V_ij|` 和相位参数的一致性约束

## R03

`demo.py` 使用 PDG 标准参数化（3 个混合角 + 1 个 CP 相位）：

- 参数：`theta12, theta13, theta23, delta`
- 记 `sij = sin(thetaij)`, `cij = cos(thetaij)`

`V_CKM =`

- 第一行：`[c12 c13, s12 c13, s13 e^{-i delta}]`
- 第二行：`[-s12 c23 - c12 s23 s13 e^{i delta}, c12 c23 - s12 s23 s13 e^{i delta}, s23 c13]`
- 第三行：`[s12 s23 - c12 c23 s13 e^{i delta}, -c12 s23 - s12 c23 s13 e^{i delta}, c23 c13]`

此外给出 Wolfenstein `O(lambda^3)` 近似并与精确参数化做误差比较。

## R04

直观理解：

- CKM 的每个元素是“夸克味转换振幅”的权重
- 元素模长决定不同衰变道强弱（如 `|V_us|`、`|V_cb|`）
- 复相位 `delta` 是标准模型 CP 破坏的重要来源
- 幺正性意味着不同代夸克混合受整体约束，不能独立任意取值

## R05

正确性要点：

1. 构造出的 `V` 应数值上满足 `||V V^dagger - I||` 很小。
2. Jarlskog 不变量 `J` 可由两条路径一致给出：
   `J = Im(V_ud V_cs V_us^* V_cd^*)`
   `J = c12 c23 c13^2 s12 s23 s13 sin(delta)`
3. 两种 `J` 结果应在浮点误差范围内一致。
4. Wolfenstein 低阶展开与精确矩阵会有可量化偏差，偏差规模反映截断误差。
5. 在合成观测下，最小二乘反演应能回收接近真值的参数。

## R06

复杂度分析（单次拟合迭代，参数维度固定为 4）：

- 构造一次 CKM：`O(1)`
- 计算观测向量（9 个模长 + 1 个 `J`）：`O(1)`
- 残差评估：`O(1)`
- 总拟合成本约为 `O(k)`，其中 `k` 为求解器函数评估次数（`nfev`）

由于维度很小，这个 MVP 的瓶颈不是线性代数规模，而是非线性优化迭代次数。

## R07

标准实现步骤：

1. 设定 `theta12/theta13/theta23/delta` 的近似中心值。
2. 用标准参数化函数生成复矩阵 `V`。
3. 计算幺正残差 `||V V^dagger - I||_F`。
4. 计算 Jarlskog 不变量（矩阵法与解析法两条路径）。
5. 将角参数映射为 Wolfenstein 参数并构造 `O(lambda^3)` 近似矩阵。
6. 构造合成“实验观测”（`|V_ij|` 与 `J`）并加高斯噪声。
7. 使用带边界的 `least_squares` 反演参数。
8. 输出拟合偏差、`chi2/dof` 与求解器状态。

## R08

`demo.py` 的 MVP 设计：

- 依赖：`numpy` + `scipy.optimize.least_squares`
- 不依赖外部数据文件，全部参数内置
- 输出包含：
  - CKM 矩阵（复数）
  - 幺正残差
  - Jarlskog 一致性
  - Wolfenstein 截断误差
  - 拟合参数与真值偏差
- 运行方式：`uv run python demo.py`（无需交互输入）

## R09

`demo.py` 接口约定：

- `ckm_standard(theta12, theta13, theta23, delta) -> np.ndarray`
- `unitarity_residual(v_ckm) -> float`
- `jarlskog_from_matrix(v_ckm) -> float`
- `jarlskog_from_angles(theta12, theta13, theta23, delta) -> float`
- `angles_to_wolfenstein(theta12, theta13, theta23, delta) -> tuple[float, float, float, float]`
- `ckm_wolfenstein_o3(lam, a_param, rho, eta) -> np.ndarray`
- `observables_from_params(params) -> np.ndarray`
- `weighted_residuals(params, measured, sigma) -> np.ndarray`

## R10

测试策略：

- 结构测试：CKM 是否为 `3x3` 复矩阵
- 物理约束测试：幺正残差是否足够小
- 解析一致性测试：`J_matrix` 与 `J_angles` 是否一致
- 近似误差测试：`V_exact` 与 `V_wolf_O3` 的 Frobenius 误差是否在可解释量级
- 反演健壮性测试：带噪观测下参数能否回收且 `chi2/dof` 合理

## R11

边界条件与异常处理：

- 拟合边界设置：
  - `theta12/theta23 in [0, pi/2]`
  - `theta13 in [0, 0.3]`（覆盖物理小角区域）
  - `delta in [0, 2pi]`
- 若 `least_squares` 返回失败状态，`main()` 抛出 `RuntimeError`
- 观测噪声尺度 `sigma` 明确给定，避免未归一化残差导致优化失衡

## R12

与相关对象关系：

- CKM 是夸克部门混合矩阵；PMNS 是轻子（中微子）部门对应对象
- 二者都可写成“3 角 + 1 相位”的参数化结构，但数值层级差异明显
- CKM 呈近单位阵结构（代间混合较小），PMNS 混合通常更大
- Jarlskog 不变量是比较 CP 破坏强度的统一指标之一

## R13

`demo.py` 参数选择（近似 PDG 中心值）：

- `theta12 = 13.04 deg`
- `theta13 = 0.201 deg`
- `theta23 = 2.38 deg`
- `delta = 68.8 deg`

合成观测配置：

- 观测向量：`[|V_11|,...,|V_33|, J]` 共 10 维
- 噪声标准差：`|V_ij|` 用 `2e-4`，`J` 用 `8e-7`
- 随机种子：`69`（可复现）

## R14

工程实现注意点：

- 绝对值观测对 `delta` 的灵敏度较弱，因此将 `J` 加入观测向量以增强 CP 相位可辨识性
- 残差按 `sigma` 归一化，等价于各观测采用不同权重
- 采用 `least_squares(method="trf")` 处理带边界非线性最小二乘
- 输出角度统一转为度，便于与物理文献常见表示对照

## R15

最小示例结果应体现：

- `||V V^dagger - I||_F` 接近机器精度（数值幺正）
- 两种 Jarlskog 计算结果高度一致
- `O(lambda^3)` Wolfenstein 与精确 CKM 存在小但非零误差
- 在轻微噪声下拟合参数与真值偏差较小，`chi2/dof` 处于可解释范围

## R16

可扩展方向：

- 将 `O(lambda^3)` 扩展到更高阶 Wolfenstein 展开
- 接入真实实验输入（如 UTfit/CKMfitter 公布量）替换合成观测
- 将目标函数扩展为含协方差矩阵的全局似然
- 引入 `bootstrap` 或 MCMC 做参数不确定度传播

## R17

本条目交付说明：

- `README.md`：完成 R01-R18 的算法、物理与工程说明
- `demo.py`：可运行最小实现（构造、验证、拟合一体）
- `meta.json`：保持任务元数据一致

运行命令：

```bash
uv run python Algorithms/物理-粒子物理-0069-CKM矩阵_(CKM_Matrix)/demo.py
```

或在目录内：

```bash
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，9 步）：

1. **参数到矩阵映射**
   `ckm_standard` 先计算 `sij/cij` 与 `exp(±i delta)`，按 PDG 公式逐元素组装 `3x3` 复矩阵。

2. **幺正性数值检查**
   `unitarity_residual` 计算 `R = V V^dagger - I`，再取 `||R||_F` 作为约束满足度指标。

3. **CP 破坏量双路径计算**
   `jarlskog_from_matrix` 用四元乘积的虚部给出 `J`；`jarlskog_from_angles` 用解析式给出 `J`，用于交叉校验。

4. **标准参数到 Wolfenstein 低阶参数转换**
   `angles_to_wolfenstein` 通过 `lambda=sin(theta12)` 等关系估计 `(lambda, A, rho, eta)`。

5. **构造低阶近似矩阵并评估截断误差**
   `ckm_wolfenstein_o3` 生成 `O(lambda^3)` 近似，主程序计算 `||V_exact - V_wolf||_F`。

6. **定义拟合观测与加权残差**
   `observables_from_params` 产出 10 维观测（9 个 `|V_ij|` + 1 个 `J`），`weighted_residuals` 按 `sigma` 标准化。

7. **调用 SciPy 非线性最小二乘内核**
   `least_squares(..., method="trf")` 使用信赖域反射算法：在每次迭代里线性化残差、求解受边界约束的子问题、更新步长并检查收敛准则（`xtol/ftol/gtol`）。

8. **回收解并构造统计量**
   读取 `fit.x` 作为估计参数，重新计算预测观测，给出 `chi2` 与 `chi2/dof`。

9. **输出可审计证据链**
   主程序打印：矩阵本体、幺正残差、J 一致性、Wolfenstein 误差、拟合偏差与求解器状态，使结果可复核且可复现。

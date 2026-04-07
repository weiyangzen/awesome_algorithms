# Gauss-Newton方法

- UID: `MATH-0351`
- 学科: `数学`
- 分类: `优化`
- 源序号: `351`
- 目标目录: `Algorithms/数学-优化-0351-Gauss-Newton方法`

## R01

Gauss-Newton 方法用于求解**非线性最小二乘**问题：

`min_theta  F(theta) = 1/2 * ||r(theta)||_2^2`

其中 `r(theta)` 是残差向量（长度 `m`），`theta` 是待估参数（维度 `p`）。
与完整牛顿法不同，Gauss-Newton 用 `J^T J` 近似 Hessian（`J` 为残差 Jacobian），因此每轮只需要一阶导数信息，常用于曲线拟合与参数辨识。

## R02

方法来源于 Carl Friedrich Gauss 的最小二乘思想与 Isaac Newton 的局部线性化迭代思想的结合，属于经典数值优化算法。
在统计回归、测量平差、计算机视觉（如 bundle adjustment）和系统辨识中长期使用。
现代工程里常见其阻尼变体（Levenberg-Marquardt 可视作在 Gauss-Newton 基础上的正则化增强）。

## R03

目标问题可写成：

`F(theta) = 1/2 * sum_{i=1}^m r_i(theta)^2`

在点 `theta_k` 处线性化残差：

`r(theta_k + delta) ≈ r_k + J_k * delta`

于是子问题变成线性最小二乘：

`min_delta 1/2 * ||r_k + J_k * delta||_2^2`

其正规方程：

`(J_k^T J_k) delta = -J_k^T r_k`

更新 `theta_{k+1} = theta_k + delta`。

## R04

Gauss-Newton 与牛顿法关系：

- 牛顿法 Hessian：`H = J^T J + sum_i r_i * ∇^2 r_i`
- Gauss-Newton 近似：`H ≈ J^T J`

当残差较小或模型近似线性时，第二项影响弱，Gauss-Newton 的近似通常有效。
在本目录 MVP 中加入了阻尼项：

`(J^T J + lambda I) delta = -J^T r`

通过调整 `lambda` 提升鲁棒性，避免正规矩阵病态时直接失败。

## R05

高层流程：

1. 给定初值 `theta0`。
2. 计算残差 `r_k` 与 Jacobian `J_k`。
3. 解线性系统得到步长 `delta_k`。
4. 若新点目标值下降则接受更新，否则增大阻尼重试。
5. 满足停止条件（梯度/步长/目标下降阈值）则退出。

本实现采用“阻尼试探”策略：若当前步不下降，`lambda` 乘 `10` 重算，最多重试固定次数。

## R06

复杂度（单轮）：

- 构造 `J`：`O(m*p)`
- 形成 `J^T J`：`O(m*p^2)`
- 解 `p x p` 线性系统：`O(p^3)`

总复杂度约为 `O(T*(m*p^2 + p^3))`，`T` 为迭代轮次。
当参数维度 `p` 不大、样本 `m` 较大时，Gauss-Newton 往往很实用。

## R07

收敛性质（常见结论）：

- 在解附近且 Jacobian 满秩时，通常有较快局部收敛。
- 与完整牛顿法相比，二阶信息不完整，远离最优点时可能不稳定。
- 初值敏感，坏初值可能导致不下降、震荡或收敛到非目标局部解。

因此工程上通常配合阻尼、线搜索或信赖域机制。

## R08

本目录示例问题：拟合指数模型

`y = a * exp(bx) + c`

参数 `theta=[a,b,c]`，残差 `r_i = f(x_i,theta) - y_i`。
解析 Jacobian：

- `∂r/∂a = exp(bx)`
- `∂r/∂b = a*x*exp(bx)`
- `∂r/∂c = 1`

`demo.py` 使用带噪声合成数据，通常 5-10 轮内收敛。

## R09

前置知识：

- 最小二乘与线性代数基础（正规方程、矩阵分解）
- 多元微分（Jacobian）
- 迭代优化停止准则
- 基本数值稳定概念（病态矩阵、阻尼）

## R10

适用边界：

- 适用：可微残差模型、目标是平方残差和、参数维度中小。
- 谨慎：初值很差、Jacobian 退化、噪声极大或模型严重非线性。
- 不适合：不可微目标或非最小二乘型目标（需改用其他优化器）。

## R11

实现正确性检查点：

1. Jacobian 维度应为 `(m, p)`。
2. 残差方向统一为 `model - observation`，与梯度符号匹配。
3. 线性系统右端应为 `-J^T r`。
4. 更新必须在“下降判定”后接受。
5. 终止条件需至少覆盖梯度、步长或目标改变量之一。

本 MVP 提供有限差分 Jacobian 误差检查，用来防止导数写错。

## R12

数值稳定性策略（本实现已采用）：

- 阻尼正规方程 `(J^T J + lambda I)`。
- 当步长不下降时放大 `lambda` 并重算。
- 线性系统求解失败时回退 `lstsq`。
- 对输入做有限值检查，提前报错而非传播 `nan/inf`。

## R13

与相关方法对比：

- 对比梯度下降：Gauss-Newton 方向利用局部二次结构，常更快。
- 对比完整牛顿法：Gauss-Newton 不需要显式二阶导，代价更低。
- 对比 Levenberg-Marquardt：LM 在阻尼调度与信赖域解释上更系统，鲁棒性通常更好。

## R14

常见失效模式与防护：

- 模型初值太差，导致迭代不下降。
  - 防护：多初值重启，或先粗搜索。
- `J^T J` 近奇异。
  - 防护：阻尼、正则化、参数重标定。
- 参数尺度差异大导致数值病态。
  - 防护：做变量缩放或重参数化。

## R15

调参建议：

- `damping_init`：初始阻尼，太小可能激进，太大则收敛慢。
- `max_damping_trials`：每轮最多尝试次数，平衡鲁棒与耗时。
- `grad_tol / step_tol / loss_tol`：停止准则，建议按数据噪声水平设置。
- 初值 `theta0`：对非凸问题非常关键，建议结合领域知识设置。

## R16

工程实践建议：

- 先用有限差分校验 Jacobian，再大规模跑数据。
- 记录每轮 `obj_before/obj_after/step_norm/grad_inf`，便于诊断。
- 对参数和输入做量纲归一化，减少病态。
- 若多次出现 `no_descent_step`，优先检查模型、初值与雅可比推导。

## R17

本目录 `demo.py` 说明：

- 依赖：仅 `numpy`。
- 功能：
  - 生成指数模型合成数据；
  - 解析 Jacobian 与有限差分 Jacobian 对照；
  - 阻尼 Gauss-Newton 迭代；
  - 输出收敛原因、参数误差与 RMSE。
- 运行：

```bash
cd Algorithms/数学-优化-0351-Gauss-Newton方法
python3 demo.py
```

脚本无交互输入，运行后直接打印迭代日志与最终检查结果。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `make_synthetic_data` 生成 `x`、带噪声观测 `y_noisy`、无噪声真值 `y_clean` 与真实参数 `theta_true`。
2. `finite_difference_jacobian_error` 对初值 `theta0` 做中心差分，验证解析 Jacobian 与数值近似的一致性。
3. `gauss_newton` 在每轮先调用 `residuals` 和 `jacobian` 计算 `r_k`、`J_k`，并得到当前目标值与梯度无穷范数。
4. 若梯度已足够小，则按 `grad_tol` 直接判定收敛。
5. 否则进入阻尼试探：`solve_damped_normal_equation` 解 `(J^T J + lambda I) delta = -J^T r` 得到候选步长。
6. 计算试探点 `theta_trial = theta + delta` 的目标值；若下降则接受，否则把 `lambda` 乘 `10` 并重算，最多重试固定次数。
7. 接受步后记录日志项（目标变化、步长范数、梯度范数、阻尼），再按 `step_tol` 与 `loss_tol` 检查是否停止。
8. `main` 汇总并打印 `theta_hat`、参数误差、`rmse_noisy/rmse_clean` 与 `global checks pass`，形成可复现实验输出。

本实现没有把求解交给黑盒优化器；每个迭代环节都在源码中显式展开。

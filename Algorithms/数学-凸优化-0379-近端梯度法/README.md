# 近端梯度法

- UID: `MATH-0379`
- 学科: `数学`
- 分类: `凸优化`
- 源序号: `379`
- 目标目录: `Algorithms/数学-凸优化-0379-近端梯度法`

## R01

近端梯度法（Proximal Gradient Method）用于求解如下“光滑项 + 非光滑项”的凸优化问题：

`min_x F(x) = f(x) + g(x)`

其中 `f` 可微且 `grad f` Lipschitz 连续，`g` 可凸但不可微。  
本条目实现最小可运行 MVP：针对 Lasso 目标用 ISTA（近端梯度的经典实例）进行求解，并在可解析闭式解场景下做数值对照。

## R02

`demo.py` 求解问题为：

`min_x (1/(2m)) * ||Xx - y||_2^2 + lam * ||x||_1`

对应分解：
- `f(x) = (1/(2m)) * ||Xx-y||_2^2`（光滑凸）
- `g(x) = lam * ||x||_1`（非光滑凸）

输出包括：
- 迭代轨迹（目标值、步长变化、梯度映射范数）；
- 与闭式最优解的目标 gap；
- 参数向量误差与支撑集重叠；
- 收敛与单调性审计结果。

## R03

选择该实例的原因：
- `L1` 正则项是近端方法最典型应用，近端算子有闭式软阈值；
- ISTA 每步结构清晰，便于展示“梯度步 + 近端步”的算法本质；
- 通过构造 `(X^T X)/m = I` 的设计矩阵，可写出闭式最优解用于验证，避免黑箱依赖。

## R04

核心数学公式：

1. 光滑项梯度  
`grad f(x) = X^T(Xx-y)/m`

2. Lipschitz 常数  
`L = ||X||_2^2 / m`

3. 近端映射（`g(x)=lam||x||_1`）  
`prox_{t g}(v) = soft_threshold(v, t*lam)`  
`soft_threshold(v_i, tau) = sign(v_i) * max(|v_i|-tau, 0)`

4. ISTA 更新  
`x_{k+1} = prox_{step*g}(x_k - step * grad f(x_k))`

5. 梯度映射（最优性指标）  
`G_step(x) = (x - prox_{step*g}(x - step*grad f(x))) / step`

## R05

算法流程（高层）：

1. 校验输入维度、有限值和超参数范围。  
2. 计算 `L=||X||_2^2/m`，设置固定步长 `step=step_scale/L`。  
3. 初始化 `x0=0`。  
4. 每轮先算 `grad f(xk)`，再做软阈值近端更新得到 `x_{k+1}`。  
5. 记录 `objective`、`||dx||`、`||G_step(x)||`。  
6. 满足联合停止条件（参数变化小且梯度映射小）则退出。  
7. 返回结果并做闭式解对照、单调性检查与支撑集分析。

## R06

收敛与正确性依据（MVP 层面）：
- 目标函数是凸组合：二次损失凸且可微，`L1` 范数凸；
- 当 `step <= 1/L` 时，ISTA 对该类问题收敛到全局最优解；
- 代码中使用 `step_scale <= 1` 保证固定步长合法；
- 使用 `||G_step(x)||` 作为近似一阶最优性证据；
- 在本脚本特定数据构造下还有闭式最优解，可额外做严格数值对照。

## R07

复杂度分析：
- 设样本数 `m`、维度 `n`、迭代轮数 `T`；
- 单轮主要开销是 `Xx` 与 `X^T r`，时间 `O(mn)`；
- 软阈值是逐元素操作，时间 `O(n)`；
- 总时间复杂度 `O(Tmn)`；
- 空间复杂度 `O(mn)`（存 `X`）加 `O(n)`（变量向量）。

## R08

边界与异常处理：
- `X` 非 2D、`y` 非 1D 或维度不匹配会抛 `ValueError`；
- `X/y` 含 `nan/inf` 会抛 `ValueError`；
- `lam < 0`、`step_scale` 不在 `(0,1]`、`max_iters<=0`、`tol<=0` 会报错；
- 迭代中若目标或范数出现非有限值会抛 `RuntimeError`；
- 若最终未收敛、目标非单调或与闭式解误差过大，`main` 会显式失败。

## R09

MVP 取舍：
- 仅用 `numpy` 实现，避免额外框架；
- 不调用 `scipy/sklearn` 黑箱优化器作为主解法；
- 不实现 FISTA、回溯线搜索、随机近端等扩展版本；
- 优先保证“源码可读、流程可审计、结果可验证”。

## R10

`demo.py` 结构说明：
- `validate_inputs`：输入合法性校验；
- `soft_threshold`：`L1` 近端算子；
- `objective_lasso` / `grad_smooth`：目标函数与梯度；
- `lipschitz_constant`：计算 `L=||X||_2^2/m`；
- `proximal_gradient_mapping_norm`：计算梯度映射范数；
- `ista_lasso`：ISTA 主循环；
- `build_orthonormal_lasso_case`：构造可复现实验和闭式参考解；
- `objective_monotone_violations` / `support_set` / `print_history`：审计与展示；
- `main`：整合运行、对照、阈值断言。

## R11

运行方式：

```bash
cd Algorithms/数学-凸优化-0379-近端梯度法
uv run python demo.py
```

脚本无交互输入，直接输出结果。

## R12

输出字段说明：
- `objective`：当前迭代解的目标值；
- `||dx||`：`||x_{k+1}-x_k||_2`，反映迭代步幅；
- `||G_step(x)||`：梯度映射范数，越小表示越接近最优；
- `objective(ista)` 与 `objective(closed form)`：迭代解与闭式解目标值；
- `relative objective gap`：相对目标差；
- `||x_ista - x_closed||_2`：与闭式最优参数的距离；
- `support overlap`：估计稀疏支撑与真实支撑交集规模。

## R13

最小测试集（脚本内置）：
- 固定随机种子 `379`，构造 `m=120, n=40, sparsity=8` 的确定性样例；
- 设计矩阵满足 `(X^T X)/m = I`，使闭式解可直接计算；
- 自动执行并验证：
  - 相对目标 gap 小于阈值；
  - 与闭式解的参数误差小于阈值；
  - 目标函数单调不增；
  - 迭代触发收敛条件。

## R14

关键参数：
- `lam`：稀疏正则强度，越大越稀疏；
- `step_scale`：步长系数，默认 `0.98`（靠近 `1/L`）；
- `max_iters`：最大迭代次数；
- `tol`：停止阈值；
- `noise_std`：数据噪声强度，影响恢复难度。

调参建议：
- 收敛慢：先增大 `max_iters`；
- 若单调性不理想：减小 `step_scale`；
- 仅快速演示：可放宽 `tol` 到 `1e-7` 或 `1e-6`。

## R15

与相关方法对比：
- 相比次梯度法：近端梯度在该复合目标上通常更稳定更快；
- 相比坐标下降：坐标下降在部分 Lasso 场景更高效，但近端梯度更统一、易扩展到非坐标可分结构；
- 相比 ADMM：ADMM 更灵活（便于加约束），但实现与参数调节复杂度更高；
- 相比 FISTA：FISTA 常更快，但基础 ISTA 更直观，适合作为教学基线。

## R16

典型应用场景：
- 稀疏线性回归（Lasso）；
- 稀疏编码与字典学习的子问题；
- 图像去噪/反卷积中的 `L1` 正则化模型；
- 更复杂近端算法（FISTA、近端牛顿、分裂法）的基线实现。

## R17

可扩展方向：
- 将固定步长升级为回溯线搜索；
- 增加 Nesterov 动量得到 FISTA；
- 支持组稀疏（`L2,1`）等更一般近端算子；
- 扩展到约束版近端梯度（投影 + 近端）；
- 与 `scikit-learn` 的 Lasso 或坐标下降结果做系统 benchmark。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 调用 `build_orthonormal_lasso_case` 生成 `X, y, x_true`，并计算闭式最优解 `x_star`。  
2. `ista_lasso` 先经 `validate_inputs` 检查维度、有限性和参数范围。  
3. 用 `lipschitz_constant` 计算 `L=||X||_2^2/m`，设置 `step=step_scale/L`。  
4. 从 `x=0` 开始，每轮调用 `grad_smooth` 计算当前梯度。  
5. 执行前向-后向更新：`x_next = soft_threshold(x - step*grad, lam*step)`。  
6. 记录 `objective_lasso`、`||dx||` 与 `proximal_gradient_mapping_norm` 形成审计轨迹。  
7. 满足联合停止准则即返回 `IstaResult`，否则迭代到 `max_iters`。  
8. `main` 将迭代解和闭式解做目标/参数对照，并检查单调性与阈值，最终输出“通过/失败”。

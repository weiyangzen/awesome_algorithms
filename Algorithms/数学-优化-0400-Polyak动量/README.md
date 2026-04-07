# Polyak动量

- UID: `MATH-0400`
- 学科: `数学`
- 分类: `优化`
- 源序号: `400`
- 目标目录: `Algorithms/数学-优化-0400-Polyak动量`

## R01

Polyak 动量（又称 Heavy-Ball 方法）是在梯度下降基础上加入“速度”状态的一阶优化方法。

核心思想：
- 普通梯度下降每步只看当前梯度；
- Polyak 动量把“上一轮运动方向”也纳入更新；
- 在病态（条件数高）问题上，常能沿主方向更快推进，减少“走走停停”的收敛行为。

## R02

本目录 MVP 解决的问题：

给定强凸二次目标

`f(w) = 0.5 * w^T A w - b^T w`

其中 `A` 为对称正定矩阵，求近似最优解 `w*`，并比较：
- 纯梯度下降（GD）
- Polyak 动量（Heavy-Ball）

输出包括：
- 每轮 `objective`、`suboptimality`、`grad_norm`、`dist_to_opt`
- 两种方法达到同一精度阈值所需步数对比
- 最终误差比值（Polyak/GD）

## R03

为什么选择强凸二次目标：
- 梯度形式简单，`grad f(w)=Aw-b`，便于严格审计；
- 最优解可由线性方程 `Aw*=b` 直接求得，可作为“真值”对照；
- 病态二次目标能清楚体现动量法相对 GD 的加速优势。

## R04

本实现使用 Polyak（Heavy-Ball）写法：

- `v_{k+1} = beta * v_k - eta * grad f(w_k)`
- `w_{k+1} = w_k + v_{k+1}`

其中：
- `eta` 为步长（学习率）
- `beta in [0,1)` 为动量系数

当 `beta=0` 时，上式退化为普通梯度下降。

## R05

算法流程（高层）：

1. 构造病态对称正定矩阵 `A` 与向量 `b`。
2. 计算解析最优解 `w* = A^{-1}b`。
3. 校验输入合法性（维度、有限值、对称性、正定性）。
4. 以同一初值 `w0`、同一步长分别运行 GD 与 Polyak。
5. 每轮记录目标值、次优差、梯度范数、到真解距离。
6. 统计“达到固定次优差阈值”的首个迭代步。
7. 输出两方法末轮指标与比值。
8. 用断言检查 Polyak 是否在该案例下优于 GD。

## R06

正确性要点：
- 梯度计算：`gradient_quadratic` 直接实现 `Aw-b`；
- 最优值基准：先算 `w*`，再用 `f(w)-f(w*)` 作为次优差；
- 比较公平：GD 与 Polyak 使用同一 `A/b/w0/eta/max_iters`；
- 收敛判据一致：都以 `||grad|| <= tol` 判定收敛；
- 结果校验：脚本末尾断言 Polyak 最终次优差更小，且更快达到阈值。

## R07

复杂度分析（参数维度 `d`）：
- 每轮核心成本是一次矩阵向量乘 `Aw`，时间 `O(d^2)`；
- GD 与 Polyak 单轮时间复杂度同阶，均为 `O(d^2)`；
- 若总迭代步数为 `T`，总时间复杂度 `O(T*d^2)`；
- 额外空间：
  - 参数与梯度 `O(d)`
  - Polyak 额外速度向量 `v` 也是 `O(d)`
  - 若保留完整 history，额外 `O(T)`。

## R08

边界与异常处理：
- `A` 非方阵、`b/w0` 维度不匹配：抛 `ValueError`；
- 含 `nan/inf`：抛 `ValueError`；
- `A` 非对称或非正定：抛 `ValueError`；
- `eta <= 0` 或 `beta` 不在 `[0,1)`：抛 `ValueError`；
- 光谱估计与构造值不一致：抛 `RuntimeError`；
- 若比较条件不成立（示例退化），末尾断言会显式报错。

## R09

MVP 取舍：
- 仅使用 `numpy` + 标准库，避免黑盒优化器；
- 不调用 `scipy.optimize.minimize`、`torch.optim` 等现成接口；
- 不覆盖随机梯度、Nesterov、自适应学习率等扩展；
- 目标是“小而诚实”的可运行、可解释对照实验。

## R10

`demo.py` 主要函数职责：
- `make_ill_conditioned_quadratic`：生成可控条件数的 SPD 二次目标。
- `validate_problem_inputs`：输入合法性与正定性检查。
- `objective_quadratic`：计算 `f(w)`。
- `gradient_quadratic`：计算 `Aw-b`。
- `run_gradient_descent`：运行 GD 并记录历史。
- `run_polyak_momentum`：运行 Polyak 动量并记录历史。
- `first_step_below_suboptimality`：统计达到阈值的首轮步数。
- `print_history`：抽样打印长轨迹。
- `main`：组织实验、打印比较结果、执行断言。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0400-Polyak动量
python3 demo.py
```

脚本不需要任何交互输入，会直接打印完整实验结果。

## R12

输出字段说明：
- `objective`：当前 `f(w)`。
- `suboptimality`：`f(w)-f(w*)`，越小越好。
- `grad_norm`：`||∇f(w)||_2`。
- `dist_to_opt`：`||w-w*||_2`。
- `suboptimality ratio (Polyak/GD)`：最终误差比值，小于 1 表示 Polyak 更优。
- `first step with suboptimality <= threshold`：达到目标精度的速度对比。

## R13

最小测试集（脚本内置）：
- 维度 `d=20` 的病态 SPD 二次目标；
- 条件数 `kappa = 1000`；
- 固定随机种子，确保可复现；
- 对比方法：
  - GD：`beta=0`
  - Polyak：`beta=0.60`
- 统一步长 `eta=1/L` 与统一最大迭代步 `220`。

## R14

关键参数与调参建议：
- `beta`：动量系数，越大记忆越强，也更可能振荡；
- `step_size`：步长，过大可能发散，过小则收益不明显；
- `tol`：梯度停止阈值；
- `max_iters`：最大迭代步数。

建议顺序：
1. 先固定 `eta` 在稳定区间；
2. 从 `beta=0.3~0.8` 扫描；
3. 再根据振荡程度调整 `tol/max_iters`。

## R15

与相关方法对比：
- 对比 GD：
  - GD 无速度状态；
  - Polyak 用历史方向加速，常在病态问题上更快。
- 对比 Nesterov 动量：
  - Polyak 在当前位置算梯度；
  - Nesterov 在“前瞻点”算梯度，通常更稳。
- 对比 Adam：
  - Adam 是逐坐标自适应步长；
  - Polyak 参数更少、机制更直接，但对步长和 `beta` 更敏感。

## R16

典型应用场景：
- 强凸或近似强凸的光滑优化问题；
- 大规模机器学习中的一阶优化基线；
- 作为理解 Nesterov、Adam 等优化器的前置教材算法。

## R17

可扩展方向：
- 增加 Nesterov 版本并做三方对照（GD/Polyak/NAG）；
- 引入 mini-batch 噪声梯度场景；
- 加入学习率衰减或重启策略；
- 扩展到非二次目标（如逻辑回归）；
- 输出 CSV 并绘制收敛曲线。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 调用 `make_ill_conditioned_quadratic` 生成 `A,b,w*`，构造病态强凸目标。
2. 使用 `validate_problem_inputs` 检查维度、有限值、对称性与正定性，并得到 `mu/L`。
3. 固定同一 `w0` 与 `eta=1/L`，分别调用 `run_gradient_descent` 和 `run_polyak_momentum`。
4. 在两种循环中，每步都先用 `gradient_quadratic` 计算梯度，再计算 `objective_quadratic` 并记录 `IterRecord`。
5. GD 分支执行 `w <- w - eta * grad`。
6. Polyak 分支先更新 `velocity <- beta * velocity - eta * grad`，再做 `w <- w + velocity`。
7. `first_step_below_suboptimality` 统计两方法首次达到目标次优差阈值（默认 `1.0`）的步数。
8. `main` 打印最终指标与比值，并用断言验证 Polyak 在该实验中最终更优且更快达阈。

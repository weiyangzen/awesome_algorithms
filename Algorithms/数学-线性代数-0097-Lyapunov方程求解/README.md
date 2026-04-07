# Lyapunov方程求解

- UID: `MATH-0097`
- 学科: `数学`
- 分类: `线性代数`
- 源序号: `97`
- 目标目录: `Algorithms/数学-线性代数-0097-Lyapunov方程求解`

## R01

Lyapunov 方程是稳定性分析中的核心线性矩阵方程。

本条目聚焦连续时间形式：

`A^T X + X A + Q = 0`

其中：
- `A in R^{n x n}`；
- `Q` 通常取对称正定或半正定；
- 目标是求 `X`（在 `A` Hurwitz 且 `Q` 对称正定时，`X` 唯一且对称正定）。

## R02

历史定位（简要）：
- Lyapunov 在 19 世纪末提出稳定性理论框架；
- 该方程把“动力系统稳定性”转化为“线性代数求解问题”；
- 现代控制、滤波、系统辨识、鲁棒控制中广泛使用；
- 是 Riccati 方程、LQR、Kalman 滤波等主题的重要基础模块。

## R03

本目录 MVP 任务定义：
- 输入：实矩阵 `A` 与对称矩阵 `Q`；
- 输出：方程 `A^T X + X A + Q = 0` 的数值解 `X`；
- 核心实现：不直接把 SciPy 当黑盒，而是显式使用 Kronecker 向量化法构造线性系统并求解；
- 验证：优先与 `scipy.linalg.solve_continuous_lyapunov` 做结果比对；若环境无 SciPy，则退化为残差与结构性校验。

## R04

典型应用场景：
- 线性系统 `x_dot = A x` 的二次型稳定性证书构造；
- LQR/Kalman 中相关矩阵方程的子问题验证；
- 控制器设计前的可稳定性快速诊断；
- 工程建模中对“衰减速度、能量函数”的数值分析。

## R05

数学要点：
- 若 `A` 的全部特征值实部都小于 0（Hurwitz），且 `Q` 给定，则连续 Lyapunov 方程存在唯一解；
- 若 `Q = Q^T > 0` 且 `A` Hurwitz，则解 `X = X^T > 0`；
- `X` 可视为一个能量函数 `V(x)=x^T X x` 的参数矩阵；
- 方程是 Sylvester 方程的特例。

## R06

向量化转换（本实现核心）：

利用恒等式：
- `vec(A^T X) = (I ⊗ A^T) vec(X)`
- `vec(X A)   = (A^T ⊗ I) vec(X)`

得到：

`[(I ⊗ A^T) + (A^T ⊗ I)] vec(X) = -vec(Q)`

记 `K = (I ⊗ A^T) + (A^T ⊗ I)`，则求解变为标准线性系统：

`K x = b`，其中 `x = vec(X)`，`b = -vec(Q)`。

## R07

算法步骤（Kronecker 法）：
1. 校验 `A, Q` 为同阶方阵且元素有限；
2. 构造 `K = kron(I, A^T) + kron(A^T, I)`；
3. 构造右端 `b = -vec(Q)`（列优先向量化）；
4. 解线性系统 `Kx=b`；
5. 把 `x` 重排回矩阵 `X`；
6. 计算残差 `R = A^T X + X A + Q` 评估正确性。

## R08

时间复杂度：
- 构造 `K` 的维度是 `n^2 x n^2`，元素规模 `O(n^4)`；
- 稠密线性系统求解复杂度约 `O((n^2)^3)=O(n^6)`；
- 因此该方案适合教学、小规模验证，不适合大规模工程。

## R09

空间复杂度：
- 显式存储 `K` 需要 `O(n^4)` 内存；
- `X`、`Q` 等矩阵为 `O(n^2)`；
- 总体由 `K` 主导，空间瓶颈明显。

## R10

技术栈：
- Python 3
- `numpy`（Kronecker 构造、线性系统求解、残差与特征值检查）
- `scipy.linalg`（可选，仅用于参考解交叉验证，不替代核心实现）
- 标准库：`dataclasses`

## R11

运行方式：

```bash
cd Algorithms/数学-线性代数-0097-Lyapunov方程求解
python3 demo.py
```

脚本为固定示例输入，不读取命令行参数，不请求交互输入。

## R12

输出字段说明：
- `matrix_shape`：输入矩阵维度；
- `is_hurwitz`：`A` 是否 Hurwitz（最大实部 < 0）；
- `used_scipy_reference`：本次运行是否启用 SciPy 参考解；
- `residual_fro`：残差 Frobenius 范数 `||A^T X + X A + Q||_F`；
- `relative_residual`：相对残差 `residual_fro / ||Q||_F`；
- `max_abs_error_vs_scipy`：与 SciPy 参考解最大绝对误差（无 SciPy 时跳过）；
- `symmetry_error`：`||X - X^T||_F`；
- `min_eig_of_sym_part`：`(X+X^T)/2` 的最小特征值，用于正定性观测。

## R13

鲁棒性处理：
- 校验输入必须为同阶方阵；
- 校验矩阵元素为有限值；
- 对 `Q` 进行对称性检查（控制理论常见假设）；
- 如果 `A` 非 Hurwitz，不强行报错，但打印警告并提示“稳定性结论不成立”；
- 用断言保证残差、对照误差与对称误差在阈值内。

## R14

当前局限：
- Kronecker 显式法复杂度高，无法扩展到大规模系统；
- 只覆盖连续时间方程，不含离散 Lyapunov 版本；
- 未实现结构化/低秩/稀疏优化；
- 未实现 Schur-Bartels-Stewart 的工程级高效流程。

## R15

可扩展方向：
- 使用 Schur 分解 + Sylvester 回代（Bartels-Stewart）降低数值成本；
- 接入稀疏矩阵和低秩近似（ADI/Krylov）处理大规模问题；
- 增加离散 Lyapunov 方程 `A^T X A - X + Q = 0`；
- 与 LQR/Kalman 示例联动，展示端到端控制设计流程。

## R16

最小测试建议：
- 正确性：残差是否接近机器精度；
- 交叉验证：与 SciPy 参考解误差；
- 结构性：`Q` 对称时，`X` 应近似对称；
- 稳定性：`A` Hurwitz 且 `Q>0` 时，`X` 的对称部分最小特征值应为正；
- 失败路径：喂入非方阵、非有限值，确认抛异常。

## R17

方案对比：
- `scipy.linalg.solve_continuous_lyapunov`：调用便捷、性能更实用，但算法细节被封装（且可能受运行环境安装情况影响）；
- 本实现（Kronecker 向量化）：步骤透明、便于教学和审计；
- Schur/Bartels-Stewart：工业场景更推荐，复杂度与稳定性通常更优；
- 结论：本条目优先“可解释闭环”，再通过 SciPy 做可信交叉验算。

## R18

`demo.py` 源码级流程（9 步）：
1. `build_stable_demo_case()` 构造一个可复现的 Hurwitz 矩阵 `A` 和对称正定 `Q`。  
2. `validate_inputs()` 检查维度、有限性、`Q` 对称性，提前阻断非法输入。  
3. `solve_lyapunov_via_kronecker()` 计算 `K = kron(I, A^T) + kron(A^T, I)`。  
4. 同函数中把 `Q` 列优先向量化成 `b = -vec(Q)`，形成线性系统 `Kx=b`。  
5. 用 `numpy.linalg.solve` 求得 `x`，并重排回矩阵 `X`。  
6. `compute_residual_metrics()` 计算绝对/相对残差与对称误差，评估数值质量。  
7. `main()` 若检测到 SciPy，则调用 `scipy.linalg.solve_continuous_lyapunov(A^T, -Q)` 得到参考解；否则跳过该对照。  
8. `run_checks()` 对残差、对照误差、对称误差、正定性执行阈值断言。  
9. 打印关键指标并输出 `All checks passed.`，形成可复现最小闭环。  

说明：SciPy 在本脚本中仅承担“验算基准”角色，核心求解流程由源码显式实现。

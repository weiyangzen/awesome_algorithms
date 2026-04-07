# Remez算法

- UID: `MATH-0175`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `175`
- 目标目录: `Algorithms/数学-数值分析-0175-Remez算法`

## R01

本条目实现一个可直接运行的 Remez 交换算法 MVP，用于在区间 `[a,b]` 上构造次数为 `n` 的多项式一致逼近（minimax / Chebyshev uniform approximation）。

演示任务固定为：
- 目标函数：`f(x) = exp(x)`
- 区间：`[-1, 1]`
- 多项式次数：`5`

脚本会给出迭代历史、最终多项式系数、等波纹（equioscillation）检查，并与同次数最小二乘拟合进行 `L∞` 误差比较。

## R02

问题定义：
- 输入：函数 `f`、区间 `[a,b]`、次数 `n`、最大迭代步数、误差网格密度
- 输出：多项式 `p_n(x)` 与其最大绝对误差估计 `max_x |f(x)-p_n(x)|`
- 目标：最小化一致范数误差

数学目标可写成：

`min_{p in Π_n} ||f - p||_∞`

其中 `Π_n` 是不超过 `n` 次的多项式空间。

## R03

Remez 核心依据是等振荡定理（alternation theorem）：

若 `p*` 是 `Π_n` 上的一致最优逼近，则误差函数
`e(x)=f(x)-p*(x)`
在区间内至少存在 `n+2` 个按横坐标有序的点，使误差绝对值相同且符号交替。

据此，Remez 迭代在每一步做两件事：
1. 假设当前参考点集 `x_0,...,x_{n+1}` 已满足交替符号，解线性方程组：
   `p(x_i) + (-1)^i E = f(x_i)`
2. 在稠密网格上找 `|e(x)|` 的极值点，更新参考点，使其更接近“等幅交替”。

## R04

本实现流程（交换法）如下：
1. 用 Chebyshev 分布初始化 `n+2` 个参考点。
2. 构造并求解 `(n+2) x (n+2)` 线性系统，得到多项式系数与波纹误差 `E`。
3. 在稠密网格上计算误差 `e(x)=f(x)-p(x)`。
4. 提取 `|e(x)|` 的局部极大点（含端点）。
5. 通过“符号交替压缩”筛出交替符号的候选极值序列。
6. 若候选点过多，选择长度为 `n+2` 且最小误差幅值最大的窗口；若不足则用最大误差点补齐。
7. 以新参考点进入下一轮。
8. 当波纹误差与网格最大误差足够接近且参考点移动很小，或达到迭代上限时结束。

## R05

关键数据结构（均为 `numpy.ndarray`）：
- `x_ref: (n+2,)` 当前参考点
- `A: (n+2, n+2)` Remez 线性系统矩阵
- `coeffs_asc: (n+1,)` 多项式系数（升幂：`c0, c1, ...`）
- `grid: (M,)` 误差搜索网格
- `err: (M,)` 网格误差值
- `history: list[(iter, max|err|, |E|, gap, move)]` 迭代诊断信息

## R06

正确性检查方式：
- 线性方程组满足：在参考点上 `f(x_i)-p(x_i)` 与 `(-1)^i` 对应，且幅值接近 `|E|`。
- 终止后验证：`|E|` 与网格上 `max|err|` 的差距（`gap`）应较小。
- 打印最终参考点误差符号，检查是否出现接近交替的结构。
- 与同次数最小二乘多项式对比 `L∞` 误差，通常 Remez 的一致误差更小。

## R07

复杂度估计（设次数 `n`，网格点数 `M`，迭代轮数 `K`）：
- 每轮解线性方程组：`O((n+2)^3)`
- 每轮网格误差评估与极值扫描：`O(M * n) + O(M)`
- 总时间复杂度：`O(K * ((n+2)^3 + M*n))`
- 空间复杂度：`O(M + n^2)`

在本 MVP（`n=5`）中，成本主要来自网格扫描，线性系统成本很小。

## R08

边界与异常处理：
- `degree < 0`：抛 `ValueError`
- `a >= b` 或区间端点非有限值：抛 `ValueError`
- `max_iter <= 0` 或 `tol <= 0`：抛 `ValueError`
- `grid_size` 过小导致极值检测不稳定：抛 `ValueError`
- 线性系统病态或奇异：抛 `RuntimeError`

这保证脚本在参数异常时尽早失败，而不是静默输出不可解释结果。

## R09

MVP 取舍说明：
- 仅依赖 `numpy` 与标准库，避免黑盒数值优化器。
- 多项式基选择单项式升幂 + Horner 评估，代码最短且便于审查。
- 误差极值点通过稠密网格近似，不做连续域精确极值求解。
- 用 `exp(x)` 做固定演示，保证无需交互输入即可复现实验。

## R10

`demo.py` 函数职责：
- `validate_inputs`：参数合法性校验
- `chebyshev_reference_points`：生成初始参考点
- `eval_poly_asc`：升幂系数下的 Horner 评估
- `build_remez_system`：构造 Remez 线性系统
- `solve_remez_step`：求单轮线性子问题
- `local_abs_extrema_indices`：提取 `|err|` 局部极大点
- `enforce_alternation`：压缩为符号交替序列
- `best_alternating_window`：过多候选时挑选最佳窗口
- `select_reference_points`：生成下一轮参考点
- `remez_minimax`：主迭代流程
- `least_squares_poly`：最小二乘基线模型
- `main`：组织演示输出

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0175-Remez算法
python3 demo.py
```

脚本无交互输入，直接打印迭代与误差对比结果。

## R12

输出字段解释：
- `iter`：Remez 第几轮
- `max|err|`：当前轮在稠密网格上的最大绝对误差
- `|ripple|`：由线性系统估计的等幅误差绝对值
- `gap`：`max|err|` 与 `|ripple|` 的相对差
- `max_ref_move`：本轮参考点最大位移
- `coeffs_asc`：最终多项式升幂系数
- `err(x_ref)`：最终参考点处误差值
- `Remez sup-norm error / Least-squares sup-norm err`：一致误差比较

## R13

建议最小测试集：
- 正常路径：默认参数执行并输出完整迭代记录
- 参数异常：`degree=-1`、`a>=b`、`tol<=0`
- 网格异常：`grid_size` 设得很小触发保护
- 数值鲁棒：提高次数（如 `degree=10`）观察是否触发病态系统异常
- 结果合理性：检查 `err(x_ref)` 是否近似等幅且符号交替

## R14

关键可调参数：
- `degree`：多项式次数，越高拟合能力越强但系统条件数可能变差
- `max_iter`：交换迭代上限
- `grid_size`：误差极值搜索分辨率，越大越稳但更慢
- `tol`：收敛阈值（波纹差距）
- 目标函数与区间 `f, [a,b]`：决定逼近难度与误差形态

实践上通常先用中等 `grid_size` 快速验证，再加密网格确认误差。

## R15

与常见方案对比：
- 最小二乘拟合（`L2`）：整体平均误差更小，但 `L∞` 峰值不一定最优
- Chebyshev 插值：节点选择优于等距插值，但仍非严格 minimax
- Remez（本条目）：直接面向 `L∞` 目标，典型特征是误差“等波纹”分布

因此在“最坏误差控制”任务中，Remez 通常比同次数 `L2` 拟合更合适。

## R16

典型应用场景：
- 数学函数库近似（如 `exp/log/sin` 的多项式核）
- DSP 中 FIR 等波纹设计的思想基础（Parks-McClellan 同类交换思想）
- 嵌入式/硬件中用低次数多项式替代表达式
- 对最大绝对误差有硬约束的工程近似

## R17

可扩展方向：
- 将单项式基替换为 Chebyshev 基，改善高次稳定性
- 用分段多项式或有理函数版本（rational Remez）提升逼近能力
- 将网格极值搜索替换为连续优化，减少离散化误差
- 增加多目标输出（误差分布图、自动精度报告）

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 固定 `f(x)=exp(x)`、区间 `[-1,1]` 和次数 `5`，调用 `remez_minimax`。  
2. `remez_minimax` 先用 `chebyshev_reference_points` 生成 `n+2` 初始参考点。  
3. 每轮调用 `solve_remez_step`：`build_remez_system` 构建矩阵并解出 `coeffs` 与 `ripple E`。  
4. 用 `eval_poly_asc` 在稠密网格上计算 `err(x)=f(x)-p(x)`，得到 `max|err|` 与 `gap`。  
5. `local_abs_extrema_indices` 找 `|err|` 局部极大点（含端点），`enforce_alternation` 压缩成符号交替序列。  
6. `best_alternating_window`/`select_reference_points` 选出新的 `n+2` 参考点并进入下一轮交换。  
7. 收敛后再解一次最终线性系统，输出最终系数、波纹误差、参考点与误差历史。  
8. `main` 额外调用 `least_squares_poly` 做同次数 `L2` 基线，对比两者 `L∞` 误差。  

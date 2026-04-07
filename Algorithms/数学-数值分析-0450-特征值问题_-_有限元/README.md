# 特征值问题 - 有限元

- UID: `MATH-0450`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `450`
- 目标目录: `Algorithms/数学-数值分析-0450-特征值问题_-_有限元`

## R01

有限元特征值问题的典型形式是广义矩阵本征问题：

`K u_h = lambda_h M u_h`

其中：
- `K` 是刚度矩阵（来自双线性型 `a(u,v)`）
- `M` 是质量矩阵（来自内积 `(u,v)`）
- `lambda_h`、`u_h` 是离散本征值和本征向量

本条目给出一个 1D 最小可运行 MVP，重点展示从弱形式到离散本征问题的完整链路。

## R02

MVP 目标问题：

- 微分方程：`-u''(x) = lambda u(x), x in (0,1)`
- 边界条件：`u(0)=u(1)=0`（齐次 Dirichlet）
- 精确本征值：`lambda_k = (k*pi)^2, k=1,2,...`
- 精确本征函数：`u_k(x)=sin(k*pi*x)`（差一个归一化常数）

这样能直接对比数值本征值与解析值，验证收敛性。

## R03

离散设置：

- 空间：`V_h` 取 1D 线性有限元空间（`P1`）
- 网格：`[0,1]` 均匀剖分为 `n_elem` 个单元
- 局部刚度矩阵：`K_e = (1/h) * [[1,-1],[-1,1]]`
- 局部质量矩阵：`M_e = (h/6) * [[2,1],[1,2]]`

装配后得到全局 `K, M`，再消去边界自由度得到内部广义本征问题。

## R04

弱形式（连续问题）：在 `H_0^1(0,1)` 中找 `u!=0` 与 `lambda`，使

`int_0^1 u'(x)v'(x) dx = lambda int_0^1 u(x)v(x) dx, for all v in H_0^1(0,1)`。

有限元离散后，把测试函数限制到 `V_h`，得到矩阵形式 `K u_h = lambda_h M u_h`。

## R05

算法核心思想：

1. 用有限元离散把“无穷维本征问题”变成“有限维广义本征问题”。
2. 利用 `K` 对称正定、`M` 对称正定（Dirichlet 后）性质做稳定求解。
3. 网格加密（`h` 变小）时，本征值误差通常按 `O(h^2)` 衰减（对光滑模态和线性元）。

## R06

MVP 伪代码：

```text
input: n_elem_list, num_modes
for each n_elem:
    assemble global K, M from P1 local matrices
    eliminate boundary dof -> K_in, M_in
    solve K_in v = lambda M_in v for smallest num_modes
    compare lambda with exact (k*pi)^2
compute empirical rates from consecutive meshes
print report and run checks
```

## R07

正确性直觉：

- 有限元本质上是 Rayleigh-Ritz 逼近；离散本征值来自有限维子空间上的最优近似。
- 对本问题，`K` 与 `M` 都是对称矩阵，`M` 还是正定，因此本征值实且可排序。
- 随着网格细化，离散空间扩张，低阶模态的本征值和本征函数逐步逼近真值。

## R08

复杂度（本 MVP 使用 dense 矩阵）：

- 装配：`O(n_elem)`（每个单元常数规模操作）
- 本征求解：`O(n^3)`（`n = n_elem - 1`，dense 广义特征值）
- 存储：`O(n^2)`（dense `K, M`）

工程规模通常应切换到稀疏存储和稀疏本征求解器。

## R09

数值稳定性与实现细节：

- 必须正确消去 Dirichlet 边界自由度，否则会引入零模态或奇异结构。
- 本征向量应进行 `M`-范数归一化（`v^T M v = 1`）以便稳定比较模态。
- 在无 SciPy 环境下，`demo.py` 用 Cholesky 降阶把广义问题转为标准对称问题作为回退方案。

## R10

常见失败模式：

1. 局部矩阵装配索引错位，导致 `K`/`M` 非对称。
2. 边界条件处理遗漏，出现错误的最小本征值。
3. 误把高阶模态排序当作低阶，导致误差评估混乱。
4. 只看单层网格结果，无法判断是否满足期望收敛阶。

## R11

`demo.py` 的定位：

- 不是大型 PDE 框架封装，而是“手写可审计”最小实现。
- 明确包含：装配、边界处理、广义本征值求解、误差统计、收敛率估计、自动检查。
- 无交互输入，运行即可复现结果。

## R12

运行方式：

```bash
cd Algorithms/数学-数值分析-0450-特征值问题_-_有限元
python3 demo.py
```

默认网格：`n_elem = [16, 32, 64, 128]`，默认比较前 `4` 个模态。

## R13

输出字段说明：

- `lam{i}`：第 `i` 个离散本征值
- `rel{i}`：相对误差 `|lam_i - exact_i| / exact_i`
- `rate{i}`：相邻网格层经验收敛阶 `log2(err_h / err_{h/2})`

一般会看到低阶模态 `rate` 接近 2，符合线性元本征值误差阶预期。

## R14

本条目与其他 FEM 主题关系：

- 与 h/p/hp 版本 FEM：它们偏重边值问题求解；本条目聚焦本征谱。
- 与模态分析/振动问题：矩阵形式相同，常见于结构固有频率计算。
- 与谱方法对比：有限元更易处理复杂几何和局部加密。

## R15

工程扩展建议：

1. 改用稀疏矩阵 `scipy.sparse` 与 `scipy.sparse.linalg.eigsh` 求大规模低频模态。
2. 扩展到 2D/3D（三角形/四面体）并引入网格质量控制。
3. 加入 shift-invert 与预条件技术提升高频模态求解效率。
4. 在模态对比时加入 MAC（Modal Assurance Criterion）做模态匹配。

## R16

验收检查建议：

1. `python3 demo.py` 无需输入且可直接运行。
2. 输出本征值严格递增且均为正。
3. `rel1` 随网格加密单调下降。
4. 末层 `rate1` 接近二阶（例如 >1.8）。
5. 脚本最终打印 `All checks passed.`。

## R17

当前 MVP 的边界：

- 仅 1D 区间，且仅齐次 Dirichlet 边界。
- 仅 `P1` 线性单元，未含高阶单元与非均匀自适应网格。
- 主目标是教学与验证，不追求大规模性能最优。

该范围有意保持最小化，以确保代码可读和流程透明。

## R18

`demo.py` 源码级算法流程（9 步，非黑盒）：

1. `main()` 创建 `ExperimentConfig`（网格层级和模态数），调用 `run_refinement_study()`。
2. `run_refinement_study()` 对每个 `n_elem` 调用 `assemble_p1_matrices()`，逐单元累加 `K_e` 和 `M_e` 到全局 `K, M`。
3. 在 `run_refinement_study()` 中裁剪内部自由度，形成 `K_in, M_in`（等价施加 Dirichlet 边界）。
4. 调用 `solve_generalized_eigenproblem(K_in, M_in, num_modes)` 求最小几个本征对。
5. 若 SciPy 可用，使用 `scipy.linalg.eigh(K_in, M_in, subset_by_index=...)` 直接解对称广义本征问题。
6. 若 SciPy 不可用，先对 `M_in` 做 Cholesky 分解 `M=LL^T`，构造 `L^{-1} K L^{-T}`，再用 `numpy.linalg.eigh` 解标准对称本征问题并回代本征向量。
7. 对求得本征向量逐个做 `M`-范数归一化（`v^T M v = 1`），保证模态比较稳定。
8. 计算解析本征值 `(k*pi)^2` 与相对误差 `rel_err`，并在 `convergence_rates()` 中计算经验收敛阶。
9. `print_report()` 输出表格，`run_checks()` 验证正性、单调性、误差下降和收敛阶阈值，最终打印 `All checks passed.`。

# 多重网格方法 (Multigrid Method)

- UID: `PHYS-0339`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `350`
- 目标目录: `Algorithms/物理-计算物理-0350-多重网格方法_(Multigrid_Method)`

## R01

多重网格方法（Multigrid, MG）是求解椭圆型偏微分方程离散线性系统的经典高效算法。其核心思想是：
- 高频误差在细网格上用平滑器快速衰减；
- 低频误差在粗网格上变成相对高频并被高效消除；
- 通过“细网格-粗网格”往返校正，达到接近 `O(N)` 的整体复杂度。

本条目实现一个最小可运行 MVP，演示 1D Poisson 方程上的 V-cycle 多重网格求解。

## R02

MVP 问题定义：

求解边值问题

`-u''(x)=f(x), x in (0,1), u(0)=u(1)=0`

选择制造解（manufactured solution）：

`u_exact(x)=sin(pi x)`，因此 `f(x)=pi^2 sin(pi x)`。

离散后得到三对角线性系统 `A u = b`，目标是用多重网格快速逼近离散解，并和解析/参考解做对照验证。

## R03

离散化采用二阶中心差分：

`(A u)_i = (2u_i - u_{i-1} - u_{i+1}) / h^2`。

在最细层网格中取 `n_finest=255` 个内部点（满足 `2^k-1` 结构，便于逐层二分粗化），边界值固定为零（Dirichlet）。

该设置既保持数值精度，也使层次构造和转移算子表达清晰。

## R04

本实现采用标准 V-cycle 结构：

1. 细网格预平滑（weighted Jacobi）。
2. 计算残差 `r=b-Au`。
3. 残差限制（restriction）到粗网格，形成误差方程右端。
4. 在粗网格递归求解误差近似。
5. 将粗网格误差延拓（prolongation）回细网格并校正。
6. 细网格后平滑。

这一流程对应“平滑 + 粗网格校正”的经典两阶段机制。

## R05

平滑器选用加权 Jacobi：

`u_i^(new) = (1-omega) u_i + omega * 0.5 * (u_{i-1}+u_{i+1}+h^2 b_i)`。

其中 `omega=2/3`。理由：
- 实现简单、向量化友好；
- 对高频误差有稳定衰减；
- 适合教学型 MVP 展示多重网格主链路。

脚本默认预平滑和后平滑步数均为 3。

## R06

网格转移算子：

1. 限制（Full Weighting）：

`r_c[j] = 0.25*r_f[2j] + 0.5*r_f[2j+1] + 0.25*r_f[2j+2]`。

2. 延拓（Linear Interpolation）：
- 粗细重合点注入；
- 相邻粗点之间取线性平均；
- 靠边界半点按零边界处理。

这两者组合在 1D Poisson 场景下稳定且常用。

## R07

层次结构自动生成：

从 `n=255` 开始，按 `n_c=(n_f-1)/2` 递归粗化到最粗层 `n<=3`。本配置得到层级：

`[255, 127, 63, 31, 15, 7, 3]`。

最粗层不再迭代，直接线性代数求解（`np.linalg.solve`），作为 V-cycle 终止条件。

## R08

复杂度分析（`N` 为最细层未知量数）：

- 单层平滑/残差/转移操作均为 `O(n_l)`；
- 全层求和 `sum_l O(n_l) = O(N)`；
- 单个 V-cycle 近似线性复杂度 `O(N)`；
- 空间复杂度为多层向量总和，仍为 `O(N)` 量级。

这也是多重网格相比单层迭代法的核心优势。

## R09

数值稳定与工程细节：

- 只允许 `n_finest=2^k-1`，避免非法粗化；
- 参数检查覆盖平滑步数、`omega`、循环次数、容差；
- 采用离散 `L2` 范数 `sqrt(h * sum(v_i^2))` 监控收敛；
- 残差满足阈值 `residual_tol` 时提前停止。

这些约束使得脚本可复现且便于自动验证。

## R10

MVP 技术栈：

- `numpy`：网格运算、平滑器、递归 V-cycle 核心计算
- `pandas`：收敛历史表格化输出
- `scipy.sparse` + `scipy.sparse.linalg.spsolve`：只用于离散系统参考解交叉校验

算法主流程（平滑、限制、延拓、递归）均在源码显式实现，不依赖黑箱多重网格库。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0350-多重网格方法_(Multigrid_Method)
uv run python demo.py
```

脚本无交互输入，直接打印参数、按 cycle 的收敛表、最终指标和 `Validation: PASS/FAIL`。

## R12

输出表字段解释：

- `cycle`: V-cycle 编号（`0` 表示初始猜测）
- `res_l2`: 当前残差离散 `L2` 范数
- `res_reduction`: 相对初始残差缩减比
- `err_l2`: 与解析解 `sin(pi x)` 的离散 `L2` 误差
- `max_abs_err`: 点值最大绝对误差（`L_inf` 指标）

此外还会输出 `|u_mg - u_scipy|_inf` 作为离散参考一致性指标。

## R13

内置验收阈值：

1. `final res_reduction < 1e-8`
2. `final res_l2 < 1e-7`
3. `final err_l2 < 2e-5`
4. `final max_abs_err < 4e-5`
5. `|u_mg-u_scipy|_inf < 1e-10`

全部满足则输出 `Validation: PASS`，否则抛出断言错误并返回非零退出码。

## R14

当前 MVP 局限：

- 仅覆盖 1D Poisson + 零 Dirichlet 边界；
- 使用固定 V-cycle，不含 W-cycle/FMG；
- 平滑器仅为 weighted Jacobi，未比较 GS/红黑 GS；
- 未实现变系数算子、非线性问题或自适应网格。

因此它是“算法可审计原型”，不是通用工业级 PDE 求解框架。

## R15

可扩展方向：

- 扩展到 2D/3D Poisson 或 Helmholtz 类问题；
- 支持几何多重网格（GMG）与代数多重网格（AMG）对比；
- 增加 V/W/FMG 周期、平滑器种类与参数扫描；
- 引入并行（NumPy+Numba/CUDA）与稀疏块结构优化。

## R16

典型应用场景：

- 计算物理中的稳态扩散、静电势、压力泊松方程；
- CFD 投影法中的压力校正子问题；
- 大规模有限差分/有限元离散线性系统预条件；
- 数值方法课程中“误差频谱 + 层次求解”教学演示。

## R17

与常见方案比较：

- 相比 Jacobi/Gauss-Seidel 单层迭代：MG 收敛更快，网格加密后优势明显。
- 相比一次性稠密直接法：MG 内存更省，适合大规模问题。
- 相比纯黑箱求解器调用：本实现完整暴露了平滑、限制、延拓、递归逻辑，便于审计与改造。

本条目强调“可运行 + 可解释 + 可验证”的最小闭环。

## R18

`demo.py` 源码级流程拆解（9 步）：

1. `MGConfig` 定义最细网格规模、平滑参数、循环次数和收敛阈值，并在 `validate()` 做参数合法性检查。  
2. `build_levels` 根据 `n_c=(n_f-1)/2` 生成从细到粗的层次结构，直到最粗层 `n<=3`。  
3. 在 `main` 中构造制造解测试：`u_exact=sin(pi x)` 与右端 `f=pi^2 sin(pi x)`，并以零向量作为初值。  
4. `apply_operator` 实现一维 Poisson 三对角算子，供残差计算 `r=b-Au` 使用。  
5. `weighted_jacobi` 执行预/后平滑，优先抑制细网格高频误差。  
6. `restrict_full_weighting` 把细层残差投影到粗层，形成粗网格误差方程右端。  
7. `v_cycle` 递归调用粗层求解；最粗层用 `direct_solve_coarsest` 直接解线性系统，再经 `prolong_linear` 延拓回细层做校正。  
8. `main` 逐个 cycle 记录 `res_l2/res_reduction/err_l2/max_abs_err` 到 `pandas` 表，形成可审计收敛历史。  
9. 用 `scipy_reference_solution` 对同一离散系统做参考交叉校验，并执行 5 条断言阈值，最终输出 `Validation: PASS/FAIL`。

上述流程完整展示了多重网格的源码级算法链路，第三方库仅用于基础数值和参考比对，不掩盖核心步骤。

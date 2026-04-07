# 谐振腔 (Cavity Resonator)

- UID: `PHYS-0179`
- 学科: `物理`
- 分类: `电磁学`
- 源序号: `180`
- 目标目录: `Algorithms/物理-电磁学-0180-谐振腔_(Cavity_Resonator)`

## R01

谐振腔用于在封闭导体结构中支持离散电磁本征模。对理想导体矩形腔，场只能在特定本征频率上稳定振荡。本条目给出一个最小可运行 MVP：把二维横向 Helmholtz 本征值问题离散成稀疏矩阵特征值问题，数值求解前几个 TM 模式频率，并与解析公式逐项对比。

## R02

对 z 向电场主分量（TM\(_z\)）可写为：
\[
-\nabla_t^2 E_z = k_c^2 E_z,
\]
其中 \(\nabla_t^2\) 是横向二维 Laplace 算子，\(k_c\) 为截止波数。矩形 PEC 边界满足：
\[
E_z=0\quad \text{on boundary}.
\]
对应本征频率：
\[
f=\frac{c}{2\pi}\,k_c.
\]
解析 TM\(_{mn}\) 模式（\(m,n\ge1\)）满足：
\[
k_{c,mn}^2=\left(\frac{m\pi}{a}\right)^2+\left(\frac{n\pi}{b}\right)^2,
\quad
f_{mn}=\frac{c}{2\pi}\sqrt{k_{c,mn}^2}.
\]

## R03

MVP 问题设置：
- 几何：二维矩形横截面 \([0,a]\times[0,b]\)。
- 边界：全边界 Dirichlet（对应 PEC 条件下 TM\(_z\) 的 \(E_z=0\)）。
- 离散：内部网格点 `nx × ny`，五点差分离散 \(-\nabla_t^2\)。
- 求解：计算最小的 `num_modes` 个本征值 \(\lambda_i\approx k_{c,i}^2\)。
- 验证：把数值频率和解析 TM\(_{mn}\) 频率按升序对齐并计算相对误差。

## R04

二维五点差分（内部节点 \((i,j)\)）写作：
\[
-\nabla_t^2 E_{i,j}
\approx
\left(\frac{2}{h_x^2}+\frac{2}{h_y^2}\right)E_{i,j}
-\frac{1}{h_x^2}(E_{i-1,j}+E_{i+1,j})
-\frac{1}{h_y^2}(E_{i,j-1}+E_{i,j+1}).
\]
将所有内部未知量按向量堆叠后得到稀疏矩阵本征问题：
\[
A\mathbf{u}=\lambda\mathbf{u},\qquad A\approx-\nabla_t^2.
\]
实现采用 Kronecker 结构：
\[
A=-(I_y\otimes L_x + L_y\otimes I_x).
\]

## R05

该 MVP 是静态本征值求解，不涉及 CFL 稳定性约束；关键数值控制在于：
- 网格分辨率（`nx`,`ny`）影响本征频率误差；
- 所求模态数 `num_modes` 不能大于系统维度减一；
- 解析枚举上限 `max_mode_index` 要足够覆盖前 `num_modes` 个低阶模态。

## R06

数值性质：
- 空间离散二阶精度，频率误差随网格加密下降；
- 稀疏对称正定矩阵适合 Lanczos/Arnoldi 类迭代本征求解；
- 当腔体接近简并几何（如正方形）时，相邻模可能接近，模态匹配应更谨慎；
- 当前示例取非方形腔 (`a != b`)，减少低阶模式简并。

## R07

复杂度（设内部未知总数 \(N=nx\cdot ny\)）：
- 构建稀疏矩阵：时间 \(O(N)\)，空间 \(O(N)\)（五点模板非零元约 \(5N\)）。
- 迭代求前 `k` 个小本征值：典型代价近似 \(O(k\cdot \text{nnz}\cdot it)\)，其中 `it` 为迭代轮数。
- 总体内存主要由稀疏矩阵和 `k` 个本征向量主导。

## R08

核心数据结构：
- `CavityConfig`：几何、网格、模态数、光速等参数。
- `AnalyticMode`：解析模式 `(m,n)`、`lambda_mn`、`frequency_hz`。
- `scipy.sparse.csr_matrix`：二维离散 \(-\nabla^2\) 稀疏矩阵。
- `eigenvalues/eigenvectors`：数值本征值和本征向量。
- `result: dict`：汇总误差、网格步长、正交性检查等指标。

## R09

伪代码：

```text
input a, b, nx, ny, num_modes
build 1D Dirichlet Laplacian Lx, Ly
build 2D sparse matrix A = -(Iy⊗Lx + Ly⊗Ix)
solve smallest k eigenpairs of A
convert lambda_i -> f_i = c/(2π)*sqrt(lambda_i)
enumerate analytic TM_mn frequencies
sort and take first k analytic modes
compute relative error |f_num - f_ana| / f_ana
report frequency table and orthogonality diagnostics
```

## R10

`demo.py` 默认参数：
- `a = 0.22 m`
- `b = 0.10 m`
- `c = 299792458 m/s`
- `nx = 80`, `ny = 40`（内部网格）
- `num_modes = 6`
- `max_mode_index = 8`

这组参数可在较短时间内得到稳定、可读的模式频率对比结果。

## R11

脚本输出内容：
- 几何与网格：`a,b,nx,ny,hx,hy`。
- 线性系统规模：`system_size` 与求解模态数。
- 频率对比表：每阶模式的 `numerical(GHz)`、`analytic(GHz)`、`rel_error`。
- 向量质量检查：
  - `max|diag(V^T V)-1|`
  - `max|offdiag(V^T V)|`
  - 第一模态边界值最大绝对值（应接近 0）。

## R12

`demo.py` 函数分工：
- `_laplacian_1d_dirichlet`：构建 1D 三对角差分算子。
- `build_negative_laplacian_2d`：用 Kronecker 构建二维稀疏算子。
- `analytic_tm_modes`：生成并排序解析 TM\(_{mn}\) 模式。
- `solve_cavity_modes`：执行本征求解、误差评估、正交性检查。
- `main`：组织参数并打印结果摘要。

## R13

运行方式：

```bash
uv run python demo.py
```

在当前算法目录执行即可，无需交互输入。

## R14

常见错误与规避：
- 把边界点也当成未知量会破坏 Dirichlet 条件。
- `hx = a/nx` 与 `hx = a/(nx+1)` 混淆会导致频率系统性偏差。
- 直接用稠密矩阵做大网格本征分解会导致不必要的内存和时间开销。
- 解析模式枚举上限过小，可能误把高阶解析模和低阶数值模对齐。

## R15

最小验证策略：
1. 使用默认参数运行，确认前 6 阶相对误差均显著小于 1%。
2. 提高网格到 `nx=120, ny=60`，观察低阶模式误差进一步下降。
3. 把 `a,b` 改成接近值（如 `a=0.2,b=0.2`）测试近简并场景，并检查模式排序稳定性。
4. 确认 `V^T V` 的对角接近 1、非对角接近 0。

## R16

适用范围与局限：
- 适用：矩形谐振腔低阶模态教学、数值离散验证、频率基线估计。
- 局限：
  - 仅处理二维横向 TM\(_z\) 归约模型；
  - 未包含损耗、耦合端口、材料色散与非理想导体；
  - 不覆盖复杂三维腔体（圆柱、椭圆、任意曲面）。

## R17

可扩展方向：
- 扩展到 TE/TM 全家族并引入三维腔体本征求解。
- 引入有限元离散以支持复杂几何边界。
- 加入品质因数（Q）与耗散模型，研究线宽和衰减。
- 增加模态场分布可视化与节点/腹点自动识别。
- 将模式求解与端口激励耦合，做频率扫描与 S 参数近似。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 创建 `CavityConfig`，调用 `solve_cavity_modes`。
2. `solve_cavity_modes` 先校验 `a,b,c,num_modes` 等参数合法性。
3. 调用 `build_negative_laplacian_2d`：
   - 先由 `_laplacian_1d_dirichlet` 构建 `Lx/Ly`；
   - 再用 Kronecker 组合得到二维稀疏矩阵 `A`。
4. 设定 `k=min(num_modes, N-1)`，调用 `scipy.sparse.linalg.eigsh(A, which="SM")` 求最小 `k` 个本征对。
5. 对本征值排序并取非负部分，按 \(f_i=\frac{c}{2\pi}\sqrt{\lambda_i}\) 转成数值频率。
6. 调用 `analytic_tm_modes` 枚举 `(m,n)` 解析模式，按频率排序并截取前 `k` 个。
7. 逐项计算数值频率与解析频率的相对误差，同时计算 `V^T V` 的对角偏差与非对角最大值检查正交性。
8. 重构第一模态到二维网格并补零边界，统计边界最大值，最终在 `main` 打印频率表与质量指标。

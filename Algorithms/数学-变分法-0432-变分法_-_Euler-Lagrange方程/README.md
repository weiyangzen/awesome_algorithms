# 变分法 - Euler-Lagrange方程

- UID: `MATH-0432`
- 学科: `数学`
- 分类: `变分法`
- 源序号: `432`
- 目标目录: `Algorithms/数学-变分法-0432-变分法_-_Euler-Lagrange方程`

## R01

本条目实现一个可运行的 Euler-Lagrange 方程最小示例（MVP）：

- 目标泛函：
  `J[y] = ∫_0^1 (1/2 * y'(x)^2 + 1/2 * λ * y(x)^2) dx`
- 边界条件：
  `y(0) = y0, y(1) = y1`
- 对应 Euler-Lagrange 方程：
  `y''(x) - λ y(x) = 0`

`demo.py` 用有限差分离散后求解边值线性系统，并与解析解对比，输出误差、残差、泛函值差异。

## R02

问题定义（对应本目录实现）：

- 输入：
  - `lam (λ) >= 0`：势能项系数；
  - `y0, y1`：两端固定边界值；
  - `num_points >= 3`：网格点总数（含两端边界点）。
- 输出：
  - 数值解曲线 `y_num`（网格上）；
  - 解析解 `y_ref`（同一网格）；
  - 误差指标（`max_abs_err`, `L2 error`）；
  - Euler-Lagrange 离散残差无穷范数；
  - 泛函值 `J[y_num]` 与 `J[y_ref]`。

脚本内置多组固定案例，无需任何交互输入。

## R03

数学基础：

1. 拉格朗日量：`L(y, y') = 1/2 * y'^2 + 1/2 * λ y^2`。  
2. Euler-Lagrange 方程：
   `d/dx(∂L/∂y') - ∂L/∂y = 0`。  
3. 对当前 `L`，有 `∂L/∂y' = y'`，`∂L/∂y = λ y`，故：
   `y'' - λ y = 0`。  
4. 当 `λ > 0` 时，解析解形如：
   `y(x) = c1 sinh(sqrt(λ)x) + c2 cosh(sqrt(λ)x)`，结合边界可定 `c1, c2`。  
5. 当 `λ -> 0` 时退化为自由粒子情形，解近似线性插值。

## R04

离散算法流程（高层）：

1. 在 `[0,1]` 上构造均匀网格，步长 `h = 1/(N-1)`。  
2. 用中心差分近似二阶导：
   `y''(x_i) ≈ (y_{i+1} - 2y_i + y_{i-1}) / h^2`。  
3. 代入 `y'' - λy = 0`，得到内部点方程：
   `-y_{i-1} + (2 + λh^2)y_i - y_{i+1} = 0`。  
4. 把边界值 `y0, y1` 移到右端，形成线性系统 `A y_interior = rhs`。  
5. 调用 `numpy.linalg.solve` 求内部未知量。  
6. 组装完整解向量 `[y0, y_interior..., y1]`。  
7. 计算解析解、误差、残差与泛函值用于验证。

## R05

核心数据结构：

- `CaseConfig`（`dataclass`）：
  - `name`: 案例名；
  - `lam`: `λ`；
  - `y0`, `y1`: 边界条件；
  - `num_points`: 网格规模。
- 线性系统：
  - `A in R^((N-2)*(N-2))`：三对角稠密矩阵；
  - `rhs in R^(N-2)`：右端项（只在首尾含边界贡献）。
- 结果字典：
  `max_abs_err`, `l2_err`, `residual_inf`, `action_gap`, `pass_flag`。

## R06

正确性要点：

- 变分条件正确：由 Euler-Lagrange 公式直接推得 `y'' - λy = 0`。  
- 离散方程正确：中心差分为二阶精度，离散误差量级 `O(h^2)`。  
- 边界处理正确：首尾内部方程把 `y0`, `y1` 纳入 `rhs`。  
- 结果可检验：
  - 与解析解逐点比较；
  - 检查离散 EL 残差 `||r||_inf`；
  - 比较数值解/解析解的泛函值。

## R07

复杂度分析（`N = num_points`, 内部未知数 `M = N-2`）：

- 构建矩阵和右端：`O(M)`（本实现采用稠密构造，实际写入成本约 `O(M^2)` 但非主导）。  
- 线性求解（稠密 `np.linalg.solve`）：`O(M^3)` 时间，`O(M^2)` 空间。  
- 误差与泛函后处理：`O(N)`。  

如果将来改为专用三对角追赶法（Thomas），可降到 `O(M)` 时间与 `O(M)` 空间。

## R08

边界与异常处理：

- `num_points < 3`：抛 `ValueError`。  
- `lam < 0`：抛 `ValueError`（本 MVP 约束 `λ >= 0`）。  
- `lam/y0/y1` 非有限值：抛 `ValueError`。  
- `lam` 极小（接近 0）时，解析解分支回退到线性插值，避免 `sinh` 分母不稳定。  
- 所有案例都固定参数运行，不依赖外部输入状态。

## R09

MVP 取舍说明：

- 聚焦一维固定端点问题，不扩展到高维或非线性拉格朗日量。  
- 使用 `numpy` 即可完成最小实现，不引入额外复杂框架。  
- 线性系统暂用通用稠密求解器，代码更直接，便于教学和审计。  
- 目标是“可读、可验证、可复现”，而非覆盖所有变分法工程场景。

## R10

`demo.py` 主要函数职责：

- `validate_config`：校验参数合法性。  
- `build_linear_system`：从离散 Euler-Lagrange 方程构造 `A, rhs`。  
- `solve_discrete_euler_lagrange`：求解内部未知并拼接完整边值解。  
- `analytic_solution`：计算同一边界条件下的闭式参考解。  
- `action_value`：计算离散泛函值 `J_h[y]`。  
- `euler_lagrange_residual`：评估离散方程残差。  
- `run_case`：单案例执行、打印指标并返回结果。  
- `main`：组织固定案例并输出汇总。

## R11

运行方式：

```bash
cd Algorithms/数学-变分法-0432-变分法_-_Euler-Lagrange方程
uv run python demo.py
```

脚本不读取命令行参数，不请求用户输入。

## R12

输出字段说明：

- `grid step h`：网格步长。  
- `max |y_num - y_ref|`：数值解与解析解最大绝对误差。  
- `L2 error`：网格上的离散 `L2` 误差。  
- `max interior EL residual`：离散 Euler-Lagrange 残差无穷范数。  
- `J[y_num]`, `J[y_ref]`：数值解与解析解对应的离散泛函值。  
- `|J[y_num]-J[y_ref]|`：泛函值差异。  
- `pass finite-diff check`：是否满足基于 `O(h^2)` 的误差阈值与残差阈值。  
- `Summary`：跨案例最大误差、最大残差、最大泛函差、总通过标记。

## R13

内置最小测试集：

- `Baseline (lambda=2)`：标准指数/双曲函数型解。  
- `Stronger potential (lambda=10)`：势能更强，验证曲线弯曲更明显场景。  
- `Near free particle (lambda~0)`：接近线性解，验证小参数稳定性。  

建议补充异常测试（可自行扩展）：
- `lam < 0`（应报错）；  
- `num_points = 2`（应报错）；  
- 非有限边界值（应报错）。

## R14

可调参数：

- `lam`：控制势能项强度，越大通常曲率影响越明显。  
- `y0, y1`：边界值，直接决定解的端点锚定。  
- `num_points`：网格细化程度，越大精度更高但求解更慢。  

调参建议：
- 精度不足：增大 `num_points`。  
- 需要快速试算：先用较小 `num_points`（如 51/81），确认逻辑后再加密。  
- 若追求更高性能：改三对角专用求解器替代稠密 `solve`。

## R15

方法对比：

- 对比直接变分推导：
  - 推导给出连续方程；
  - 本实现给出可计算离散版本。  
- 对比射击法（shooting）：
  - 射击法把 BVP 转成 IVP + 根搜索；
  - 本实现直接解线性方程，稳定且实现简单。  
- 对比有限元：
  - 有限元更通用（复杂几何/边界）；
  - 本示例更轻量，适合教学和最小验证。

## R16

典型应用场景：

- 变分法教学中演示“泛函极值 -> Euler-Lagrange 方程 -> 数值离散”。  
- 一维平滑曲线设计中的基础正则项模型。  
- 更复杂 PDE/最优控制问题前的最小原型验证。  
- 数值方法课程中比较差分、有限元、射击法的基准例题。

## R17

可扩展方向：

- 把常系数 `λ` 扩展为位置依赖 `λ(x)` 或更一般 `L(x,y,y')`。  
- 从均匀网格扩展到非均匀网格。  
- 采用三对角追赶法降低复杂度。  
- 增加 `matplotlib` 可视化解曲线和误差分布。  
- 扩展到二维/高维变分问题与有限元离散。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 构造多个 `CaseConfig`，固定 `λ`、边界与网格规模，确保复现性。  
2. `run_case` 调用 `solve_discrete_euler_lagrange`，先进入 `validate_config` 做输入检查。  
3. `build_linear_system` 依据离散 EL 方程
   `-y_{i-1} + (2 + λh^2)y_i - y_{i+1}=0`
   生成三对角矩阵 `A` 与右端 `rhs`。  
4. `numpy.linalg.solve(A, rhs)` 触发线性代数内核（LAPACK `gesv` 路径）：先做 LU 分解，再执行前后代回，得到内部解 `y_interior`。  
5. 将边界值 `y0, y1` 与 `y_interior` 拼接为完整离散曲线 `y_num`。  
6. `analytic_solution` 按 `λ` 分支计算闭式参考曲线 `y_ref`（`λ≈0` 走线性回退，其他走 `sinh/cosh` 公式）。  
7. `euler_lagrange_residual` 计算内点残差 `r_i = y''_h - λy`，并统计 `||r||_inf`；同时计算逐点误差与 `L2` 误差。  
8. `action_value` 分别计算 `J[y_num]` 与 `J[y_ref]`，`main` 汇总最大误差/残差/泛函差并输出总通过标志。

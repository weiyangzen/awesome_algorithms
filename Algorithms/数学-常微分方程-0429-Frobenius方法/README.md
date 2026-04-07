# Frobenius方法

- UID: `MATH-0429`
- 学科: `数学`
- 分类: `常微分方程`
- 源序号: `429`
- 目标目录: `Algorithms/数学-常微分方程-0429-Frobenius方法`

## R01

Frobenius 方法用于求解在某点（通常取 `x=0`）具有正则奇点的线性常微分方程。其核心思想是把解写成：
`y(x)=x^r \sum_{k=0}^{\infty} a_k x^k`，其中 `r` 由指标方程确定。

本条目提供一个最小可运行 MVP：
- 选取经典正则奇点方程 Bessel 方程：`x^2 y'' + x y' + (x^2-\nu^2)y=0`；
- 手写 Frobenius 推导后的系数递推，不依赖黑盒 ODE 数值求解器；
- 用 `scipy.special.jv` 仅作为参考真值，验证误差与方程残差。

## R02

问题定义（本目录实现）：
- 输入：
  - 参数 `nu >= 0`；
  - 截断阶数 `N`；
  - 正实数网格 `x`（严格递增，`x>0`）。
- 输出：
  - 指标根 `r1, r2`；
  - 选定分支 `r` 下的 Frobenius 系数 `a_0...a_N`；
  - 截断级数解 `y_series`；
  - 参考解 `y_reference = J_nu(x)`；
  - 绝对误差与 ODE 残差。

`demo.py` 为固定样例，无需命令行参数和交互输入。

## R03

数学基础（以 Bessel 方程为例）：

1. 方程：`x^2 y'' + x y' + (x^2-\nu^2)y=0`。  
2. 设 `y = x^r\sum_{k=0}^{\infty} a_k x^k`。  
3. 代入并按幂次合并，最低次项给出指标方程：`r^2-\nu^2=0`，故 `r=\pm\nu`。  
4. 对 `k>=2` 的同次幂系数，得到递推：  
   `a_k((k+r)^2-\nu^2) + a_{k-2} = 0`，
   即 `a_k = -a_{k-2}/((k+r)^2-\nu^2)`。  
5. 本实现选 `r=\nu`，并用
   `a_0 = 1 / (2^\nu \Gamma(\nu+1))` 归一化，使级数对应 `J_\nu` 的主支。

## R04

算法流程（高层）：
1. 检查网格：一维、严格递增、有限值、且 `x>0`。  
2. 求指标根 `(\nu, -\nu)`，选择 `r=\nu`。  
3. 初始化 `a_0` 与 `a_1`，并循环应用二阶递推计算 `a_k`。  
4. 在网格上计算截断级数 `y_series = \sum_{k=0}^N a_k x^{k+r}`。  
5. 计算参考解 `y_reference = scipy.special.jv(\nu, x)`。  
6. 计算误差 `|y_series-y_reference|` 与相对 `L2` 误差。  
7. 对 `y_series` 做数值微分并代回原方程，得到残差。  
8. 输出系数、抽样表、汇总指标及 PASS/FAIL。

## R05

核心数据结构：
- `FrobeniusResult`（`dataclass`）：
  - `nu, x, n_terms`：问题参数；
  - `indicial_roots, r_used`：指标根与选用根；
  - `coefficients`：截断系数数组；
  - `y_series, y_reference`：级数解与参考解；
  - `abs_error, residual`：误差与方程残差。
- `pandas.DataFrame`：用于打印采样点结果（`x/y_series/y_reference/error/residual`）。

## R06

正确性要点：
- 理论层：指标方程与递推关系来自 Frobenius 展开逐项配平。  
- 实现层：`demo.py` 显式实现了系数递推，而非调用黑盒“直接解 ODE”。  
- 数值层：使用参考 `J_\nu` 计算误差，并独立计算 ODE 残差。  
- 诊断层：同时要求误差与残差过阈值，避免仅凭单一指标误判。

## R07

复杂度分析（设网格点数为 `m`，截断阶数为 `N`）：
- 系数递推：`O(N)`；
- 级数求值（矩阵化幂次计算）：`O(mN)`；
- 数值微分与残差：`O(m)`；
- 总时间复杂度：`O(mN)`；
- 额外空间复杂度：`O(mN)`（幂次矩阵）+ `O(m)`（结果向量）。

## R08

边界与异常处理：
- `nu < 0`：抛 `ValueError`（本 MVP 只覆盖 `nu>=0`）。
- `n_terms < 2`：抛 `ValueError`。
- 网格不合法（非一维、不递增、含 `nan/inf`、含 `x<=0`）：抛 `ValueError`。
- 递推分母接近 0：抛 `ValueError`，防止不稳定除法。
- 验证失败：抛 `RuntimeError`，防止静默输出错误结果。

## R09

MVP 取舍说明：
- 只针对一个代表性正则奇点方程（Bessel）做完整闭环。  
- 不实现一般方程的自动符号匹配与通用递推生成器，保持最小可理解实现。  
- 引入 `scipy.special.jv` 仅用于验算，不参与核心系数计算流程。  
- 使用 `pandas` 输出小表格，便于快速人工检查。

## R10

`demo.py` 主要函数职责：
- `_check_grid`：校验网格合法性。  
- `indicial_roots_bessel`：返回 Bessel 指标根。  
- `frobenius_coefficients_bessel_j`：执行 Frobenius 系数递推。  
- `evaluate_frobenius_series`：在网格上计算截断级数值。  
- `solve_bessel_with_frobenius`：整合求解、参考值、误差和残差。  
- `make_report_table`：构造采样输出表。  
- `main`：配置样例并输出 PASS/FAIL。

## R11

运行方式：

```bash
cd Algorithms/数学-常微分方程-0429-Frobenius方法
uv run python demo.py
```

脚本为非交互式执行，直接打印结果。

## R12

输出字段说明：
- 头部信息：
  - `nu`、`N`、指标根、选定 `r`。
- 系数预览：
  - 打印前若干非零 `(k, a_k)`，验证递推是否合理（Bessel 场景通常同奇偶分离）。
- 表格列：
  - `x`：采样点；
  - `y_series`：Frobenius 截断值；
  - `y_reference`：`J_nu(x)` 参考值；
  - `abs_error`：点误差；
  - `residual`：代回 ODE 的残差。
- 汇总指标：
  - 相对 `L2` 误差、最大绝对误差、内部区间最大残差、`PASS`。

## R13

内置测试场景：
- ODE：`x^2 y'' + x y' + (x^2-\nu^2)y=0`；
- 参数：`nu = 0.5`；
- 网格：`x in [1e-4, 8]`，共 600 点；
- 截断：`N = 32`。

该设置能体现 Frobenius 的非整数根特征（`r=0.5`），并与 `J_{0.5}` 高精度对齐。

## R14

可调参数：
- `nu`：阶数参数，控制指标根与递推分母。
- `n_terms`：截断阶数，越大通常越精确但计算量更高。
- `x` 的区间与密度：影响误差评估与残差稳定性。
- 通过阈值（`main` 中）：
  - `relative L2 error < 1e-6`；
  - `max absolute error < 1e-6`；
  - `max interior residual < 2e-3`。

## R15

方法对比：
- 对比直接数值积分（如 RK 方法）：
  - 数值积分通用性更强；
  - Frobenius 在奇点附近有解析结构优势，并能显式看到解的级数构造。  
- 对比纯符号推导：
  - 纯符号更“解析”；
  - 本实现是解析递推 + 数值验证，工程可执行性更强。

## R16

典型应用场景：
- 常微分方程课程中展示正则奇点解法。  
- 特殊函数（Bessel、Legendre 等）级数展开的数值验证。  
- 科学计算代码中的“可解释基线实现”，用于回归测试。  
- 需要在奇点附近保持结构可解释性的建模任务。

## R17

可扩展方向：
- 扩展到更一般的正则奇点方程，自动生成指标方程与递推关系；
- 增加第二根分支和根差为整数时的对数项处理；
- 支持多精度（`mpmath`）以验证更高阶截断；
- 输出误差-阶数曲线，自动给出截断阶数建议。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 固定 `nu=0.5`、`N=32` 与正实网格 `x`。  
2. `solve_bessel_with_frobenius` 调用 `_check_grid`，确保网格适合奇点问题数值处理。  
3. `frobenius_coefficients_bessel_j` 先由 `indicial_roots_bessel` 得到指标根 `(\nu,-\nu)`，再选 `r=\nu`。  
4. 以 `a_0 = 1/(2^\nu\Gamma(\nu+1))` 初始化，随后按
   `a_k = -a_{k-2}/((k+r)^2-\nu^2)` 递推到 `k=N`。  
5. `evaluate_frobenius_series` 计算 `y_series(x)=\sum_{k=0}^N a_k x^{k+r}`。  
6. 使用 `scipy.special.jv` 仅作参考真值 `y_reference`，并计算逐点误差与相对 `L2`。  
7. 对 `y_series` 用 `np.gradient` 估计导数，构造残差
   `x^2 y'' + x y' + (x^2-\nu^2)y`。  
8. `main` 打印系数预览、采样表和汇总指标，并按阈值输出 `PASS` 或抛错。

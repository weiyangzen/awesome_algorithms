# 矩阵条件数计算

- UID: `MATH-0073`
- 学科: `数学`
- 分类: `数值线性代数`
- 源序号: `73`
- 目标目录: `Algorithms/数学-数值线性代数-0073-矩阵条件数计算`

## R01

本条目实现“矩阵条件数计算”的最小可运行 MVP，聚焦数值线性代数里最常用的稳定性指标：
- `kappa_2(A) = sigma_max / sigma_min`（2-范数条件数）；
- `kappa_p(A) = ||A||_p * ||A^{-1}||_p`（`p=1,∞,F`）。

目标是给出一个透明、可复现、可直接运行的实现，而不是仅调用黑盒 API。

## R02

问题定义：
- 输入：实数方阵 `A in R^(n x n)`；
- 输出：
  - `kappa_2, kappa_1, kappa_inf, kappa_F`；
  - 最大/最小奇异值 `sigma_max, sigma_min`；
  - 可选的右端项扰动实验结果（验证条件数与误差放大的关系）。

若矩阵奇异或数值上近奇异，则条件数记为 `inf`。

## R03

数学背景：
- 条件数描述输入微小扰动对输出误差的放大能力；
- 对线性系统 `Ax=b`，在典型近似关系下有
  `relative_error(x) <= kappa(A) * relative_error(data)`；
- `kappa_2` 越大，系统越病态，解对噪声越敏感。

因此条件数是判断“问题是否数值稳定”的核心诊断量。

## R04

算法思路（本实现）：
1. 校验 `A`：二维、非空、有限值、方阵；
2. 对 `A` 做 SVD，得到奇异值序列；
3. 用阈值 `tol = rcond * sigma_max` 判断是否近奇异；
4. 若近奇异：返回 `kappa_2=kappa_1=kappa_inf=kappa_F=inf`；
5. 若非奇异：
   - 用 `sigma_max/sigma_min` 得 `kappa_2`；
   - 通过解方程得到 `A^{-1}`；
   - 分别计算 `||A||_p * ||A^{-1}||_p`（`p=1,∞,F`）；
6. 在 demo 中额外做 `b -> b+db` 扰动，观测解误差放大。

## R05

核心数据结构：
- `ConditionNumberReport`（`dataclass`）：
  - `n, rank`；
  - `sigma_max, sigma_min`；
  - `cond_2, cond_1, cond_inf, cond_fro`；
  - `reciprocal_cond_2`；
  - `singular`。
- `PerturbationReport`（`dataclass`）：
  - `rel_rhs_perturb`（右端扰动相对量）；
  - `rel_solution_change`（解变化相对量）；
  - `bound_by_kappa2`（`kappa_2 * rel_rhs`）；
  - `amplification_factor`（观测放大量）。

## R06

正确性要点：
- `kappa_2 = sigma_max/sigma_min` 是标准定义；
- `kappa_p = ||A||_p ||A^{-1}||_p` 在 `A` 可逆时成立；
- `sigma_min` 很小时，问题会对扰动高度敏感，代码用阈值统一归类为“近奇异”；
- 扰动实验可实证“高条件数 -> 更大误差放大”。

## R07

复杂度分析（`n x n` 方阵）：
- SVD：`O(n^3)`；
- 求逆（通过 `solve(A, I)`）：`O(n^3)`；
- 范数计算：`O(n^2)`；
- 总体由 `O(n^3)` 主导；
- 空间复杂度约 `O(n^2)`（存储矩阵、逆矩阵和中间结果）。

## R08

边界与异常处理：
- 非二维输入：`ValueError`；
- 空矩阵：`ValueError`；
- 含 `NaN/Inf`：`ValueError`；
- 非方阵：`ValueError`；
- 非法 `rcond` 或 `perturb_scale`：`ValueError`；
- 条件数为无穷时不执行扰动实验（避免无意义比较）。

## R09

MVP 取舍：
- 主计算逻辑不依赖 `np.linalg.cond` 黑盒；
- 仅使用 `numpy`，保持依赖最小；
- 仅覆盖实数稠密方阵，暂不扩展到稀疏矩阵与大规模迭代估计；
- 保留 `np.linalg.cond` 仅用于交叉校验展示，不参与主实现。

## R10

`demo.py` 函数职责：
- `_as_finite_2d_array` / `_require_square`：输入合法性检查；
- `_effective_rcond`：统一近奇异阈值策略；
- `build_matrix_with_target_cond`：生成指定条件数量级的测试矩阵；
- `hilbert_matrix`：生成经典病态样例；
- `compute_condition_report`：核心条件数计算；
- `rhs_perturbation_experiment`：误差放大实验；
- `run_case`：单个案例的完整评估与打印；
- `main`：组织三组案例并输出结果。

## R11

运行方式：

```bash
cd Algorithms/数学-数值线性代数-0073-矩阵条件数计算
python3 demo.py
```

脚本无需交互输入，会自动生成测试矩阵并输出报告。

## R12

输出解读：
- `kappa_2 (manual SVD)`：本实现主结果；
- `kappa_2 (np.linalg.cond check)`：校验值；
- `|difference|`：两者差异，通常接近机器精度；
- `kappa_1 / kappa_inf / kappa_F`：不同范数下的敏感度；
- `reciprocal kappa_2`：倒条件数，越接近 0 越病态；
- 扰动实验中：
  - `relative perturbation in b` 是输入扰动；
  - `relative change in solution` 是输出误差；
  - `kappa_2 * rel_b` 是理论上界量级参考。

## R13

最小测试建议：
- 条件数较小矩阵（例如目标约 `1e1~1e2`）；
- 条件数很大矩阵（例如目标约 `1e8`）；
- Hilbert 矩阵（经典病态）；
- 异常输入（非方阵、空矩阵、含 `NaN/Inf`）。

当前 `main` 已覆盖前三类正常路径。

## R14

可调参数：
- `build_matrix_with_target_cond(n, target_cond, seed)`：控制规模和病态程度；
- `compute_condition_report(..., rcond=...)`：控制近奇异判定阈值；
- `rhs_perturbation_experiment(..., perturb_scale=...)`：控制扰动强度；
- `seed`：控制随机可复现性。

实践中可通过增大 `target_cond` 或 `n` 来构造更困难案例。

## R15

方法对比：
- 与直接 `np.linalg.cond` 比较：
  - 黑盒方式代码短，但内部流程不透明；
  - 本实现能显式展示 SVD、阈值判定和逆矩阵范数链路。
- 与仅看行列式比较：
  - 行列式接近 0 只能粗略提示奇异性；
  - 条件数直接量化“误差放大倍数”，更适合数值稳定性分析。
- 与迭代估计法（如大规模 1-范数估计）比较：
  - 迭代法更适合超大稀疏矩阵；
  - 本 MVP 优先清晰与教学可读性。

## R16

典型应用：
- 在线性方程求解前评估问题稳定性；
- 在最小二乘/回归中识别特征矩阵病态性；
- 在算法基准测试中区分“算法误差”与“问题本身病态”；
- 在教学中演示奇异值谱与数值稳定性的关系。

## R17

可扩展方向：
- 增加稀疏矩阵场景下的条件数估计（避免显式求逆）；
- 增加按批量矩阵输出统计结果（均值、分位数）；
- 补充随机扰动多次试验的置信区间；
- 增加对复数矩阵和伪逆条件数的支持。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 调用 `build_matrix_with_target_cond` 与 `hilbert_matrix` 构造三组测试矩阵。  
2. `main` 对每组矩阵调用 `run_case`，进入统一评估流程。  
3. `run_case` 先调用 `compute_condition_report`。  
4. `compute_condition_report` 使用 `_require_square` 做输入合法性校验，并通过 `_effective_rcond` 生成阈值。  
5. `compute_condition_report` 对矩阵做 SVD，读取 `sigma_max/sigma_min`，依据 `sigma_min <= rcond*sigma_max` 判定是否近奇异。  
6. 若非奇异，`compute_condition_report` 通过 `solve(A, I)` 得到逆矩阵并计算 `kappa_1/kappa_inf/kappa_F`，同时用 `sigma_max/sigma_min` 得 `kappa_2`。  
7. `run_case` 将手工 `kappa_2` 与 `np.linalg.cond(..., p=2)` 做差值校验，并打印全部指标。  
8. 对非奇异矩阵，`run_case` 进一步调用 `rhs_perturbation_experiment`：生成 `b, db`，分别求解 `Ax=b` 与 `Ax=b+db`，输出输入扰动与解变化的放大关系。  

# 自适应Simpson积分

- UID: `MATH-0135`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `135`
- 目标目录: `Algorithms/数学-数值分析-0135-自适应Simpson积分`

## R01

本条目实现自适应 Simpson 积分（Adaptive Simpson Integration）的最小可运行版本。

目标：
- 在给定区间 `[a, b]` 上近似计算定积分 `I = \int_a^b f(x)dx`；
- 用“区间二分 + 局部误差估计”自动把计算预算分配到更困难的子区间；
- 在 `demo.py` 输出近似值、误差估计、函数评估次数与递归深度，便于验证算法行为。

## R02

问题定义：
- 输入：
  - 被积函数 `f(x)`；
  - 有限区间 `[a, b]`；
  - 绝对误差容忍阈值 `tol`；
  - 最大递归深度 `max_depth`。
- 输出：
  - 积分近似值 `estimate`；
  - 全局误差估计 `abs_error_est`；
  - 统计信息：`eval_count`、`accepted_intervals`、`max_depth_reached`。

## R03

核心公式（Simpson 与误差估计）：

1) 单区间 Simpson 公式  
`S(a,b) = (b-a)/6 * [f(a) + 4f((a+b)/2) + f(b)]`。

2) 二分后两段 Simpson 值  
`S_left = S(a,m)`，`S_right = S(m,b)`，其中 `m=(a+b)/2`。

3) 局部误差指标  
`delta = (S_left + S_right) - S(a,b)`，  
若 `|delta| <= 15 * tol_local` 则接受当前区间。

4) Richardson 修正  
接受区间时返回  
`S_corrected = S_left + S_right + delta/15`，  
并用 `|delta|/15` 作为该区间误差估计。

## R04

算法流程：
1. 校验边界与参数（端点有限、`tol > 0`、`max_depth >= 0`）。
2. 处理退化与方向：
   - `a == b` 直接返回 `0`；
   - `b < a` 时交换端点并在最终结果乘 `-1`。
3. 先计算 `f(a), f((a+b)/2), f(b)`，构造整段 Simpson 估计。
4. 递归函数中将区间二分，新增两个四分点函数值。
5. 计算 `S_left, S_right, delta`，检查收敛条件或深度上限。
6. 若满足条件，接受该区间并返回修正积分值与误差估计。
7. 否则将局部容差平分到左右子区间继续递归。
8. 汇总所有子区间结果，得到全局积分近似和统计信息。

## R05

核心数据结构：
- `AdaptiveSimpsonResult`（`dataclass`）：
  - `estimate: float` 最终积分近似；
  - `abs_error_est: float` 误差估计累加值；
  - `eval_count: int` 函数总评估次数；
  - `accepted_intervals: int` 被接受的叶子区间数量；
  - `max_depth_reached: int` 实际递归最大深度。
- 测试样例 `cases`：元组列表，包含函数名、函数对象、区间与解析真值。

## R06

正确性要点：
- 每个叶子区间都以 Simpson 公式近似，并通过 `delta` 量化局部离散误差；
- 终止条件 `|delta| <= 15*tol_local` 来自 Simpson 误差阶与 Richardson 修正关系；
- 非终止区间继续细分，意味着困难区域会获得更密集采样；
- 全局结果由所有叶子区间积分值相加得到，误差估计也按叶子区间累加。

## R07

复杂度分析（设最终接受叶子区间数为 `K`）：
- 时间复杂度：
  - 函数评估次数与叶子区间规模同阶，整体约 `O(K)`；
  - 对光滑函数通常 `K` 较小，对尖峰/振荡函数 `K` 增大。
- 空间复杂度：
  - 递归栈深度为 `O(D)`，`D <= max_depth`；
  - 除递归栈外仅常数级额外变量。

## R08

边界与异常处理：
- 非有限端点（`nan/inf`）抛出 `ValueError`；
- `tol <= 0` 或 `max_depth < 0` 抛出 `ValueError`；
- 函数值若出现 `nan/inf`，`safe_eval` 抛出 `RuntimeError`；
- 达到 `max_depth` 时即使未满足误差阈值也会强制接受当前区间（保证终止）。

## R09

MVP 取舍：
- 只使用 Python 标准库，避免外部黑盒积分器；
- 保留必要统计指标，不引入复杂框架；
- 优先可读与可验证，不做并行化或向量化优化；
- 演示样例均给出解析真值，便于直接核验误差数量级。

## R10

`demo.py` 函数职责：
- `safe_eval`：函数值评估与有限性检查；
- `simpson_from_values`：根据端点/中点值计算单区间 Simpson 估计；
- `adaptive_simpson_integral`：完整自适应 Simpson 主流程（递归 + 停止条件 + 统计）；
- `relative_error`：相对误差计算；
- `run_case`：单样例执行与结果打印；
- `main`：组织固定样例并批量运行，无交互输入。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0135-自适应Simpson积分
python3 demo.py
```

脚本会自动执行全部样例并打印每个样例的误差与统计信息。

## R12

输出字段解读：
- `estimate`：最终积分近似；
- `exact`：解析真值；
- `abs_error`：绝对误差 `|estimate-exact|`；
- `rel_error`：相对误差；
- `error_estimate`：算法内部误差估计（叶子区间累加）；
- `eval_count`：函数调用次数；
- `accepted_segments`：递归接受的叶子区间数量；
- `max_depth_reached`：实际达到的最大深度。

## R13

建议最小测试集（`demo.py` 已覆盖）：
- `sin(x)` on `[0, pi]`，真值 `2`；
- `exp(x)` on `[0, 1]`，真值 `e-1`；
- `1/(1+x^2)` on `[0, 1]`，真值 `pi/4`；
- `sqrt(x)` on `[0, 1]`，真值 `2/3`（端点导数奇异，检验自适应能力）；
- `cos(20x)` on `[0, 1]`，真值 `sin(20)/20`（振荡函数）；
- 反向区间 `sin(x)` on `[pi, 0]`，真值 `-2`。

## R14

可调参数：
- `tol`：目标误差阈值，默认 `1e-10`；
- `max_depth`：递归最大深度，默认 `20`。

调参建议：
- 若函数较平滑，可适当放宽 `tol` 以减少函数评估；
- 若函数局部变化剧烈，可提高 `max_depth` 以避免过早截断。

## R15

方法对比：
- 与固定步长复合 Simpson 相比：
  - 自适应 Simpson 在“困难区间”自动细分，通常更省评估次数；
  - 固定步长实现更简单，但容易在局部尖峰处精度不足。
- 与 `scipy.integrate.quad` 相比：
  - 本实现更透明，便于教学与调试；
  - `quad` 更通用、鲁棒性更强，工程上可作为生产替代方案。

## R16

应用场景：
- 数值分析课程中的误差控制与自适应细分演示；
- 中小规模科学计算任务中的一维定积分近似；
- 需要“误差估计 + 过程可解释”的算法原型验证阶段。

## R17

后续扩展方向：
- 增加“最小区间长度”阈值，避免极端情况下的过深递归；
- 改写为显式栈迭代版本，规避 Python 递归栈限制；
- 增加向量化函数评估（`numpy`）提升吞吐；
- 增加与 `scipy.integrate.quad` 的自动对照测试。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 定义多个有解析解的积分样例，并设置 `tol` 与 `max_depth`。  
2. 每个样例在 `run_case` 中调用 `adaptive_simpson_integral` 执行积分。  
3. `adaptive_simpson_integral` 校验参数、处理 `a==b`/反向区间，并计算初始三点函数值。  
4. 通过 `simpson_from_values` 得到整段 Simpson 估计，随后进入递归 `recurse`。  
5. `recurse` 在四分点新增两次函数评估，计算左右子区间 Simpson 值与 `delta`。  
6. 若 `|delta| <= 15*tol_local` 或达到 `max_depth`，则接受当前区间并返回修正值 `S_left+S_right+delta/15`。  
7. 否则将容差平分（`tol_local/2`）后递归处理左右子区间，再累加积分值与误差估计。  
8. 全部递归返回后汇总 `estimate/error_estimate/eval_count` 等指标，`run_case` 输出误差与统计结果。  

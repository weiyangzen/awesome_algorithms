# Monte Carlo积分

- UID: `MATH-0138`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `138`
- 目标目录: `Algorithms/数学-数值分析-0138-Monte_Carlo积分`

## R01

本条目实现一维定积分的 Monte Carlo（随机采样）最小可运行版本，目标是：
- 在有限区间 `[a,b]` 上估计 `I=\int_a^b f(x)dx`；
- 明确展示“均匀采样 -> 样本均值 -> 积分估计 -> 误差条（标准误差/置信区间）”的完整链路；
- 用固定随机种子保证演示可复现、可对比。

## R02

问题定义（MVP 范围）：
- 输入：
  - 可调用函数 `f: R -> R`；
  - 有界区间 `a < b`；
  - 采样数 `n >= 2`；
  - 随机种子 `seed`（用于复现）。
- 输出：
  - 积分估计值 `\hat I_n`；
  - 样本标准误差 `SE(\hat I_n)`；
  - 95% 近似置信区间；
  - 少量采样点与函数值预览（便于排查）。

## R03

数学基础：

1) 令 `U ~ Uniform(a,b)`，则
`I = \int_a^b f(x)dx = (b-a) E[f(U)]`。

2) 用 `n` 个独立同分布样本 `U_1,...,U_n`，构造估计器
`\hat I_n = (b-a) * (1/n) * sum_{i=1}^n f(U_i)`。

3) 无偏性：
`E[\hat I_n] = I`。

4) 方差与标准误差：
`Var(\hat I_n) = (b-a)^2 * Var(f(U))/n`，
所以误差尺度约为 `O(n^{-1/2})`。

5) 大样本下（CLT 近似）
`\hat I_n ± 1.96 * SE` 可作为 95% 置信区间。

## R04

算法流程（MVP）：
1. 校验参数：`a,b` 有限且 `a<b`，`n>=2`。
2. 初始化伪随机数发生器 `rng = numpy.random.default_rng(seed)`。
3. 在 `[a,b]` 上均匀采样 `n` 个点 `x_i`。
4. 计算函数值 `y_i=f(x_i)`，并检查是否有限。
5. 计算样本均值 `mean(y)` 与样本方差 `var(y, ddof=1)`。
6. 估计积分 `estimate = (b-a)*mean(y)`。
7. 估计标准误差 `se = (b-a)*sqrt(var(y)/n)`。
8. 构造 95% 区间 `[estimate-1.96*se, estimate+1.96*se]` 并输出。

## R05

核心数据结构：
- `numpy.ndarray`：
  - `samples_x: shape=(n,)`，随机采样点；
  - `samples_fx: shape=(n,)`，函数采样值。
- `MonteCarloResult`（`dataclass`）：
  - 记录 `estimate`、`std_error`、`ci95_low/high`、`sample_mean/sample_var`、`n/a/b/seed` 以及样本预览。

## R06

正确性要点：
- 由期望积分等价关系可知估计器形式正确；
- 样本均值是 `E[f(U)]` 的一致估计，故 `\hat I_n` 随 `n` 增大收敛到真值；
- 标准误差按样本方差估计，能反映随机误差量级；
- 固定 `seed` 不改变算法定义，仅用于复现实验。

## R07

复杂度：
- 时间复杂度：`O(n)`（采样 + 函数评估 + 统计量计算）；
- 空间复杂度：`O(n)`（保存样本和函数值）。

说明：MVP 为可解释性保留样本向量。若追求内存，可用流式统计把空间降为 `O(1)`。

## R08

边界与异常处理：
- `a` 或 `b` 非有限值 -> `ValueError`；
- `a >= b` -> `ValueError`；
- `n < 2` -> `ValueError`；
- 函数值出现 `nan/inf` -> `ValueError`。

## R09

MVP 取舍：
- 只实现最基础、透明的一维均匀采样 Monte Carlo；
- 使用 `numpy` 生成随机数与向量统计，不引入复杂框架；
- 不做重要性采样/控制变量/分层采样，先保证基线可复现可验证；
- 输出包含误差条，避免只给单点估计造成误判。

## R10

`demo.py` 职责划分：
- `check_interval`：区间合法性校验；
- `monte_carlo_integrate`：执行一次完整 Monte Carlo 积分并返回结构化结果；
- `print_result_summary`：打印估计值、真值、绝对误差、标准误差与置信区间；
- `run_examples`：组织多组函数在多种 `n` 下演示收敛；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0138-Monte_Carlo积分
python3 demo.py
```

脚本无交互输入，会自动打印多组样例结果。

## R12

输出字段解读：
- `estimate`：积分估计值；
- `reference`：解析或高精度参考值；
- `abs_error`：`|estimate-reference|`；
- `std_error`：估计标准误差；
- `ci95`：95% 近似置信区间；
- `sample_x_head`、`sample_fx_head`：前若干采样点与函数值。

关注点：
- 随 `n` 增大，`std_error` 常按 `1/sqrt(n)` 下降；
- 单次实验 `abs_error` 不一定单调，但总体会趋于缩小。

## R13

建议最小测试集：
- 多项式：`f(x)=x^2`，`[0,1]`，真值 `1/3`；
- 三角函数：`f(x)=sin(x)`，`[0,pi]`，真值 `2`；
- 光滑非多项式：`f(x)=exp(-x^2)`，`[0,1]`，真值 `0.5*sqrt(pi)*erf(1)`；
- 异常输入：`n=1`、`a>=b`、区间端点为 `nan/inf`、函数返回非有限值。

## R14

可调参数：
- `n_values`：采样规模列表（如 `[500, 5000, 50000]`）；
- `seed_base`：基础随机种子，便于固定实验；
- `preview_k`：样本预览长度，控制日志输出量。

实践建议：先用小 `n` 检查流程，再提升到大 `n` 观察误差条缩小趋势。

## R15

方法对比：
- 相比梯形/辛普森：
  - Monte Carlo 收敛阶更慢（通常 `O(n^{-1/2})`）；
  - 但在高维积分时不易遭遇维度灾难爆炸。
- 相比 SciPy 黑盒积分器：
  - 黑盒在一维常更高效；
  - 本实现更适合教学、可解释分析和随机误差建模。

## R16

应用场景：
- 概率期望估计与不确定性传播；
- 金融衍生品定价中的路径采样积分；
- 复杂模型中难以解析积分的近似计算；
- 数值分析课程中随机积分方法教学。

## R17

后续扩展：
- 重要性采样（降低方差）；
- 分层采样与拉丁超立方；
- 控制变量与反变量（variance reduction）；
- 多次重复试验与批量统计（更稳定地评估误差）。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `run_examples` 组装样例函数、区间、参考真值与 `n_values`。  
2. 对每个 `(example, n)`，先在 `monte_carlo_integrate` 中调用 `check_interval(a,b)` 并校验 `n>=2`。  
3. 用 `numpy.random.default_rng(seed)` 初始化可复现随机数发生器。  
4. 调用 `rng.uniform(a, b, size=n)` 生成独立均匀样本 `x_i`。  
5. 在每个样本点上评估 `f(x_i)`，得到 `f` 值向量并检查有限性。  
6. 计算 `sample_mean` 与 `sample_var(ddof=1)`，据此得到积分估计 `estimate=(b-a)*sample_mean`。  
7. 计算 `std_error=(b-a)*sqrt(sample_var/n)`，再给出 `95%` 区间 `estimate ± 1.96*std_error`。  
8. 把结果封装为 `MonteCarloResult`，`print_result_summary` 输出估计、误差、置信区间与样本预览。  

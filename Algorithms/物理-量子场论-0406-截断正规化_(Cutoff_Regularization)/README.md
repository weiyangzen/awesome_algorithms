# 截断正规化 (Cutoff Regularization)

- UID: `PHYS-0387`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `406`
- 目标目录: `Algorithms/物理-量子场论-0406-截断正规化_(Cutoff_Regularization)`

## R01

截断正规化（Cutoff Regularization）的核心是给发散动量积分引入一个显式 UV 截止 `Lambda`，把“无穷高动量”的贡献先裁掉，再在 `Lambda -> infinity` 的过程中识别并分离发散项。

本条目选用最小代表积分（欧氏空间一圈标量积分）：

`I(Lambda,m) = ∫_{|k|<Lambda} d^4k/(2pi)^4 * 1/(k^2 + m^2)^2`

该积分在 4 维是对数发散，非常适合演示 cutoff regularization 的全过程。

## R02

目标是把“算法闭环”做完整，而不是只给定义：

1. 用径向数值积分计算 `I(Lambda,m)`；
2. 用闭式解析公式计算同一量并与数值对照；
3. 给出大 `Lambda` 渐近式，验证误差按 `O(m^2/Lambda^2)` 缩放；
4. 显式构造 cutoff 反项并得到有限重整化结果。

## R03

将四维积分写成径向形式：

- `d^4k = 2pi^2 * k^3 dk`
- 所以  
`I(Lambda,m) = 1/(8pi^2) * ∫_0^Lambda k^3/(k^2+m^2)^2 dk`

这就是 `demo.py` 中数值积分函数的直接来源，没有省略中间步骤。

## R04

同一积分的闭式结果：

`I(Lambda,m) = 1/(16pi^2) * [ ln(1 + Lambda^2/m^2) - Lambda^2/(Lambda^2+m^2) ]`

它明确显示了两件事：

- `Lambda -> infinity` 时是对数发散；
- 发散的主导项来自 `ln(Lambda^2/m^2)`，并带一个有限常数 `-1`。

## R05

大 cutoff 渐近展开（保留到 `O(m^2/Lambda^2)`）：

`I(Lambda,m) = 1/(16pi^2) * [ ln(Lambda^2/m^2) - 1 + 2m^2/Lambda^2 + O(m^4/Lambda^4) ]`

`demo.py` 同时输出 exact 与 asymptotic，并验证两者差值随 `Lambda` 增大而下降。

## R06

在本 MVP 中采用一个显式 subtraction 方案：

`CT(Lambda,mu) = 1/(16pi^2) * [ ln(Lambda^2/mu^2) - 1 ]`

定义重整化后量：

`I_R(Lambda,m;mu) = I(Lambda,m) - CT(Lambda,mu)`

于是

`lim_{Lambda->infinity} I_R = 1/(16pi^2) * ln(mu^2/m^2)`，

脚本会用数值验证该极限。

## R07

`demo.py` 提供的最小能力：

1. `scipy.integrate.quad` 径向积分；
2. 闭式公式实现与数值比对；
3. 渐近展开对照；
4. `pandas` 结果表输出；
5. 自动断言（正确性、发散行为、有限化行为）。

## R08

复杂度（`N` 为 cutoff 采样点数量）：

- 数值积分部分：`O(N * Q)`，`Q` 为单次 `quad` 自适应采样成本；
- 解析式与重整化计算：`O(N)`；
- 空间复杂度：`O(N)`（用于保存结果表）。

## R09

输入参数与约束：

- `mass > 0`：传播子质量参数；
- `mu_ren > 0`：重整化尺度；
- `cutoffs`：正且严格递增；
- 数值积分容差 `epsabs/epsrel`：默认 `1e-11/1e-10`。

不满足约束会抛出 `ValueError`，防止沉默错误。

## R10

运行方式（无交互）：

```bash
cd Algorithms/物理-量子场论-0406-截断正规化_(Cutoff_Regularization)
uv run python demo.py
```

或在仓库根目录直接运行：

```bash
uv run python Algorithms/物理-量子场论-0406-截断正规化_(Cutoff_Regularization)/demo.py
```

## R11

输出表含以下关键列：

- `I_numeric`：径向数值积分；
- `I_analytic`：闭式精确值；
- `I_asymptotic`：大 cutoff 近似；
- `I_ren_subtracted`：减法后有限化结果；
- `abs_err_num_vs_analytic`、`abs_err_asymptotic`、`abs_err_ren_to_limit`：三类误差。

末尾打印 `All checks passed.` 表示所有断言通过。

## R12

正确性验证策略：

1. `I_numeric` 与 `I_analytic` 最大绝对误差足够小；
2. `I_analytic` 随 `Lambda` 单调上升（反映对数发散）；
3. `I_ren_subtracted` 在大 `Lambda` 接近理论极限；
4. 渐近误差随 `Lambda` 增长而变小。

## R13

适用场景：

- 量子场论课程里展示 cutoff regularization 的第一性例子；
- 验证程序里的 UV 发散分离与反项减法逻辑；
- 作为后续重整化群/跑动耦合示例的前置模板。

不适用场景：

- 多圈图与多尺度阈值问题；
- 需要保持规范对称性的高精度现象学计算（通常更偏好维度正规化）；
- 非微扰问题。

## R14

常见失效模式：

1. 忘记角向因子 `2pi^2` 或 `(2pi)^4` 归一化；
2. 把积分上限错写为无穷（那就不是 cutoff regularization）；
3. `mass <= 0` 导致表达式失去当前实数分支假设；
4. cutoff 未递增，单调性断言会失败；
5. 数值容差过宽，数值-解析误差不再可控。

## R15

可扩展方向：

1. 增加对比：硬 cutoff vs 维度正规化 vs Pauli-Villars；
2. 推广到带外动量的两点函数并提取波函数重整化常数；
3. 引入 `sympy` 自动做高阶渐近展开；
4. 结合 Wilsonian 视角，把 cutoff 当作有效场论尺度。

## R16

相关主题：

- 重整化（Renormalization）；
- 重整化群与跑动耦合；
- 有效场论中的 UV 截断；
- 维度正规化与 MS/MS-bar 方案。

## R17

MVP 交付清单：

1. 可直接运行的 `demo.py`；
2. 明确的 cutoff 正规化积分定义与闭式公式；
3. 数值与解析交叉验证；
4. 显式反项减法与有限极限校验；
5. 结构化输出与自动断言。

该目录可独立验证，不依赖交互输入。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `CutoffConfig` 定义质量、重整化尺度、cutoff 网格与积分容差。  
2. `validate_config` 检查参数合法性（正值、递增）。  
3. `cutoff_integrand_radial` 给出被积函数 `k^3/(k^2+m^2)^2`。  
4. `one_loop_cutoff_numeric` 调用 `scipy.integrate.quad` 在 `[0,Lambda]` 做径向积分并乘 `1/(8pi^2)`。  
5. `one_loop_cutoff_analytic` 按闭式公式计算精确值，作为基准。  
6. `one_loop_cutoff_asymptotic` 计算大 `Lambda` 渐近近似，用于误差阶验证。  
7. `counterterm_log` 与 `renormalized_subtracted` 构造并应用 cutoff 反项，`renormalized_limit` 给出理论极限。  
8. `build_report` 组装 `pandas.DataFrame`，`main` 打印结果并执行断言。  

第三方库不是黑盒：`quad` 仅做基础数值求积，`numpy/pandas` 仅做数组和表格；发散结构、反项定义、极限验证都由源码显式实现。

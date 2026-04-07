# 临界指数 (Critical Exponents)

- UID: `PHYS-0295`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `298`
- 目标目录: `Algorithms/物理-统计力学-0298-临界指数_(Critical_Exponents)`

## R01

临界指数用于描述系统在二级相变临界点附近的幂律行为。令约化温度
`t = (T - Tc) / Tc`，典型关系包括：

- 磁化强度: `M ~ (-t)^beta` (`t < 0`)
- 磁化率: `chi ~ |t|^-gamma`
- 比热: `C ~ |t|^-alpha`
- 关联长度: `xi ~ |t|^-nu`

本 MVP 的目标是给定临界附近观测点，估计指数 `beta/gamma/alpha/nu`，并演示在未知 `Tc` 时通过扫描候选 `Tc` 进行联合估计。

## R02

输入与输出约定如下：

- 输入:
  - 一组温度 `T`
  - 对应可观测量 `y`
  - 相对临界点方向 `side in {below, above}`
  - 该量是否发散 `diverges`（例如 `chi`、`C`、`xi`）
- 输出:
  - `estimated_exponent`：估计到的临界指数
  - `amplitude`：幂律前因子 `A`
  - `r2_logspace`：在对数域拟合优度
  - 可选 `estimated_tc`：扫描候选 `Tc` 的最优结果

## R03

核心思路是把幂律回归转成线性回归：

- 原模型: `y = A * |t|^p`
- 取对数: `log(y) = log(A) + p * log(|t|)`

其中 `p` 与目标临界指数的映射：

- 若量随临界点发散（`diverges=True`），目标指数为 `-p`
- 否则目标指数为 `p`

## R04

数学建模细节：

1. 计算 `t = (T - Tc) / Tc`。
2. 按 `side` 过滤：
   - `below`: `t < 0`
   - `above`: `t > 0`
3. 使用窗口 `t_min <= |t| <= t_max`，避免远离临界区和过近导致噪声放大。
4. 仅保留 `y > 0` 以支持对数变换。
5. 在 `(x, z) = (log|t|, log y)` 上回归，求斜率和截距。

## R05

伪代码：

```text
for each observable dataset:
    choose side/diverges
    filter points by side and t-window
    x = log(|t|), z = log(y)

    # 方法1: sklearn 线性回归
    fit z = b0 + b1*x
    p = b1

    # 方法2: scipy 非线性曲线拟合
    fit y = A * t^p

    # 方法3: torch 梯度下降（对数域）
    optimize MSE(log(y), log(A) + p*x)

    exponent = -p if diverges else p

if Tc unknown:
    scan candidate Tc grid
    pick Tc with largest log-space R^2
```

## R06

正确性要点：

- 幂律模型在对数域可线性化，斜率直接对应指数。
- 使用同一窗口与同一侧数据，保证比较公平。
- 对发散量统一做 `exponent = -p`，避免符号混乱。
- `R^2` 在对数域评估，更贴合幂律拟合目标。

## R07

复杂度分析（单个可观测量）：

- 预处理过滤: `O(n)`
- `sklearn` 线性回归: `O(n)`
- `scipy curve_fit`: 迭代法，经验上 `O(k*n)`
- `torch` 梯度下降: `O(steps*n)`
- `Tc` 网格扫描: `O(m*n)`（`m` 为候选 `Tc` 数）

## R08

数值稳定性与工程处理：

- 对数域要求 `y > 0`，示例采用乘性噪声保证正值。
- 通过 `t_window` 避免 `|t|` 过小导致数值不稳定。
- `curve_fit` 施加参数边界，避免异常发散。
- `torch` 使用 Adam 优化器与固定步数，保证可复现。

## R09

关键超参数：

- `t_window=(0.015, 0.22)`：临界区拟合窗口。
- `tc_grid=np.linspace(2.20, 2.34, 120)`：`Tc` 扫描范围。
- `torch steps=1200, lr=0.05`：对数域线性模型优化配置。
- synthetic 噪声尺度 `noise_sigma`：模拟实验测量波动。

## R10

`demo.py` 的 MVP 内容：

- 生成 3D Ising 风格指数的合成数据。
- 对四个量（`M, chi, C, xi`）分别估计指数。
- 并行演示三种估计器：
  - `sklearn` 线性回归
  - `scipy` 非线性拟合
  - `torch` 梯度下降
- 给出已知 `Tc` 与扫描 `Tc` 两组结果表。

## R11

运行方式：

```bash
uv run python Algorithms/物理-统计力学-0298-临界指数_(Critical_Exponents)/demo.py
```

运行后会打印两张表：

- 已知 `Tc` 时各方法指数估计对比
- 扫描 `Tc` 后的最优 `Tc` 与指数

## R12

结果解释建议：

- `abs_error` 越小越好，代表估计指数更接近真值。
- `r2_logspace` 越接近 1 越好，说明幂律直线关系更成立。
- 不同方法在噪声与初始化下会有小偏差，属正常现象。

## R13

局限与失败模式：

- 临界区数据不足时拟合会退化。
- 若真实行为不是单纯幂律（含对数修正项），模型会系统偏差。
- 远离临界区的数据混入会拉偏指数。
- `Tc` 扫描范围若不覆盖真值，会导致错误最优解。

## R14

与第三方库关系（非黑箱原则）：

- `sklearn` 仅做线性最小二乘，输入是手工构造的 `log|t|` 与 `log y`。
- `scipy curve_fit` 只负责参数优化，目标函数 `A * t^p` 明确定义在代码内。
- `torch` 仅用于演示梯度下降求解同一回归问题，不引入复杂网络结构。

因此核心算法流程、变量定义和物理映射均在本仓实现，不是“一行黑箱调用”。

## R15

验证策略：

- 用已知真值的合成数据做回归测试。
- 对比三种方法估计值的一致性。
- 检查 `R^2` 是否在合理区间（通常高于 0.95）。
- 观察扫描 `Tc` 是否回到真实 `Tc` 附近。

## R16

可扩展方向：

- 加入有限尺度标度（FSS）并同时估计 `nu`。
- 加入 bootstrap 置信区间。
- 支持真实实验/蒙特卡洛数据读入（CSV/Parquet）。
- 引入稳健回归（Huber、RANSAC）降低异常点影响。

## R17

工程化落地建议：

- 将数据预处理、拟合、评估拆成独立模块。
- 对拟合窗口、扫描区间做配置化管理。
- 输出结构化结果（DataFrame/JSON）供后续管道使用。
- 固定随机种子，确保实验可复现。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. 在 `generate_synthetic_dataset` 中构造四类临界附近观测数据，并注入乘性噪声。
2. 在 `_prepare_log_features` 中按 `side` 和 `t_window` 过滤点集，构建 `log|t|` 与 `log y`。
3. `fit_exponent_sklearn` 调用线性回归，得到斜率 `p` 与截距 `log(A)`。
4. `fit_exponent_scipy` 以 `y=A*t^p` 做非线性拟合，得到另一组 `A,p`。
5. `fit_exponent_torch` 在对数域定义参数 `logA,p`，用 Adam 最小化 MSE。
6. 对每种方法统一执行符号映射：`diverges=True` 时指数取 `-p`，否则取 `p`。
7. `scan_tc_by_best_r2` 在候选 `Tc` 网格上重复第 2-6 步，按 `R^2` 选最优 `Tc`。
8. `main` 汇总为 `pandas.DataFrame`，打印指数误差表与 `Tc` 扫描结果表，形成可复核输出。

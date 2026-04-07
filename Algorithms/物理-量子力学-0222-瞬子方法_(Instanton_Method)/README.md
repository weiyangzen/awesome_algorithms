# 瞬子方法 (Instanton Method)

- UID: `PHYS-0221`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `222`
- 目标目录: `Algorithms/物理-量子力学-0222-瞬子方法_(Instanton_Method)`

## R01

瞬子方法（Instanton Method）用于处理量子隧穿这类非微扰效应，核心思想是在欧氏时间（`t -> -iτ`）下寻找连接不同经典真空（或势阱极小值）的有限作用量经典轨道（瞬子）。

本条目 MVP 聚焦一维双阱势中的基态劈裂（`ΔE = E1 - E0`）：
- 用瞬子近似给出 `ΔE` 的半经典估计；
- 用有限差分 + 稀疏本征求解给出数值“真值”对照；
- 展示 `ΔE` 随 `1/ħ` 呈近似指数衰减。

## R02

MVP 的具体问题设置：

1. 势函数：`V(x) = λ(x²-a²)²`，`m=1`。
2. 目标：对多个 `ħ` 值计算
- 数值劈裂：`ΔE_numeric = E1 - E0`；
- 瞬子估计：`ΔE_instanton`。
3. 输出：结果表（`pandas.DataFrame`）与误差、线性拟合摘要。

该任务属于瞬子方法最经典的入门验证：双阱中本应简并的局域基态因隧穿耦合产生非零劈裂。

## R03

为何选这个设定：

- 双阱势是瞬子教材级模型，解析形式清晰，能把“路径积分非微扰鞍点”落成可运行代码；
- `E0/E1` 的数值求解门槛低（稀疏三对角矩阵），便于与半经典公式做同台对比；
- 通过扫描 `ħ`，可以直观看到 `exp(-S0/ħ)` 的主导结构，而不是只给单点结果。

## R04

本 MVP 采用的关键公式：

1. 欧氏时间瞬子轨道（双阱）：
`x_I(τ) = a * tanh(k(τ-τ0))`，其中 `k = a*sqrt(2λ/m)`。

2. 阱底小振动频率：
`ω0 = sqrt(V''(±a)/m) = sqrt(8λa²/m)`。

3. 单瞬子作用量：
`S0 = ∫_{-a}^{a} dx sqrt(2mV(x)) = (4/3)a³sqrt(2mλ)`。

4. 劈裂估计（本 MVP 采用的常见一环量级写法）：
`ΔE_instanton ≈ 2ω0 * sqrt(S0/(2πħ)) * exp(-S0/ħ)`。

说明：不同教材在前因子归一化上会有常数差异，本 MVP 重点验证指数主导与量级一致性。

## R05

算法主流程：

1. 固定 `m,a,λ`，计算 `ω0` 与解析 `S0`。
2. 在欧氏时间网格上生成解析瞬子轨道 `x_I(τ)`。
3. 用 `scipy.integrate.simpson` 数值积分欧氏作用量，核对解析 `S0`。
4. 用 `torch` 复算离散欧氏作用量，并计算内部节点梯度范数（检查轨道接近平稳路径）。
5. 对一组 `ħ`，构建有限差分哈密顿量并求最小两个本征值。
6. 得到 `ΔE_numeric`，同时用瞬子公式计算 `ΔE_instanton`。
7. 用 `pandas` 汇总误差表。
8. 用 `sklearn.linear_model.LinearRegression` 拟合 `log(ΔE_numeric)` 对 `1/ħ` 直线，验证斜率约为 `-S0`。

## R06

正确性要点：

- 数值“基准”来自薛定谔方程离散化本征谱，不依赖瞬子公式；
- 瞬子侧同时做解析 `S0` 与数值积分 `S0` 互检，降低推导/实现错位风险；
- 增加 `torch` 的离散作用量梯度检查，确保给定轨道确实接近欧氏作用驻值轨道；
- 对多个 `ħ` 做趋势验证，而非单点巧合。

## R07

复杂度分析：

设空间网格点数为 `N`，`ħ` 采样数为 `K`。

- 每个 `ħ` 上本征求解采用稀疏 `eigsh(k=2)`，主成本近似 `O(iter * N)` 到 `O(iter * nnz)`；
- 总体复杂度约 `O(K * eigsolve(N))`；
- 作用量积分与表格处理开销可忽略（`O(N)` 或 `O(K)`）。

在本 MVP 默认参数中（`N=1200, K=6`）可在较短时间完成。

## R08

边界与异常处理：

- `mass/a/lambda_` 必须正值；
- `ħ` 必须正值；
- `n_grid` 过小会导致本征值不稳定，代码中做下限检查；
- 结果输出前执行 `all_finite` 与 `positive_splitting` 检查。

## R09

MVP 取舍：

- 只实现一维对称双阱，不覆盖反瞬子相互作用、非对称势阱、多瞬子重求和；
- 前因子采用常见近似写法，不做完整涨落行列式比值的严格数值求解；
- 重点是“可复现的半经典 vs 数值”对照闭环，而非高阶修正。

## R10

`demo.py` 主要函数职责：

- `potential` / `potential_torch`：定义双阱势；
- `omega_small_oscillation`：计算 `ω0`；
- `instanton_profile`：生成解析瞬子轨道；
- `instanton_action_analytic`：解析 `S0`；
- `instanton_action_numeric`：SciPy 数值积分 `S0`；
- `instanton_action_torch_and_grad`：Torch 版离散作用量与梯度检查；
- `numerical_splitting_fdm`：有限差分哈密顿量求 `E0,E1,ΔE`；
- `instanton_splitting_estimate`：瞬子劈裂估计；
- `build_comparison_table`：汇总多 `ħ` 对照表；
- `fit_log_splitting_line`：拟合 `log(ΔE)` 与 `1/ħ` 关系；
- `main`：组织运行与打印验收输出。

## R11

运行方式：

```bash
cd Algorithms/物理-量子力学-0222-瞬子方法_(Instanton_Method)
uv run python demo.py
```

脚本无需任何交互输入。

## R12

输出字段说明：

- `S0 (analytic)`：解析瞬子作用量；
- `S0 (numeric, scipy)`：对解析轨道做数值积分得到的作用量；
- `S0 (numeric, torch)`：Torch 离散作用量；
- `|grad S_E| ...`：离散欧氏作用量对路径内部节点梯度范数；
- 表格列：
  - `hbar`
  - `E0_numeric`, `E1_numeric`
  - `DeltaE_numeric`
  - `DeltaE_instanton`
  - `relative_error`
- 拟合行：`log(DeltaE_numeric) ~ slope*(1/hbar)+intercept`。

## R13

最小验收项（脚本内可直接看到）：

1. `check_all_finite=True`。
2. `check_positive_splitting=True`。
3. `S0` 的解析值与数值积分值接近。
4. 线性拟合斜率 `slope` 与 `-S0` 同号且量级接近。

## R14

关键参数与调参建议：

- `a`：阱位置，增大后通常使势垒更宽，`S0` 增大；
- `lambda_`：势阱刚度，增大后势垒更高，`S0` 增大；
- `hbar_values`：越小越接近半经典极限，但数值上 `ΔE` 更小，对精度更敏感；
- `n_grid`：过小会拉大离散误差，过大则增加本征求解耗时。

实务建议：先固定 `a, lambda_`，用网格收敛性测试确定 `n_grid`，再做 `ħ` 扫描。

## R15

与其他近似方法对比：

- 微扰论：适合单阱小扰动，不擅长描述双阱隧穿劈裂；
- WKB：可给出隧穿指数结构，但在多路径/场论语境扩展性不如瞬子语言统一；
- 瞬子法：天然是非微扰鞍点展开框架，可系统接入多瞬子与涨落修正。

## R16

典型应用场景：

- 量子力学双阱隧穿与能级劈裂估计；
- 量子场论中的真空跃迁、假真空衰变（bounce/instanton 语境）；
- 统计物理与凝聚态中的拓扑激发、非平庸鞍点贡献分析。

## R17

可扩展方向：

- 用 Gelfand-Yaglom 方法更严格计算涨落行列式比值；
- 实现多瞬子-反瞬子构型并估计高阶修正；
- 推广到非对称双阱和有限温度（周期欧氏时间）情形；
- 用变分/神经网络路径表示（例如 PyTorch 优化路径）替代解析 `tanh` 轨道。

## R18

`demo.py` 的源码级流程（9 步）：

1. 在 `run_instanton_mvp` 中设定双阱参数 `m,a,lambda_` 并校验输入。
2. 调用 `omega_small_oscillation` 与 `instanton_action_analytic` 得到 `ω0` 和解析 `S0`。
3. 在欧氏时间网格上由 `instanton_profile` 生成解析瞬子轨道 `x_I(τ)`。
4. 用 `instanton_action_numeric`（SciPy Simpson 积分）计算轨道欧氏作用量，做解析-数值核对。
5. 用 `instanton_action_torch_and_grad` 在 Torch 中离散化同一路径，得到作用量与内部梯度范数，检查路径接近驻值。
6. 在 `build_comparison_table` 中遍历多个 `ħ`，每次调用 `numerical_splitting_fdm`：
   组装有限差分哈密顿量并用 `eigsh` 求最低两本征值 `E0,E1`，得到 `ΔE_numeric`。
7. 同时调用 `instanton_splitting_estimate` 计算 `ΔE_instanton`，并写入 `pandas` 表格与相对误差。
8. 用 `fit_log_splitting_line`（`sklearn` 线性回归）拟合 `log(ΔE_numeric)` 对 `1/ħ` 的直线，提取斜率并对照 `-S0`。
9. `main` 统一打印作用量核对、对照表、拟合结果和有限性检查，完成可复现 MVP 验收。

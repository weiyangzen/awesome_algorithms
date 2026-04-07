# 核聚变 (Nuclear Fusion)

- UID: `PHYS-0426`
- 学科: `物理`
- 分类: `核物理`
- 源序号: `446`
- 目标目录: `Algorithms/物理-核物理-0446-核聚变_(Nuclear_Fusion)`

## R01

本条目实现一个“核聚变（D-T 受控热核聚变）”最小可运行算法闭环，关注三件可计算核心：
1. 反应性 `⟨σv⟩(T)` 的可解释拟合（不是直接调用黑盒数据库）。
2. 功率平衡 `P_alpha(T)` 与 `P_brem(T)` 的温度扫描。
3. Lawson 风格约束 `nτ_E` 与 `nTτ_E` 的数值诊断。

目标是用少量代码跑通“数据锚点 -> 反应性模型 -> 点火可行域”流程，而不是构建完整托卡马克输运求解器。

## R02

D-T 反应主通道：

`D + T -> alpha (3.5 MeV) + n (14.1 MeV)`，总释放能约 `17.6 MeV`。

对磁约束点火判据而言，等离子体可自持主要依赖 alpha 粒子沉积加热 `P_alpha`。若 `P_alpha` 无法覆盖辐射损失与能量外输，系统不能进入燃烧区。

## R03

本 MVP 使用的主公式：

1. 反应率密度（50-50 D/T，准中性）：
`R = (n_e^2 / 4) * ⟨σv⟩`

2. 聚变功率密度与 alpha 加热功率：
`P_fus = R * E_fus`
`P_alpha = R * E_alpha`

3. 近似轫致辐射损失：
`P_brem = C_brem * Z_eff * n_e^2 * sqrt(T_e[eV])`

4. 热能密度（`Ti=Te=T` 近似）：
`W = 3 * n_e * T[J]`

5. 最小约束时间（当 `P_alpha > P_brem`）：
`tau_E,min = W / (P_alpha - P_brem)`

并据此计算 `nτ_E`、`nTτ_E` 作为 Lawson 风格指标。

## R04

`demo.py` 的输入/输出设计：

- 输入：脚本内置 D-T 反应性锚点（`T_keV` 与 `⟨σv⟩`），固定 `n_e`、`Z_eff`、随机种子。
- 输出：
  1. 三种反应性拟合路径（SciPy/Sklearn/PyTorch）在锚点上的误差对比；
  2. 温度扫描下的 `P_alpha`、`P_brem`、`P_net`；
  3. break-even 温度、最小 `nTτ_E` 点等关键工况摘要。

脚本无交互输入，运行即打印结果。

## R05

变量与单位约定：

- `T_keV`：等离子体温度，单位 `keV`
- `⟨σv⟩`：热平均反应性，单位 `m^3/s`
- `n_e`：电子密度，单位 `m^-3`
- `P_*`：功率密度，单位 `W/m^3`
- `tau_E`：能量约束时间，单位 `s`
- `n_tau_E`：单位 `m^-3 s`
- `nTtau`：单位 `keV s m^-3`

常量中显式使用 `E_fus=17.6 MeV`、`E_alpha=3.5 MeV`。

## R06

反应性数据策略：

- 使用一组代表性 D-T 锚点（4~90 keV 区间）作为“可审计输入”；
- 不联网、不依赖外部数据库下载；
- 通过三条拟合路线交叉校验，避免单模型过拟合造成误判。

这使得条目可复现、可移植，并可在后续替换为更精确实验表。

## R07

反应性拟合算法：

参数化模型采用 Gamow 风格简式：
`⟨σv⟩(T) = a*T^2*exp(-b/T^(1/3)) / (1 + c*T + d*T^2)`

实现三条路径：
1. `SciPy least_squares`：在对数残差上拟合 `(a,b,c,d)`；
2. `scikit-learn LinearRegression`：在 log-space 特征上做基线拟合；
3. `PyTorch Adam`：对同一参数化进行可微细化。

最终温度扫描使用 PyTorch 细化后的参数。

## R08

Lawson 风格诊断流程：

1. 在 `T=4..100 keV` 网格计算 `⟨σv⟩(T)`；
2. 计算 `P_alpha(T)` 与 `P_brem(T)`；
3. 仅在 `P_net = P_alpha - P_brem > 0` 区域定义 `tau_E,min`；
4. 从可行域提取：
   - 首次正净加热点（break-even 近似）；
   - 最小 `n_tau_E` 点；
   - 最小 `nTtau` 点。

## R09

复杂度（`N` 为锚点数量，`G` 为温度网格点，`K` 为优化迭代步数）：

- SciPy 拟合：`O(N*K)`（每步含一次模型评估）
- Sklearn 拟合：`O(N)`（小规模线性回归）
- PyTorch 拟合：`O(N*K)`
- 功率平衡扫描：`O(G)`

本实现默认 `N=13`、`G=240`，运行耗时很低。

## R10

数值稳定性处理：

- 所有温度输入统一 `clip(T, 1e-6, +inf)`，避免 `T^(−1/3)` 奇异；
- 对对数残差使用 `clip(pred, 1e-40, +inf)` 防止 `log(0)`；
- 对 `tau_E` 在 `P_net<=0` 区域置为 `inf`，避免伪物理解；
- 对 torch 优化采用参数指数映射，保证 `a,b,c,d` 始终正值。

## R11

正确性检查策略（脚本断言）：

1. PyTorch 拟合相对 RMSE 需低于阈值（默认 `< 20%`）；
2. 温度扫描中必须出现 `P_net>0` 的可行区；
3. 最优 `nTtau` 点必须有限且温度落在扫描区间内；
4. `nτ_E`、`nTτ_E` 在可行区为正值。

这些检查保证“模型拟合 + 物理判据 + 程序逻辑”三者连通。

## R12

依赖栈与职责分工：

- `numpy`：向量化物理计算与网格扫描
- `scipy`：`least_squares` 非线性拟合
- `pandas`：锚点/诊断表输出
- `scikit-learn`：log-space 线性回归基线
- `torch`：可微参数细化（Adam）

MVP 不依赖额外数据文件，目录可独立运行。

## R13

运行方式：

```bash
uv run python Algorithms/物理-核物理-0446-核聚变_(Nuclear_Fusion)/demo.py
```

或在该目录下运行：

```bash
uv run python demo.py
```

脚本固定参数执行，不需要命令行输入。

## R14

物理假设与边界：

- 只演示 D-T 单通道，不含 D-D、D-He3 分支；
- 采用均匀 0D 体模型，不解输运方程与剖面；
- `P_brem` 用简化经验式，忽略回旋辐射、线辐射等复杂损失；
- 默认 `Ti=Te` 且 `Z_eff` 常数。

因此结果用于算法演示与量级判断，不替代工程设计代码。

## R15

可扩展方向：

1. 替换锚点为公开反应率数据库并增加不确定度传播；
2. 引入外加加热 `P_aux` 与总增益 `Q` 路径；
3. 扩展到 D-He3、p-B11 反应并比较最优温区；
4. 接入简化输运模型，做 `n-T-tau` 三维操作窗评估。

## R16

工程复现性：

- 随机种子固定为 `20260407`；
- 参数、常量、锚点均在源码显式声明；
- 输出以文本表格为主，便于批处理日志留痕与自动校验。

## R17

非黑盒说明：

- 反应性函数形式在源码中显式定义（非外部神秘 API）；
- SciPy/Sklearn/PyTorch 只承担“数值优化与回归”角色，物理方程与功率平衡计算均为手写；
- `P_alpha -> P_brem -> tau_E` 的每个中间量均在 DataFrame 列中可追踪。

这保证了可解释性与可审计性。

## R18

`demo.py` 源码级流程可拆为 8 步：

1. `build_anchor_dataset()` 构造 D-T 锚点温度与反应性表。  
2. `fit_reactivity_scipy()` 用对数残差调用 `scipy.optimize.least_squares`，得到首版参数。  
3. `fit_reactivity_sklearn()` 在 log-space 特征上用 `LinearRegression` 给出线性基线。  
4. `fit_reactivity_torch()` 用 Adam 最小化对数 MSE，细化同一参数化反应率模型。  
5. `build_fit_comparison_table()` 逐锚点汇总三种拟合预测与相对误差。  
6. `build_operating_scan()` 在温度网格计算 `⟨σv⟩`、`P_alpha`、`P_brem`、`P_net`、`tau_E`、`nτ_E`、`nTτ_E`。  
7. `extract_key_points()` 从可行域提取 break-even、最小 `nτ_E` 和最小 `nTτ_E` 工况。  
8. `main()` 打印结果并执行断言，确保脚本对验证环境可直接运行。  

第三方库算法追踪补充：
- SciPy `least_squares` 在内部执行“残差评估 -> 雅可比近似/更新 -> 信赖域步进”的迭代流程；
- Sklearn 线性回归等价于最小二乘闭式/数值求解；
- PyTorch 路径显式暴露“前向计算 -> 反向梯度 -> 参数更新”三步。

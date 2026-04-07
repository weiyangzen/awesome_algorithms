# 格临爱森参数 (Grüneisen Parameter)

- UID: `PHYS-0445`
- 学科: `物理`
- 分类: `固体物理`
- 源序号: `468`
- 目标目录: `Algorithms/物理-固体物理-0468-格临爱森参数_(Grüneisen_Parameter)`

## R01

格临爱森参数（Grüneisen Parameter, `gamma`）用于刻画晶格振动频率对体积变化的敏感性，是固体非谐性的重要量化指标。常用定义有两层：

1. 模态定义（第 `i` 个声子模）：
`gamma_i = -(V/omega_i) * (d omega_i / dV)`

2. 宏观热力学定义：
`gamma(T) = alpha(T) * K_T * V / C_V(T)`

其中 `alpha` 为体膨胀系数，`K_T` 为等温体模量，`C_V` 为定容热容。

## R02

该参数在固体物理中的价值：

- 它把“微观频率随体积漂移”与“宏观热膨胀-热容-弹性”直接连接；
- 是分析晶格非谐效应、热膨胀、热输运的重要桥梁；
- 常用于高温热力学建模、地球物理状态方程、材料热稳定性评估。

因此，`gamma` 可以看作“材料非谐程度”的可计算摘要量。

## R03

本目录 MVP 目标：

- 构造一个可复现的多模态模型 `theta_i(V)=theta_i0*(V/V0)^(-gamma_i)`；
- 用有限差分从模态定义估计 `gamma_i`；
- 用 Einstein 模态热容构造热容加权 `gamma(T)`；
- 用 `gamma=alpha*K_T*V/C_V` 正反计算并做一致性断言。

约束：不调用黑盒材料数据库/声子软件，公式与数值流程全部在 `demo.py` 展开。

## R04

本实现使用的关键公式：

1. 模态体积依赖：
`theta_i(V) = theta_i0 * (V/V0)^(-gamma_i)`

2. 中心差分估计模态 `gamma_i`：
`gamma_i^FD = -(V0/theta_i(V0)) * [theta_i(V+)-theta_i(V-)]/(V+-V-)`

3. Einstein 单模热容：
`C_{V,i}(T)=g_i*R*x_i^2*exp(x_i)/(exp(x_i)-1)^2`, `x_i=theta_i0/T`

4. 热容加权平均：
`gamma(T)=sum_i[gamma_i*C_{V,i}(T)] / sum_i[C_{V,i}(T)]`

5. 热力学关系：
`alpha(T)=gamma(T)*C_V(T)/(K_T*V0)`

## R05

实现级算法流程：

1. 校验配置与模态数组合法性；
2. 生成 `V-`, `V0`, `V+` 三个体积点；
3. 计算每个模态的 `theta(V-)`, `theta(V0)`, `theta(V+)`；
4. 用中心差分估计 `gamma_i^FD` 并形成模态误差表；
5. 在温度网格上计算每个模态热容 `C_{V,i}(T)`；
6. 汇总得到 `C_V(T)` 与 `gamma(T)`；
7. 由热力学关系得到 `alpha(T)`，再反推 `gamma_recovered(T)`；
8. 输出样例表并执行物理断言。

## R06

复杂度（`N_m` 模态数，`N_T` 温度点数）：

- 模态差分估计：`O(N_m)`；
- 温度扫描热容与加权：`O(N_m * N_T)`；
- 空间复杂度：`O(N_m * N_T)`（主要是模态热容矩阵）。

默认 `N_m=3`, `N_T=140`，运行耗时很小，适合批量校验。

## R07

`demo.py` 输出三部分：

1. 参数摘要：`V0`, `K_T`, 差分步长、温区等；
2. `[mode_gamma_estimation]`：每个模态的 `gamma_true`, `gamma_fd`, `rel_err`；
3. `[temperature_samples]`：`T`, `C_V`, `gamma(T)`, `alpha(T)`, `gamma_recovered`；
4. `[sanity_checks]`：模式误差、高温极限误差、反演一致性、正性检查。

## R08

前置知识：

- 格临爱森参数的模态定义与热力学定义；
- Einstein 热容核函数；
- 中心差分与基础数值稳定性。

运行环境：

- Python `>=3.10`
- `numpy`
- `pandas`

## R09

适用场景：

- 教学演示 `gamma_i` 与 `gamma(T)` 的关系；
- 在没有第一性原理声子谱时，搭建可审计的非谐基线模型；
- 验证热膨胀-热容-体模量关系的维度与数值一致性。

不适用场景：

- 需要材料级高精度 `gamma(q,nu)` 分布；
- 需要显式非谐三声子/四声子散射过程；
- 直接替代实验反演或 DFT 声子计算。

## R10

正确性直觉：

1. 若 `theta(V)` 已按幂律构造，有限差分应恢复输入 `gamma_i`；
2. `C_V(T)` 作为权重应让 `gamma(T)` 在高温趋向退化度加权平均；
3. `alpha = gamma*C_V/(K_T*V)` 与其反推式应互相一致；
4. 在本设定 `gamma_i>0` 下，应有 `C_V>0`, `alpha>0`。

`demo.py` 中的断言逐一覆盖这些物理条件。

## R11

数值稳定性处理：

- `C_V` 核函数使用 `exp(-x)` 与 `expm1`，避免大 `x` 溢出和小 `x` 消减；
- 小 `x` 分支使用级数近似 `1 - x^2/12 + x^4/240`；
- 差分步长限制在 `(0, 0.2)`，避免过大截断误差与过小舍入噪声；
- 对所有温度、体积、模态参数执行正值检查。

## R12

关键参数（`GruneisenConfig`）：

- `v0_m3_per_mol`：参考摩尔体积；
- `bulk_modulus_pa`：等温体模量；
- `finite_diff_fraction`：有限差分体积扰动比例；
- `theta_modes_k`：各模态参考温度（等价频率尺度）；
- `gamma_modes_true`：构造数据用的真实模态 `gamma_i`；
- `degeneracies`：模态简并度（默认和为 3）；
- `t_min/t_max/n_temps`：温度扫描范围与采样密度。

## R13

保证类型说明：

- 近似比保证：N/A（非优化问题）。
- 随机成功率保证：N/A（确定性计算）。

本 MVP 的可执行保证：

- 无交互输入；
- 输出固定结构结果表；
- 若关键物理关系不成立，脚本会触发断言失败。

## R14

常见失效模式：

1. 体积、体模量、热容单位不统一导致 `alpha` 数量级错误；
2. 差分步长过大，`gamma_i^FD` 截断误差变大；
3. 差分步长过小，浮点舍入噪声放大；
4. 模态简并度设置不当导致高温 `C_V` 目标值偏移；
5. 温区过窄，无法观察高温极限行为。

脚本通过参数检查与高温/一致性断言覆盖 2-5。

## R15

可扩展方向：

1. 从 Einstein 离散模态替换为离散声子谱 `omega_{q,nu}`；
2. 引入温度依赖体模量 `K_T(T)` 与体积 `V(T)`；
3. 用实验 `alpha(T), C_P(T)` 反演 `gamma(T)`；
4. 加入不确定度传播（参数采样、误差条）；
5. 与德拜模型/准谐近似（QHA）联动做材料拟合。

## R16

相关主题：

- 德拜模型（Debye Model）
- 爱因斯坦模型（Einstein Model）
- 准谐近似（Quasi-Harmonic Approximation）
- 热膨胀与体模量温度依赖
- 声子非谐效应与热输运

## R17

运行方式（无交互）：

```bash
cd Algorithms/物理-固体物理-0468-格临爱森参数_(Grüneisen_Parameter)
uv run python demo.py
```

交付核对：

- `README.md` 的 `R01-R18` 已完整填写；
- `demo.py` 可直接运行并输出结果；
- `meta.json` 与任务元数据保持一致；
- 目录可独立验证。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 创建 `GruneisenConfig`，调用 `validate_config` 与 `unpack_mode_arrays` 做参数检查。  
2. `build_mode_summary_table` 生成 `V-`, `V0`, `V+`，并用 `theta_at_volume` 计算三点模态温度。  
3. `estimate_mode_gamma_fd` 用中心差分估计每个模态 `gamma_i^FD`，输出模态误差表。  
4. `compute_bulk_properties` 在温度网格上调用 `einstein_mode_cv_molar` 计算各模态 `C_{V,i}(T)`。  
5. 同函数按热容权重汇总出 `C_V(T)` 与 `gamma_weighted(T)`。  
6. 用 `alpha = gamma*C_V/(K_T*V0)` 得到 `alpha(T)`，再反推 `gamma_recovered(T)`。  
7. `run_sanity_checks` 依次检查：模态差分误差、高温 `gamma` 极限、高温 `C_V` 极限、反推一致性与正性。  
8. `main` 打印参数摘要、模态表、温度样例表和检查指标，形成可复现交付。  

第三方库角色说明：`numpy` 用于数组和数值计算，`pandas` 只用于表格展示；格临爱森参数的核心流程（体积差分、热容加权、热力学反推、断言）全部在源码中显式实现，不依赖黑盒算法包。

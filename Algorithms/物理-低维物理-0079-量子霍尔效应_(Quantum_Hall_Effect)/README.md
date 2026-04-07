# 量子霍尔效应 (Quantum Hall Effect)

- UID: `PHYS-0079`
- 学科: `物理`
- 分类: `低维物理`
- 源序号: `79`
- 目标目录: `Algorithms/物理-低维物理-0079-量子霍尔效应_(Quantum_Hall_Effect)`

## R01

量子霍尔效应（这里聚焦整数型 IQHE）描述二维电子气在强磁场与低温下出现的横向电导量子化现象：

`sigma_xy = nu * e^2 / h`，其中 `nu` 为接近整数的填充因子。

同时，纵向电导 `sigma_xx` 在平台区接近零，表现为“台阶 + 峰值”的典型输运特征：

- `sigma_xy(B)` 随磁场呈分段平台；
- `sigma_xx(B)` 在平台间跃迁处出现峰值；
- 电阻张量中 `rho_xy` 对应量子化值，`rho_xx` 在平台区小、在跃迁处增大。

## R02

本条目目标是构建一个可运行、可审计的最小数值 MVP，而非完整实验级仿真：

- 输入：电子面密度 `n_e` 与磁场扫描区间 `B`；
- 输出：`nu(B)`、`sigma_xy`、`sigma_xx`、`rho_xy`、`rho_xx` 以及平台诊断；
- 自动验证：
  - 平台区量子化误差足够小；
  - `sigma_xy` 对 `B` 单调（本配置下随 `B` 增大而非增）；
  - `sigma_xx` 峰值位置贴近半整数填充因子。

## R03

MVP 的建模取舍：

- 连续填充因子使用经典关系：`nu_cont = n_e h / (eB)`；
- 不直接做无序势微观哈密顿量对角化，而用“平滑台阶”近似 Landau 能级占据；
- 通过 `sigmoid` 聚合得到量子化填充 `nu_quantized`；
- 用占据导数型项构造 `sigma_xx` 峰值，模拟扩展态主导的跃迁带。

该取舍能保留 IQHE 的关键可解释结构，同时保持脚本短小、稳定、可复现实验逻辑。

## R04

核心公式：

1. 填充因子：`nu_cont(B) = n_e h / (eB)`
2. Landau 能级：`E_n = (n + 1/2) hbar * omega_c`，`omega_c = eB / m*`
3. 平滑占据：`occ_n(nu) = sigmoid((nu - (n+1/2)) / w)`
4. 量子化填充：`nu_q = sum_n occ_n`
5. 纵向电导（无量纲，单位 `e^2/h`）：`sigma_xx_tilde = a * sum_n occ_n(1-occ_n)`
6. 横向电导：`sigma_xy = nu_q * e^2/h`
7. 张量反演：
   - `rho_xy = sigma_xy / (sigma_xx^2 + sigma_xy^2)`
   - `rho_xx = sigma_xx / (sigma_xx^2 + sigma_xy^2)`

## R05

复杂度（`N_B` 为磁场采样数，`L` 为考虑的 Landau 层数）：

- 占据与输运主计算：`O(N_B * L)`；
- 峰值检测：`O(N_B)`；
- 统计诊断：`O(N_B)`。

总体时间复杂度 `O(N_B * L)`，空间复杂度 `O(N_B * L)`（若保留全部占据矩阵；当前实现仅保留必要中间量，近似 `O(N_B * L)` 上界）。

## R06

`demo.py` 默认执行内容：

- 使用 `n_e = 3e15 m^-2`，`B in [2, 12] T`，`600` 个采样点；
- 生成 `nu_cont`、`nu_quantized`、`sigma_xy`、`sigma_xx`、`rho_xy`、`rho_xx`；
- 给出与整数量子化参考 `rho_xy = h/(nu_int e^2)` 的平台误差；
- 检测跃迁峰并比较其与半整数 `nu = k + 0.5` 的偏差；
- 输出代表性平台点表格与诊断指标，最后断言通过则打印 `All checks passed.`。

## R07

优点：

- 每个公式都可追溯到显式代码函数，黑盒程度低；
- 同时覆盖电导、电阻、平台、峰值位置四类结果；
- 作为教学或单元测试基线非常轻量。

局限：

- 未显式求解无序散射与边缘态传播；
- 未引入温度、Zeeman 劈裂、自旋分辨平台；
- 参数是“物理启发 + 工程稳定”而非某具体样品拟合。

## R08

前置知识与运行依赖：

- 低维电子系统、Landau 量子化、电导/电阻张量；
- Python `>=3.10`；
- `numpy`, `pandas`, `scipy`（`constants`, `special.expit`, `signal.find_peaks`）。

运行方式：

```bash
cd Algorithms/物理-低维物理-0079-量子霍尔效应_(Quantum_Hall_Effect)
uv run python demo.py
```

## R09

适用场景：

- 课程演示 IQHE 平台为何出现；
- 在更复杂微观代码前，先验证输运后处理逻辑；
- 作为 CI 中“量子霍尔结构性回归测试”的快速样例。

不适用场景：

- 要求与具体材料样品做高精度拟合；
- 需要分数霍尔效应（FQHE）关联效应；
- 需要边缘态电流分布、非平衡输运、时域动力学。

## R10

正确性直觉：

1. `nu_cont ~ 1/B`：磁场增大时可填充 Landau 层数下降；
2. 每跨越半整数填充，系统从一个平台过渡到下一个平台；
3. 跃迁区态密度有效导通增强，`sigma_xx` 出峰；
4. 平台区 `sigma_xx` 小，故 `rho_xy ~ 1/sigma_xy ~ h/(nu e^2)`；
5. 若数值结果同时满足“平台量子化 + 峰值半整数对齐 + Hall 单调性”，说明主链路一致。

## R11

数值稳定策略：

- 使用足够密集的 `B` 网格（默认 `600` 点）避免假峰；
- 过渡宽度 `transition_width_nu` 控制台阶锐度，避免过陡导致离散采样振荡；
- 峰值检测设置最小高度与最小间距，防止噪声重复计峰；
- 所有关键配置参数在入口做显式校验，防止非法输入（负密度、非法磁场范围）。

## R12

关键参数与调参建议：

- `electron_density_m2`：越大则同一 `B` 下 `nu` 越高；
- `b_min_t`, `b_max_t`：决定覆盖多少个平台；
- `transition_width_nu`：越小台阶越锐利，`sigma_xx` 峰越尖；
- `sigma_xx_peak_scale`：控制纵向电导峰高度；
- `max_level`：需要高于扫描区间中的最大 `nu`，否则高填充端会截断。

## R13

可提供的工程保证（当前模型语义下）：

- 固定参数与依赖版本时，输出确定可复现；
- 平台区 Hall 电阻与 `h/(nu_int e^2)` 的相对误差受断言约束；
- 至少检测到多个跃迁峰，且平均偏离半整数不超过阈值。

注意：这是“模型内保证”，不是对真实实验样品的普适误差保证。

## R14

常见失效模式：

1. `max_level` 太小，导致高填充侧平台被截断；
2. `transition_width_nu` 过大，平台被抹平、峰值模糊；
3. `transition_width_nu` 过小且采样不足，出现数值锯齿或漏峰；
4. 错把 `rho_xy` 与 `sigma_xy` 的量纲关系混用；
5. 忽略 `sigma_xx` 非零对 `rho_xy` 的修正，误判量子化误差。

## R15

可扩展方向：

- 增加温度依赖（Fermi 函数）与无序展宽参数拟合；
- 引入自旋与 Zeeman 劈裂，得到更细平台结构；
- 将 `sigma_xx` 由经验峰替换为 Kubo/Boltzmann 近似计算；
- 加入真实数据读入，用最小二乘或贝叶斯反演参数。

## R16

相关主题：

- 分数量子霍尔效应（FQHE）
- Chern 数与拓扑不变量
- Landauer-Büttiker 边缘态输运
- Shubnikov-de Haas 振荡与低维电子气表征

## R17

`demo.py` 功能清单：

- `QHEConfig`：集中管理密度、磁场、展宽与峰值阈值；
- `filling_factor`：计算连续填充 `nu_cont`；
- `quantized_filling_and_sigma_xx`：从平滑占据构造 `nu_quantized` 与 `sigma_xx`；
- `conductivity_tensor` / `resistivity_tensor`：在电导与电阻表象间转换；
- `run_qhe_mvp`：生成完整 DataFrame 与诊断；
- `run_checks`：执行自动断言；
- `summarize_plateaus`：提取最接近整数平台的代表点并打印。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `validate_config` 检查所有物理与数值参数的合法范围。
2. `magnetic_field_grid` 构造磁场数组 `B`，作为统一自变量。
3. `filling_factor` 计算连续填充 `nu_cont(B)=n_e h/(eB)`。
4. `quantized_filling_and_sigma_xx` 对每个 Landau 层构造 `occ_n=sigmoid((nu-(n+1/2))/w)`，并聚合得到 `nu_quantized` 与 `sigma_xx` 峰。
5. `conductivity_tensor` 将无量纲结果转为 SI 电导：`sigma_xy=nu_q e^2/h`，`sigma_xx=sigma_xx_tilde e^2/h`。
6. `resistivity_tensor` 通过张量反演得到 `rho_xy` 与 `rho_xx`。
7. `run_qhe_mvp` 计算整数量子化参考 `rho_xy_ref=h/(nu_int e^2)`，并汇总成 `pandas` 报表（含 `E0/E1/E2` Landau 能级参考）。
8. 在 `run_qhe_mvp` 内执行峰值检测与诊断统计：平台占比、平台误差、Hall 单调性、半整数峰偏差。
9. `run_checks` 对核心诊断做断言，`main` 打印代表性平台表与诊断，最终输出 `All checks passed.`。

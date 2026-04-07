# 史密斯圆图 (Smith Chart)

- UID: `PHYS-0182`
- 学科: `物理`
- 分类: `电磁学`
- 源序号: `183`
- 目标目录: `Algorithms/物理-电磁学-0183-史密斯圆图_(Smith_Chart)`

## R01

史密斯圆图是把复阻抗/复导纳映射到反射系数平面（`Γ` 平面）的几何工具，常用于传输线阻抗匹配。

核心思想：
- 先把负载阻抗归一化：`z = Z_L / Z_0 = r + jx`；
- 再映射到反射系数：`Γ = (z - 1) / (z + 1)`；
- 在无耗传输线上移动等价于 `Γ` 在单位圆内旋转。

因此，史密斯圆图把“代数匹配问题”转成“圆与旋转的几何问题”。

## R02

本条目 MVP 解决的问题是：

给定 `Z_0`、复负载 `Z_L` 和工作频率 `f`，求一组“传输线长度 + 串联电抗”使输入端精确匹配到 `Z_0`。

更具体地：
1. 沿线寻找 `Re(z_in)=1` 的位置（史密斯图上即与 `r=1` 圆交点）；
2. 在该位置补一个串联电抗 `jX_s` 使虚部抵消；
3. 输出可实现器件（电感或电容）及参数。

脚本使用内置案例，`uv run python demo.py` 可直接运行，无交互输入。

## R03

使用到的关键公式：

1. 阻抗与反射系数双向映射：
   - `Γ = (z - 1) / (z + 1)`
   - `z = (1 + Γ) / (1 - Γ)`
2. 无耗传输线长度变换（`θ = βl`）：
   - `z_in(θ) = (z_L + j tanθ) / (1 + j z_L tanθ)`
3. 等价旋转关系：
   - `Γ_in = Γ_L * exp(-j2θ)`
4. 驻波比：
   - `VSWR = (1 + |Γ|) / (1 - |Γ|)`（`|Γ| < 1`）
5. 史密斯圆图显式几何方程：
   - 常阻圆：`(u - r/(1+r))^2 + v^2 = (1/(1+r))^2`
   - 常抗圆：`(u - 1)^2 + (v - 1/x)^2 = (1/x)^2`（`x != 0`）

## R04

算法高层流程：

1. 输入 `Z_0, Z_L, f` 并归一化得到 `z_L`；
2. 计算 `Γ_L`、`VSWR`，并做 `z -> Γ -> z` 回代一致性检查；
3. 在 `θ ∈ [0, π]` 网格扫描目标函数 `g(θ)=Re(z_in(θ))-1`；
4. 发现变号区间后用二分法求精确根（交点）；
5. 对每个根计算 `z_in`；
6. 令 `x_s = -Im(z_in)` 得到串联归一化电抗；
7. 还原为欧姆电抗 `X_s = x_s * Z_0`，再换算理想 L/C；
8. 用 `|z_match-(1+j0)|`、旋转误差、圆方程残差做质量门限断言。

## R05

核心数据结构：

- `SmithMatchCase`（`dataclass`）：
  - `name`：案例名称；
  - `z0_ohm`：特征阻抗；
  - `z_load_ohm`：复负载阻抗；
  - `freq_hz`：频率；
  - `grid_points`：根搜索网格数。
- `run_case` 返回字典：
  - 负载层指标：`z_load_norm`, `gamma_load`, `vswr`；
  - 校验指标：`roundtrip_error`, `circle_residual_r/x`；
  - 解集合 `solutions`：每个解含 `theta_rad`, `line_len_lambda`, `x_series_ohm`, `component_type/value` 等。

## R06

正确性依据：

- 变换正确性：阻抗-反射系数双向公式显式实现并做回代误差检查；
- 几何一致性：用常阻/常抗圆方程残差验证点确实落在对应圆上；
- 物理一致性：沿线变换同时用 `z_in(θ)` 与 `Γ` 旋转两条路径计算并互相核对；
- 匹配一致性：到 `r=1` 交点后补 `-jIm(z_in)` 必须得到 `1+j0`（数值误差门限内）。

## R07

复杂度分析（`N = grid_points`，根数通常 `K=2`）：

- 网格扫描 `g(θ)`：`O(N)`；
- 二分法求根：每个根 `O(log(1/eps))`，总 `O(K log(1/eps))`；
- 解后处理与指标计算：`O(K)`。

总时间复杂度约 `O(N + K log(1/eps))`，空间复杂度 `O(N)`（保存网格采样值）。

## R08

边界与异常处理：

- `z0_ohm <= 0` 或非有限：抛 `ValueError`；
- `Z_L` 非有限复数：抛 `ValueError`；
- 映射奇点：
  - `z+1≈0` 时 `Γ=(z-1)/(z+1)` 不可用；
  - `1-Γ≈0` 时 `z=(1+Γ)/(1-Γ)` 不可用；
- `grid_points < 101`：抛 `ValueError`；
- 二分区间端点不异号：抛 `ValueError`；
- `[0, π]` 无 `Re(z_in)=1` 交点：抛 `RuntimeError`；
- `freq_hz <= 0`：抛 `ValueError`。

## R09

MVP 取舍：

- 只做无耗传输线（不含衰减常数 `α`）；
- 只做“线段 + 串联单电抗”匹配，不做多节网络优化；
- 不引入 RF 专用黑盒库，全部用 `numpy` 基础复数运算和自写二分法；
- 不画图，仅输出可审计数值，保证自动化验证稳定。

## R10

`demo.py` 函数职责：

- `normalize_impedance`：`Z_L -> z_L`。
- `gamma_from_z_norm` / `z_norm_from_gamma`：双向映射。
- `input_impedance_norm`：按线长计算 `z_in(θ)`。
- `gamma_after_line`：按旋转公式计算 `Γ_in`。
- `smith_circle_residuals`：检查常阻/常抗圆方程残差。
- `find_r_one_intersections`：搜索 `Re(z_in)=1` 交点。
- `_bisect_root`：交点区间二分求精。
- `series_reactance_for_match`：求补偿电抗。
- `reactance_to_component`：把 `X` 换算为理想 L/C。
- `run_case` / `print_case_report` / `main`：组织案例、输出、质量门限。

## R11

运行方式：

```bash
cd Algorithms/物理-电磁学-0183-史密斯圆图_(Smith_Chart)
uv run python demo.py
```

脚本无交互输入，不读取网络资源。

## R12

输出字段说明：

- `zL = ZL/Z0`：归一化负载。
- `Gamma_L`、`|Gamma_L|`、`VSWR`：负载反射特性。
- `z<->Gamma roundtrip`：双向映射回代误差。
- `r/x-circle residual`：圆方程残差，越接近 0 越好。
- 每个匹配解：
  - `theta`：电长度（弧度）；
  - `line length`：`l/λ`；
  - `z_in_norm`：该长度处输入阻抗；
  - `series X`：串联补偿电抗（欧姆）；
  - `component`：L 或 C 及其理想值；
  - `|z_match-(1+j0)|`：匹配误差。
- `Summary`：全案例最大误差与通过标志。

## R13

内置最小测试集：

- `Case-A`：`Z_L = 30 - j40 Ω, Z_0 = 50 Ω, f = 1 GHz`（偏容性）；
- `Case-B`：`Z_L = 120 + j80 Ω, Z_0 = 50 Ω, f = 2.4 GHz`（偏感性）。

每个案例通常产生两组可行解（`0 ~ λ/2` 内）。脚本最终断言：
- `max roundtrip error < 1e-12`
- `max circle residual < 1e-10`
- `max matching error < 1e-9`
- `max gamma rotation error < 1e-10`

## R14

可调参数：

- `grid_points`：根搜索分辨率，越大越稳但更慢；
- 负载 `z_load_ohm`：控制史密斯图上的起点；
- `z0_ohm`：参考阻抗；
- `freq_hz`：只影响器件值换算，不影响归一化几何路径。

调参建议：
- 若漏检根，先增大 `grid_points`（如 `6001 -> 12001`）；
- 若器件值过大，可改变工作频率或切换到另一根对应的解；
- 若数值误差偏大，先检查负载是否接近映射奇点。

## R15

方法对比：

- 对比“直接代数一次解”：
  - 本方法保留史密斯图几何结构，可直接解释“旋转 + 圆交点 + 补偿”；
- 对比 RF 黑盒工具：
  - 本实现不依赖黑盒优化器，关键步骤全部可追踪；
- 对比大规模网络综合：
  - 本 MVP 覆盖单频单节匹配，复杂度低、验证直接，适合作为后续扩展基线。

## R16

典型应用场景：

- 射频前端单频阻抗匹配教学；
- 传输线长度与补偿元件的快速估算；
- 史密斯圆图几何关系的程序化验证；
- 更复杂匹配器（多节、宽带、含损耗）前的基线模块。

## R17

可扩展方向：

- 扩展到有耗线：引入传播常数 `γ = α + jβ`；
- 支持并联 stub 匹配与导纳域工作流；
- 加入频率扫描，生成 `S11(f)` 与带宽指标；
- 增加 CSV/JSON 输入输出和批量案例评估；
- 可选接入可视化（生成史密斯图轨迹）。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 定义两个负载案例，调用 `run_case` 执行求解。  
2. `run_case` 先用 `normalize_impedance` 得到 `z_L`，再计算 `Γ_L` 和 `VSWR`。  
3. 通过 `z_norm_from_gamma(gamma_from_z_norm(z_L))` 做往返检查，得到映射误差。  
4. 调用 `smith_circle_residuals`，把 `z_L` 映射到 `Γ=u+jv` 后代入常阻/常抗圆方程，验证几何一致性。  
5. 调用 `find_r_one_intersections`：在 `[0, π]` 网格评估 `Re(z_in(θ))-1`，找到变号区间并用 `_bisect_root` 精确求根。  
6. 对每个根计算 `z_in(θ)`，用 `series_reactance_for_match` 求 `x_s=-Im(z_in)`，构成 `z_match = z_in + jx_s`。  
7. 把 `x_s` 还原为欧姆电抗并由 `reactance_to_component` 换算成理想电感/电容值，同时用 `gamma_after_line` 验证旋转公式与阻抗变换一致。  
8. `main` 汇总所有案例的最大误差并执行断言门限，确保一次运行即可完成正确性验证。

第三方库未被当成黑盒：`numpy` 只承担基础复数运算、三角函数和数组处理，史密斯圆图映射、根搜索、匹配综合逻辑全部在源码中显式实现。

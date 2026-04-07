# 水平集方法 (Level Set)

- UID: `MATH-0165`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `165`
- 目标目录: `Algorithms/数学-数值分析-0165-水平集方法_(Level_Set)`

## R01

水平集方法（Level Set）用一个高维标量函数 `phi(x,t)` 的零等值面 `phi=0` 来隐式表示界面。与显式参数曲线相比，它天然支持拓扑变化（合并、分裂）和法向几何量计算。

本条目给出一个最小可运行 MVP：
- 2D 圆形界面在常法向速度下传播；
- 用一阶时间推进 + Godunov 上风离散更新 `phi`；
- 通过“面积等效半径”与解析半径对比验证数值合理性。

## R02

本实现求解的模型方程：

- `phi_t + F * |grad(phi)| = 0`

其中：
- `F` 为常数法向速度；
- 初始 `phi` 取圆的有符号距离函数（内部负、外部正）；
- 零等值面随时间演化为半径 `R(t)=R0+F*t` 的圆（在边界影响可忽略时）。

## R03

离散设定：
- 空间：二维笛卡尔网格 `nx x ny`，步长 `dx, dy`；
- 时间：显式 Euler，`phi^{n+1} = phi^n - dt * F * |grad(phi)|_upwind`；
- 边界：零法向导数（Neumann），用边界复制实现。

稳定性约束采用 CFL 形式：
- `dt <= cfl * min(dx,dy) / |F|`（`F != 0` 时）。

## R04

Godunov 上风梯度（Hamiltonian `H=F|grad(phi)|`）按 `F` 符号分支：

- 当 `F >= 0`：
  - `|grad(phi)|^2 = max(Dx-,0)^2 + min(Dx+,0)^2 + max(Dy-,0)^2 + min(Dy+,0)^2`
- 当 `F < 0`：
  - `|grad(phi)|^2 = min(Dx-,0)^2 + max(Dx+,0)^2 + min(Dy-,0)^2 + max(Dy+,0)^2`

其中 `Dx- / Dx+`、`Dy- / Dy+` 分别是后向/前向差分。该形式是 Level Set 一阶单调格式的常见基线。

## R05

`demo.py` 的核心数据对象：
- `phi: np.ndarray`：当前水平集函数；
- `Snapshot(dataclass)`：记录 `step/time`、数值半径、解析半径、误差、包围面积；
- `snapshots: list[Snapshot]`：演化过程采样序列。

用面积估算界面半径：
- `area = count(phi<=0) * dx * dy`
- `R_numeric = sqrt(area/pi)`

## R06

正确性验证路径：
- 初值是圆的 signed distance，解析界面仍是圆；
- 对常速传播，解析半径 `R_exact(t)=R0+F*t`；
- 脚本输出若干时刻 `R_numeric` 与 `R_exact` 的绝对误差，作为 MVP 的定量检查。

说明：本实现强调“算法流程透明”，并非高阶/高精度工业求解器。

## R07

复杂度分析：
- 单步时间复杂度：`O(nx*ny)`（局部差分与逐点更新）；
- 总时间复杂度：`O(steps*nx*ny)`；
- 空间复杂度：`O(nx*ny)`（若含临时数组，常数倍扩展）。

由于采用显式格式，计算成本与 CFL 限制导致的步数直接相关。

## R08

异常与边界处理：
- 网格尺度过小、区间非法会抛 `ValueError`；
- `phi0` 非有限值会抛 `ValueError`；
- `final_time < 0`、`cfl` 非法会抛 `ValueError`；
- 时间推进后若出现非有限值会抛 `RuntimeError`；
- 每步调用 `apply_neumann_boundary` 保持边界处零法向梯度近似。

## R09

MVP 取舍：
- 只依赖 `numpy`，避免黑盒 PDE 框架；
- 使用一阶上风格式，优先稳健与可读；
- 不引入重初始化（reinitialization）、ENO/WENO、曲率速度等进阶模块；
- 使用面积等效半径做轻量验证，不做复杂几何重构。

## R10

主要函数职责：
- `make_grid`：构建规则网格和步长；
- `signed_distance_circle`：生成圆形初值；
- `apply_neumann_boundary`：施加边界条件；
- `godunov_grad_norm`：计算上风 `|grad(phi)|`；
- `evolve_level_set`：执行时间迭代并采样快照；
- `estimate_radius_from_area`：由 `phi<=0` 区域估算半径；
- `print_report`：输出结构化结果；
- `main`：配置参数并运行完整实验。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0165-水平集方法_(Level_Set)
python3 demo.py
```

脚本无交互输入，会直接打印网格参数、采样时刻半径误差和最终误差统计。

## R12

输出字段说明：
- `step/time`：离散步和对应物理时间；
- `R_numeric`：由数值 `phi` 估算的界面半径；
- `R_exact`：解析半径 `R0+F*t`；
- `abs_error`：`|R_numeric - R_exact|`；
- `area_inside`：`phi<=0` 的面积估计。

若算法与参数合理，误差应保持在与网格分辨率一致的量级。

## R13

建议最小测试：
- 稳定性：把 `cfl` 从 `0.2` 提到 `0.45`，观察误差变化；
- 网格收敛趋势：比较 `81x81`、`161x161`、`241x241`；
- 速度符号：令 `F<0`，检查界面收缩（半径随时间减小）；
- 极端参数：`final_time=0`、`speed=0` 验证退化场景。

## R14

关键可调参数：
- `nx, ny`：空间分辨率，影响精度与成本；
- `speed`：法向传播速度；
- `final_time`：演化总时间；
- `cfl`：显式格式稳定性与步数控制；
- `snapshot_count`：输出采样密度。

建议先保持 `cfl<=0.5`，再通过加密网格评估误差下降。

## R15

与相关方法对比：
- 显式前沿追踪：界面点维护简单场景效率高，但拓扑变化处理困难；
- 水平集法：表示统一、拓扑变化自然，但要在整个欧拉网格更新 `phi`；
- 相场法：界面更“厚”，数值稳定性好但参数与物理尺度耦合更强。

本条目定位在“Level Set 离散骨架”演示，不覆盖全套高阶几何数值技巧。

## R16

典型应用场景：
- 界面运动与形状演化（材料、流体、图像分割）；
- 基于法向速度或几何速度（如曲率）的前沿传播；
- 需要处理界面断裂/并合的计算几何与 PDE 问题。

## R17

可扩展方向：
- 增加 signed-distance 重初始化方程以改善长期稳定性；
- 升级到二阶 TVD RK + ENO/WENO 空间离散；
- 支持空间变速场 `F(x,y,t)` 与曲率项；
- 增加窄带（narrow band）以降低整体计算量；
- 增加单元测试与基准误差曲线输出。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 设定网格、初始半径、速度 `F`、终止时间和 `cfl`，并调用 `make_grid`。
2. `signed_distance_circle` 构造初始 `phi0`，其零等值面是圆界面。
3. `evolve_level_set` 根据 CFL 计算 `dt` 与 `steps`，初始化快照并施加 Neumann 边界。
4. 每一时间步先在 `godunov_grad_norm` 内计算 `Dx-/Dx+/Dy-/Dy+` 差分，再按 `F` 符号选 Godunov 上风组合得到 `|grad(phi)|`。
5. 用显式公式 `phi <- phi - dt * F * |grad(phi)|` 更新全场，并再次执行 `apply_neumann_boundary`。
6. 在采样步调用 `estimate_radius_from_area`，把 `phi<=0` 区域面积换算为 `R_numeric`。
7. 同步计算解析半径 `R_exact=R0+F*t` 与绝对误差，写入 `Snapshot` 序列。
8. `print_report` 逐行打印采样结果，并汇总最终误差与最大误差，完成可复现验证闭环。

# Parrinello-Rahman方法 (Parrinello-Rahman Method)

- UID: `PHYS-0330`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `337`
- 目标目录: `Algorithms/物理-计算物理-0337-Parrinello-Rahman方法_(Parrinello-Rahman_Method)`

## R01

Parrinello-Rahman 方法用于在分子动力学中让**模拟盒形状和体积都成为动力学变量**，从而描述各向异性应力下的晶胞响应。

本条目给出一个可运行的最小 MVP：
- 周期性 Lennard-Jones 粒子系；
- 计算内部应力张量；
- 用 Parrinello-Rahman 风格的矩阵方程推进胞矩阵 `h`；
- 输出应力误差与晶胞形变，自动给出 `PASS/FAIL`。

## R02

问题定义（MVP 范围）：
- 输入：`demo.py` 内固定参数（粒子数、温度、时间步、目标应力、barostat 质量等）。
- 输出：采样轨迹表（能量、温度、体积、胞参数、应力分量、应力误差范数）和验证指标。
- 目标：展示“粒子动力学 + 内部应力估计 + 变胞动力学 + 可审计验证”的完整闭环。

本实现强调算法透明，不追求工程级高精度势场与长时间稳定性极限。

## R03

核心变量：
- 分数坐标 `s_i` 与笛卡尔坐标 `r_i` 的关系：`r_i = h s_i`；
- 胞矩阵 `h` 为 `3x3`，可编码长度与剪切；
- 内部应力 `sigma_int` 由动能项与 virial 项组成。

MVP 使用的简化演化关系：
- 粒子（笛卡尔）采用 Langevin-Euler 形式：
  `v <- v + (F/m) dt`，再加阻尼/热噪声，`r <- r + v dt`。
- 胞矩阵采用 Parrinello-Rahman 风格：
  `Gdot = (sigma_int - sigma_target) / W`，`h <- h + (G h) dt`，其中 `W` 为 barostat 质量参数。

## R04

周期边界与最小镜像在分数坐标中实现：
- 粒子对差值 `ds = s_j - s_i`；
- 最小镜像：`ds <- ds - round(ds)`；
- 转回笛卡尔位移：`dr = h ds`。

这样可自然支持非正交（含剪切）晶胞，而不仅限立方盒。

## R05

粒子相互作用使用截断平移的 Lennard-Jones 势：
- `U(r) = 4*eps*((sigma/r)^12 - (sigma/r)^6) - U(rc)`，`r < rc`；
- `U(r)=0`，`r>=rc`。

内部应力张量采用：
- `sigma_int = (K + W_virial) / V`；
- `K = sum_i m v_i v_i^T`；
- `W_virial = sum_{i<j} r_ij ⊗ f_ij`；
- `V = det(h)`。

`demo.py` 中这些量都显式计算，没有黑箱 MD 引擎调用。

## R06

`demo.py` 输入输出约定：
- 输入：无需命令行参数，不需要交互。
- 输出：
1. 参数回显（粒子数、步数、目标应力、barostat 参数）；
2. 轨迹采样表；
3. 验证指标；
4. `Validation: PASS/FAIL`。

校验失败时脚本以非零状态退出，便于自动化流水线检测。

## R07

高层流程：
1. 初始化分数坐标、初始各向异性胞矩阵 `h`、粒子速度。
2. 按当前 `h` 与周期边界计算 LJ 力、势能、virial。
3. 由速度与 virial 计算内部应力 `sigma_int`。
4. 更新粒子速度与位置，并回写到分数坐标（含 wrap）。
5. 由 `sigma_int - sigma_target` 更新 barostat 速度矩阵。
6. 推进胞矩阵 `h`，并做速度仿射修正。
7. 周期采样统计量写入 `pandas.DataFrame`。
8. 汇总验证并输出 `PASS/FAIL`。

## R08

设粒子数为 `N`，步数为 `T`：
- 力与 virial 的两两计算复杂度 `O(N^2)`；
- 总时间复杂度 `O(T * N^2)`；
- 空间复杂度约 `O(N + T_sample)`。

MVP 取 `N=8`，朴素 `O(N^2)` 实现足够快且易审计。

## R09

数值稳定性措施：
- 较小时间步 `dt=0.002`；
- 粒子与 barostat 都加入阻尼项，抑制高频振荡；
- 胞速度矩阵每步对称化，去除无关旋转分量；
- 每步检查体积正性，防止 `det(h)<=0`；
- 验证阶段检查有限性与体积范围。

## R10

MVP 技术栈：
- Python 3
- `numpy`：向量化数值计算（力、应力、变胞更新）
- `pandas`：采样结果表格化与输出
- 标准库 `dataclasses`

算法核心完全在源码中展开，实现路径透明。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0337-Parrinello-Rahman方法_(Parrinello-Rahman_Method)
uv run python demo.py
```

预期输出包含轨迹采样、验证指标与最终 `Validation` 结果。

## R12

采样输出字段含义：
- `step`, `time`: 步号与物理时间
- `potential`, `kinetic`, `temperature`: 势能、动能、瞬时温度
- `volume`: 体积 `det(h)`
- `a_len`, `b_len`, `c_len`: 三个晶胞基矢长度
- `h_xy`, `h_xz`, `h_yz`: 胞矩阵上三角剪切分量
- `stress_xx`, `stress_yy`, `stress_zz`: 内部应力对角分量
- `stress_error_norm`: `||sigma_int - sigma_target||_F`

## R13

内置验证规则：
1. 全表必须有限（无 `nan/inf`）。
2. 体积最小值大于 `min_volume`。
3. 体积最大值小于 `max_volume`。
4. 晶胞变化量 `||h_final - h0||` 大于阈值（确认变胞机制有效）。
5. 最终应力误差范数小于初始值（确认有向目标应力收敛趋势）。

全部满足则输出 `Validation: PASS`。

## R14

当前实现局限：
- 采用教学型简化 PR 方程，未完整包含严格拉格朗日形式中的全部耦合项；
- 粒子积分器使用简单 Euler-Maruyama，而非高阶辛积分；
- 仅演示小体系、短时程，不用于生产级材料定量预测；
- 未使用邻居表、并行、长程电势等工程优化。

## R15

可扩展方向：
- 改为 velocity-Verlet / Trotter splitting 形式的更稳定 NPT 积分器；
- 增加 Nosé-Hoover 链与更严格 PR 扩展变量耦合；
- 引入邻居表（cell list）把大体系计算效率显著提升；
- 支持非对角目标应力与弹性常数反演实验；
- 增加 RDF、MSD、应力自相关等物理分析模块。

## R16

典型应用：
- 晶体在外压或各向异性载荷下的晶胞形变模拟；
- 固液相转换时体积与形状自由变化过程；
- 材料弹性响应与应力控制下结构松弛教学演示；
- 计算物理课程中的 NPT/变胞动力学入门实验。

## R17

方法对比：
- 相比固定盒 NVT/NVE：Parrinello-Rahman 能模拟应力驱动的盒形变化，但参数调节更敏感。
- 相比仅各向同性 barostat：PR 可处理剪切与各向异性形变，不局限体积标量缩放。
- 相比黑箱 MD 软件：本条目更小、更透明，便于学习公式到代码的映射。

## R18

`demo.py` 源码级算法流（9 步）：
1. `PRConfig` 定义粒子、LJ 势、积分、barostat、目标应力与验证阈值。
2. `make_initial_fractional_positions` 生成 2x2x2 分数坐标晶格初态。
3. `lj_forces_energy_virial` 在周期边界下计算 LJ 力、势能和 virial 张量。
4. `internal_stress_tensor` 用动能张量与 virial 张量构建内部应力。
5. `run_simulation` 粒子子步：速度更新、Langevin 阻尼/噪声、坐标推进与分数坐标回绕。
6. `run_simulation` 变胞子步：由 `sigma_int - sigma_target` 更新 barostat 速度矩阵，并推进 `h`。
7. `run_simulation` 执行速度仿射修正并按间隔记录 `potential/volume/stress_*` 等观测量。
8. `validate` 检查有限性、体积上下界、应力误差下降和晶胞变化量。
9. `main` 打印采样表与验证指标，输出 `Validation: PASS/FAIL`，失败时返回非零退出码。

该流程把 Parrinello-Rahman 的关键思想（应力反馈驱动胞矩阵演化）拆成可逐行检查的实现步骤。

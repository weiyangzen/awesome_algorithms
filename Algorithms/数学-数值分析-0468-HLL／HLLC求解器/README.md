# HLL/HLLC求解器

- UID: `MATH-0468`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `468`
- 目标目录: `Algorithms/数学-数值分析-0468-HLL／HLLC求解器`

## R01

HLL/HLLC 是双曲守恒律（尤其可压缩流体 Euler 方程）中最常用的近似黎曼求解器族：
- HLL（Harten-Lax-van Leer）只保留左右两条外波，鲁棒、稳定、实现简单；
- HLLC（HLL-Contact）在 HLL 基础上补回接触波，能显著改善接触间断分辨率。

本条目给出一个最小可运行 MVP：
- 方程：1D Euler 方程；
- 场景：Sod 激波管；
- 离散：一阶有限体积 Godunov 更新；
- 对比：同网格下 HLL vs HLLC 的守恒误差、正性、接触带厚度。

## R02

控制方程（守恒形式）：
\[
\partial_t \mathbf U + \partial_x \mathbf F(\mathbf U)=0,
\]
其中
\[
\mathbf U=\begin{bmatrix}\rho\\\rho u\\E\end{bmatrix},\quad
\mathbf F(\mathbf U)=\begin{bmatrix}
\rho u\\
\rho u^2+p\\
u(E+p)
\end{bmatrix},\quad
p=(\gamma-1)\left(E-\frac12\rho u^2\right).
\]

这里 \(\rho\) 为密度，\(u\) 为速度，\(p\) 为压力，\(E\) 为总能量密度。

## R03

测试问题（Sod shock tube）：
- 区间：\(x\in[0,1]\)
- 初始分界：\(x_0=0.5\)
- 左侧状态：\((\rho,u,p)=(1.0,0.0,1.0)\)
- 右侧状态：\((\rho,u,p)=(0.125,0.0,0.1)\)
- 比热比：\(\gamma=1.4\)
- 边界：外流（零梯度）边界

该问题会产生稀疏波、接触间断和激波，适合检验黎曼求解器质量。

## R04

有限体积离散：
\[
\mathbf U_i^{n+1} = \mathbf U_i^n - \frac{\Delta t}{\Delta x}
\left(\hat{\mathbf F}_{i+1/2}-\hat{\mathbf F}_{i-1/2}\right),
\]
其中 \(\hat{\mathbf F}_{i+1/2}\) 由 HLL 或 HLLC 计算。

时间步由 CFL 条件给出：
\[
\Delta t = \text{CFL}\cdot \frac{\Delta x}{\max_i(|u_i|+a_i)},\quad
a_i=\sqrt{\gamma p_i/\rho_i}.
\]

## R05

HLL 通量思想：仅使用两条波速 \(S_L,S_R\)。
- 若 \(S_L\ge 0\)，取左通量 \(F_L\)；
- 若 \(S_R\le 0\)，取右通量 \(F_R\)；
- 否则取中间 HLL 混合通量：
\[
\hat F_{HLL}=\frac{S_R F_L-S_L F_R+S_LS_R(U_R-U_L)}{S_R-S_L}.
\]

本实现使用 Davis 估计：
\[
S_L=\min(u_L-a_L,\,u_R-a_R),\quad
S_R=\max(u_L+a_L,\,u_R+a_R).
\]

## R06

HLLC 在 HLL 中恢复接触波 \(S_*\)，将中间区分为左星区和右星区：
\[
S_*=
\frac{p_R-p_L+\rho_Lu_L(S_L-u_L)-\rho_Ru_R(S_R-u_R)}
{\rho_L(S_L-u_L)-\rho_R(S_R-u_R)}.
\]

并据此构造 \(U_{*L},U_{*R}\) 与分段通量：
- \(S_L\le 0\le S_*\)：\(F=F_L+S_L(U_{*L}-U_L)\)
- \(S_*\le 0\le S_R\)：\(F=F_R+S_R(U_{*R}-U_R)\)

这样可保留接触不连续信息，通常比 HLL 更锐利。

## R07

HLL 与 HLLC 对比要点：
- HLL：更耗散，接触间断被涂抹更宽，优点是稳健；
- HLLC：接触分辨率更高，密度/速度接触层更窄；
- 两者都远比“中心差分 + 无耗散”更稳定，且可用于激波问题。

在一阶空间重构下，二者都会有数值扩散，但 HLLC 通常仍明显更清晰。

## R08

复杂度分析（每个时间步）：
- 网格单元数为 \(N\)，界面数约 \(N+1\)
- 每个界面做常数次代数运算
- 单步复杂度 \(O(N)\)，总复杂度 \(O(NN_t)\)
- 存储主要为 \(U\) 与界面通量 \(F\)，空间复杂度 \(O(N)\)

## R09

算法伪代码：

```text
输入: nx, gamma, cfl, t_end, scheme in {HLL,HLLC}
构造网格与 Sod 初值 U
while t < t_end:
    由 U 计算 max(|u|+a)
    dt = cfl * dx / max_speed
    若 t+dt > t_end: dt = t_end - t
    施加外流 ghost cell
    对每个界面 i+1/2:
        取 UL, UR
        用 HLL 或 HLLC 计算界面通量 F[i]
    用守恒更新 U <- U - dt/dx * (F[i+1]-F[i])
    检查 rho,p 正性与有限性
输出守恒误差、最小 rho/p、接触区指标
```

## R10

`demo.py` 模块组织：
- `primitive_to_conservative` / `conservative_to_primitive`：变量互转；
- `flux`：Euler 物理通量；
- `hll_flux`：HLL 界面通量；
- `hllc_flux`：HLLC 界面通量（含星区状态）；
- `finite_volume_step`：一次 Godunov 更新；
- `run_solver`：完整时间推进；
- `contact_region_metrics`：接触层宽度/梯度诊断；
- `main`：固定参数运行 HLL 与 HLLC 并打印对比表。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0468-HLL／HLLC求解器
python3 demo.py
```

代码无交互输入，直接输出实验结果。

## R12

默认参数：
- `nx = 400`
- `cfl = 0.45`
- `t_end = 0.20`
- `gamma = 1.4`

该设置可在很短时间内给出稳定的激波管剖面，并可直观看到 HLL/HLLC 差异。

## R13

输出指标说明：
- `mass_err / mom_err / energy_err`：总质量、总动量、总能量初末差；
- `min_rho / min_p`：正性检查；
- `contact_cells`：接触区域混合单元计数（越小通常越锐利）；
- `max|drho|`：接触附近最大密度梯度（越大通常越锐利）；
- `Mean |rho_HLLC - rho_HLL|`：两求解器最终密度剖面的平均差异。

说明：本 MVP 使用外流边界，`mom_err` 反映“边界通量作用后的动量变化”，不应机械地按封闭系统守恒误差解读。

## R14

最小验证清单：
1. 两种求解器均完成时间推进，无异常退出；
2. `min_rho > 0` 且 `min_p > 0`；
3. 守恒误差保持在可接受的小量级；
4. HLLC 在接触分辨率指标上通常优于 HLL（更少混合单元或更大局部梯度）。

## R15

常见实现错误：
- 把保守量和原始量混用，导致压力计算错误；
- 漏掉最后一步 `dt=t_end-t` 截断，终止时间漂移；
- HLLC 星区能量公式写错，造成负压；
- 波速估计或分支条件次序错误，导致通量不连续；
- 边界未处理，界面访问越界或出现非物理反射。

本实现包含正性/有限性检查，便于快速定位问题。

## R16

与相关方法关系：
- HLL 可视为 Roe/Exact 的更鲁棒近似版本，代价低、实现短；
- HLLC 在 HLL 基础上恢复接触波，常作为可压缩流体一阶基线；
- 若需更高精度，可在此基础上叠加 MUSCL 重构 + 限制器，形成二阶 TVD 方案。

## R17

可扩展方向：
- 增加 MUSCL-Hancock（空间二阶）并比较 HLL/HLLC 差异是否进一步放大；
- 引入其他 Riemann 求解器（Roe、AUSM）做统一基准；
- 扩展到二维 Euler（方向分裂）；
- 增加与解析 Sod 解的误差量化（L1/L2）。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 固定 `nx/cfl/t_end`，分别调用 `run_solver(..., scheme="HLL")` 与 `run_solver(..., scheme="HLLC")`。  
2. `run_solver` 在单元中心生成网格，并通过 `sod_initial_condition` 构造初始保守量 `U=[rho,mom,E]`。  
3. 每个时间步先用 `max_signal_speed` 计算全局最大传播速度 `max(|u|+a)`，再按 CFL 公式得到 `dt`，最后一步截断到 `t_end`。  
4. `finite_volume_step` 先调用 `apply_outflow_ghosts` 施加外流 ghost cell，再逐界面读取 `UL/UR`。  
5. 若方案是 HLL：`hll_flux` 计算 `S_L,S_R` 并按三段分支返回界面通量；若是 HLLC：`hllc_flux` 额外计算 `S_*`、`U_*L/U_*R` 后按四段分支返回通量。  
6. 用守恒更新式 `U^{n+1}=U^n-(dt/dx)(F_{i+1/2}-F_{i-1/2})` 更新全域保守量。  
7. 更新后立刻做物理约束检查：`rho>0`、`p>0`、所有状态有限；若失败直接抛错终止。  
8. 时间推进结束后统计守恒误差、最小 `rho/p`、接触层指标（`contact_cells` 与 `max|drho|`），并打印 HLL 与 HLLC 对照结果。  

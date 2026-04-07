# Roe近似Riemann解

- UID: `MATH-0467`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `467`
- 目标目录: `Algorithms/数学-数值分析-0467-Roe近似Riemann解`

## R01

Roe近似Riemann解是一类用于双曲守恒律数值离散的界面通量构造方法。核心思想是在每个网格界面把非线性系统局部线性化为一个常系数Riemann问题，再用该线性问题的特征分解给出数值耗散项，从而得到稳定且分辨率较高的Godunov型通量。

## R02

本条目采用的一维模型是浅水方程（无底床项）：

- 守恒变量：`U=[h, hu]^T`
- 物理通量：`F(U)=[hu, hu^2/h + 0.5*g*h^2]^T`

其中 `h` 是水深，`u` 是速度，`g` 是重力常数。该系统是典型双曲守恒律系统，适合展示Roe通量的两特征波结构。

## R03

有限体积半离散形式：

`dU_i/dt = -(1/dx) * (F_{i+1/2} - F_{i-1/2})`

关键在于界面通量 `F_{i+1/2}`。Roe方法把它写成“中心通量 - 特征耗散”：

`F_roe = 0.5*(F_L + F_R) - 0.5*D_roe(U_L,U_R)`

其中 `D_roe` 由Roe平均雅可比矩阵的特征分解构造。

## R04

对浅水方程，Roe平均量采用：

- `u_tilde = (sqrt(h_L)*u_L + sqrt(h_R)*u_R) / (sqrt(h_L)+sqrt(h_R))`
- `h_tilde = 0.5*(h_L + h_R)`
- `c_tilde = sqrt(g*h_tilde)`

对应特征值：

- `lambda_1 = u_tilde - c_tilde`
- `lambda_2 = u_tilde + c_tilde`

右特征向量可取：

- `r_1 = [1, u_tilde-c_tilde]^T`
- `r_2 = [1, u_tilde+c_tilde]^T`

## R05

设 `Delta U = U_R-U_L = [Delta h, Delta m]^T`（`m=hu`），将跳变分解到特征向量基：

- `alpha_1 = ((u_tilde+c_tilde)*Delta h - Delta m)/(2*c_tilde)`
- `alpha_2 = (Delta m - (u_tilde-c_tilde)*Delta h)/(2*c_tilde)`

耗散项为：

`D_roe = |lambda_1|*alpha_1*r_1 + |lambda_2|*alpha_2*r_2`

最终界面通量：

`F_roe = 0.5*(F_L+F_R) - 0.5*D_roe`

## R06

纯Roe在跨声速稀疏波附近可能出现熵违例。MVP中对特征值使用Harten熵修正：

- 若 `|lambda| >= eps`，用 `|lambda|`
- 若 `|lambda| < eps`，用 `0.5*(lambda^2/eps + eps)`

这会在小特征速度区间引入平滑耗散，避免非物理解。

## R07

空间离散采用一阶有限体积Godunov框架：

- 单元平均量 `U_i`
- 界面左右态直接取相邻单元值（MUSCL重构未启用）
- 每步根据所有界面Roe通量统一更新

该实现是“最小可运行版本”，优先保证算法主干清晰。

## R08

时间推进使用显式前向Euler，步长由CFL条件控制：

`dt = CFL * dx / max_i(|u_i| + sqrt(g*h_i))`

并在最后一步截断 `dt` 以精确到达 `t_final`。

## R09

边界条件采用外流（零梯度）幽灵单元：

- 左侧幽灵单元复制第一个物理单元
- 右侧幽灵单元复制最后一个物理单元

实现简单且适合标准dam-break演示。

## R10

稳定性与数值鲁棒处理：

- `h_floor` 防止干床导致除零
- 所有 `h` 参与除法/开方前都与 `h_floor` 比较取大
- 更新后再次裁剪 `h>=h_floor`

这些处理不改变Roe核心结构，但显著降低MVP在极端状态下崩溃概率。

## R11

复杂度分析（`N` 为网格数，`T` 为时间步数）：

- 单步计算：`O(N)`
- 总复杂度：`O(N*T)`
- 存储复杂度：`O(N)`

向量化实现使常数项较小，`python3 demo.py` 在普通CPU上可快速完成。

## R12

`demo.py` 中实现内容：

- Roe界面通量（含两特征波分解）
- Harten熵修正
- 一阶有限体积更新
- CFL自适应时间步
- dam-break初值测试
- 输出 `result.csv`（列：`x,h,u,momentum`）

## R13

运行方式：

```bash
cd Algorithms/数学-数值分析-0467-Roe近似Riemann解
python3 demo.py
```

无交互输入。运行完成后会在同目录生成 `result.csv`。

## R14

默认测试设置：

- 区间：`[0,1]`
- 网格：`400` 个单元
- 初值：左侧 `h=2`，右侧 `h=1`，速度均为 `0`
- 终止时刻：`t=0.15`

这是浅水激波管（dam-break）经典构型，可观察稀疏波/间断传播。

## R15

可用以下方式做快速正确性检查：

- 检查控制台输出的 `水深范围` 是否均为正
- 检查 `总质量` 与初值质量相比仅有小量误差
- 检查 `result.csv` 中界面附近是否出现合理波系结构而非随机振荡

## R16

常见实现错误：

- 直接用 `h` 做除法，未做 `h_floor` 防护
- `alpha_1/alpha_2` 公式符号写反，导致耗散项方向错误
- 接口通量数组与单元更新下标错位（`i+1/2` 与 `i-1/2`）
- 忽略熵修正导致局部非物理膨胀激波

## R17

可扩展方向：

- 二阶MUSCL重构 + 限幅器（如minmod/van Leer）
- SSP-RK2/RK3时间推进
- 湿干前沿专用处理（well-balanced与positivity-preserving）
- 扩展到Euler方程（3变量）的Roe求解器

## R18

`demo.py` 的源码级算法流程可拆为8步：

1. `initial_condition` 构造单元中心初值 `U=[h,hu]`（dam-break左右常值）。
2. `simulate` 进入时间循环，先由 `max_wave_speed` 计算全局最大传播速度 `|u|+c`。
3. 按CFL公式得到 `dt`，并在最后一步裁剪到 `t_final`。
4. `finite_volume_step` 通过复制端点构造两侧幽灵单元，形成所有界面的 `U_left/U_right`。
5. `roe_flux` 逐界面计算 Roe 平均 `u_tilde,h_tilde,c_tilde` 与特征值 `lambda_1/lambda_2`。
6. `roe_flux` 将 `Delta U` 分解成 `alpha_1/alpha_2`，再与特征向量组合出耗散项，并构造 `F_roe`。
7. `finite_volume_step` 用 `U^{n+1}=U^n-(dt/dx)*(F_{i+1/2}-F_{i-1/2})` 完成守恒更新，再做 `h_floor` 裁剪。
8. 达到终止时刻后，`save_csv` 输出 `x,h,u,momentum` 到 `result.csv`，主程序打印范围与近似守恒量统计。

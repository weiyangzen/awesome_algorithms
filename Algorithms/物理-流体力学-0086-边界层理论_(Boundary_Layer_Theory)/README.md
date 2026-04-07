# 边界层理论 (Boundary Layer Theory)

- UID: `PHYS-0086`
- 学科: `物理`
- 分类: `流体力学`
- 源序号: `86`
- 目标目录: `Algorithms/物理-流体力学-0086-边界层理论_(Boundary_Layer_Theory)`

## R01

边界层理论用于描述高雷诺数外流中“壁面附近薄层黏性区 + 外部近似无黏区”的分层结构。核心思想是：
- 绝大部分速度变化集中在薄边界层内；
- 层外流速约为自由来流 `U_inf`；
- 阻力、剪切应力、换热等关键工程量由边界层主导。

## R02

历史与定位：
- Ludwig Prandtl 在 1904 年提出边界层分离思想，解释了经典无黏理论无法解释的阻力问题；
- 边界层理论成为空气动力学、船舶水动力学、传热传质与湍流建模的基础；
- 在工程实践中，平板边界层解常用于初步估算壁面摩擦与层流尺度。

## R03

本条目 MVP 的目标是实现一个可运行、可核查的“层流平板 Blasius 解”流程：

1. 通过射击法求解 Blasius 相似常微分方程；
2. 得到 `u/U_inf = f'(eta)` 的速度剖面；
3. 在若干 `x` 位置输出 `Re_x`、`C_f`、`tau_w`、`delta_99`、`delta*`、`theta`、`H`；
4. 用断言验证数值解与经典工程公式的一致性。

## R04

零压梯度、二维、稳态、不可压层流边界层方程：

- 连续性：`du/dx + dv/dy = 0`
- 动量：`u*du/dx + v*du/dy = nu*d2u/dy2`

边界条件：
- 壁面无滑移：`u(x,0)=0`
- 壁面不可渗：`v(x,0)=0`
- 层外匹配：`u(x,inf)=U_inf`

## R05

Blasius 相似变换：

- `eta = y * sqrt(U_inf / (nu*x))`
- `u/U_inf = f'(eta)`

代入可得 ODE：

`f''' + 0.5*f*f'' = 0`

配套边界条件：
- `f(0)=0`
- `f'(0)=0`
- `f'(inf)=1`

MVP 通过射击变量 `s = f''(0)`，将边值问题转为初值问题并用 `brentq` 找根。

## R06

由相似解投影到量纲量（平板局部量）：

- `Re_x = U_inf*x/nu`
- `C_f,x = 2*f''(0)/sqrt(Re_x) ≈ 0.664/sqrt(Re_x)`
- `tau_w = 0.5*rho*U_inf^2*C_f,x`
- `delta_99 = eta_99*sqrt(nu*x/U_inf)`（`u/U_inf=0.99` 对应厚度）
- `delta* = sqrt(nu*x/U_inf) * integral(1-u/U_inf) d_eta`
- `theta = sqrt(nu*x/U_inf) * integral((u/U_inf)*(1-u/U_inf)) d_eta`
- `H = delta*/theta`

## R07

`demo.py` 的工况策略：
- 固定流体参数：`U_inf=15 m/s`、`nu=1.5e-5 m^2/s`、`rho=1.225 kg/m^3`；
- 取 5 个平板位置：`x = [0.03, 0.06, 0.10, 0.20, 0.30] m`；
- 所有工况保持 `Re_x < 5e5`，对应经典层流边界层近似更合理。

## R08

复杂度分析（`N_eta` 为相似坐标离散点数，`N_x` 为站位数）：
- ODE 单次积分：`O(N_eta)`；
- 射击法根搜索：约 `K` 次积分，总体 `O(K*N_eta)`；
- 站位量纲化与表格输出：`O(N_x)`；
- 总体复杂度：`O(K*N_eta + N_x)`，空间复杂度 `O(N_eta + N_x)`。

## R09

数值稳定与鲁棒性处理：
- 用高精度 `solve_ivp(DOP853)` 积分 Blasius ODE；
- 用 `brentq` 在有符号变化区间内找 `f''(0)`，避免无约束迭代漂移；
- 通过 `f'(eta_max)-1` 检查“截断无穷远边界”是否满足；
- 对 `f'` 的有界性、单调性、有限值进行断言，防止静默错误。

## R10

技术栈：
- Python 3
- `numpy`：数组运算、插值、积分与断言
- `scipy.integrate`：ODE 积分
- `scipy.optimize`：射击法找根
- `pandas`：结果表组织与打印

说明：核心算法（Blasius 方程构造、射击法、相似量到量纲量映射、校验）均在源码显式实现，无黑盒流体求解器。

## R11

运行方式：

```bash
cd Algorithms/物理-流体力学-0086-边界层理论_(Boundary_Layer_Theory)
uv run python demo.py
```

脚本不需要参数输入，不会请求交互。

## R12

输出包含两部分：

1. `similarity_metrics`：
- `fpp0`：求得的 `f''(0)`
- `eta_99`：`u/U_inf=0.99` 的相似坐标
- `displacement_eta`、`momentum_eta`：相似坐标积分量
- `shape_factor_H`：形状因子

2. `station_table`：
- `x_m`、`Re_x`
- `Cf_similarity`、`Cf_classic`、`Cf_rel_error`
- `tau_w_Pa`
- `delta99_mm`、`delta_star_mm`、`theta_mm`
- `shape_factor_H`

## R13

正确性校验（脚本内自动执行）：
1. `f'(eta_max)` 必须接近 1；
2. 速度比 `f'` 必须非负、近似单调增加；
3. `f''(0)` 必须落在 Blasius 经典值附近；
4. `C_f` 相似解与 `0.664/sqrt(Re_x)` 的相对误差必须很小；
5. `H` 必须接近层流平板典型值 `~2.59`；
6. `Re_x` 上限必须低于常见层流转捩阈值 `5e5`。

## R14

当前 MVP 局限：
- 仅覆盖零压梯度层流平板，不含不利压梯度和分离；
- 未覆盖湍流边界层与转捩过程；
- 未加入压缩性、温度变化和可变黏度；
- 采用二维自相似框架，不适用于三维复杂流场。

## R15

可扩展方向：
1. 引入 Falkner-Skan 方程处理非零压梯度边界层；
2. 增加湍流经验模型（如 `1/7` 幂律）做层流/湍流对比；
3. 耦合温度方程，输出热边界层厚度与 `Nu_x`；
4. 加入实验数据拟合，对比 `tau_w` 与测量值偏差。

## R16

工程使用注意事项：
- 特征长度 `x` 必须与平板前缘定义一致，否则 `Re_x` 与厚度量会失真；
- 若 `Re_x` 接近或超过转捩阈值，层流 Blasius 结果仅可作下限/参考；
- 当外流压梯度显著时，本模型的 `C_f` 和 `delta` 误差会增大；
- 计算 `tau_w` 时需确保 `rho`、`U_inf` 单位一致。

## R17

最小测试建议：
1. 单元测试 `blasius_rhs` 维度与符号是否正确；
2. 射击法回归测试：`f''(0)` 是否稳定在 `0.332` 附近；
3. 边界条件测试：`f'(eta_max)` 是否随 `eta_max` 增大而收敛到 1；
4. 公式一致性测试：`Cf_similarity` 与经典相关式误差是否在阈值内。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main()` 构造 `BoundaryLayerConfig`，设定 `U_inf, nu, rho, eta_max, x_stations`。  
2. `solve_blasius()` 先在 `f''(0)` 的上下界上调用 `shooting_residual()`，确认 `f'(eta_max)-1` 存在变号区间。  
3. `brentq` 在该区间迭代求根，得到最优 `f''(0)`；每次残差评估都调用 `integrate_blasius()`。  
4. `integrate_blasius()` 用 `solve_ivp` 积分三维一阶系统 `[f, f', f'']`，得到完整相似速度剖面。  
5. `compute_similarity_metrics()` 从 `f'(eta)` 插值计算 `eta_99`，并对 `(1-f')` 与 `f'(1-f')` 做数值积分得到 `delta*`、`theta` 的相似系数。  
6. `build_station_table()` 将相似结果映射到各 `x` 站位，计算 `Re_x, C_f, tau_w, delta_99, delta*, theta, H`，并与经典 `0.664/sqrt(Re_x)` 对照。  
7. `run_checks()` 执行物理与数值断言：边界条件、单调性、经典常数范围、相关式一致性、层流 `Re_x` 范围。  
8. `main()` 打印相似常数和站位结果表，所有断言通过后输出 `All checks passed.`。  

第三方库边界：`numpy/pandas/scipy` 仅承担数值基础能力；边界层算法主逻辑与验证链条由源码逐步展开。

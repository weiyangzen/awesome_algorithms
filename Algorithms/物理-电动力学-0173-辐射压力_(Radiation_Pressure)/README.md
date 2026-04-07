# 辐射压力 (Radiation Pressure)

- UID: `PHYS-0172`
- 学科: `物理`
- 分类: `电动力学`
- 源序号: `173`
- 目标目录: `Algorithms/物理-电动力学-0173-辐射压力_(Radiation_Pressure)`

## R01

**问题定义**  
辐射压力描述电磁波携带的动量流对物体表面产生的机械压力。给定辐照强度 `I`（W/m²）、表面反射率 `R`、入射角 `theta` 与受光面积 `A`，计算：

1. 法向辐射压力 `p`（Pa）；
2. 由压力产生的法向力 `F = pA`（N）；
3. 在典型场景（吸收体、镜面反射、斜入射）下的可验证数值结果。

## R02

**物理背景**  
电磁辐射不仅传输能量，也传输动量。单位时间单位面积上的动量通量对应“压力”。

- 完全吸收表面（正入射）：`p = I/c`。
- 完全反射表面（正入射）：`p = 2I/c`。
- 斜入射时，表面截获的能流和法向动量变化都带 `cos(theta)` 因子，因此法向压力按 `cos^2(theta)` 缩放。

该机制是太阳帆推进、激光微操纵和辐射平衡问题中的基础组件。

## R03

**数学模型**  
设 `R` 为反射率，`A_abs` 为吸收率，`T` 为透射率，满足 `A_abs + R + T = 1`。

一般法向压力可写为：

`p_n = (I/c) * cos^2(theta) * (A_abs + 2R)`。

本 MVP 采用**不透明表面**（`T = 0`），因此 `A_abs = 1 - R`，得到：

`p_n = (I/c) * (1 + R) * cos^2(theta)`。

并定义法向力：

`F_n = p_n * area`。

## R04

**MVP 输入/输出定义**  
`demo.py` 为无交互脚本，内部固定参数执行。

输入（脚本内部配置）：

1. 一组辐照强度数组 `I_list`；
2. 一组反射率 `R_list`；
3. 入射角 `theta`；
4. 一个太阳帆样例（面积、反射率、入射角、质量）。

输出：

1. 压力对照表（不同 `I` 与 `R`）；
2. 物理一致性断言结果（不通过即抛错）；
3. 太阳帆样例中的压力、力与加速度。

## R05

**建模假设**  
1. 辐射源为单向准直光束，`I` 视作已知平均强度。  
2. 表面宏观平整，仅关注法向压力。  
3. 反射近似镜面且无频散；透射忽略（不透明近似）。  
4. 不考虑热形变、材料退化和多次散射。  
5. 采用 SI 制，真空光速 `c` 为常量。

## R06

**关键公式与约束**  
1. 压力（不透明表面）：

`p = (I/c) * (1 + R) * cos^2(theta)`

2. 力：

`F = p * A`

3. 光子图像（用于交叉验证）：

- 单光子能量：`E_ph = h*c/lambda`
- 单光子动量：`p_ph = h/lambda`
- 单位面积每秒入射光子数：`Phi = I*cos(theta)/E_ph`
- 单光子法向动量改变量（吸收+反射统一写法）：`Delta p_n = (1+R)*p_ph*cos(theta)`
- 压力：`p = Phi * Delta p_n`

4. 参数约束：`I >= 0`，`0 <= R <= 1`，`0 <= theta <= 90 deg`，`A >= 0`。

## R07

**算法思路**  
1. 实现核心函数 `radiation_pressure`，直接按闭式公式计算。  
2. 实现 `radiation_force`，由压力乘面积得到力。  
3. 实现光子路径函数（光子能量、光子通量、动量改变量），独立算出 `photon_based_pressure`。  
4. 在 `main` 中做四项校验：
   - 反射体压力是吸收体的 2 倍（正入射）；
   - 压力随角度 obey `cos^2(theta)`；
   - 连续介质公式与光子计数公式一致；
   - `R=0, theta=0` 时 `p = I/c`。  
5. 输出压力表和太阳帆样例指标。

## R08

**伪代码**

```text
define c, h

function radiation_pressure(I, R, theta_deg):
    validate I, R, theta
    theta = deg2rad(theta_deg)
    return (I / c) * (1 + R) * cos(theta)^2

function radiation_force(I, R, theta_deg, area):
    return radiation_pressure(I, R, theta_deg) * area

function photon_based_pressure(I, wavelength, R, theta_deg):
    E_ph = h*c/wavelength
    p_ph = h/wavelength
    theta = deg2rad(theta_deg)
    Phi = I*cos(theta)/E_ph
    Delta_p_n = (1+R)*p_ph*cos(theta)
    return Phi * Delta_p_n

main:
    build pressure table for multiple I and R
    assert reflector pressure = 2 * absorber pressure at theta=0
    assert oblique pressure follows cos^2(theta)
    assert continuum pressure == photon-based pressure
    assert absorber-normal pressure equals I/c
    compute solar-sail pressure, force, acceleration
    print diagnostics
```

## R09

**实现说明（对应 demo.py）**  
1. `SailCase`：封装太阳帆场景参数。  
2. `_validate_common_inputs(...)`：统一输入约束检查。  
3. `radiation_pressure(...)`：核心闭式压力公式（支持 NumPy 向量）。  
4. `radiation_force(...)`：压力转力。  
5. `photon_energy(...)`、`photon_flux_on_surface(...)`、`photon_based_pressure(...)`：独立光子链路。  
6. `build_pressure_table(...)`：生成多组参数对照表（`pandas.DataFrame`）。  
7. `main()`：执行断言、打印表格与太阳帆数值结果。

## R10

**运行方式**

```bash
uv run python Algorithms/物理-电动力学-0173-辐射压力_(Radiation_Pressure)/demo.py
```

预期：脚本直接输出若干数值与检查状态；若关系不成立，`numpy.testing.assert_allclose` 将抛错并非零退出。

## R11

**复杂度分析**  
设压力表中强度点数为 `N`、反射率点数为 `M`。

1. 时间复杂度：`O(N*M)`（逐组合计算闭式公式）。  
2. 空间复杂度：`O(N*M)`（保存输出表）。

核心公式本身为 `O(1)`，无迭代求解或矩阵分解。

## R12

**数值稳定性与单位检查**  
1. 全程 SI 单位：`I(W/m^2)`, `p(Pa)`, `F(N)`, `A(m^2)`。  
2. 计算仅包含乘除与三角函数，条件数良好。  
3. `theta` 接近 `90 deg` 时 `cos(theta)` 很小，结果应自然趋近 0；代码允许该极限。  
4. 断言采用严格相对误差阈值（`1e-12`），验证解析关系。

## R13

**验证策略**  
1. **吸收/反射比验证**：`R=1` 与 `R=0` 在正入射下满足倍数关系 2。  
2. **角度缩放验证**：同一 `I,R` 下比较 `theta=0` 与 `theta=60`，应满足 `cos^2`。  
3. **双模型一致性验证**：连续介质公式结果应与光子动量法一致。  
4. **能量密度关系验证**：吸收体正入射 `p = I/c`。

## R14

**边界与局限**  
1. 仅覆盖平面波照射平面表面的法向分量。  
2. 未建模粗糙表面、漫反射 BRDF 与偏振依赖。  
3. 未考虑介质内部传播导致的折射/透射动量项。  
4. 未处理脉冲激光的瞬态峰值和热耦合力学反馈。

## R15

**可扩展方向**  
1. 将反射率扩展为波长函数 `R(lambda)` 并做谱积分。  
2. 加入透射率 `T`，使用一般式 `A_abs + R + T = 1`。  
3. 对接轨道动力学，计算长期太阳帆轨道漂移。  
4. 增加随机角度分布，估计姿态抖动下平均推力。

## R16

**工程化检查清单**  
1. `README.md` 与 `demo.py` 不含待填充占位符。  
2. `demo.py` 可非交互运行。  
3. 关键物理关系均由断言覆盖。  
4. 依赖为轻量科学栈（NumPy/SciPy/Pandas）。  
5. 所有改动仅位于本题所属目录。

## R17

**参考资料**  
1. D. J. Griffiths, *Introduction to Electrodynamics*, 4th ed.  
2. J. D. Jackson, *Classical Electrodynamics*, 3rd ed.  
3. Solar sail radiation pressure basic relations in classical EM momentum flux formulation.

## R18

**源码级算法流程拆解（3-10步）**  
1. `main()` 初始化强度数组、反射率数组和场景参数，准备批量计算。  
2. 调用 `build_pressure_table`，内部循环 `R` 并向量化调用 `radiation_pressure`，得到 `(I, R, theta, p)` 表。  
3. `radiation_pressure` 先做参数合法性检查，再按 `p=(I/c)(1+R)cos^2(theta)` 返回压力。  
4. `main()` 用 `assert_allclose` 验证 `R=1` 时压力是 `R=0` 的两倍。  
5. `main()` 验证同一 `I,R` 下斜入射压力与 `cos^2(theta)` 比例一致。  
6. 调用 `photon_based_pressure`，其内部依次计算光子能量、光子通量、单光子法向动量改变量，再组合为压力。  
7. `main()` 比较连续介质公式结果与光子链路结果，确保两条独立推导一致。  
8. 使用 `radiation_force` 计算太阳帆样例中的力与加速度并打印，形成可审计输出。

第三方库未被当作黑盒求解器：

- `numpy` 仅用于数组与基础数值运算；
- `scipy.constants` 仅提供物理常数；
- `pandas` 仅用于结果表格展示；
- 压力公式、校验逻辑和光子链路均在源码中显式实现。

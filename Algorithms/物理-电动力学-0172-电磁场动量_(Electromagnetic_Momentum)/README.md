# 电磁场动量 (Electromagnetic Momentum)

- UID: `PHYS-0171`
- 学科: `物理`
- 分类: `电动力学`
- 源序号: `172`
- 目标目录: `Algorithms/物理-电动力学-0172-电磁场动量_(Electromagnetic_Momentum)`

## R01

**问题定义**  
给定真空中的电场 `E(x)` 与磁场 `B(x)`，计算电磁场动量密度
\[
\mathbf{g}(x) = \epsilon_0\,\mathbf{E}(x)\times\mathbf{B}(x),
\]
并数值积分得到总电磁动量
\[
\mathbf{P}_{\mathrm{em}}=\int_V \mathbf{g}\,dV.
\]
本题要求用可运行最小脚本验证：
1. 局部关系 `g = S/c^2`；
2. 单向波包满足 `P_em = U/c`；
3. 对称反向波包总动量近零但总能量非零。

## R02

**物理背景**  
电磁场携带能量和动量。真空中，Poynting 向量
\[
\mathbf{S}=\frac{1}{\mu_0}(\mathbf{E}\times\mathbf{B})
\]
表示能流密度，而电磁动量密度与其满足
\[
\mathbf{g}=\frac{\mathbf{S}}{c^2}=\epsilon_0(\mathbf{E}\times\mathbf{B}).
\]
因此，辐射压力、光压推进、光与物质作用中的冲量传递，都可通过 `g` 或 `S` 等价描述。

## R03

**数学模型**  
在真空中采用以下点态量：

1. 能量密度：
\[
u = \frac{1}{2}\left(\epsilon_0|\mathbf{E}|^2 + \frac{|\mathbf{B}|^2}{\mu_0}\right)
\]
2. Poynting 向量：
\[
\mathbf{S}=\frac{1}{\mu_0}(\mathbf{E}\times\mathbf{B})
\]
3. 动量密度：
\[
\mathbf{g}=\epsilon_0(\mathbf{E}\times\mathbf{B})=\frac{\mathbf{S}}{c^2}
\]
4. 体积分量（MVP 用一维网格 + 单位截面积）：
\[
U=\int u\,dx,\qquad \mathbf{P}_{\mathrm{em}}=\int \mathbf{g}\,dx.
\]

## R04

**MVP 输入/输出定义**  
`demo.py` 无交互输入，内部固定构造两类场：

1. 单个高斯包络平面波（沿 `+x` 传播）；
2. 对称反向传播双波包（`+x` 与 `-x`）。

输出：
1. 每个场景的总能量 `U_total`；
2. `x` 方向总动量 `P_x`；
3. `U/c` 与 `P_x` 的比值；
4. `max|g - S/c^2|` 一致性误差；
5. 断言校验结果（失败即抛异常）。

## R05

**建模假设**  
1. 介质取真空，使用常数 `epsilon_0`, `mu_0`, `c`。  
2. 波包取解析构造，不求解完整 Maxwell PDE 时间推进。  
3. 横向截面积固定为 1 m²（因此积分量单位为 J/m²、N·s/m²）。  
4. 忽略介质色散、耗散、边界反射与量子修正。  
5. 采用双精度浮点数进行离散积分。

## R06

**关键公式与约束**  
1. 高斯包络载波：
\[
E_y(x,t)=E_0\exp\!\left(-\frac{\xi^2}{2\sigma^2}\right)\cos(k\xi),\quad
\xi=x-x_0-sct,
\]
其中 `s=+1/-1` 对应传播方向。  
2. 对应磁场：
\[
B_z = s\,\frac{E_y}{c}.
\]
3. 数值约束：`direction ∈ {-1, +1}`，场数组形状必须为 `(..., 3)`。  
4. 守恒关系检查：
   - 单向波包 `P_x ≈ U/c`；
   - 对称反向波包 `P_x ≈ 0`。

## R07

**算法思路**  
1. 构造离散空间网格 `x`。  
2. 生成前向高斯平面波 `E_fwd, B_fwd`。  
3. 生成后向高斯平面波 `E_bwd, B_bwd` 并线性叠加。  
4. 分别计算 `u`, `S`, `g` 的逐点分布。  
5. 用梯形积分得到 `U_total` 与 `P_total`。  
6. 汇总为 DataFrame 并执行物理断言。  
7. 打印可审计结果表。

## R08

**伪代码**

```text
def build_wave_packet(x, t, e0, wavelength, sigma, x0, direction):
    xi = x - x0 - direction * c * t
    Ey = e0 * exp(-xi^2/(2*sigma^2)) * cos(k*xi)
    Bz = direction * Ey / c
    return E, B

def evaluate_case(E, B):
    u = 0.5 * (eps0*|E|^2 + |B|^2/mu0)
    S = cross(E, B) / mu0
    g = eps0 * cross(E, B)
    U_total = integrate(u, x)
    P_total = integrate(g, x)
    return metrics

# case A
E_fwd, B_fwd = build_wave_packet(..., direction=+1)
# case B
E_bwd, B_bwd = build_wave_packet(..., direction=-1)
E_pair = E_fwd + E_bwd
B_pair = B_fwd + B_bwd

metrics_fwd = evaluate_case(E_fwd, B_fwd)
metrics_pair = evaluate_case(E_pair, B_pair)

assert P_fwd_x ~= U_fwd/c
assert g ~= S/c^2
assert P_pair_x ~= 0
print(metrics)
```

## R09

**实现说明（对应 demo.py）**  
1. `VacuumMedium`：封装真空参数 `epsilon`, `mu`。  
2. `electromagnetic_energy_density(...)`：计算 `u`。  
3. `poynting_vector(...)`：计算 `S`。  
4. `electromagnetic_momentum_density(...)`：计算 `g = epsilon * (E×B)`。  
5. `build_gaussian_plane_wave(...)`：按方向生成 `Ey/Bz` 波包。  
6. `integrate_scalar_over_x(...)` 与 `integrate_vector_over_x(...)`：梯形积分。  
7. `summarize_case(...)`：汇总每个场景的 `U、P、误差指标`。  
8. `main()`：构造两类场景、断言验证、输出表格。

## R10

**运行方式**

```bash
uv run python Algorithms/物理-电动力学-0172-电磁场动量_(Electromagnetic_Momentum)/demo.py
```

脚本无需交互输入；若物理约束不满足，程序会通过断言非零退出。

## R11

**复杂度分析**  
设网格点数为 `N`。

1. 单场景逐点计算 `u,S,g` 为 `O(N)`；  
2. 数值积分为 `O(N)`；  
3. 两个场景总时间复杂度 `O(N)`（常数倍）；  
4. 空间复杂度 `O(N)`（存储向量场与密度数组）。

## R12

**数值稳定性与单位检查**  
1. SI 单位：`E(V/m)`, `B(T)`, `u(J/m^3)`, `g(N·s/m^3)`。  
2. 采用 `scipy.constants` 避免手写常数误差。  
3. 积分区间选取远大于 `sigma`，降低边界截断误差。  
4. 对 `P_x = U/c` 采用相对误差断言，对 `P_pair ≈ 0` 使用绝对误差断言。

## R13

**验证策略**  
1. **局部一致性**：检查 `max|g - S/c^2|` 是否接近 0。  
2. **单向波包积分关系**：验证 `P_x` 与 `U/c` 一致。  
3. **反向对消**：等幅反向波包应出现 `P_x ≈ 0`。  
4. **方向正确性**：横向动量分量应为 0（`g_y, g_z` 接近 0）。

## R14

**边界与局限**  
1. 仅覆盖真空，不涉及介质中 Abraham/Minkowski 形式差异。  
2. 波包由解析式直接构造，未包含离散 Maxwell 更新误差分析。  
3. 使用 1D 空间采样，未建模复杂 3D 边界与散射体。  
4. 未耦合力学方程，不直接输出物体受力位移演化。

## R15

**可扩展方向**  
1. 增加介质参数空间分布 `epsilon(x), mu(x)`。  
2. 接入 FDTD（Yee 网格）并验证离散动量守恒。  
3. 加入导体/介质边界，计算辐射压力与冲量交换。  
4. 输出时序数据到 CSV/Parquet，供后续统计分析。

## R16

**工程化检查清单**  
1. `README.md` 与 `demo.py` 不含 待填充占位符。  
2. `uv run python demo.py` 可一次成功运行。  
3. 关键物理关系有断言覆盖。  
4. 依赖轻量（NumPy + SciPy + Pandas），实现透明。  
5. 代码与文档改动仅位于本题专属目录。

## R17

**参考资料**  
1. D. J. Griffiths, *Introduction to Electrodynamics*, 4th ed.  
2. J. D. Jackson, *Classical Electrodynamics*, 3rd ed.  
3. R. P. Feynman et al., *The Feynman Lectures on Physics*, Vol. II (EM momentum and radiation pressure sections).

## R18

**源码级算法流程拆解（3-10步）**  
1. `main()` 初始化真空常数封装、空间网格和波包参数。  
2. 调用 `build_gaussian_plane_wave(..., direction=+1)` 构造前向波包 `E_fwd/B_fwd`。  
3. 调用同一函数但 `direction=-1` 构造后向波包，并与前向波包线性叠加得到 `E_pair/B_pair`。  
4. `summarize_case` 内部先调用 `electromagnetic_energy_density` 逐点计算 `u`。  
5. 同一函数中调用 `poynting_vector` 和 `electromagnetic_momentum_density`，分别得到 `S` 与 `g`。  
6. 使用 `integrate_scalar_over_x` 与 `integrate_vector_over_x` 对 `u`、`g` 做梯形积分，得到 `U_total` 与 `P_total`。  
7. 计算误差指标 `max|g-S/c^2|` 与横向动量分量上界，封装为场景统计字典。  
8. `main()` 对两场景执行断言：前向波包 `P_x≈U/c`、对称反向波包 `P_x≈0`，最后打印 DataFrame。

第三方库未被当作黑盒求解器：

- `numpy`：数组运算、向量叉积、积分网格；
- `scipy.constants`：提供 `c`, `epsilon_0`, `mu_0` 常数；
- `pandas`：仅用于输出汇总表；
- 电磁动量与能量计算逻辑全部在 `demo.py` 显式实现。

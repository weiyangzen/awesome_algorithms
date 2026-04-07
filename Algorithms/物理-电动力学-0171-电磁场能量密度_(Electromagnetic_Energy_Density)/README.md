# 电磁场能量密度 (Electromagnetic Energy Density)

- UID: `PHYS-0170`
- 学科: `物理`
- 分类: `电动力学`
- 源序号: `171`
- 目标目录: `Algorithms/物理-电动力学-0171-电磁场能量密度_(Electromagnetic_Energy_Density)`

## R01

**问题定义**  
给定某介质中的电场 \(\mathbf{E}\) 与磁场 \(\mathbf{B}\)，计算空间上每个点的电磁场能量密度 \(u\)（单位 J/m\(^3\)），并用一个可运行的最小示例验证公式在典型场景（平面波、静态电场）下的正确性。

## R02

**物理背景**  
电磁场储能由电场储能和磁场储能两部分组成。在线性、各向同性介质中，能量密度写为
\[
u = \frac{1}{2}\left(\epsilon |\mathbf{E}|^2 + \frac{|\mathbf{B}|^2}{\mu}\right).
\]
其中 \(\epsilon\) 为介电常数，\(\mu\) 为磁导率。  
在真空平面波中，电场与磁场满足 \(B = E/c\)，且电、磁能量密度瞬时相等。

## R03

**数学模型**  
1. 向量场输入：\(\mathbf{E}(x), \mathbf{B}(x)\in\mathbb{R}^3\)。  
2. 点态能量密度：
\[
u(x)=\frac{1}{2}\left(\epsilon \,\mathbf{E}(x)\cdot \mathbf{E}(x)+\frac{\mathbf{B}(x)\cdot \mathbf{B}(x)}{\mu}\right).
\]
3. 真空常数：\(\epsilon_0=8.8541878128\times10^{-12}\), \(\mu_0=1.25663706212\times10^{-6}\), \(c=1/\sqrt{\epsilon_0\mu_0}\)。

## R04

**MVP 输入/输出定义**  
`demo.py` 不需要交互输入，内部构造两组测试场：  
1. 真空一维平面波（沿 +x 传播，\(E_y\)、\(B_z\)）。  
2. 理想平行板电容器近似（均匀静电场，\(B=0\)）。  

输出：  
1. 数值均值能量密度与理论值对比。  
2. 电磁波中电能与磁能平衡误差。  
3. Poynting 关系 \(S_x\approx c u\) 的误差。  
4. 静态电场能量密度与理论值对比。

## R05

**建模假设**  
1. 介质为线性、各向同性、均匀介质（MVP 取真空）。  
2. 不考虑色散、损耗、非线性效应。  
3. 平面波采用解析表达式，不做 FDTD 时域迭代。  
4. 数值验证以双精度浮点计算。

## R06

**关键公式与物理约束**  
1. 电磁能量密度：
\[
u = u_E + u_B = \frac{1}{2}\epsilon |\mathbf{E}|^2 + \frac{1}{2}\frac{|\mathbf{B}|^2}{\mu}.
\]
2. 平面波关系（真空）：
\[
\mathbf{B}=\frac{1}{c}\hat{\mathbf{k}}\times \mathbf{E}.
\]
3. Poynting 向量：
\[
\mathbf{S}=\frac{1}{\mu}\mathbf{E}\times\mathbf{B},\quad \text{平面波中}\;\mathbf{S}=cu\,\hat{\mathbf{k}}.
\]

## R07

**算法思路**  
1. 用 NumPy 构造离散空间网格 \(x_i\)。  
2. 在网格上生成解析电磁场样本 \(\mathbf{E}_i,\mathbf{B}_i\)。  
3. 对每个网格点计算 \(|\mathbf{E}_i|^2\)、\(|\mathbf{B}_i|^2\)。  
4. 代入能量密度公式得到 \(u_i\)。  
5. 汇总统计（均值、最大误差、积分）并与理论值比对。  
6. 通过断言确保结果处在容许误差内。

## R08

**伪代码**

```text
define constants eps0, mu0, c0
build x-grid

function energy_density(E, B, eps, mu):
    e2 = sum(E*E, axis=-1)
    b2 = sum(B*B, axis=-1)
    return 0.5 * (eps*e2 + b2/mu)

# case A: plane wave
Ew, Bw = build_plane_wave_fields(x, t, E0, f)
uw = energy_density(Ew, Bw, eps0, mu0)
check mean(uw) against 0.5*eps0*E0^2
check electric part ~= magnetic part
check Sx ~= c0*uw

# case B: static field
Es, Bs = build_static_fields(x, E0_static)
us = energy_density(Es, Bs, eps0, mu0)
check mean(us) against 0.5*eps0*E0_static^2

print metrics
```

## R09

**实现说明（对应 demo.py）**  
1. `LinearIsotropicMedium`：封装 \(\epsilon,\mu\)。  
2. `electromagnetic_energy_density(...)`：核心计算函数，支持形状 `(..., 3)` 的向量场批量计算。  
3. `build_plane_wave_fields(...)`：构造 \(E_y,B_z\) 同相正弦平面波。  
4. `build_capacitor_like_fields(...)`：构造均匀静电场。  
5. `main()`：执行两组场景、做数值断言并打印可读指标。

## R10

**运行方式**

```bash
uv run python Algorithms/物理-电动力学-0171-电磁场能量密度_(Electromagnetic_Energy_Density)/demo.py
```

预期：程序直接结束并输出若干数值；若物理关系不成立会触发 `assert_allclose` 报错。

## R11

**复杂度分析**  
设网格点数量为 \(N\)。  
1. 时间复杂度：\(O(N)\)（逐点向量运算与统计）。  
2. 空间复杂度：\(O(N)\)（存储 E/B/u 数组）。  
无矩阵求逆或迭代优化过程。

## R12

**数值稳定性与单位检查**  
1. 使用 SI 单位：E(V/m)、B(T)、u(J/m\(^3\))。  
2. 全流程双精度浮点，避免手动单位换算误差。  
3. 平面波用解析关系 \(B=E/c\) 生成，可减少离散微分误差来源。  
4. 校验采用相对误差阈值，兼容有限采样误差。

## R13

**验证策略**  
1. **平面波均值验证**：\(\langle u\rangle\) 对比 \(\frac{1}{2}\epsilon_0 E_0^2\)。  
2. **能量分配验证**：检查 \(u_E\) 与 \(u_B\) 的逐点差值应接近 0。  
3. **能流关系验证**：检查 \(S_x\) 与 \(cu\) 的一致性。  
4. **静态场验证**：\(B=0\) 时应退化为 \(u=\frac{1}{2}\epsilon E^2\)。

## R14

**边界与局限**  
1. 未覆盖非线性介质（如 \(\epsilon(E)\)）。  
2. 未覆盖耗散介质中的复介电常数与时间平均功耗。  
3. 未做边界条件复杂结构（波导、谐振腔）仿真。  
4. 仅演示 1D 采样，不是通用 PDE 求解器。

## R15

**可扩展方向**  
1. 增加频域时均能量密度（复场相量形式）。  
2. 接入 FDTD 更新方程，验证 Poynting 定理守恒。  
3. 支持空间分布的 \(\epsilon(\mathbf{r}), \mu(\mathbf{r})\)。  
4. 增加 CSV/Parquet 输出，便于后续数据分析。

## R16

**工程化检查清单**  
1. `README.md` 与 `demo.py` 不含未填充占位符。  
2. 脚本可通过 `uv run python demo.py` 非交互执行。  
3. 关键物理关系有断言保障。  
4. 代码仅依赖 NumPy，环境要求低。  
5. 文件均位于本题目专属目录内。

## R17

**参考资料**  
1. D. J. Griffiths, *Introduction to Electrodynamics*, 4th ed.  
2. J. D. Jackson, *Classical Electrodynamics*, 3rd ed.  
3. 经典电动力学中关于 Poynting 向量与能量密度的标准推导。

## R18

**源码级算法流程拆解（3-10步）**  
1. 在 `main()` 中初始化介质参数（\(\epsilon_0,\mu_0\)）与空间网格 `x`。  
2. 调用 `build_plane_wave_fields`：按解析相位 \(kx-\omega t\) 生成 `E` 与 `B` 向量数组。  
3. 调用 `electromagnetic_energy_density`：先做逐点平方和，再按 \(u=\frac{1}{2}(\epsilon|E|^2+|B|^2/\mu)\) 计算 `u`。  
4. 在 `main()` 中单独计算 `u_E`、`u_B`，检查电磁波里两部分能量密度平衡。  
5. 计算 `S = (E×B)/mu`，并检验 `Sx` 与 `c*u` 的一致性。  
6. 构造静态场 `build_capacitor_like_fields`，复用同一能量密度函数得到 `u_static`。  
7. 对平面波与静态场分别执行 `np.testing.assert_allclose` 与理论值对齐。  
8. 打印关键指标（均值、误差、积分量）作为最小可审计输出。

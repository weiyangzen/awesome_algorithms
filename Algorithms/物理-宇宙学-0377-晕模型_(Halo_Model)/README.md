# 晕模型 (Halo Model)

- UID: `PHYS-0359`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `377`
- 目标目录: `Algorithms/物理-宇宙学-0377-晕模型_(Halo_Model)`

## R01

晕模型（Halo Model）把非线性物质分布分解为“晕内相关 + 晕间相关”两部分：`P(k)=P_1h(k)+P_2h(k)`。本 MVP 的目标是给出一个可运行、可检查、不过度黑盒化的 `z=0` 非线性物质功率谱计算流程，明确展示从线性谱 `P_lin(k)`、质量函数 `dn/dM`、晕偏置 `b(M)` 到最终 `P_total(k)` 的数值路径。

## R02

本实现采用最小但完整的物理闭环：

1. 背景宇宙学：平直 `LCDM`，`Omega_m=0.315, Omega_Lambda=0.685, h=0.674`
2. 线性谱：`P_lin(k)=A*k^{n_s}*T_BBKS(k)^2`，`n_s=0.965`
3. 归一化：通过 `sigma8=0.811` 反解振幅 `A`
4. 晕丰度：Sheth-Tormen 质量函数（比原始 PS 更贴近模拟）
5. 晕偏置：Sheth-Tormen 偏置公式
6. 晕内部结构：NFW 密度分布 + `c(M,z)` 幂律近似
7. 有限质量区间修正：对 `dn/dM` 与 `b(M)` 做乘性重标定，使质量矩与偏置质量矩在积分区间内满足归一化约束
8. 组合：`P_1h` 与 `P_2h` 叠加得到 `P_total`

## R03

`demo.py` 输入/输出约定（无交互）：

- 固定输入（脚本内设置）：
1. `k_sigma`：`1e-4` 到 `1e2` (`1/Mpc`)，用于 `sigma(M)` 积分
2. `masses`：`1e10` 到 `1e16` (`Msun`)，用于质量积分
3. `k_eval`：`10^-2.5` 到 `10^1.2` (`1/Mpc`)，用于最终谱输出
4. 红移：`z=0`

- 终端输出：
1. 归一化前后 `sigma8`（自检）
2. 有限质量区间的质量矩与偏置质量矩积分
3. `P_lin`, `P_1h`, `P_2h`, `P_total`, `Delta^2` 样例表

## R04

MVP 使用的核心公式：

1. 线性方差定义：
`sigma^2(R)=∫ dk [k^2 P(k) W^2(kR)]/(2*pi^2)`

2. 质量-尺度映射：
`M=(4/3)*pi*rho_m0*R^3`

3. 晕模型分解：
`P_1h(k)=∫ dM n(M) (M/rho_m0)^2 |u(k|M)|^2`
`P_2h(k)=[∫ dM n(M) b(M) (M/rho_m0) u(k|M)]^2 * P_lin(k)`

4. Sheth-Tormen 质量函数（以 `nu=delta_c/sigma`）：
`dn/dM=(rho_m0/M^2) * f(nu) * |dlnsigma/dlnM|`

5. NFW 傅里叶核：
`u(k|M)` 用 `Si/Ci` 的解析表达并对小 `k*r_s` 做极限保护，使 `u->1`

## R05

`demo.py` 主流程：

1. 设定宇宙学参数、`k` 网格与质量网格
2. 计算 BBKS 线性谱形状并按 `sigma8` 归一化
3. 数值积分得到 `sigma(M,z=0)`
4. 由 `sigma(M)` 得到 `dn/dM`、`b(M)`、`nu`
5. 对 `dn/dM` 与 `b(M)` 执行有限区间质量/偏置约束重标定
6. 由 `M` 计算 `R_vir`、`c(M)`、NFW 的 `u(k|M)`
7. 对质量积分得到 `P_1h(k)` 与 `I_2(k)`
8. 计算 `P_2h(k)=I_2(k)^2*P_lin(k)`，再得 `P_total`
9. 输出诊断量与功率谱样例表

## R06

正确性依据：

1. `sigma(R)` 与顶帽窗积分遵循标准线性理论定义
2. 线性谱幅度由 `sigma8` 锚定，避免任意缩放
3. `P_1h` 和 `P_2h` 明确对应晕模型两项，不依赖隐式库黑箱
4. 先输出原始质量矩与偏置质量矩，再执行重标定并输出重标定后积分，确保约束可审计
5. 输出低 `k` 处 `P_total/P_lin`，用于验证大尺度极限是否接近线性理论

## R07

复杂度（`N_M` 为质量点数，`N_k` 为 `sigma` 积分波数点数，`N_ke` 为输出波数点数）：

- `sigma(M)` 计算：时间 `O(N_M*N_k)`，空间 `O(N_M*N_k)`
- NFW `u(k|M)` 计算：时间 `O(N_M*N_ke)`，空间 `O(N_M*N_ke)`
- `P_1h/P_2h` 积分：时间 `O(N_M*N_ke)`

当前参数 `N_M=240, N_k=4096, N_ke=140` 在普通 CPU 上可快速运行。

## R08

数值稳定性处理：

1. 顶帽窗 `W(x)` 在 `x->0` 用泰勒展开，避免 `0/0`
2. `sigma`、`nu`、`a*nu^2` 都设下界，避免对数和幂运算奇异
3. NFW 傅里叶核中 `x`、`(1+c)x` 使用安全下界；`x<1e-4` 直接置 `u=1`
4. 所有积分使用 `scipy.integrate.simpson`，降低粗网格积分偏差
5. `q=k/(Gamma h)` 做下界裁剪，避免 BBKS 传输函数奇异点

## R09

单位体系：

1. 质量：`Msun`
2. 长度：`Mpc`
3. 波数：`1/Mpc`
4. 密度：`Msun/Mpc^3`
5. 功率谱：`Mpc^3`

在该体系下，`(M/rho_m0)` 具有体积量纲，`P_1h/P_2h` 与 `P_lin` 的量纲一致。

## R10

运行方式：

```bash
uv run python Algorithms/物理-宇宙学-0377-晕模型_(Halo_Model)/demo.py
```

或在该目录内：

```bash
uv run python demo.py
```

## R11

输出字段说明：

1. `P_lin`：线性理论物质功率谱
2. `P_1h`：同一晕内部相关贡献（小尺度主导）
3. `P_2h`：不同晕之间相关贡献（大尺度主导）
4. `P_total`：`P_1h+P_2h`
5. `Delta2_total = k^3 P_total/(2*pi^2)`：无量纲谱强度
6. `Raw integral n(M)M/rho_m dM`：重标定前质量矩
7. `Raw integral n(M)b(M)M/rho_m dM`：重标定前偏置质量矩
8. `Renorm integral n(M)M/rho_m dM`：重标定后质量矩（应接近 1）
9. `Renorm integral n(M)b(M)M/rho_m dM`：重标定后偏置质量矩（应接近 1）

## R12

可预期结果特征：

1. 小 `k` 端通常 `P_2h` 主导，经过质量/偏置重标定后 `P_total` 会更接近 `P_lin`
2. 大 `k` 端 `P_1h` 逐渐抬升并主导
3. `Delta2_total(k)` 随 `k` 增大先上升并进入非线性增强区
4. `nu(M)` 随 `M` 上升而增大（高质量晕更稀有）

## R13

模型边界与简化：

1. 线性谱使用 BBKS 近似，未显式建模 BAO 细节
2. 未区分不同 halo mass definition（如 `M200c`, `Mvir`）的严格转换
3. `c(M,z)` 用简化幂律，不含散度与环境依赖
4. 仅计算 `z=0`，未输出红移演化序列
5. 对有限质量积分区间引入了乘性重标定，属于工程化闭环修正而非第一性原理推导
6. 结果用于教学、流程验证与工程起点，不等价于高精度模拟拟合器（如 HMCode）

## R14

可能失败模式：

1. 质量区间过窄会让质量矩积分偏离 1，从而扭曲大尺度归一化
2. `k_sigma` 区间过窄会导致 `sigma(M)` 积分漏掉低/高频贡献
3. 质量网格过疏会让 `dlnsigma/dlnM` 数值噪声放大，造成 `dn/dM` 抖动
4. 若 `c(M)` 参数异常，NFW 傅里叶核会在中高 `k` 产生不合理形状
5. 将本参数直接用于非平直宇宙或极端红移会引入系统误差

## R15

最小测试建议：

1. 归一化测试：`sigma8(after normalization)` 应接近 `0.811`
2. 正值测试：`P_1h`, `P_2h`, `P_total` 应非负
3. 结构测试：`P_total` 在高 `k` 端相对 `P_lin` 应出现明显非线性增强
4. 诊断测试：重标定后 `Renorm integral n(M)M/rho_m dM`、`Renorm integral n(M)b(M)M/rho_m dM` 应接近 1
5. 极限测试：`k` 最小点处 `P_total/P_lin` 应接近 1 且不出现数量级异常

## R16

可扩展方向：

1. 用 CAMB/CLASS 线性谱替换 BBKS 近似
2. 把 ST 改为 Tinker 等质量函数并比较 `P_1h/P_2h` 差异
3. 引入 redshift grid，输出 `P(k,z)` 演化
4. 引入 concentration scatter 或 baryonic feedback 修正
5. 采用 FFTLog/Hankel 等方法加速多红移批量计算

## R17

应用场景：

1. 大尺度结构课程中讲解“线性谱 -> 非线性谱”的桥梁模型
2. 作为 N-body 仿真分析前的快速数量级估计器
3. 半解析星系形成流程中的暗物质背景模块
4. 复杂宇宙学管线中的回归测试 baseline（检查谱形与参数敏感性）

## R18

`demo.py` 源码级算法流（显式展开第三方调用）：

1. 在 `linear_power_unnormalized` 中逐点构造 `k^{n_s}` 与 BBKS `T(k)`，得到单位振幅线性谱形。
2. 在 `sigma_r` 中构造 `x=outer(R,k)`，先算顶帽窗 `W(x)`，再形成二维被积函数矩阵 `k^2 P(k)W^2/(2pi^2)`。
3. `scipy.integrate.simpson` 沿 `k` 轴做分段抛物线求积，得到每个 `R` 的 `sigma^2`，再开方得到 `sigma(M)`。
4. `normalize_linear_power_to_sigma8` 在 `R=8/h` 处测得 `sigma8_unit`，反解振幅 `A=(sigma8_target/sigma8_unit)^2` 并缩放 `P_lin`。
5. `sheth_tormen_mass_function_and_bias` 用 `np.gradient(log sigma, log M)` 求斜率项，代入 ST 的 `f(nu)` 和偏置公式，得到 `dn/dM` 与 `b(M)`。
6. `renormalize_mass_and_bias_constraints` 先做 `simpson` 得到原始质量矩与偏置质量矩，再对 `dn/dM`、`b(M)` 做乘性重标定以满足有限区间约束。
7. `nfw_u_of_km` 中为每个质量计算 `r_vir -> r_s -> x=k*r_s`，再调用 `scipy.special.sici` 计算 `Si/Ci`，按 NFW 解析式拼出 `u(k|M)`。
8. `halo_power_terms` 对质量执行两次 `simpson`：一次积分 `n(M)(M/rho)^2u^2` 得 `P_1h`，一次积分 `n(M)b(M)(M/rho)u` 得 `I_2`。
9. 主函数把 `P_2h=I_2^2*P_lin` 与 `P_1h` 相加，生成 `P_total` 与 `Delta^2`，并用 `pandas.DataFrame` 输出固定格式样例表与诊断量。

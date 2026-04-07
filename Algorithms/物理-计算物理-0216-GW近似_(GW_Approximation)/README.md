# GW近似 (GW Approximation)

- UID: `PHYS-0215`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `216`
- 目标目录: `Algorithms/物理-计算物理-0216-GW近似_(GW_Approximation)`

## R01

GW 近似是多体微扰理论里用于计算准粒子能级修正的核心方法，
其基本思想是把电子自能写成：

`Sigma = i G W`

其中 `G` 是单粒子格林函数，`W` 是经屏蔽后的库仑相互作用。和常见 DFT 本征值相比，GW 更直接面向“激发能/带隙”问题。

本条目实现一个可运行、可审计的 `G0W0` 最小原型：用两能级模型显式走通 `P0 -> W -> Sigma -> Z -> E_QP` 的完整链路。

## R02

MVP 任务定义：

1. 构造一个包含价带态/导带态的两能级 Kohn-Sham 起点；
2. 在实频网格上计算独立粒子极化 `P0(omega)`；
3. 通过 RPA 公式求屏蔽相互作用 `W(omega)`；
4. 数值积分得到相关自能 `Sigma_c(omega)`；
5. 加上静态交换项 `Sigma_x`，并线性化准粒子方程得到 `E_QP`；
6. 比较 KS 带隙与 QP 带隙并做阈值校验。

该模型是教学/算法验证性质，不是材料级第一性原理生产计算。

## R03

本实现采用的核心表达式：

- 独立粒子极化（单主跃迁 `v->c`）：
`P0(omega) = 2M^2 [ 1/(omega-Delta+i*eta) - 1/(omega+Delta-i*eta) ]`
- 屏蔽相互作用（RPA）：
`W(omega) = v / (1 - v P0(omega))`
- 相关部分：
`W_c(omega) = W(omega) - v`

其中：
- `Delta = eps_c - eps_v`
- `M` 是模型跃迁强度
- `eta` 是频域展宽参数

## R04

相关自能在实轴上离散为：

`Sigma_c(omega) = i/(2pi) * integral G0(omega + omega') * W_c(omega') d omega'`

总自能：

`Sigma(omega) = Sigma_x + Sigma_c(omega)`

准粒子线性化修正（`G0W0` 常用近似）：

- `Z = 1 / [1 - d ReSigma / domega |_(eps_KS)]`
- `E_QP = eps_KS + Z * [ReSigma(eps_KS) - v_xc]`

代码中通过中心差分计算导数，不使用黑盒电子结构库。

## R05

模型假设与边界：

- 两能级 toy model（一个占据价带态 + 一个非占据导带态）；
- `Sigma_x` 使用预设 Fock 矩阵元近似（`-U_vv`、`-U_cv`）；
- 频域有限区间 `[-omega_max, omega_max]` 上数值积分；
- 仅做 one-shot `G0W0`，不做自洽 `GW0`/`scGW`；
- 未包含完整矩阵元结构、k 点采样与实际材料基组细节。

## R06

`demo.py` 输入输出约定：

- 输入：全部参数内置在 `GWConfig`（无需交互）；
- 输出：
1. 各态的 `Sigma_x`、`ReSigma_c`、`Z`、`E_QP` 等表格；
2. `W(0)`、`W(0)/v`、`min|1-vP0|`、KS/QP 带隙摘要；
3. 阈值检查项与 `Validation: PASS/FAIL`。

脚本可直接执行，失败时以非零退出码终止。

## R07

算法主流程（高层）：

1. 读取 `GWConfig`，建立价带/导带态与频率网格；
2. 由 `Delta`、`M`、`eta` 计算 `P0(omega)`；
3. 计算 `W(omega)=v/(1-vP0)` 及 `W_c=W-v`；
4. 对每个态构建 `G0(omega+omega')`；
5. 用数值积分求 `Sigma_c(eps_KS)` 与邻近频点自能；
6. 用中心差分得到 `dReSigma/domega` 并求 `Z`；
7. 线性化更新得到 `E_QP`；
8. 汇总状态表、带隙和稳定性指标，执行验收阈值。

## R08

复杂度分析（`Nw` 为频率网格点数，`Ns` 为状态数，本 MVP 中 `Ns=2`）：

- 计算 `P0`、`W`：`O(Nw)`；
- 单次自能积分：`O(Nw)`；
- 每个态需要 `omega-h, omega, omega+h` 三次自能：`O(3Nw)`；
- 总体：`O(Ns * Nw)`，常数项小。

空间复杂度主要来自频率数组与复数核，为 `O(Nw)`。

## R09

数值稳定策略：

- 强制 `n_omega` 为奇数，保证 Simpson 积分可用；
- 显式检查 `min|1-vP0|`，避免屏蔽分母接近奇异；
- 使用有限展宽 `eta` 抑制实轴极点病态；
- `Z` 通过局部中心差分而非全局拟合，降低噪声传播；
- 在验收中约束 `|ImSigma|` 与 `Z` 取值区间。

## R10

MVP 技术栈：

- `numpy`：复数向量化、频率网格、代数操作
- `scipy.integrate.simpson`：实轴自能积分
- `pandas`：结果表格与摘要打印

没有调用“一键 GW”黑盒包；关键链路 `P0 -> W -> Sigma -> Z -> E_QP` 全部由源码函数显式实现。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0216-GW近似_(GW_Approximation)
uv run python demo.py
```

若阈值检查全部通过，末尾输出 `Validation: PASS`。

## R12

关键输出字段说明：

- `eps_KS`：起点 Kohn-Sham 能级
- `v_xc`：起点交换相关势期望值（双计数扣除）
- `Sigma_x`：静态交换自能
- `ReSigma_c`：相关自能实部
- `ReSigma_total` / `ImSigma_total`：总自能实部/虚部
- `dReSigma_domega`：自能实部频率导数
- `Z`：准粒子重整化因子
- `E_QP`：G0W0 修正后的准粒子能级
- `QP_minus_KS`：能级相对修正量
- `KS gap` / `QP gap`：修正前后带隙

## R13

demo 内置验收条件：

1. `min|1-vP0| > 5e-2`（屏蔽分母稳定）
2. `W(0)/v` 在 `(0,1)`（静态屏蔽合理）
3. 所有 `Z` 在 `(0, 1.2)`
4. `QP gap > KS gap`
5. 两个态 `|ImSigma_total| < 5e-2`
6. 价带下移（`E_QP < eps_KS`）
7. 导带上移（`E_QP > eps_KS`）

全部满足才输出 `Validation: PASS`。

## R14

当前实现局限：

- 两能级模型无法反映真实多能带/多 k 点结构；
- 仅 one-shot `G0W0`，未做自洽更新 `G` 或 `W`；
- 未包含顶点修正（vertex correction）；
- 实轴积分和展宽参数对数值有模型依赖；
- 自能矩阵在本实现里退化为标量态分辨，非完整矩阵元处理。

## R15

可扩展方向：

- 从两能级扩展到多能级矩阵形式 `Sigma_nm(omega)`；
- 引入等离子体极点模型（PPM）或虚频轴 + 解析延拓；
- 加入 `GW0`/`scGW` 迭代闭环；
- 与 DFT 网格/基组模块对接，读取真实 `eps_nk` 与矩阵元；
- 接入 Bethe-Salpeter Equation（BSE）计算激子光谱。

## R16

典型应用语境：

- 半导体与绝缘体带隙修正（DFT 常低估带隙）；
- 准粒子能谱与光电子实验（ARPES）对比；
- 作为多体电子结构工作流中 DFT 后处理步骤；
- 教学环境下验证 `G0W0` 的最小可运行算法链。

## R17

与相关方法简要对比：

- DFT-LDA/GGA：基态总能高效，但激发能级常偏差较大；
- HF：交换处理好但缺乏动态屏蔽，常高估带隙；
- GW：通过动态屏蔽 `W(omega)` 引入频率依赖自能，通常更可靠地描述准粒子能级。

本条目选择最小 `G0W0`，是为了在可控复杂度下保留 GW 的核心物理机制与可审计计算流程。

## R18

`demo.py` 的源码级算法流（9 步）：

1. `GWConfig` 定义 KS 能级、`v_xc`、Fock 交换矩阵元和频域积分参数；`states` 产出价带/导带 `StateSpec`。  
2. `build_frequency_grid` 构建奇数点实频网格，确保 `simpson` 积分前提成立。  
3. `independent_particle_polarization` 按单跃迁公式计算 `P0(omega)`。  
4. `screened_interaction` 计算 `W(omega)=v/(1-vP0)`，并检查分母 `1-vP0` 的最小模避免奇异。  
5. 通过 `w_corr = W - v` 得到相关核；`correlation_self_energy` 对 `G0*(W-v)` 做实轴积分得到 `Sigma_c`。  
6. `total_self_energy` 把静态 `Sigma_x` 与 `Sigma_c` 相加，得到总自能 `Sigma(omega)`。  
7. `quasiparticle_update` 在 `eps_KS-h, eps_KS, eps_KS+h` 三点评估自能，中心差分求 `dReSigma/domega` 并得到 `Z`。  
8. 同一函数内执行线性化更新：`E_QP = eps_KS + Z*(ReSigma(eps_KS)-v_xc)`，输出每个态的修正明细。  
9. `main` 汇总状态表和带隙摘要，执行 7 条阈值检查，打印 `Validation: PASS/FAIL`。

说明：唯一第三方积分调用是 `scipy.integrate.simpson`，其前后物理量构造和公式映射在源码中逐步展开，并非黑盒求解。

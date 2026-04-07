# 重整化群 (Renormalization Group)

- UID: `PHYS-0297`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `300`
- 目标目录: `Algorithms/物理-统计力学-0300-重整化群_(Renormalization_Group)`

## R01

重整化群（Renormalization Group, RG）研究的是“系统在尺度变换下参数如何流动”。
在统计力学中，RG 解释了临界现象中的普适性：不同微观模型在粗粒化后，会流向相同的固定点，从而共享相同临界指数。

本条目采用二维 Ising 模型做一个可运行、可追踪的最小 RG 演示。

## R02

MVP 目标不是复现高精度文献指数，而是把 RG 的关键机制串起来并可直接运行验证：

- 构造经验 RG 映射 `K -> K'`（`K=J/(k_B T)`）；
- 观察 `K' - K` 的变号，定位非平凡固定点 `K*`；
- 在固定点附近线性化，给出 `lambda = dK'/dK` 与 `nu ≈ ln(b)/ln(lambda)`（`b=2`）。

## R03

物理模型（二维 Ising，无外场，`J=1, k_B=1`）：

- 自旋：`s_i ∈ {+1,-1}`
- 哈密顿量：`H = - Σ_<ij> s_i s_j`
- 周期边界条件
- 耦合常数：`K = 1/T`

用于参考的二维 Ising 精确临界耦合：

- `Kc = 0.5 * ln(1 + sqrt(2)) ≈ 0.4406868`

## R04

本实现的 RG 变换由两步组成：

1. 实空间粗粒化：`2x2` block-spin 多数表决（平票随机打破）。
2. 逆问题估计：对粗粒化自旋场做 Ising 伪似然（pseudolikelihood）拟合，得到等效耦合 `K'`。

因此每个输入耦合 `K` 都会生成一个“观测到的” `K'`，从而形成离散 RG 映射数据表。

## R05

为什么用伪似然而不是直接黑盒库：

- 条件分布有显式形式：
  `P(s_i | nn) = 1 / (1 + exp(-2 K s_i h_i))`, `h_i = Σ_nn s_j`
- 令 `z_i = s_i h_i`，则目标函数可写为：
  `NLL(K) = mean(log(1 + exp(-2 K z_i)))`
- 只需一维标量优化即可得到 `K'`，物理意义和数值过程都透明。

## R06

数值流程（端到端）：

1. 对一组 `K` 网格逐点运行 Metropolis 采样，收集细网格快照。
2. 对每个快照执行 `2x2` block-spin 变换，得到粗网格快照。
3. 在粗网格上提取所有局部变量 `z=s_i*sum_nn s_j`。
4. 最小化伪似然负对数，求出 `K'`。
5. 汇总成 RG 映射表：`K, K_prime, delta_K=K'-K`。
6. 用 `delta_K` 变号区间线性插值估计 `K*`。
7. 用局部斜率估计 `lambda` 与 `nu`。
8. 用插值映射迭代 `K_{n+1}=R(K_n)`，展示流向高温/低温固定点的趋势。

## R07

复杂度估计（设 `L` 为边长，`M` 为 K 网格数）：

- 单个 Metropolis sweep：`O(L^2)`。
- 单个 `K` 的采样：`O((warmup + snapshots*stride) * L^2)`。
- 全部扫描：乘以 `M`。
- 伪似然拟合是一维优化，单次评估 `O(samples * (L/2)^2)`，总体相对采样成本较小。

空间复杂度主要来自快照存储：`O(snapshots * L^2)`。

## R08

`demo.py` 结构：

- `RGConfig`：统一设置晶格、K 网格、采样长度、随机种子；
- `Ising2D`：模型状态与 Metropolis 更新；
- `collect_snapshots`：在给定 `K` 下采样快照；
- `block_majority_2x2` / `coarse_grain_snapshots`：实空间粗粒化；
- `build_local_alignment_values`：构造伪似然变量 `z`；
- `estimate_effective_coupling_from_pseudolikelihood`：求 `K'`；
- `run_rg_map`：批量构建 `K -> K'` 表；
- `estimate_nontrivial_fixed_point`：估计 `K*、lambda、nu`；
- `iterate_flow_by_interpolation`：展示 RG 迭代流轨迹。

## R09

默认参数（偏重可运行与趋势清晰）：

- `lattice_size = 20`
- `coupling_values = (0.24, 0.30, 0.36, 0.42, 0.46, 0.52, 0.62, 0.76)`
- `warmup_sweeps = 120`
- `snapshots_per_k = 48`
- `sweeps_between_snapshots = 4`
- `seed = 20260407`

这组设置通常可以得到“低 K 向下流、高 K 向上流、临界附近变号”的可读结果。

## R10

伪代码：

```text
for K in coupling_grid:
    fine_snapshots = MetropolisSample(K)
    coarse_snapshots = BlockMajority2x2(fine_snapshots)
    z_values = flatten(s_i * sum_nn s_j over all coarse sites/snapshots)
    K_prime = argmin_K mean(log(1 + exp(-2*K*z_values)))
    record (K, K_prime, K_prime-K)

find crossing interval where (K_prime-K) changes sign
linearly interpolate K* in this interval
estimate lambda ~ dK'/dK and nu ~ ln(2)/ln(lambda)
iterate K_{n+1}=R(K_n) by interpolation to show RG flows
```

## R11

可复现性策略：

- 顶层固定 `seed`；
- 每个 `K` 使用确定性偏移种子；
- tie-break 随机也使用确定性 RNG；
- 全流程无交互输入，重复运行可得到统计上接近且路径一致的结果。

## R12

输出包含：

1. RG 映射表：
   - `K`, `K_prime`, `delta_K`
   - `abs_m_fine`, `abs_m_coarse`, `nll`
2. 固定点摘要：
   - `K*`, `lambda`, `nu`
   - 与 `Kc_exact` 的差值
3. 流轨迹：
   - 若干 `K0` 的迭代序列 `K0 -> K1 -> ...`
4. 自检：
   - 单调性检查
   - 相方向检查（低 K 下流，高 K 上流）

## R13

数值注意事项：

- 本实现属于教学级近似 RG，`K*` 与 `nu` 仅作趋势展示；
- 采样长度不足会导致 `K'` 噪声增大；
- block-majority 与伪似然逆映射本身会引入近似误差；
- 若需要更稳定的临界指数，应增加 `L`、快照数，并做误差条估计（bootstrap/binning）。

## R14

方法对比：

- 当前方案（MC + 实空间 block + 伪似然逆推）
  - 优点：机制直观、实现短、可运行验证 RG 流
  - 缺点：系统误差较大，精度有限
- 解析/半解析方案（如 Migdal-Kadanoff）
  - 优点：更快、公式明确
  - 缺点：近似假设更强，不直接使用采样数据
- 高精度方案（Wolff + 大尺寸 FSS）
  - 优点：指数估计更准
  - 缺点：实现和计算成本更高

## R15

可扩展方向：

1. 把 Metropolis 换成 Wolff/Swendsen-Wang 以降低临界慢化；
2. 对 `K'` 做多副本估计并输出置信区间；
3. 引入外场项和多参数 RG 流（`K,h`）；
4. 比较不同 block 规则（多数表决、加权表决）；
5. 在固定点附近做更细 K 网格以改进 `lambda` 与 `nu` 稳定性。

## R16

运行方式（无交互）：

```bash
uv run python Algorithms/物理-统计力学-0300-重整化群_(Renormalization_Group)/demo.py
```

若当前目录已在该算法目录下：

```bash
uv run python demo.py
```

## R17

交付核对：

- `README.md`：`R01-R18` 已完整填充；
- `demo.py`：已替换占位符并可直接运行；
- `meta.json`：任务元数据（UID/学科/分类/源序号/目录）保持一致；
- 仅修改本算法自有目录文件。

## R18

`demo.py` 源码级算法流（8 步，含第三方函数调用拆解）：

1. `main` 创建 `RGConfig`，调用 `run_rg_map` 对整组 `K` 扫描构建经验映射。  
2. `run_rg_map` 对每个 `K` 调 `collect_snapshots`，执行 Metropolis 预热和采样，得到细网格快照。  
3. `coarse_grain_snapshots` 对每个快照调用 `block_majority_2x2`，把 `LxL` 压缩为 `(L/2)x(L/2)`。  
4. `build_local_alignment_values` 计算全部粗网格局部量 `z=s_i*sum_nn s_j`，作为伪似然输入。  
5. `estimate_effective_coupling_from_pseudolikelihood` 定义 `NLL(K)=mean(logaddexp(0,-2Kz))`，并调用 `scipy.optimize.minimize_scalar(..., method="bounded")` 在 `[k_min_fit,k_max_fit]` 内做一维有界最小化得到 `K'`。  
6. `run_rg_map` 汇总 `K, K_prime, delta_K, nll` 等列形成 DataFrame，即离散 RG 映射。  
7. `estimate_nontrivial_fixed_point` 在 `delta_K` 变号区间线性插值求 `K*`，并用相邻点斜率近似 `lambda=dK'/dK`，再由 `nu=ln(b)/ln(lambda)` 给出临界指数估计。  
8. `iterate_flow_by_interpolation` 用 `np.interp` 近似 `R(K)` 并迭代 `K_{n+1}=R(K_n)`；`main` 输出映射表、固定点和流轨迹，以及两个自检结果。  

上述流程没有把 RG 交给黑盒函数：粗粒化规则、逆耦合目标函数、固定点估计和流迭代都在源码中显式展开。
